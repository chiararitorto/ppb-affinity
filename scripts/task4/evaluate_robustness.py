# ======================================================================================
# SCRIPT: evaluate_robustness.py (Task 4 - verifica di robustezza dei risultati)
#
# Ripete la cross-validation della CNN 2D (stesso protocollo di train_final.py: split
# interno 85/15 per il checkpoint, Z-score sul solo train interno, augmentation) su
# più seed e con due tipi di fold:
#   - "random":   KFold(5, shuffle=True), come in train_final.py;
#   - "homology": GroupKFold(5) sui gruppi di omologia di sequenza in
#                 data/homology_groups.csv (nessun gruppo condiviso fra train e test).
# Sugli STESSI fold valuta anche tre baseline semplici:
#   - media del training set;
#   - ridge sul numero di residui di interfaccia (ligando + recettore);
#   - ridge su dimensione + composizione dell'interfaccia (12 feature).
#
# Uso (dalla radice del repo):
#   PYTHONPATH=scripts/task4 python scripts/task4/evaluate_robustness.py --seeds 5
# OUTPUT: outputs/task4_final/robustness_runs.csv (una riga per seed/split/modello)
#         outputs/task4_final/robustness_summary.csv (media e dev. std sui seed)
# ======================================================================================

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold, GroupKFold, train_test_split
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

from models import ProteinInterfaceDataset, InterfaceCNN, compute_channel_stats

CSV_PATH = "data/affinity_dataset.csv"
MAPS_DIR = "outputs/task3/interface_maps/"
TASK1_DIR = "outputs/task1/"
GROUPS_PATH = "data/homology_groups.csv"
OUTPUT_DIR = "outputs/task4_final/"

BATCH_SIZE = 8
LEARNING_RATE = 0.001
WEIGHT_DECAY = 2e-3
EPOCHS = 100
NUM_FOLDS = 5
INNER_VAL_FRACTION = 0.15

HYDRO = {'ILE': 4.5, 'VAL': 4.2, 'LEU': 3.8, 'PHE': 2.8, 'CYS': 2.5, 'MET': 1.9, 'ALA': 1.8,
         'GLY': -0.4, 'THR': -0.7, 'SER': -0.8, 'TRP': -0.9, 'TYR': -1.3, 'PRO': -1.6,
         'HIS': -3.2, 'GLU': -3.5, 'GLN': -3.5, 'ASP': -3.5, 'ASN': -3.5, 'LYS': -3.9, 'ARG': -4.5}
CHARGE = {'ARG': 1.0, 'LYS': 1.0, 'HIS': 0.1, 'ASP': -1.0, 'GLU': -1.0}
AROMATIC = {'PHE', 'TYR', 'TRP', 'HIS'}

device = torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def augment(batch):
    out = batch.clone()
    for i in range(out.size(0)):
        if random.random() > 0.5:
            out[i] = torch.flip(out[i], dims=[2])
        if random.random() > 0.5:
            out[i] = torch.flip(out[i], dims=[1])
        k = random.randint(0, 3)
        if k > 0:
            out[i] = torch.rot90(out[i], k=k, dims=[1, 2])
    return out


def train_cnn_fold(dataset, train_idx, test_idx, seed):
    """Un fold esterno con lo stesso protocollo di train_final.py. Restituisce le predizioni sul test."""
    inner_train, inner_val = train_test_split(train_idx, test_size=INNER_VAL_FRACTION, random_state=seed)
    means, stds = compute_channel_stats(dataset, inner_train)
    dataset.standardize_with(means, stds)
    train_loader = DataLoader(Subset(dataset, inner_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(Subset(dataset, inner_val), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=BATCH_SIZE, shuffle=False)

    model = InterfaceCNN().to(device)
    crit = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=7)
    best, best_state = float('inf'), None
    for _ in range(EPOCHS):
        model.train()
        for x, y in train_loader:
            opt.zero_grad()
            loss = crit(model(augment(x)), y)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = sum(crit(model(x), y).item() * x.size(0) for x, y in val_loader) / len(inner_val)
        sched.step(vl)
        if vl < best:
            best = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(x).numpy().ravel() for x, _ in test_loader])


def record_folder(sample):
    return os.path.basename(sample['path']).replace('_tensor.npy', '')


def interface_features(folder):
    """Dimensione e composizione dell'interfaccia dai file della Task 1."""
    feats = {}
    sides = {}
    for side in ('ligand', 'receptor'):
        res = pd.read_csv(os.path.join(TASK1_DIR, folder, f"{side}_interface_residues.csv"))
        names = res['residue_name'].astype(str).str.upper().str[:3]
        sides[side] = names
        feats[f'n_{side}'] = len(names)
        feats[f'hydro_{side}'] = names.map(HYDRO).fillna(0).mean()
        feats[f'charge_{side}'] = names.map(CHARGE).fillna(0).sum()
        feats[f'frac_charged_{side}'] = names.isin(['ARG', 'LYS', 'ASP', 'GLU']).mean()
        feats[f'frac_aromatic_{side}'] = names.isin(AROMATIC).mean()
    feats['charge_product'] = feats['charge_ligand'] * feats['charge_receptor']
    feats['n_total'] = feats['n_ligand'] + feats['n_receptor']
    return feats


def fit_predict_ridge(X, y, train_idx, test_idx):
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))
    model.fit(X[train_idx], y[train_idx])
    return model.predict(X[test_idx])


def metrics(pred, y):
    r = pearsonr(pred, y)[0] if np.std(pred) > 0 else np.nan
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
    return r, rmse, r2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--splits', nargs='+', default=['random', 'homology'])
    args = ap.parse_args()

    dataset = ProteinInterfaceDataset(csv_affinity_path=CSV_PATH, maps_dir=MAPS_DIR)
    n = len(dataset)
    folders = [record_folder(s) for s in dataset.valid_samples]
    y = np.array([s['target'] for s in dataset.valid_samples])

    groups_df = pd.read_csv(GROUPS_PATH).set_index('record_id')
    groups = groups_df.loc[folders, 'homology_group'].values

    feat_df = pd.DataFrame([interface_features(f) for f in folders])
    X_size = feat_df[['n_total']].values
    comp_cols = ['n_ligand', 'n_receptor', 'hydro_ligand', 'hydro_receptor', 'charge_ligand',
                 'charge_receptor', 'charge_product', 'frac_charged_ligand', 'frac_charged_receptor',
                 'frac_aromatic_ligand', 'frac_aromatic_receptor', 'n_total']
    X_comp = feat_df[comp_cols].values

    print(f"Complessi: {n} | gruppi di omologia: {len(set(groups))} | gruppo maggiore: "
          f"{pd.Series(groups).value_counts().iloc[0]}")

    rows = []
    for split in args.splits:
        for seed in range(args.seeds):
            set_seed(seed)
            if split == 'random':
                folds = list(KFold(NUM_FOLDS, shuffle=True, random_state=seed).split(np.arange(n)))
            else:
                folds = list(GroupKFold(NUM_FOLDS, shuffle=True, random_state=seed).split(np.arange(n), groups=groups))
            preds = {k: np.zeros(n) for k in ['mean', 'ridge_residues', 'ridge_size_composition', 'cnn_2d']}
            for tr, te in folds:
                preds['mean'][te] = y[tr].mean()
                preds['ridge_residues'][te] = fit_predict_ridge(X_size, y, tr, te)
                preds['ridge_size_composition'][te] = fit_predict_ridge(X_comp, y, tr, te)
                preds['cnn_2d'][te] = train_cnn_fold(dataset, tr, te, seed)
            for model_name, p in preds.items():
                r, rmse, r2 = metrics(p, y)
                rows.append({'split': split, 'seed': seed, 'model': model_name, 'pearson_r': r,
                             'rmse': rmse, 'r2': r2, 'pred_std': float(np.std(p))})
                print(f"[{split} seed {seed}] {model_name:24s} r={r:+.3f} RMSE={rmse:.3f}")
            pd.DataFrame({'record': folders, 'pkd_real': y, **{f'pred_{k}': v for k, v in preds.items()}}) \
                .to_csv(os.path.join(OUTPUT_DIR, f"robustness_oof_{split}_seed{seed}.csv"), index=False)

    runs = pd.DataFrame(rows)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    runs.to_csv(os.path.join(OUTPUT_DIR, "robustness_runs.csv"), index=False)
    summary = runs.groupby(['split', 'model']).agg(
        r_mean=('pearson_r', 'mean'), r_std=('pearson_r', 'std'),
        rmse_mean=('rmse', 'mean'), r2_mean=('r2', 'mean'), n_seeds=('seed', 'count')).reset_index()
    summary.to_csv(os.path.join(OUTPUT_DIR, "robustness_summary.csv"), index=False)
    print("\n" + summary.to_string(index=False))


if __name__ == '__main__':
    main()
