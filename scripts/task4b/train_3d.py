# ======================================================================================
# SCRIPT: train_3d.py (Task 4b - Geometric DL, baseline 3D)
#
# Stesso protocollo a due livelli di train_final.py (Task 4), adattato al modello 3D:
# split interno 85/15 per la selezione del checkpoint, standardizzazione Z-score
# calcolata solo sul training set di ciascun fold, augmentation geometrica, test
# esterno mai usato per decisioni durante l'addestramento.
#
# OUTPUT:
#   outputs/task4_dl/cnn3d_fold_{1..5}.pth
#   outputs/task4_dl/oof_predictions_3d.csv
# ======================================================================================

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # deve precedere gli import di numpy/torch/scipy

import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold, train_test_split
from scipy.stats import pearsonr, spearmanr

from models_3d import ProteinInterfaceDataset3D, Interface3DCNN, compute_channel_stats_3d

CSV_PATH = "data/affinity_dataset.csv"
MAPS_DIR = "outputs/task4_dl/voxel_maps/"
OUTPUT_DIR = "outputs/task4_dl/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Stessa esclusione della Task 4: 1NVU ha geometria (catena Q) e target sperimentale
# disallineati nel dataset originale. 1UUG viene escluso automaticamente dal Dataset
# perche' il suo KD("<1E-13") non e' un valore numerico valido.
EXCLUDED_PDB_IDS = {'1NVU'}

# MPS non supporta Conv3d in questa versione di PyTorch (RuntimeError confermato); si forza
# quindi la CPU per questo script, a differenza della CNN 2D della Task 4 che può usare MPS.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device per l'addestramento: {device}")

BATCH_SIZE = 8
LEARNING_RATE = 0.001
WEIGHT_DECAY = 2e-3
EPOCHS = 100
NUM_FOLDS = 5
INNER_VAL_FRACTION = 0.15
OUTER_SEED = 42
INNER_SEED = 123


def apply_geometry_augmentation_3d(batch_inputs):
    """Rotazioni di 90 gradi e ribaltamenti casuali sui tre assi del volume 3D."""
    augmented = batch_inputs.clone()
    for i in range(augmented.size(0)):
        # Ribaltamenti casuali sui tre assi spaziali (dims 1,2,3 = D,H,W; dim 0 = canale)
        for dim in (1, 2, 3):
            if random.random() > 0.5:
                augmented[i] = torch.flip(augmented[i], dims=[dim])
        # Rotazione casuale di 90/180/270 gradi su una coppia di assi scelta a caso
        axis_pair = random.choice([(1, 2), (1, 3), (2, 3)])
        k_rot = random.randint(0, 3)
        if k_rot > 0:
            augmented[i] = torch.rot90(augmented[i], k=k_rot, dims=axis_pair)
    return augmented


dataset = ProteinInterfaceDataset3D(csv_affinity_path=CSV_PATH, maps_dir=MAPS_DIR,
                                     excluded_pdb_ids=EXCLUDED_PDB_IDS)
n_samples = len(dataset)

kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=OUTER_SEED)

all_oof_predictions = np.zeros(n_samples)
all_oof_targets = np.zeros(n_samples)

print(f"=== INIZIO ADDESTRAMENTO 3D ({NUM_FOLDS} FOLD ESTERNI) ===")

for fold, (train_idx, test_idx) in enumerate(kf.split(np.arange(n_samples)), 1):
    print(f"\n--- FOLD [{fold}/{NUM_FOLDS}] ---")

    inner_train_idx, inner_val_idx = train_test_split(
        train_idx, test_size=INNER_VAL_FRACTION, random_state=INNER_SEED
    )
    print(f"    Train interno: {len(inner_train_idx)} | Val interno: {len(inner_val_idx)} | Test esterno: {len(test_idx)}")

    means, stds = compute_channel_stats_3d(dataset, inner_train_idx)
    dataset.standardize_with(means, stds)
    print(f"    Medie canale (train interno): {means}")
    print(f"    Dev.std canale (train interno): {stds}")

    train_loader = DataLoader(Subset(dataset, inner_train_idx), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(Subset(dataset, inner_val_idx), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=BATCH_SIZE, shuffle=False)

    model = Interface3DCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=7)

    best_val_loss = float('inf')
    fold_weights_path = os.path.join(OUTPUT_DIR, f"cnn3d_fold_{fold}.pth")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs_aug = apply_geometry_augmentation_3d(inputs)

            optimizer.zero_grad()
            outputs = model(inputs_aug)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        val_loss /= len(inner_val_idx)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), fold_weights_path)

    print(f"    -> Fold {fold} completato. Miglior Val Loss (interna): {best_val_loss:.4f}")

    model.load_state_dict(torch.load(fold_weights_path))
    model.eval()

    fold_preds = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            fold_preds.extend(outputs.cpu().numpy().flatten())

    all_oof_predictions[test_idx] = fold_preds
    all_oof_targets[test_idx] = [dataset.valid_samples[i]['target'] for i in test_idx]

print("\n" + "=" * 60)
print("METRICHE FINALI 3D (OOF CUMULATIVO, SENZA LEAKAGE)")
print("=" * 60)

global_rmse = np.sqrt(np.mean((all_oof_predictions - all_oof_targets) ** 2))
global_pearson, p_pearson = pearsonr(all_oof_predictions, all_oof_targets)
global_spearman, p_spearman = spearmanr(all_oof_predictions, all_oof_targets)

print(f" RMSE:              {global_rmse:.4f}")
print(f" Pearson R:         {global_pearson:.4f}  (p={p_pearson:.4g})")
print(f" Spearman rho:      {global_spearman:.4f}  (p={p_spearman:.4g})")
print("=" * 60)
print(f"Pesi salvati in: {OUTPUT_DIR}")

oof_df = pd.DataFrame({
    'pdb_id': [s['pdb_id'] for s in dataset.valid_samples],
    'pkd_real': all_oof_targets,
    'pkd_predicted': all_oof_predictions,
})
oof_df.to_csv(os.path.join(OUTPUT_DIR, "oof_predictions_3d.csv"), index=False)
print(f"Predizioni OOF salvate in: {os.path.join(OUTPUT_DIR, 'oof_predictions_3d.csv')}")