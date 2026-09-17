# ======================================================================================
# SCRIPT: train_final.py (Task 4 - Pipeline corretta, senza leakage)
#
# DESCRIZIONE GENERALE:
# Unifica le buone idee di train_kfold.py e train_advanced.py (5-fold CV, standardizzazione
# Z-score, data augmentation geometrica, LR scheduler) correggendo due problemi identificati
# nelle versioni precedenti:
#
#   1. Standardizzazione Z-score calcolata SOLO sul training set di ciascun fold esterno
#      (non più su tutti i 118 complessi prima dello split).
#   2. Selezione del checkpoint migliore tramite uno split di VALIDATION interno, ricavato
#      dal training set del fold esterno -- non più tramite il fold di test stesso.
#      Il fold di test esterno non viene mai usato per prendere decisioni sul modello,
#      solo per la valutazione finale OOF.
#
# Struttura per ciascun fold esterno (5-fold KFold, seed fisso):
#   train_idx (esterno) --split 85/15--> inner_train_idx, inner_val_idx
#   test_idx (esterno)  --mai toccato fino alla valutazione finale--
#
# OUTPUT:
# - outputs/task4_final/cnn_final_fold_{1..5}.pth
# - Metriche OOF finali (RMSE, Pearson, Spearman) su tutti i 118 complessi.
# ======================================================================================

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # DEVE precedere gli import di numpy/torch/scipy,
                                               # altrimenti il conflitto OpenMP avviene comunque
                                               # durante l'import stesso (bug ereditato da
                                               # train_advanced.py, dove la riga era messa dopo
                                               # gli import e quindi non aveva alcun effetto).

import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold, train_test_split
from scipy.stats import pearsonr, spearmanr

from models import ProteinInterfaceDataset, InterfaceCNN, compute_channel_stats

CSV_PATH = "data/affinity_dataset.csv"
MAPS_DIR = "outputs/task3/interface_maps/"
OUTPUT_DIR = "outputs/task4_final/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"Device per l'addestramento: {device}")

BATCH_SIZE = 8
LEARNING_RATE = 0.001
WEIGHT_DECAY = 2e-3
EPOCHS = 100
NUM_FOLDS = 5
INNER_VAL_FRACTION = 0.15   # frazione del training set esterno usata come validation interno
OUTER_SEED = 42             # seed dello split K-Fold esterno
INNER_SEED = 123            # seed dello split train/val interno (fisso per riproducibilità)


def apply_geometry_augmentation(batch_inputs):
    """Rotazioni ortogonali e ribaltamenti casuali sui tensori d'interfaccia 2D."""
    augmented = batch_inputs.clone()
    for i in range(augmented.size(0)):
        if random.random() > 0.5:
            augmented[i] = torch.flip(augmented[i], dims=[2])
        if random.random() > 0.5:
            augmented[i] = torch.flip(augmented[i], dims=[1])
        k_rot = random.randint(0, 3)
        if k_rot > 0:
            augmented[i] = torch.rot90(augmented[i], k=k_rot, dims=[1, 2])
    return augmented


# Dataset grezzo: nessuna standardizzazione applicata qui, verrà impostata per ciascun fold
dataset = ProteinInterfaceDataset(csv_affinity_path=CSV_PATH, maps_dir=MAPS_DIR)
n_samples = len(dataset)

kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=OUTER_SEED)

all_oof_predictions = np.zeros(n_samples)
all_oof_targets = np.zeros(n_samples)

print(f"=== INIZIO ADDESTRAMENTO ({NUM_FOLDS} FOLD ESTERNI, split interno per il checkpoint) ===")

for fold, (train_idx, test_idx) in enumerate(kf.split(np.arange(n_samples)), 1):
    print(f"\n--- FOLD [{fold}/{NUM_FOLDS}] ---")

    # Split interno: dal training set esterno ricaviamo un validation set
    # usato SOLO per scegliere il checkpoint migliore. Il test_idx esterno
    # non viene mai toccato in questa fase.
    inner_train_idx, inner_val_idx = train_test_split(
        train_idx, test_size=INNER_VAL_FRACTION, random_state=INNER_SEED
    )
    print(f"    Train interno: {len(inner_train_idx)} | Val interno: {len(inner_val_idx)} | Test esterno: {len(test_idx)}")

    # Standardizzazione Z-score calcolata SOLO sul training set interno di questo fold
    means, stds = compute_channel_stats(dataset, inner_train_idx)
    dataset.standardize_with(means, stds)
    print(f"    Medie canale (train interno): {means}")
    print(f"    Dev.std canale (train interno): {stds}")

    train_loader = DataLoader(Subset(dataset, inner_train_idx), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(Subset(dataset, inner_val_idx), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=BATCH_SIZE, shuffle=False)

    model = InterfaceCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=7)

    best_val_loss = float('inf')
    fold_weights_path = os.path.join(OUTPUT_DIR, f"cnn_final_fold_{fold}.pth")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs_aug = apply_geometry_augmentation(inputs)

            optimizer.zero_grad()
            outputs = model(inputs_aug)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        # --- SELEZIONE DEL CHECKPOINT: valutata sul VALIDATION INTERNO, non sul test esterno ---
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

    # --- VALUTAZIONE FINALE: SOLO ORA si usa il test esterno, mai visto prima ---
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
print("METRICHE FINALI (OOF CUMULATIVO, SENZA LEAKAGE)")
print("=" * 60)

global_rmse = np.sqrt(np.mean((all_oof_predictions - all_oof_targets) ** 2))
global_pearson, p_pearson = pearsonr(all_oof_predictions, all_oof_targets)
global_spearman, p_spearman = spearmanr(all_oof_predictions, all_oof_targets)

print(f" RMSE:              {global_rmse:.4f}")
print(f" Pearson R:         {global_pearson:.4f}  (p={p_pearson:.4g})")
print(f" Spearman rho:      {global_spearman:.4f}  (p={p_spearman:.4g})")
print("=" * 60)
print(f"Pesi salvati in: {OUTPUT_DIR}")

# Salvataggio delle predizioni OOF per eventuali grafici successivi (scatter reale vs predetto)
oof_df = pd.DataFrame({
    'pdb_id': [s['pdb_id'] for s in dataset.valid_samples],
    'pkd_real': all_oof_targets,
    'pkd_predicted': all_oof_predictions,
})
oof_df.to_csv(os.path.join(OUTPUT_DIR, "oof_predictions.csv"), index=False)
print(f"Predizioni OOF salvate in: {os.path.join(OUTPUT_DIR, 'oof_predictions.csv')}")
