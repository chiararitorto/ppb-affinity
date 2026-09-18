# ======================================================================================
# MODULO: models_3d.py (Task 4b - Geometric DL, baseline 3D)
#
# Analogo di scripts/task4/models.py ma per tensori voxel 3D (4, 16, 16, 16). Segue la
# stessa filosofia senza leakage: il Dataset non calcola statistiche di standardizzazione
# da solo; vanno calcolate esternamente sul solo training set di ciascun fold tramite
# compute_channel_stats_3d() e applicate con standardize_with().
# ======================================================================================

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F


class ProteinInterfaceDataset3D(Dataset):
    """Dataset PyTorch per i tensori voxel 3D. Nessuna standardizzazione di default."""

    def __init__(self, csv_affinity_path, maps_dir, excluded_pdb_ids=None):
        self.maps_dir = maps_dir
        self.df_affinity = pd.read_csv(csv_affinity_path, sep=';')
        self.valid_samples = []
        self.means = None
        self.stds = None
        self.excluded = set(excluded_pdb_ids) if excluded_pdb_ids else set()

        self.file_mapping = {}
        if os.path.exists(maps_dir):
            for f in os.listdir(maps_dir):
                if f.endswith('_voxel.npy'):
                    parts = f.split('_')
                    if len(parts) >= 2:
                        self.file_mapping[parts[1].upper()] = f

        for _, row in self.df_affinity.iterrows():
            if pd.isna(row['PDB']) or pd.isna(row['KD(M)']):
                continue
            pdb_id = str(row['PDB']).strip().upper()

            if pdb_id in self.excluded:
                continue

            if pdb_id in self.file_mapping:
                tensor_path = os.path.join(self.maps_dir, self.file_mapping[pdb_id])
                try:
                    kd_value = float(row['KD(M)'])
                    if kd_value <= 0:
                        continue
                    pkd_target = -np.log10(kd_value)
                    self.valid_samples.append({
                        'path': tensor_path,
                        'target': pkd_target,
                        'pdb_id': pdb_id
                    })
                except ValueError:
                    continue

        print(f"-> Dataset 3D pronto: {len(self.valid_samples)} complessi caricati "
              f"(nessuna standardizzazione applicata di default).")

    def standardize_with(self, means, stds):
        stds = np.array(stds, dtype=np.float32).copy()
        stds[stds == 0] = 1.0
        self.means = np.array(means, dtype=np.float32)
        self.stds = stds

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        sample = self.valid_samples[idx]
        tensor = np.load(sample['path']).astype(np.float32)
        target = np.array(sample['target'], dtype=np.float32)

        if self.means is not None:
            for c in range(tensor.shape[0]):
                tensor[c] = (tensor[c] - self.means[c]) / self.stds[c]

        x_tensor = torch.from_numpy(tensor)
        y_tensor = torch.tensor(target, dtype=torch.float32).unsqueeze(0)
        return x_tensor, y_tensor


def compute_channel_stats_3d(dataset, indices):
    """Media e deviazione standard per canale sui soli campioni indicati (training set
    del fold corrente), leggendo i tensori grezzi da disco."""
    tensors = []
    for i in indices:
        t = np.load(dataset.valid_samples[i]['path']).astype(np.float32)
        tensors.append(t)
    stacked = np.stack(tensors, axis=0)  # [N, 4, D, D, D]
    means = np.mean(stacked, axis=(0, 2, 3, 4))
    stds = np.std(stacked, axis=(0, 2, 3, 4))
    return means, stds


class Interface3DCNN(nn.Module):
    """
    CNN 3D su voxel grid (4, 16, 16, 16). A differenza della pipeline 2D principale,
    non riceve uno score di complementarita' Zernike pre-calcolato: i canali 0-1
    (densita' ligando/recettore) sono le due superfici separate nello stesso volume,
    e la rete deve imparare la complementarita' geometrica direttamente dalle
    convoluzioni 3D congiunte.
    """

    def __init__(self, in_channels=4):
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(16)

        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(32)

        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(64)

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Risoluzione: 16 -> 8 -> 4 -> 2. Feature spianate = 64 * 2^3 = 512.
        self.fc1 = nn.Linear(64 * 2 * 2 * 2, 128)
        self.dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # [Batch, 16, 8, 8, 8]
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # [Batch, 32, 4, 4, 4]
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # [Batch, 64, 2, 2, 2]

        x = x.view(x.size(0), -1)                        # [Batch, 512]

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)                                   # [Batch, 1]

        return x
