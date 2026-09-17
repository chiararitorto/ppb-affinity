# ======================================================================================
# MODULO: models.py (Task 4) -- VERSIONE CORRETTA (fix leakage nella standardizzazione)
# COMPONENTI: ProteinInterfaceDataset (Classe), InterfaceCNN (Classe)
#
# DESCRIZIONE GENERALE:
# Questo modulo contiene le fondamenta dell'architettura di Deep Learning per il Task 4.
# Definisce come i dati estratti nel Task 3 vengono digeriti, normalizzati e fatti confluire
# all'interno di una rete neurale convoluzionale (CNN) per compiti di regressione.
#
# CORREZIONE RISPETTO ALLA VERSIONE PRECEDENTE:
# La versione precedente calcolava media e deviazione standard globali su TUTTI i campioni
# dentro __init__, prima di qualunque split train/test. Questo costituisce un lieve data
# leakage: le statistiche di standardizzazione "vedono" anche i dati di test.
# In questa versione il Dataset carica i tensori grezzi (nessuna standardizzazione di
# default); le statistiche vanno calcolate ESTERNAMENTE, sul solo sottoinsieme di training
# di ciascun fold, tramite compute_channel_stats(), e poi applicate al dataset con
# dataset.standardize_with(means, stds) prima di costruire i DataLoader di quel fold.
# ======================================================================================

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F


class ProteinInterfaceDataset(Dataset):
    """
    Dataset PyTorch per le mappe di interfaccia 2D (Task 3). Non applica alcuna
    standardizzazione finché non viene chiamato standardize_with(means, stds).
    """

    # 1NVU compare due volte nel benchmark originale (catena ligando Q, KD=3.6e-06 M,
    # e catena ligando R, KD=1.9e-06 M; stesso recettore S). La pipeline della Task 1
    # indicizza l'output per PDB ID, quindi una sola delle due geometrie sopravvive su
    # disco (verificato: la catena Q). Il target di affinità nel CSV corrisponde però
    # alla catena R: il tensore e il target sono quindi disallineati per questo
    # complesso. Escluso esplicitamente per integrità dei dati (1 campione su 118).
    EXCLUDED_PDB_IDS = {'1NVU'}

    def __init__(self, csv_affinity_path, maps_dir):
        self.maps_dir = maps_dir
        self.df_affinity = pd.read_csv(csv_affinity_path, sep=';')
        self.valid_samples = []
        self.means = None
        self.stds = None

        # Indicizzazione dei file .npy realmente presenti su disco
        self.file_mapping = {}
        if os.path.exists(maps_dir):
            for f in os.listdir(maps_dir):
                if f.endswith('_tensor.npy'):
                    parts = f.split('_')
                    if len(parts) >= 2:
                        self.file_mapping[parts[1].upper()] = f

        # Costruzione della lista dei campioni validi (path del tensore + target pKd)
        for _, row in self.df_affinity.iterrows():
            if pd.isna(row['PDB']) or pd.isna(row['KD(M)']):
                continue
            pdb_id = str(row['PDB']).strip().upper()

            if pdb_id in self.EXCLUDED_PDB_IDS:
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

        print(f"-> Dataset pronto: {len(self.valid_samples)} complessi dimerici caricati "
              f"(nessuna standardizzazione applicata di default).")

    def standardize_with(self, means, stds):
        """Imposta le statistiche di standardizzazione da usare in __getitem__.
        Vanno calcolate con compute_channel_stats() sul solo training set del fold corrente."""
        stds = np.array(stds, dtype=np.float32).copy()
        stds[stds == 0] = 1.0  # evita divisioni per zero su canali a varianza nulla
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
                tensor[c, :, :] = (tensor[c, :, :] - self.means[c]) / self.stds[c]

        x_tensor = torch.from_numpy(tensor)
        y_tensor = torch.tensor(target, dtype=torch.float32).unsqueeze(0)
        return x_tensor, y_tensor


def compute_channel_stats(dataset, indices):
    """Calcola media e deviazione standard per canale sui soli campioni indicati
    (tipicamente il training set di un fold), leggendo i tensori GREZZI da disco
    indipendentemente da eventuali statistiche già impostate sul dataset."""
    tensors = []
    for i in indices:
        t = np.load(dataset.valid_samples[i]['path']).astype(np.float32)
        tensors.append(t)
    stacked = np.stack(tensors, axis=0)  # [N, 3, 32, 32]
    means = np.mean(stacked, axis=(0, 2, 3))
    stds = np.std(stacked, axis=(0, 2, 3))
    return means, stds


class InterfaceCNN(nn.Module):
    """
    Rete Neurale Convoluzionale a 3 stadi ottimizzata per immagini d'interfaccia 32x32.
    Esegue una riduzione spaziale progressiva fino alla regressione lineare finale.
    (Architettura invariata rispetto alla versione precedente.)
    """
    def __init__(self):
        super(InterfaceCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # [Batch, 16, 16, 16]
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # [Batch, 32, 8, 8]
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # [Batch, 64, 4, 4]

        x = x.view(x.size(0), -1)                       # [Batch, 1024]

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)                                 # [Batch, 1]

        return x
