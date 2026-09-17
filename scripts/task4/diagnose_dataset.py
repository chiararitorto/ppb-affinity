"""Diagnostica: quali complessi vengono scartati da ProteinInterfaceDataset, e perché."""
import os
import pandas as pd

CSV_PATH = "data/affinity_dataset.csv"
MAPS_DIR = "outputs/task3/interface_maps/"

df = pd.read_csv(CSV_PATH, sep=';')

# 1. Quanti file .npy ci sono realmente su disco?
npy_files = [f for f in os.listdir(MAPS_DIR) if f.endswith('_tensor.npy')]
print(f"File .npy trovati in {MAPS_DIR}: {len(npy_files)}")

file_mapping = {}
for f in npy_files:
    parts = f.split('_')
    if len(parts) >= 2:
        pdb = parts[1].upper()
        if pdb in file_mapping:
            print(f"  ATTENZIONE: PDB ID duplicato tra i file .npy: {pdb} "
                  f"(già mappato a {file_mapping[pdb]}, ora anche {f})")
        file_mapping[pdb] = f

print(f"PDB ID unici nei file .npy: {len(file_mapping)}")
print()

# 2. Per ogni riga del CSV, verifica se viene inclusa o esclusa, e perché
excluded_manual = {'1NVU'}
included, excluded_bad_kd, excluded_manual_list = [], [], []

for _, row in df.iterrows():
    if pd.isna(row['PDB']) or pd.isna(row['KD(M)']):
        continue
    pdb_id = str(row['PDB']).strip().upper()

    if pdb_id in excluded_manual:
        excluded_manual_list.append(pdb_id)
        continue

    if pdb_id not in file_mapping:
        continue  # non e' un dimero processato, ignoriamo (non e' un errore)

    try:
        kd_value = float(row['KD(M)'])
        if kd_value <= 0:
            excluded_bad_kd.append(pdb_id)
            continue
        included.append(pdb_id)
    except ValueError:
        excluded_bad_kd.append(pdb_id)

print(f"Campioni inclusi: {len(included)}")
print(f"Esclusi manualmente (1NVU): {excluded_manual_list}")
print(f"Esclusi per KD non valido (<=0 o non numerico): {excluded_bad_kd}")

# 3. PDB presenti tra i file .npy ma MAI trovati tra le righe incluse
missing = set(file_mapping.keys()) - set(included) - excluded_manual
print(f"\nPDB con tensore su disco ma NESSUNA riga valida corrispondente nel CSV: {sorted(missing)}")
