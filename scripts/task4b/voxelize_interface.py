# ======================================================================================
# SCRIPT: voxelize_interface.py (Task 4b - Geometric DL, baseline 3D)
#
# DESCRIZIONE:
# Converte le patch di superficie prodotte dalla Task 1 (ligand_surface_patch.csv,
# receptor_surface_patch.csv) in un tensore 3D a 4 canali (4, GRID_SIZE, GRID_SIZE, GRID_SIZE),
# senza passare dal matching Zernike 2D ne' dalla proiezione PCA su piano medio usati dalla
# pipeline principale (Task 2-3). A differenza della Task 3, i canali di carica e idrofobicita'
# usano la VERA identita' del residuo per ciascun punto (colonna 'residue_name', presente nei
# file della Task 1 ma persa nel sottocampionamento della Task 2) -- non l'assegnazione per
# interleaving descritta in Sezione 7.2 del report.
#
# CANALI:
#   0. Densita' ligando   (occupazione locale, zero lontano dal ligando)
#   1. Densita' recettore (idem)
#   2. Carica elettrostatica (nuvola combinata ligando+recettore)
#   3. Idrofobicita' Kyte-Doolittle (idem)
#
# INPUT:
#   outputs/task1/<cartella>/ligand_surface_patch.csv
#   outputs/task1/<cartella>/receptor_surface_patch.csv
#
# OUTPUT:
#   outputs/task4_dl/voxel_maps/<cartella>_voxel.npy   (tensore [4, GRID_SIZE, GRID_SIZE, GRID_SIZE])
# ======================================================================================

import os
import glob
import numpy as np
import pandas as pd
from scipy.interpolate import griddata

GRID_SIZE = 16
BOX_MARGIN_ANGSTROM = 2.0  # margine attorno al bounding box dei punti campionati
INPUT_DIR = "outputs/task1"
OUTPUT_DIR = "/content/drive/MyDrive/ppb-affinity-voxel"
os.makedirs(OUTPUT_DIR, exist_ok=True)

HYDROPHOBICITY_SCALE = {
    'ILE': 4.5, 'VAL': 4.2, 'LEU': 3.8, 'PHE': 2.8, 'CYS': 2.5, 'MET': 1.9, 'ALA': 1.8,
    'GLY': -0.4, 'THR': -0.7, 'SER': -0.8, 'TRP': -0.9, 'TYR': -1.3, 'PRO': -1.6,
    'HIS': -3.2, 'GLU': -3.5, 'GLN': -3.5, 'ASP': -3.5, 'ASN': -3.5, 'LYS': -3.9, 'ARG': -4.5
}
CHARGE_SCALE = {
    'ARG': 1.0, 'LYS': 1.0, 'HIS': 0.1, 'ASP': -1.0, 'GLU': -1.0,
    'ALA': 0.0, 'ASN': 0.0, 'CYS': 0.0, 'GLN': 0.0, 'GLY': 0.0, 'ILE': 0.0, 'LEU': 0.0,
    'MET': 0.0, 'PHE': 0.0, 'PRO': 0.0, 'SER': 0.0, 'THR': 0.0, 'TRP': 0.0, 'TYR': 0.0, 'VAL': 0.0
}


def load_patch(path):
    df = pd.read_csv(path)
    res = df['residue_name'].astype(str).str.upper().str.strip()
    df['charge'] = res.map(CHARGE_SCALE).fillna(0.0)
    df['hydrophobicity'] = res.map(HYDROPHOBICITY_SCALE).fillna(0.0)
    return df


def interpolate_field(points_own, values_own, grid_points, fill_value=0.0):
    """Interpolazione lineare; fuori dall'inviluppo convesso riempie con fill_value
    (non con 'nearest'): usato per i canali di densita', che devono realisticamente
    annullarsi lontano dai punti campionati del proprio partner."""
    vals = griddata(points_own, values_own, grid_points, method='linear')
    vals = np.where(np.isnan(vals), fill_value, vals)
    return vals


def interpolate_field_nearest_fallback(points, values, grid_points):
    """Come sopra ma con fallback 'nearest' (coerente con generate_interface_images.py):
    usato per i canali chimici, che si estendono ragionevolmente su tutta l'interfaccia."""
    vals = griddata(points, values, grid_points, method='linear')
    if np.isnan(vals).any():
        nearest_vals = griddata(points, values, grid_points, method='nearest')
        vals = np.where(np.isnan(vals), nearest_vals, vals)
    return vals


def voxelize_complex(folder):
    lig_path = os.path.join(folder, 'ligand_surface_patch.csv')
    rec_path = os.path.join(folder, 'receptor_surface_patch.csv')
    if not (os.path.exists(lig_path) and os.path.exists(rec_path)):
        return None

    df_lig = load_patch(lig_path)
    df_rec = load_patch(rec_path)
    if len(df_lig) == 0 or len(df_rec) == 0:
        return None

    coords_lig = df_lig[['x', 'y', 'z']].values
    coords_rec = df_rec[['x', 'y', 'z']].values
    coords_all = np.vstack([coords_lig, coords_rec])

    mins = coords_all.min(axis=0) - BOX_MARGIN_ANGSTROM
    maxs = coords_all.max(axis=0) + BOX_MARGIN_ANGSTROM

    axes = [np.linspace(mn, mx, GRID_SIZE) for mn, mx in zip(mins, maxs)]
    gx, gy, gz = np.meshgrid(*axes, indexing='ij')
    grid_points = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

    tensor = np.zeros((4, GRID_SIZE, GRID_SIZE, GRID_SIZE), dtype=np.float32)

    # Canale 0: densita' ligando (solo dai punti del ligando, zero fuori dal suo inviluppo)
    tensor[0] = interpolate_field(
        coords_lig, df_lig['density'].values, grid_points, fill_value=0.0
    ).reshape(GRID_SIZE, GRID_SIZE, GRID_SIZE)

    # Canale 1: densita' recettore
    tensor[1] = interpolate_field(
        coords_rec, df_rec['density'].values, grid_points, fill_value=0.0
    ).reshape(GRID_SIZE, GRID_SIZE, GRID_SIZE)

    # Canali 2-3: proprieta' chimiche sulla nuvola combinata, con vera identita' di residuo
    df_all = pd.concat([df_lig, df_rec], ignore_index=True)
    tensor[2] = interpolate_field_nearest_fallback(
        coords_all, df_all['charge'].values, grid_points
    ).reshape(GRID_SIZE, GRID_SIZE, GRID_SIZE)
    tensor[3] = interpolate_field_nearest_fallback(
        coords_all, df_all['hydrophobicity'].values, grid_points
    ).reshape(GRID_SIZE, GRID_SIZE, GRID_SIZE)

    return tensor


def main():
    folders = sorted(f for f in glob.glob(os.path.join(INPUT_DIR, '*')) if os.path.isdir(f))
    print(f"Trovate {len(folders)} cartelle da voxelizzare.\n")

    n_ok = 0
    for i, folder in enumerate(folders, 1):
        name = os.path.basename(folder)
        try:
            tensor = voxelize_complex(folder)
            if tensor is None:
                print(f"[{i}/{len(folders)}] Salto {name}: file mancanti o vuoti")
                continue
            np.save(os.path.join(OUTPUT_DIR, f"{name}_voxel.npy"), tensor)
            n_ok += 1
            if n_ok % 20 == 0 or i == len(folders):
                print(f"[{i}/{len(folders)}] {n_ok} complessi voxelizzati con successo finora")
        except Exception as e:
            print(f"[{i}/{len(folders)}] Errore in {name}: {e}")

    print(f"\nCompletato: {n_ok}/{len(folders)} complessi voxelizzati in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
