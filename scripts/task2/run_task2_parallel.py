import os
import pandas as pd
import numpy as np
import zepyros as zp
from pathlib import Path
import warnings
from multiprocessing import Pool, cpu_count
from scipy.spatial.distance import cdist

warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- CONFIGURAZIONE ---
INPUT_DIR = Path("outputs/task1")
STRIDE = 20
RADIUS = 6.0
ORDER = 20
OUTPUT_BASE_DIR = Path("outputs/task2")
OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
# ----------------------

def process_single_complex(folder_name):
    """Elaborazione simmetrica andata + ritorno"""
    pdb_input_path = INPUT_DIR / folder_name
    complex_output_dir = OUTPUT_BASE_DIR / folder_name
    complex_output_dir.mkdir(parents=True, exist_ok=True)

    # Cerchiamo il nuovo file simmetrico
    output_file = complex_output_dir / "zernike_complementarity_symmetric.csv"

    if output_file.exists():
        return f"Già fatto: {folder_name}"

    try:
        df_lig = pd.read_csv(pdb_input_path / "ligand_surface_patch.csv")
        df_rec = pd.read_csv(pdb_input_path / "receptor_surface_patch.csv")

        idx_lig = np.arange(0, len(df_lig), STRIDE)
        idx_rec = np.arange(0, len(df_rec), STRIDE)

        # Calcolo Descrittori Ligando
        coords_norm_lig = df_lig[['x', 'y', 'z', 'nx', 'ny', 'nz']].values
        desc_lig = []
        for i in idx_lig:
            res = zp.get_zernike(coords_norm_lig, RADIUS, int(i), ORDER, 1)
            desc_lig.append(res[0] if isinstance(res, tuple) else res)

        # Calcolo Descrittori Recettore
        coords_norm_rec = df_rec[['x', 'y', 'z', 'nx', 'ny', 'nz']].values
        desc_rec = []
        for i in idx_rec:
            res = zp.get_zernike(coords_norm_rec, RADIUS, int(i), ORDER, -1)
            desc_rec.append(res[0] if isinstance(res, tuple) else res)

        desc_lig = np.array(desc_lig)
        desc_rec = np.array(desc_rec)

        # Matching globale
        dists = cdist(desc_lig, desc_rec, metric='euclidean')

        # VERSO 1: Ligando -> Recettore
        min_dists_lig = np.min(dists, axis=1)
        results_lig = pd.DataFrame({
            'source': 'ligand',
            'center_index': idx_lig,
            'x': df_lig.iloc[idx_lig]['x'].values,
            'y': df_lig.iloc[idx_lig]['y'].values,
            'z': df_lig.iloc[idx_lig]['z'].values,
            'zernike_distance': min_dists_lig,
            'complementarity_score': -min_dists_lig
        })

        # VERSO 2: Recettore -> Ligando
        min_dists_rec = np.min(dists, axis=0)
        results_rec = pd.DataFrame({
            'source': 'receptor',
            'center_index': idx_rec,
            'x': df_rec.iloc[idx_rec]['x'].values,
            'y': df_rec.iloc[idx_rec]['y'].values,
            'z': df_rec.iloc[idx_rec]['z'].values,
            'zernike_distance': min_dists_rec,
            'complementarity_score': -min_dists_rec
        })

        # Unione simmetrica
        results_symmetric = pd.concat([results_lig, results_rec], ignore_index=True)
        results_symmetric.to_csv(output_file, index=False)

        return f"Completato (Simmetrico): {folder_name}"

    except Exception as e:
        return f"Errore in {folder_name}: {str(e)}"

def main():
    folders = [f for f in os.listdir(INPUT_DIR) if os.path.isdir(INPUT_DIR / f)]
    print(f"Trovate {len(folders)} proteine da processare.")

    # Sincronizzazione core con Slurm
    num_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", 48))
    print(f"Avvio elaborazione parallela su {num_cores} core di Leonardo...")

    with Pool(num_cores) as p:
        for result in p.imap_unordered(process_single_complex, folders):
            print(result)

if __name__ == "__main__":
    main()
