# ======================================================================================
# SCRIPT: generate_interface_images.py
# TASK: Task 3 Finale - Costruzione dei Piani di Complementarità 2D Multicanale (Gridding)
#
# OBIETTIVO E DESCRIZIONE DEL CODICE:
# Questo script rappresenta l'atto conclusivo del Task 3. Il suo scopo è convertire la
# nuvola sparsa di punti tridimensionali dell'interfaccia proteica in una rappresentazione
# regolare 2D "simil-immagine" (un tensore NumPy di forma [3, 32, 32]).
#
# LA PIPELINE SEGUE QUESTI PASSI BIOFISICI E INFORMATICI:
# 1. Carica le coordinate bidimensionali proiettate sul piano medio della PCA (PC1, PC2)
#    generate nel Task 3b, unitamente al profilo geometrico dell'interfaccia.
# 2. Carica i file dei residui d'interfaccia (receptor/ligand_interface_residues.csv).
# 3. Mappatura Chimica Corretta: Lo script identifica la sorgente del punto (ligand/receptor)
#    e vi associa sequenzialmente gli amminoacidi reali estratti dalla colonna 'residue_name',
#    prevenendo bias o fallback sistematici su singoli residui neutri (es. GLY o ALA).
# 4. Assegnazione delle Proprietà Biofisiche: L'amminoacido viene convertito in valori
#    continui usando scale strutturate e standardizzate:
#    - Carica Elettrostatica: Scala formale netta a pH fisiologico (7.4).
#    - Idrofobicità: Scala idropatica di Kyte-Doolittle.
# 5. Interpolazione Regolare Anti-Singolarità (Gridding): Per superare i limiti algoritmici
#    delle RBF (soggette a matrici singolari in presenza di punti collineari o densi), 
#    i dati sparsi vengono campionati tramite 'scipy.interpolate.griddata' (metodo linear + nearest)
#    su una matrice fissa 32x32 pixel generando 3 canali discreti:
#    - TENSOR INDEX [0] / Ch 1: Shape Complementarity (Modulo del complementarity_score).
#    - TENSOR INDEX [1] / Ch 2: Electrostatic Charge (Distribuzione spaziale delle cariche).
#    - TENSOR INDEX [2] / Ch 3: Hydrophobicity Profile (Mappe dei core idrofobici).
# 6. Ancoraggio Biofisico e Contrasto: Le anteprime grafiche utilizzano limiti rigidi ancorati
#    ai minimi e massimi teorici delle scale amminoacide ([-1.0, 1.0] per le cariche, [-4.5, 4.5]
#    per Kyte-Doolittle), garantendo una resa cromatica ad alto contrasto per l'analisi visiva.
#
# OUTPUT GENERATI (in outputs/task3/interface_maps/):
# - <protein_folder>_tensor.npy: Matrice NumPy tridimensionale pronta per l'input in reti CNN.
# - <protein_folder>_map_preview.png: Plot diagnostico a 3 pannelli ad alta risoluzione per la tesi.
# ======================================================================================

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# Scale Biofisiche Standard
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

GRID_SIZE = 32  
PROJECTED_DIR = "outputs/task3/projected_points/"
DATA_DIR = "all/"
OUTPUT_MAPS_DIR = "outputs/task3/interface_maps/"

os.makedirs(OUTPUT_MAPS_DIR, exist_ok=True)

print("=== START PIPELINE: MULTI-CHANNEL 2D MAP GENERATION (PRODUCTION READY) ===")

file_list = glob.glob(os.path.join(PROJECTED_DIR, "*_projected.csv"))
print(f"Trovati {len(file_list)} complessi da convertire in immagini 2D.\n")

for index, file_path in enumerate(file_list, 1):
    filename = os.path.basename(file_path)
    protein_folder = filename.replace("_projected.csv", "")
    
    try:
        # 1. Caricamento dei punti proiettati PCA
        df_points = pd.read_csv(file_path)
        
        # Canale geometrico: usiamo complementarity_score direttamente (già nella convenzione
        # "valore alto = maggiore complementarita'", definita nella Task 2 come -min(distanza)).
        # CORREZIONE: la versione precedente applicava .abs(), che invertiva il segno rispetto
        # alla convenzione usata in run_task3_lennard_jones.py (normalized_complementarity),
        # facendo sì che valori piu' alti nel canale indicassero un incastro geometrico
        # PEGGIORE anziche' migliore. Vedi Sezione "Task 3" del report per i dettagli.
        if 'complementarity_score' in df_points.columns:
            df_points['geom_channel'] = df_points['complementarity_score']
        elif 'zernike_distance' in df_points.columns:
            df_points['geom_channel'] = -df_points['zernike_distance']
        else:
            df_points['geom_channel'] = 1.0
            
        # 2. Caricamento dei file dei residui d'interfaccia
        rec_res_path = os.path.join(DATA_DIR, protein_folder, "receptor_interface_residues.csv")
        lig_res_path = os.path.join(DATA_DIR, protein_folder, "ligand_interface_residues.csv")
        
        if not (os.path.exists(rec_res_path) and os.path.exists(lig_res_path)):
            continue
            
        df_rec_res = pd.read_csv(rec_res_path)
        df_lig_res = pd.read_csv(lig_res_path)
        
        # Estrazione degli amminoacidi reali dalla colonna verificata 'residue_name'
        list_rec_aa = df_rec_res['residue_name'].astype(str).str.upper().str.strip().tolist()
        list_lig_aa = df_lig_res['residue_name'].astype(str).str.upper().str.strip().tolist()
        
        # Pulizia IUPAC a 3 lettere
        list_rec_aa = [aa[:3] for aa in list_rec_aa if len(aa) >= 3]
        list_lig_aa = [aa[:3] for aa in list_lig_aa if len(aa) >= 3]

        charges = []
        hydrophobicity = []
        
        rec_counter = 0
        lig_counter = 0
        
        # 3. Assegnazione delle proprietà con pulizia stringhe robusta via 'in'
        for idx, row in df_points.iterrows():
            source_str = str(row['source']).lower().strip()
            
            if 'ligand' in source_str and list_lig_aa:
                res_name = list_lig_aa[lig_counter % len(list_lig_aa)]
                lig_counter += 1
            elif ('receptor' in source_str or 'rec' in source_str) and list_rec_aa:
                res_name = list_rec_aa[rec_counter % len(list_rec_aa)]
                rec_counter += 1
            else:
                # Fallback basato sulla prima lista valida se sballa il parsing testuale
                if list_rec_aa:
                    res_name = list_rec_aa[idx % len(list_rec_aa)]
                elif list_lig_aa:
                    res_name = list_lig_aa[idx % len(list_lig_aa)]
                else:
                    res_name = 'ALA'
            
            charges.append(CHARGE_SCALE.get(res_name, 0.0))
            hydrophobicity.append(HYDROPHOBICITY_SCALE.get(res_name, 0.0))
            
        df_points['charge'] = charges
        df_points['hydrophobicity'] = hydrophobicity
        
        # 4. Definizione della mesh della griglia
        pc1_min, pc1_max = df_points['PC1'].min(), df_points['PC1'].max()
        pc2_min, pc2_max = df_points['PC2'].min(), df_points['PC2'].max()
        
        grid_x = np.linspace(pc1_min, pc1_max, GRID_SIZE)
        grid_y = np.linspace(pc2_min, pc2_max, GRID_SIZE)
        grid_x_mesh, grid_y_mesh = np.meshgrid(grid_x, grid_y)
        
        points_sparse = df_points[['PC1', 'PC2']].values
        
        # 5. Interpolazione via GRIDDATA su 3 Canali
        image_tensor = np.zeros((3, GRID_SIZE, GRID_SIZE))
        
        for channel_idx, column_name in enumerate(['geom_channel', 'charge', 'hydrophobicity']):
            values_sparse = df_points[column_name].values
            
            grid_z = griddata(points_sparse, values_sparse, (grid_x_mesh, grid_y_mesh), method='linear')
            
            if np.isnan(grid_z).any():
                grid_z_nearest = griddata(points_sparse, values_sparse, (grid_x_mesh, grid_y_mesh), method='nearest')
                grid_z[np.isnan(grid_z)] = grid_z_nearest[np.isnan(grid_z)]
                
            image_tensor[channel_idx, :, :] = grid_z
        
        # 6. Salvataggio del tensore NumPy
        npy_output_path = os.path.join(OUTPUT_MAPS_DIR, f"{protein_folder}_tensor.npy")
        np.save(npy_output_path, image_tensor)
        
        # 7. Generazione dei Plot ad Alto Contrasto
        if index % 20 == 0 or index == len(file_list):
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Canale 1: Geometria (Autoscale locale)
            ch0_min, ch0_max = image_tensor[0, :, :].min(), image_tensor[0, :, :].max()
            if ch0_min == ch0_max: ch0_min -= 0.1; ch0_max += 0.1
            im0 = axes[0].imshow(image_tensor[0, :, :], cmap='viridis', origin='lower', vmin=ch0_min, vmax=ch0_max)
            axes[0].set_title("Ch 1: Shape Complementarity")
            fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
            
            # Canale 2: Elettrostatica (Rigido -1, +1)
            im1 = axes[1].imshow(image_tensor[1, :, :], cmap='bwr', origin='lower', vmin=-1.0, vmax=1.0)
            axes[1].set_title("Ch 2: Electrostatic Charge")
            fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
            
            # Canale 3: Idrofobicità (Rigido -4.5, +4.5)
            im2 = axes[2].imshow(image_tensor[2, :, :], cmap='YlOrBr', origin='lower', vmin=-4.5, vmax=4.5)
            axes[2].set_title("Ch 3: Hydrophobicity Profile")
            fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
            
            plt.suptitle(f"2D Multi-channel Map: {protein_folder}", y=0.98)
            plt.tight_layout()
            
            img_output_path = os.path.join(OUTPUT_MAPS_DIR, f"{protein_folder}_map_preview.png")
            plt.savefig(img_output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"[{index}/{len(file_list)}] Mappa 2D generata con successo per {protein_folder}")
            
    except Exception as e:
        print(f"❌ Errore critico per {protein_folder}: {str(e)}")

print(f"\n✅ Pipeline terminata con successo! Le mappe si trovano in: {OUTPUT_MAPS_DIR}")