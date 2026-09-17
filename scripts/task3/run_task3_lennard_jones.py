# ======================================================================================
# SCRIPT: run_task3_lennard_jones.py
# TASK: Task 3 Finale - Integrazione Energetica di Lennard-Jones e Correlazione Globale
#
# COSA FA QUESTO CODICE:
# 1. Carica l'elenco dei file proiettati della PCA (Task 3b) e l'Excel dell'Affinità Sperimentale.
# 2. Per ciascuna delle 118 proteine, estrae i punti spaziali e calcola la normalizzazione locale.
# 3. Utilizza la logica Cross-Interfaccia tramite due KD-Tree di SciPy:
#    - Punti Ligando -> Distanza minima dagli atomi del Recettore PDB.
#    - Punti Recettore -> Distanza minima dagli atomi del Ligando PDB.
# 4. Sintonizza ogni punto con il potenziale di Lennard-Jones (LJ) (sigma = 3.5 Å):
#    - Peso = 0 per r < 2.0 Å (repulsione) e r > 6.0 Å (interazione nulla).
#    - Peso compreso tra 0 e 1 per la buca attrattiva di van der Waals.
# 5. Calcola la COMPLEMENTARITÀ GEOMETRICA MEDIA PESATA ENERGETICAMENTE per il complesso.
# 6. Unisce i risultati numerici con le affinità sperimentali (Inibizione Costante Ki/Kd espressa come logaritmo o dG).
# 7. Calcola il Coefficiente di Correlazione di Pearson finalizzato per verificare l'impatto della fisica.
#
# INPUT: 
# - outputs/task3/projected_points/*_projected.csv (Coordinate geometriche)
# - all/<protein_folder>/receptor_interface_patch.pdb & ligand_interface_patch.pdb (Strutture)
# - data/Affinity_Benchmark_v5.5.xlsx (Affinità sperimentali da correlare)
#
# OUTPUT:
# - outputs/task3/summary_lennard_jones_affinity.csv (Tabella finale dei complessi con score chimico-geometrico)
# - Stampa a terminale del valore di Pearson (R) e del p-value.
# ======================================================================================

import os
import glob
import pandas as pd
import numpy as np
from Bio.PDB import PDBParser
from scipy.spatial import KDTree
from scipy.stats import pearsonr

# =====================================================================
# CONFIGURAZIONE PERCORSI
# =====================================================================
PROJECTED_DIR = "outputs/task3/projected_points/"
PDB_DIR = "all/"
EXCEL_PATH = "data/Affinity Benchmark v5.5.xlsx"  # corretto: nome file con spazi, coerente con scripts/task1/run_interface_extraction.py
OUTPUT_FILE = "outputs/task3/summary_lennard_jones_affinity.csv"

print("=== START PIPELINE: LENNARD-JONES ENERGY FILTER & PEARSON CORRELATION ===")

# Caricamento del benchmark sperimentale delle affinità
df_excel = pd.read_excel(EXCEL_PATH)
# Pulizia dell'ID PDB portandolo in maiuscolo per evitare mancate corrispondenze
df_excel['PDB'] = df_excel['PDB'].str.upper().str.strip()

# Raccolta di tutti i file geometrici proiettati nel Task 3b
file_list = glob.glob(os.path.join(PROJECTED_DIR, "*_projected.csv"))
print(f"Trovati {len(file_list)} file proiettati da analizzare energeticamente.\n")

parser = PDBParser(QUIET=True)
protein_results = []

# Funzione ausiliaria per estrarre le coordinate tridimensionali dai file PDB
def get_pdb_coords(folder, pdb_name):
    path = os.path.join(PDB_DIR, folder, pdb_name)
    coords = []
    if os.path.exists(path):
        structure = parser.get_structure(pdb_name, path)
        for atom in structure.get_atoms():
            coords.append(atom.get_coord())
    return np.array(coords)

# =====================================================================
# LOOP ENERGETICO SU TUTTE LE PROTEINE
# =====================================================================
for index, file_path in enumerate(file_list, 1):
    filename = os.path.basename(file_path)
    protein_folder = filename.replace("_projected.csv", "")
    protein_id = protein_folder.split('_')[-1].upper()
    
    try:
        # 1. Caricamento dei punti geometrici proiettati
        df_points = pd.read_csv(file_path)
        
        # 2. Normalizzazione locale della complementarità (1 = massimo incastro geometrico)
        z_min = df_points['zernike_distance'].min()
        z_max = df_points['zernike_distance'].max()
        # Gestione caso limite divisione per zero
        if z_max == z_min:
            df_points['normalized_complementarity'] = 1.0
        else:
            df_points['normalized_complementarity'] = 1 - (df_points['zernike_distance'] - z_min) / (z_max - z_min)
            
        # 3. Estrazione atomi reali dal PDB per la logica Cross
        atoms_rec = get_pdb_coords(protein_folder, "receptor_interface_patch.pdb")
        atoms_lig = get_pdb_coords(protein_folder, "ligand_interface_patch.pdb")
        
        # Salto l'analisi se uno dei due file PDB fondamentali è vuoto o mancante
        if len(atoms_rec) == 0 or len(atoms_lig) == 0:
            print(f"⚠️ Salto {protein_folder}: Atomi PDB mancanti.")
            continue
            
        # 4. Costruzione dei KD-Tree Spaziali
        tree_rec = KDTree(atoms_rec)
        tree_lig = KDTree(atoms_lig)
        
        weights = []
        sigma = 3.5
        
        # 5. Calcolo delle distanze incrociate e applicazione del filtro energetico
        for idx, row in df_points.iterrows():
            point_xyz = row[['x', 'y', 'z']].values.astype(float)
            
            if row['source'] == 'ligand':
                dist, _ = tree_rec.query(point_xyz)
            else:
                dist, _ = tree_lig.query(point_xyz)
                
            # Equazione troncata di Lennard-Jones
            if dist < 2.0 or dist > 6.0:
                w = 0.0
            else:
                w = -( ((sigma / dist)**12) - 2 * ((sigma / dist)**6) )
                w = max(0.0, min(1.0, w))
            weights.append(w)
            
        df_points['lj_weight'] = weights
        # Calcolo della complementarità pesata localmente
        df_points['weighted_comp'] = df_points['normalized_complementarity'] * df_points['lj_weight']
        
        # 6. CALCOLO METRICHE INTEGRATE MEDIE PER IL COMPLESSO
        # Se la somma dei pesi è zero (caso patologico), usiamo la media aritmetica semplice
        sum_weights = df_points['lj_weight'].sum()
        if sum_weights > 0:
            weighted_score_mean = df_points['weighted_comp'].sum() / sum_weights
        else:
            weighted_score_mean = 0.0
            
        # Memorizzazione dello score chimico-geometrico combinato
        protein_results.append({
            'protein_id': protein_id,
            'folder_name': protein_folder,
            'pure_geometric_mean': df_points['normalized_complementarity'].mean(),
            'weighted_energy_geometric_mean': weighted_score_mean
        })
        
        if index % 20 == 0 or index == len(file_list):
            print(f"[{index}/{len(file_list)}] Calcolata energia LJ per {protein_folder}")
            
    except Exception as e:
        print(f"❌ Errore nel complesso {protein_folder}: {str(e)}")

# Creazione del dataframe dei risultati aggregati
df_results = pd.DataFrame(protein_results)

# =====================================================================
# MERGE CON I DATI SPERIMENTALI E CALCOLO CORRELAZIONE DI PEARSON
# =====================================================================
# Uniamo la nostra tabella geometrica con l'Excel usando l'ID PDB come chiave di giunzione
df_final = pd.merge(df_results, df_excel, left_on='protein_id', right_on='PDB', how='inner')

if df_final.empty:
    print("❌ Errore critico: Nessuna corrispondenza trovata tra gli ID PDB degli script e l'Excel.")
else:
    print(f"\n✅ Merge completato! {len(df_final)} complessi mappati con successo con i dati sperimentali.")
    
    # Rilevamento dinamico della colonna di affinità sperimentale
    affinity_col = [col for col in df_final.columns if any(k in str(col).upper() for k in ['DG', 'AFFINITY', 'PK', 'KD', 'KI', 'VALUE'])]
    
    if affinity_col:
        target_y = affinity_col[0]
        print(f"Colonna sperimentale rilevata per la correlazione: '{target_y}'")
        
        # --- CORREZIONE DEL BUG DI TIPO (UFuncNoLoopError) ---
        # Convertiamo la colonna in numerica, forzando i testi non validi a NaN
        df_final[target_y] = pd.to_numeric(df_final[target_y], errors='coerce')
        
        # Eliminiamo eventuali righe che contengono NaN nella colonna sperimentale o nei nostri score
        df_final = df_final.dropna(subset=[target_y, 'pure_geometric_mean', 'weighted_energy_geometric_mean'])
        print(f"Punti validi post-pulizia numerica: {len(df_final)}")
        
        # --- AGGIUNTA FISICA: CALCOLO DEL LOGARITMO DI KD ---
        # Se la colonna è proprio il KD, per avere senso fisico dobbiamo analizzare il logaritmo (proporzionale a dG)
        if 'KD' in target_y.upper() or 'KI' in target_y.upper():
            # Evitiamo logaritmi di zero o valori negativi fisicamente impossibili
            df_final = df_final[df_final[target_y] > 0]
            df_final['log_affinity'] = np.log10(df_final[target_y])
            y_data_col = 'log_affinity'
            print("🔬 Rilevata costante di dissociazione: la correlazione verrà calcolata su log10(KD) per linearità biofisica.")
        else:
            y_data_col = target_y

        # Calcolo Pearson per la geometria PURA (Task 2)
        r_geo, p_geo = pearsonr(df_final['pure_geometric_mean'], df_final[y_data_col])
        
        # Calcolo Pearson per la geometria PESATA CON LENNARD-JONES (Task 3)
        r_lj, p_lj = pearsonr(df_final['weighted_energy_geometric_mean'], df_final[y_data_col])
        
        print("\n=========================================================")
        print("📊 RISULTATI COMPARATIVI DI CORRELAZIONE DI PEARSON 📊")
        print("=========================================================")
        print(f"Variabile sperimentale usata: {y_data_col}")
        print(f"1. Geometria Pura (Solo Zernike):  R = {r_geo:.4f}  (p-value = {p_geo:.2e})")
        print(f"2. Geometria + Lennard-Jones:      R = {r_lj:.4f}  (p-value = {p_lj:.2e})")
        print("=========================================================")
        
        if abs(r_lj) > abs(r_geo):
            print(f"🎉 Successo! Il vincolo energetico di van der Waals ha MIGLIORATO la correlazione di {abs(r_lj)-abs(r_geo):.4f}!")
        else:
            print("Fisica inclusa, ma la geometria pura mantiene un segnale strutturale dominante.")
    else:
        print("⚠️ Impossibile calcolare Pearson: nessuna colonna di affinità riconosciuta nell'Excel.")

# Salvataggio della tabella comparativa completa per i grafici di tesi
df_final.to_csv(OUTPUT_FILE, index=False)
print(f"\nTabella comparativa finale salvata in: {OUTPUT_FILE}")