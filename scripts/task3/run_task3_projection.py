# ======================================================================================
# SCRIPT: run_task3_projection.py
# TASK: Task 3 - Proiezione 2D delle Interfacce via PCA e Calcolo delle Distanze PC3
#
# COSA FA QUESTO CODICE:
# 1. Scansiona ricorsivamente l'output del Task 2 alla ricerca dei file di complementarità
#    geometrica simmetrica per tutte le proteine del dataset.
# 2. Per ciascuna proteina, estrae le coordinate spaziali cartesiane 3D (X, Y, Z) dei punti.
# 3. Applica l'Analisi delle Componenti Principali (PCA) a 3 componenti:
#    - PC1 e PC2 definiscono gli assi ortogonali del "Piano Medio" ottimale dell'interfaccia.
#    - PC3 rappresenta l'asse ortogonale al piano; di conseguenza, la coordinata proiettata su PC3 
#      esprime la distanza geometrica esatta di ogni punto dal piano medio del complesso.
# 4. Salva un nuovo file .csv inserendo le tre nuove colonne proiettate (PC1, PC2, PC3).
# 5. Calcola la varianza spiegata dal piano (PC1 + PC2) per quantificare matematicamente 
#    il grado di planarità della patch d'interfaccia.
# 6. Genera un report statistico globale (Summary) con medie, deviazioni standard e range.
#
# INPUT: 
# - outputs/task2/*/zernike_complementarity_symmetric.csv
#
# OUTPUT:
# - outputs/task3/projected_points/<folder_name>_projected.csv (File proiettati singoli)
# - outputs/task3/summary_geometrico_pca_task3.csv (Tabella di riepilogo del dataset)
# ======================================================================================

import os
import pandas as pd
import numpy as np
import glob
from sklearn.decomposition import PCA

# =====================================================================
# CONFIGURAZIONE PERCORSI E DIRECTORY
# =====================================================================
# Pattern di ricerca ricorsivo per raccogliere tutti i file simmetrici generati nel Task 2
PATH_SYMMETRIC = "outputs/task2/*/zernike_complementarity_symmetric.csv"
OUTPUT_TASK3_DIR = "outputs/task3/projected_points/"

# Creazione della cartella di output per ospitare le nuvole di punti proiettate
os.makedirs(OUTPUT_TASK3_DIR, exist_ok=True)

print("=== START PIPELINE TASK 3b: 2D PROJECTION & PC3 DISTANCE ===")

# Generazione della lista globale di tutti i file da elaborare sul Mac
file_list = glob.glob(PATH_SYMMETRIC, recursive=True)
print(f"Trovati {len(file_list)} file simmetrici da proiettare.\n")

# Lista d'appoggio per memorizzare i dizionari dei risultati geometrici di ciascun complesso
pca_results_list = []

# =====================================================================
# PIPELINE: CICLO DI ELABORAZIONE GEOMETRICA SUL DATASET
# =====================================================================
for index, file_path in enumerate(file_list, 1):
    # Isolamento del nome della cartella nativa di Leonardo (es. 00150_2AQ3)
    protein_folder = os.path.basename(os.path.dirname(file_path))
    # Estrazione dell'ID PDB puro in lettere maiuscole per il matching finale
    protein_id = protein_folder.split('_')[-1].upper()
    
    try:
        # 1. Lettura del file di complementarità simmetrico
        df_interface = pd.read_csv(file_path)
        # Estrazione della matrice di coordinate spaziali 3D (N punti x 3 colonne)
        X_space = df_interface[['x', 'y', 'z']].values
        
        # 2. Inizializzazione e Fit della PCA a 3 Dimensioni
        pca = PCA(n_components=3)
        # Rotazione rigida dello spazio: trasforma XYZ nelle componenti principali (PC1, PC2, PC3)
        X_pca = pca.fit_transform(X_space) 
        
        # 3. Integrazione delle coordinate ruotate nel DataFrame originale
        df_interface['PC1'] = X_pca[:, 0]
        df_interface['PC2'] = X_pca[:, 1]
        df_interface['PC3'] = X_pca[:, 2] # PC3 = Distanza con segno rispetto al piano d'interfaccia
        
        # 4. Scrittura del file specifico contenente la proiezione geometrica
        output_file_name = f"{protein_folder}_projected.csv"
        df_interface.to_csv(os.path.join(OUTPUT_TASK3_DIR, output_file_name), index=False)
        
        # 5. Analisi delle frazioni di Varianza Spiegata
        var_pc1, var_pc2, var_pc3 = pca.explained_variance_ratio_
        # La planarità è definita dalla somma della varianza trattenuta dalle prime due componenti
        total_plane_var = (var_pc1 + var_pc2) * 100
        
        # Archiviazione delle feature geometriche estratte
        pca_results_list.append({
            'folder_name': protein_folder,
            'protein_id': protein_id,
            'num_points': X_space.shape[0],
            'PC1_var': var_pc1,
            'PC2_var': var_pc2,
            'PC3_var': var_pc3,
            'plane_variance_pct': total_plane_var
        })
        
        # Monitoraggio dell'avanzamento a terminale (stampa ogni 20 proteine e alla fine)
        if index % 20 == 0 or index == len(file_list):
            print(f"[{index}/{len(file_list)}] Processata {protein_folder}: Punti={X_space.shape[0]} | Varianza Piano={total_plane_var:.2f}%")
            
    except Exception as e:
        print(f"⚠️ Errore durante l'elaborazione di {protein_folder}: {str(e)}")

# Conversione dei dati geometrici aggregati in un DataFrame Pandas strutturato
df_pca_summary = pd.DataFrame(pca_results_list)

# Salvataggio del file di riepilogo statistico nella cartella di Task 3
df_pca_summary.to_csv("outputs/task3/summary_geometrico_pca_task3.csv", index=False)

# =====================================================================
# ANALISI STATISTICA DESCRITTIVA GLOBALE DEL DATASET
# =====================================================================
print("\n=== ANALISI STATISTICA GEOMETRICA DEL DATASET ===")
mean_plane_var = df_pca_summary['plane_variance_pct'].mean() # Valore medio di planarità
std_plane_var = df_pca_summary['plane_variance_pct'].std()   # Deviazione standard
min_plane_var = df_pca_summary['plane_variance_pct'].min()   # Complesso meno planare (più curvo)
max_plane_var = df_pca_summary['plane_variance_pct'].max()   # Complesso più planare (più piatto)

print(f"Varianza media spiegata dal Piano Medio (PC1+PC2): {mean_plane_var:.2f}% ± {std_plane_var:.2f}%")
print(f"Range Varianza Piano: Min = {min_plane_var:.2f}% | Max = {max_plane_var:.2f}%")

print(f"\nFile 'summary_geometrico_pca_task3.csv' salvato con successo in outputs/task3!")
print(f"Tutti i file singoli proiettati sono in: {OUTPUT_TASK3_DIR}")