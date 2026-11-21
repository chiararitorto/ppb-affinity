import pandas as pd
import shutil
import os

# 1. Configurazione percorsi
excel_path = "data/Affinity Benchmark v5.5.xlsx"
all_outputs_dir = "outputs/all"
dimers_output_dir = "outputs/task1"

# Crea la cartella se non esiste
os.makedirs(dimers_output_dir, exist_ok=True)

# 2. Leggi il file (Pandas caricherà le colonne basandosi sulla prima riga)
df = pd.read_excel(excel_path)

# 3. Filtro per i Dimeri (PDB con 1 sola catena per lato)
# Cerchiamo le righe dove non ci sono virgole nelle colonne delle catene
is_dimer = (df['Ligand Chains'].astype(str).str.len() == 1) & \
           (df['Receptor Chains'].astype(str).str.len() == 1)

dimer_df = df[is_dimer]

print(f"Trovati {len(dimer_df)} dimeri su {len(df)} proteine totali.")

# 4. Spostamento file
for index, row in dimer_df.iterrows():
    pdb_id = row['PDB'].strip() # Uso 'PDB' come visto nello screenshot
    
    # La pipeline della Task 1 scrive una cartella per RIGA del dataset ({row_index:05d}_{PDB}).
    # Si cerca quindi la cartella della riga corrente, non la prima che termina con il PDB ID:
    # altrimenti un PDB presente su piu' righe (es. 1NVU, catene ligando Q e R) verrebbe
    # copiato una sola volta.
    candidate = f"{index:05d}_{pdb_id}"
    found_folder = candidate if os.path.isdir(os.path.join(all_outputs_dir, candidate)) else None

    if found_folder:
        src = os.path.join(all_outputs_dir, found_folder)
        dst = os.path.join(dimers_output_dir, found_folder)
        
        if not os.path.exists(dst):
            shutil.copytree(src, dst)
            print(f"Spostato dimero: {found_folder}")
    else:
        print(f"Cartella non trovata per: {pdb_id}")

print("\nTask 1 (Dimeri) popolata correttamente!")