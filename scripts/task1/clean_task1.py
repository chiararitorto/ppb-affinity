import os

# Percorso della cartella da pulire
task1_dir = "outputs/task1"

# Lista dei file che vogliamo ASSOLUTAMENTE mantenere
files_to_keep = [
    "ligand_surface_patch.csv",
    "receptor_surface_patch.csv",
    "ligand_interface_residues.csv",
    "receptor_interface_residues.csv",
    "metadata.json"
]

print("Inizio pulizia della cartella task1...")

# Ciclo attraverso le cartelle dei dimeri
for folder in os.listdir(task1_dir):
    folder_path = os.path.join(task1_dir, folder)
    
    # Verifichiamo che sia effettivamente una cartella
    if os.path.isdir(folder_path):
        # Elenchiamo tutti i file all'interno della cartella del PDB
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            
            # Se il file non è nella nostra lista "salva-vita", lo eliminiamo
            if filename not in files_to_keep:
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                    elif os.path.isdir(file_path):
                        import shutil
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"Errore durante l'eliminazione di {file_path}: {e}")

print("Pulizia completata! Ora task1 contiene solo i patch delle interfacce.")