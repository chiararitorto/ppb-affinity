# ======================================================================================
# SCRIPT: plot_lennard_jones_correlation.py
# TASK: Task 3 - Generazione Grafici di Correlazione e Analisi per Cluster Chimico
#
# COSA FA QUESTO CODICE:
# 1. Carica la tabella comparativa finale generata nel passaggio precedente.
# 2. Cerca colonne di raggruppamento nell'Excel (es. 'Type', 'Category', 'Class') per 
#    separare i complessi in base alla loro famiglia biologica.
# 3. Genera un grafico a due pannelli (Subplots) affiancati:
#    - Pannello A: Geometria Pura (Zernike) vs log10(KD)
#    - Pannello B: Geometria + Lennard-Jones vs log10(KD)
# 4. Traccia le rette di regressione lineare per mostrare l'andamento del trend.
# 5. Salva l'immagine in alta risoluzione pronta per essere inserita nel capitolo 3 della tesi.
#
# INPUT: 
# - outputs/task3/summary_lennard_jones_affinity.csv
#
# OUTPUT:
# - outputs/task3/final_correlation_plot.png
# ======================================================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# Set estetico per i grafici della tesi (stile pulito ed elegante)
sns.set_theme(style="ticks", context="talk")
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.titlesize': 16
})

INPUT_PATH = "outputs/task3/summary_lennard_jones_affinity.csv"
OUTPUT_PLOT = "outputs/task3/final_correlation_plot.png"

print("=== START PIPELINE: PLOTTING CORRELATION RESULTS ===")

if not os.path.exists(INPUT_PATH):
    print(f"❌ Errore: File {INPUT_PATH} non trovato. Esegui prima lo script di calcolo.")
    exit()

# Caricamento dei dati integrati
df = pd.read_csv(INPUT_PATH)

# Identificazione dinamica delle colonne chiave salvate nel summary
y_col = 'log_affinity'
x_geo = 'pure_geometric_mean'
x_lj = 'weighted_energy_geometric_mean'

# Cerchiamo se esiste una colonna di classificazione biologica dei complessi
cluster_col = None
possible_cluster_keywords = ['TYPE', 'CATEGORY', 'CLASS', 'GROUP', 'COMPLEX']
for col in df.columns:
    if any(k in str(col).upper() for k in possible_cluster_keywords):
        cluster_col = col
        break

# Inizializzazione della figura a due pannelli
fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

# Ricalcolo al volo dei coefficienti per i titoli
r_geo, _ = pearsonr(df[x_geo], df[y_col])
r_lj, _ = pearsonr(df[x_lj], df[y_col])

# --- PANNELLO 1: GEOMETRIA PURA ---
print("Disegno Pannello A (Geometria Pura)...")
if cluster_col:
    print(f"-> Rilevato cluster biologico nella colonna: '{cluster_col}'")
    sns.scatterplot(data=df, x=x_geo, y=y_col, hue=cluster_col, palette="Set2", alpha=0.8, ax=axes[0], s=60)
    axes[0].legend(title=cluster_col, fontsize=10, title_fontsize=11, loc='best')
else:
    sns.scatterplot(data=df, x=x_geo, y=y_col, color="steelblue", alpha=0.7, ax=axes[0], s=60)

# CORREZIONE: Impacchettiamo lo stile della linea in line_kws
sns.regplot(data=df, x=x_geo, y=y_col, scatter=False, ax=axes[0], color="dimgray", robust=True, line_kws={"linestyle": "--"})
axes[0].set_title(f"A) Geometria Pura (Solo Zernike)\n$R = {r_geo:.4f}$")
axes[0].set_xlabel("Score Geometrico Medio")
axes[0].set_ylabel("Affinità Sperimentale $\log_{10}(K_D)$")
axes[0].grid(True, linestyle=":", alpha=0.5)

# --- PANNELLO 2: GEOMETRIA + LENNARD-JONES ---
print("Disegno Pannello B (Filtro Energetico)...")
if cluster_col:
    sns.scatterplot(data=df, x=x_lj, y=y_col, hue=cluster_col, palette="Set2", alpha=0.8, ax=axes[1], s=60, legend=False)
else:
    sns.scatterplot(data=df, x=x_lj, y=y_col, color="crimson", alpha=0.7, ax=axes[1], s=60)

# CORREZIONE: Impacchettiamo lo stile della linea in line_kws
sns.regplot(data=df, x=x_lj, y=y_col, scatter=False, ax=axes[1], color="dimgray", robust=True, line_kws={"linestyle": "--"})
axes[1].set_title(f"B) Geometria + Lennard-Jones\n$R = {r_lj:.4f}$")
axes[1].set_xlabel("Score Pesato Energeticamente")
axes[1].grid(True, linestyle=":", alpha=0.5)

# Titolo globale superiore e ottimizzazione spazi
plt.suptitle("Analisi Comparativa di Correlazione: Geometria vs Vincolo Biofisico", y=0.98)
plt.tight_layout()

# Salvataggio del grafico finale in alta definizione
plt.savefig(OUTPUT_PLOT, dpi=300, bbox_inches='tight')
plt.show()

print(f"✅ Grafico finale salvato con successo in: {OUTPUT_PLOT}")

# --- ANALISI AGGIUNTIVA PER CLUSTER (SE DISPONIBILE) ---
if cluster_col:
    print("\n=== STUDIO DI CORRELAZIONE DISAGGREGATO PER FAMIGLIA BROWSIANA ===")
    unique_groups = df[cluster_col].dropna().unique()
    for group in unique_groups:
        df_group = df[df[cluster_col] == group]
        if len(df_group) >= 5: # Calcoliamo Pearson solo se ci sono abbastanza campioni
            try:
                r_g_pure, _ = pearsonr(df_group[x_geo], df_group[y_col])
                r_g_lj, _ = pearsonr(df_group[x_lj], df_group[y_col])
                print(f"Famiglia: {group:20} | Campioni: {len(df_group):3} | R Pura = {r_g_pure:7.4f} | R + LJ = {r_g_lj:7.4f}")
            except:
                pass