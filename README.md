# Mapping Physicochemical Features onto 2D Interface Planes for Protein–Protein Binding Affinity Prediction

Progetto per il corso *Advanced Machine Learning for Physics* (Sapienza Università di Roma, A.A. 2025/2026).

<!--
Dopo aver creato il repository su GitHub, sostituisci chiararitorto nel link qui sotto con il tuo
username reale, così il badge apre direttamente il notebook in Google Colab.
-->
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chiararitorto/ppb-affinity/blob/main/PPB_Affinity.ipynb)

## Descrizione

Framework fisicamente motivato per la predizione dell'affinità di legame ($K_D$ / $\Delta G$) tra coppie di
proteine, a partire dal dataset *PPB-Affinity / Affinity Benchmark v5.5*, tramite mappe 2D di complementarità
(forma, elettrostatica, idrofobicità) usate come input di una CNN. Proposta di progetto: E. Milanetti, G. Ruocco.

Il report completo è in [`report/report.pdf`](report/report.pdf) (sorgente Word in `report/report.docx`).

## Come eseguire

- **Google Colab (consigliato):** clicca sul badge in alto. Il notebook clona questo repository e lavora
  direttamente sui file caricati su GitHub — nessuna installazione locale necessaria.
- **In locale:**
  ```bash
  git clone https://github.com/chiararitorto/ppb-affinity.git
  cd ppb-affinity
  pip install -r requirements.txt
  jupyter notebook PPB_Affinity.ipynb
  ```

Il notebook carica risultati già calcolati (CSV, `.npy`, pesi del modello): non richiede ChimeraX né un
cluster HPC per essere eseguito e ispezionato.

## Struttura del repository

```
ppb-affinity/
├── README.md
├── requirements.txt
├── PPB_Affinity.ipynb          # notebook narrativo, apribile in Colab
│
├── data/
│   └── affinity_dataset.csv    # Affinity Benchmark v5.5 (207 complessi, sep=';')
│
├── src/                        # moduli riutilizzabili della pipeline (Task 1)
│   ├── __init__.py
│   ├── pdb.py                  # parsing/scrittura PDB, download da RCSB
│   ├── interface.py            # identificazione residui di interfaccia (cutoff 5 Å)
│   ├── surface.py              # generazione superficie "sampler" + densità approssimata
│   ├── chimerax.py             # generazione superficie via ChimeraX headless
│   ├── dataset.py               # lettura dei record del dataset
│   └── pipeline.py             # orchestrazione della pipeline Task 1
│
├── scripts/
│   ├── task1/
│   │   ├── run_interface_extraction.py   # CLI della pipeline Task 1
│   │   ├── filter_dimers.py              # selezione dei complessi dimerici
│   │   └── clean_task1.py                # pulizia delle cartelle di output
│   ├── task2/
│   │   └── run_task2_parallel.py         # mappatura Zernike, eseguito su CINECA Leonardo
│   ├── task3/
│   │   ├── run_task3b.py                 # proiezione PCA sul piano medio di interfaccia
│   │   ├── run_task3_lennard_jones.py    # filtro energetico e correlazione con l'affinità
│   │   ├── generate_interface_images.py  # costruzione dei tensori 2D (3x32x32)
│   │   └── plot_lennard_jones_correlation.py
│   └── task4/
│       ├── models.py              # ProteinInterfaceDataset (senza leakage) + InterfaceCNN
│       ├── train_final.py         # training a due livelli: split interno per il checkpoint,
│       │                          # standardizzazione Z-score calcolata solo sul training set del fold
│       └── diagnose_dataset.py    # script diagnostico di integrità dati (esclusioni 1NVU/1UUG)
│
├── outputs/
│   ├── task1/
│   │   ├── manifest.csv                  # esito dell'elaborazione per ogni complesso
│   │   └── 00000_1KTZ/                   # esempio di output per un complesso
│   ├── task2/
│   │   └── 00000_1KTZ/                   # esempio di output: score di complementarità Zernike
│   ├── task3/
│   │   ├── summary_geometrico_pca_task3.csv       # planarità dell'interfaccia (118 complessi)
│   │   ├── summary_lennard_jones_affinity.csv     # score geometrico/energetico + affinità (118 complessi)
│   │   ├── final_correlation_plot.png             # correlazione score vs affinità sperimentale
│   │   ├── projected_points/00000_1KTZ_projected.csv
│   │   └── interface_maps/00000_1KTZ_tensor.npy   # tensore 3x32x32 di esempio
│   └── task4_final/
│       ├── cnn_final_fold_1.pth ... cnn_final_fold_5.pth   # pesi CNN, un fold ciascuno
│       └── oof_predictions.csv                              # predizioni out-of-fold (116 complessi)
│
├── figures/
│   ├── interface_hist.pdf
│   ├── interface_hist.png
│   ├── interface_comparison_1ktz_2oza.png
│   ├── task2_score_histogram.png
│   ├── task2_score_scatter3d.png
│   ├── task3_planarity_histogram.png
│   ├── task3_tensor_preview_1ktz.png
│   ├── task3_examples_grid.png
│   ├── final_correlation_plot.png
│   └── task4_oof_scatter.png
│
└── report/
    ├── report.docx
    └── report.pdf
```

## Stato di avanzamento

- [x] Task 1 — Interface Identification and Surface Patch Extraction
- [x] Task 2 — Zernike-Based Complementarity Mapping (118/120 complessi completati)
- [x] Task 3 — Construction of 2D Complementarity Planes
- [x] Task 4 — CNN-Based Binding Affinity Prediction (116/118 complessi: 2 esclusi per controllo di integrità dati; risultato finale R=0.42, R²≈18%, p<10⁻⁵)
- [ ] Task 5 (opzionale) — Affinity Maturation / Docking Pose Discrimination
