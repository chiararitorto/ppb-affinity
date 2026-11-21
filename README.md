# Mapping Physicochemical Features onto 2D Interface Planes for Protein–Protein Binding Affinity Prediction

Progetto per il corso *Advanced Machine Learning for Physics* (Sapienza Università di Roma, A.A. 2025/2026).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chiararitorto/ppb-affinity/blob/main/PPB_Affinity.ipynb)

## Descrizione

Framework fisicamente motivato per la predizione dell'affinità di legame ($K_D$ / $\Delta G$) tra coppie di
proteine, a partire dal dataset *PPB-Affinity / Affinity Benchmark v5.5*, tramite mappe 2D di complementarità
(forma, elettrostatica, idrofobicità) usate come input di una CNN. Proposta di progetto: E. Milanetti, G. Ruocco.

Il report completo è in [`report/PPB-affinity.pdf`](report/PPB-affinity.pdf) (sorgente: `report/PPB-affinity.tex`).

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
│   ├── affinity_dataset.csv    # Affinity Benchmark v5.5 (207 complessi, sep=';')
│   └── homology_groups.csv     # gruppi di omologia di sequenza dei 120 dimeri (split per omologia)
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
│   │   ├── run_task3_projection.py       # proiezione PCA sul piano medio di interfaccia
│   │   ├── run_task3_lennard_jones.py    # filtro energetico e correlazione con l'affinità
│   │   ├── generate_interface_images.py  # tensori 2D (3x32x32), residuo esatto per punto
│   │   └── plot_lennard_jones_correlation.py
│   ├── task4/
│   │   ├── models.py              # ProteinInterfaceDataset (senza leakage) + InterfaceCNN
│   │   ├── train_final.py         # training a due livelli: split interno per il checkpoint,
│   │   │                          # standardizzazione Z-score calcolata solo sul training set del fold
│   │   ├── evaluate_robustness.py # CV ripetuta su 5 seed, fold casuali e per omologia, con baseline
│   │   └── diagnose_dataset.py    # script diagnostico di integrità dati (esclusioni 1NVU/1UUG)
│   └── task4b/                    # baseline di geometric deep learning (CNN 3D), per il confronto
│       │                          # richiesto dalla traccia con la pipeline 2D principale
│       ├── voxelize_interface.py  # converte le patch della Task 1 in tensori 3D a 4 canali
│       │                          # (densita' ligando, densita' recettore, carica, idrofobicita'),
│       │                          # senza passare dal matching Zernike ne' dalla proiezione PCA
│       ├── models_3d.py           # ProteinInterfaceDataset3D (senza leakage) + Interface3DCNN
│       └── train_3d.py            # stesso protocollo a due livelli di train_final.py, adattato
│                                  # al modello 3D; su Colab legge/scrive automaticamente su
│                                  # Google Drive per sopravvivere a disconnessioni del runtime
│   └── make_report_figures.py     # rigenera le figure di Task 3, 4 e 4b usate nel report
│
├── outputs/
│   ├── task1/
│   │   ├── manifest.csv                  # esito dell'elaborazione per ogni complesso
│   │   └── {indice}_{PDB}/               # superfici e residui di interfaccia dei 120 dimeri
│   ├── task2/
│   │   └── {indice}_{PDB}/               # score di complementarità Zernike (118 complessi)
│   ├── task3/
│   │   ├── summary_geometrico_pca_task3.csv       # planarità dell'interfaccia (118 complessi)
│   │   ├── summary_lennard_jones_affinity.csv     # score geometrico/energetico + affinità (118 complessi)
│   │   ├── final_correlation_plot.png             # correlazione score vs affinità sperimentale
│   │   ├── projected_points/             # punti proiettati sul piano PCA (118 complessi)
│   │   └── interface_maps/               # tensori 3x32x32 (118 complessi)
│   ├── task4_final/
│   │   ├── cnn_final_fold_1.pth ... cnn_final_fold_5.pth   # pesi CNN 2D, un fold ciascuno
│   │   ├── oof_predictions.csv                              # predizioni out-of-fold (116 complessi)
│   │   ├── robustness_runs.csv, robustness_summary.csv      # risultati di evaluate_robustness.py
│   │   └── robustness_oof_{split}_seed{n}.csv               # predizioni OOF per seed e tipo di fold
│   └── task4_dl/                  # risultati della baseline geometric DL (Task 4b, CNN 3D)
│       ├── voxel_maps/00000_1KTZ_voxel.npy         # tensore 4x16x16x16 di esempio
│       └── results/
│           ├── cnn3d_fold_1.pth ... cnn3d_fold_5.pth   # pesi CNN 3D, un fold ciascuno
│           └── oof_predictions_3d.csv                   # predizioni out-of-fold (118 complessi)
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
│   ├── task4_oof_scatter.png
│   ├── task4_robustness.png               # CNN 2D vs baseline, fold casuali e per omologia
│   ├── task4b_voxel_preview_1ktz.png      # sezione del volume voxel della CNN 3D
│   └── task4_vs_task4b_comparison.png     # confronto affiancato CNN 2D vs CNN 3D
│
└── report/
    ├── PPB-affinity.tex
    └── PPB-affinity.pdf
```

## Stato di avanzamento

- [x] Task 1 — Interface Identification and Surface Patch Extraction
- [x] Task 2 — Zernike-Based Complementarity Mapping (118/120 complessi completati)
- [x] Task 3 — Construction of 2D Complementarity Planes
- [x] Task 4 — CNN-Based Binding Affinity Prediction (116/118 complessi: 2 esclusi per controllo di integrità dati). Su 5 seed: R = 0.40 ± 0.05 con fold casuali e R = 0.41 ± 0.09 con fold per omologia, contro 0.33 / 0.28 di una baseline ridge su dimensione e composizione dell'interfaccia valutata sugli stessi fold
- [x] Task 4b — Baseline di geometric deep learning (CNN 3D end-to-end su voxel grezzi, senza Zernike né proiezione PCA): 118 complessi, R=0.46 (run singolo), comparabile alla pipeline 2D principale
- [ ] Task 5 (opzionale) — Affinity Maturation / Docking Pose Discrimination
