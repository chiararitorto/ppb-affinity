# ======================================================================================
# SCRIPT: make_report_figures.py
# Rigenera le figure del report che dipendono dai tensori della Task 3 e dalle predizioni
# della Task 4/4b:
#   figures/task3_tensor_preview_1ktz.png, figures/task3_examples_grid.png,
#   figures/task4_oof_scatter.png, figures/task4b_voxel_preview_1ktz.png,
#   figures/task4_vs_task4b_comparison.png, figures/task4_robustness.png
# Uso (dalla radice del repo): python scripts/make_report_figures.py
# ======================================================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

MAPS = "outputs/task3/interface_maps"
FIG = "figures"
os.makedirs(FIG, exist_ok=True)


def tensor_panels(axes, tensor, name=None):
    ch0 = tensor[0]
    lo, hi = ch0.min(), ch0.max()
    if lo == hi:
        lo, hi = lo - 0.1, hi + 0.1
    specs = [(ch0, 'viridis', lo, hi, "Ch 1: Shape Complementarity"),
             (tensor[1], 'bwr', -1.0, 1.0, "Ch 2: Electrostatic Charge"),
             (tensor[2], 'YlOrBr', -4.5, 4.5, "Ch 3: Hydrophobicity Profile")]
    for ax, (img, cmap, vmin, vmax, title) in zip(axes, specs):
        im = ax.imshow(img, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


# --- Task 3: anteprima 1KTZ e griglia di esempi --------------------------------------
t = np.load(os.path.join(MAPS, "00000_1KTZ_tensor.npy"))
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
tensor_panels(axes, t)
plt.suptitle("2D Multi-channel Map: 00000_1KTZ", y=0.98)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "task3_tensor_preview_1ktz.png"), dpi=150, bbox_inches='tight')
plt.close()

examples = ["00094_3SGB", "00112_1OC0", "00145_2SIC"]
fig = plt.figure(figsize=(15, 15))
subfigs = fig.subfigures(3, 1)
for sf, name in zip(subfigs, examples):
    axes = sf.subplots(1, 3)
    tensor_panels(axes, np.load(os.path.join(MAPS, f"{name}_tensor.npy")))
    sf.suptitle(f"2D Multi-channel Map: {name}")
plt.savefig(os.path.join(FIG, "task3_examples_grid.png"), dpi=150, bbox_inches='tight')
plt.close()


# --- Task 4: scatter OOF della CNN 2D ------------------------------------------------
def scatter(ax, df, title):
    ax.scatter(df['pkd_real'], df['pkd_predicted'], alpha=0.6, color='#4C72B0',
               edgecolor='k', linewidth=0.4)
    lo = min(df['pkd_real'].min(), df['pkd_predicted'].min()) - 0.5
    hi = max(df['pkd_real'].max(), df['pkd_predicted'].max()) + 0.5
    ax.plot([lo, hi], [lo, hi], 'r--', label='Predizione perfetta')
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    r = pearsonr(df['pkd_real'], df['pkd_predicted'])[0]
    ax.set_title(f"{title} (n={len(df)}, R={r:.2f})")
    ax.set_xlabel(r"$pK_D$ sperimentale")
    ax.set_ylabel(r"$pK_D$ predetto (OOF)")
    ax.legend(loc='upper left')
    return lo, hi


oof2d = pd.read_csv("outputs/task4_final/oof_predictions.csv")
oof3d = pd.read_csv("outputs/task4_dl/results/oof_predictions_3d.csv")

fig, ax = plt.subplots(figsize=(6, 6))
scatter(ax, oof2d, "CNN 2D: predizioni OOF")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "task4_oof_scatter.png"), dpi=200)
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
l1 = scatter(axes[0], oof2d, "CNN 2D")
l2 = scatter(axes[1], oof3d, "CNN 3D")
lo, hi = min(l1[0], l2[0]), max(l1[1], l2[1])
for ax in axes:
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "task4_vs_task4b_comparison.png"), dpi=200)
plt.close()

# --- Task 4b: sezione centrale del volume voxel di 1KTZ ------------------------------
v = np.load("outputs/task4_dl/voxel_maps/00000_1KTZ_voxel.npy")
z = v.shape[3] // 2
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
lig, rec = v[0, :, :, z], v[1, :, :, z]
rgb = np.ones(lig.shape + (3,))
for dens, color in ((lig, np.array([0.12, 0.47, 0.71])), (rec, np.array([1.0, 0.5, 0.05]))):
    w = (dens / max(dens.max(), 1e-9))[..., None]
    rgb = rgb * (1 - w) + color * w
axes[0].imshow(np.clip(np.transpose(rgb, (1, 0, 2)), 0, 1), origin="lower")
axes[0].set_title("Densità ligando (blu) e recettore (arancione)")
im = axes[1].imshow(v[2, :, :, z].T, cmap='bwr', origin='lower', vmin=-1, vmax=1)
axes[1].set_title("Carica elettrostatica")
plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
im = axes[2].imshow(v[3, :, :, z].T, cmap='YlOrBr', origin='lower', vmin=-4.5, vmax=4.5)
axes[2].set_title("Idrofobicità")
plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
plt.suptitle(f"Volume voxel (4,16,16,16) di 1KTZ: sezione z = {z}", y=0.98)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "task4b_voxel_preview_1ktz.png"), dpi=150, bbox_inches='tight')
plt.close()

# --- Task 4: robustezza (seed multipli, fold casuali e per omologia) -----------------
runs_path = "outputs/task4_final/robustness_runs.csv"
if os.path.exists(runs_path):
    runs = pd.read_csv(runs_path)
    order = ['ridge_residues', 'ridge_size_composition', 'cnn_2d']
    labels = ['Ridge:\nn. residui', 'Ridge:\ndim.+composiz.', 'CNN 2D']
    splits = [('random', 'Fold casuali', '#4C72B0'), ('homology', 'Fold per omologia', '#DD8452')]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    width = 0.36
    for k, (split, lab, col) in enumerate(splits):
        sub = runs[runs.split == split]
        if sub.empty:
            continue
        means = [sub[sub.model == m].pearson_r.mean() for m in order]
        stds = [sub[sub.model == m].pearson_r.std() for m in order]
        x = np.arange(len(order)) + (k - 0.5) * width
        ax.bar(x, means, width, yerr=stds, capsize=4, label=lab, color=col)
        for m, (xi, mu) in enumerate(zip(x, means)):
            pts = sub[sub.model == order[m]].pearson_r.values
            ax.scatter(np.full(len(pts), xi), pts, color='k', s=8, zorder=3)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Pearson R (OOF)")
    ax.set_title("CNN 2D e baseline sugli stessi fold (media ± dev. std sui seed)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "task4_robustness.png"), dpi=200)
    plt.close()

print("Figure rigenerate in", FIG)
