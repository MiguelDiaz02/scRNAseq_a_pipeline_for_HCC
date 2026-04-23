#!/usr/bin/env python3
"""
CSC FIGURES — REVISED VERSION
Fixes:
  - UMAP: correct cell count (6,302 malignant), legible external legends
  - Metabolic heatmap: no label overlap, improved aesthetics
  - Dotplot: bold text, title repositioned
  - Cowplot-style 3-panel assembly (UMAP → Dotplot → Metabolic)
"""

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import matplotlib.patheffects as pe
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

sc.settings.verbosity = 1
sc.settings.set_figure_params(dpi=300, facecolor='white', frameon=False)

import os
INPUT_H5AD = os.getenv('INPUT_H5AD', '/home/mdiaz/HCC_project/MERGED_adata/scvi_integrated.h5ad')
OUTPUT_DIR = os.getenv('OUTPUT_DIR', '/home/mdiaz/manuscript_revision/new_figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)


FONT = {'family': 'sans-serif', 'weight': 'bold'}
matplotlib.rc('font', **FONT)

import os
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading data and recomputing CSC scores...")

# ── Load & subset ──────────────────────────────────────────────────────────
adata = sc.read_h5ad(INPUT_H5AD)
ct_col = 'CellType_harmonized'
adata_mal = adata[adata.obs[ct_col].str.lower().str.contains('malignan', na=False)].copy()
N_MAL = adata_mal.n_obs  # 6,302

# ── CSC signature ─────────────────────────────────────────────────────────
var_names = set(adata_mal.var_names)
csc_signature = [g for g in [
    'CD44','EPCAM','PROM1','ALDH1A1','CD24',
    'HAMP','GPC3','DNAJC6','NT5DC2','UBD','ATAD2','LAMC1','GABRE',
    'LRRC1','MUC13','STK39','SDS','PPP1R1A','TRIM22','FGFR2','SPINK1','IGF2BP2',
    'HSPB1','ADH4','FTH1','APCS',
    'MYC','CCND1','AXIN2','GLI1','PTCH1','NANOG','SOX2','KLF4'
] if g in var_names]

sc.tl.score_genes(adata_mal, csc_signature, score_name='CSC_score', use_raw=False)
threshold = np.percentile(adata_mal.obs['CSC_score'], 70)
adata_mal.obs['CSC_class'] = pd.Categorical(
    np.where(adata_mal.obs['CSC_score'] >= threshold, 'CSC-high', 'CSC-low'),
    categories=['CSC-high', 'CSC-low']
)

# ── Neighbors + UMAP + Leiden ─────────────────────────────────────────────
sc.pp.neighbors(adata_mal, use_rep='X_scVI', n_neighbors=15)
sc.tl.umap(adata_mal, min_dist=0.3)
sc.tl.leiden(adata_mal, resolution=0.8, key_added='leiden_csc')
n_clusters = adata_mal.obs['leiden_csc'].nunique()

# DEGs for dotplot
sc.tl.rank_genes_groups(adata_mal, groupby='CSC_class',
                        groups=['CSC-high'], reference='CSC-low',
                        method='wilcoxon', use_raw=False)
degs_df = sc.get.rank_genes_groups_df(adata_mal, group='CSC-high')
top_markers = [g for g in degs_df['names'].head(15).tolist() if g in var_names][:10]

print(f"  Malignant cells: {N_MAL:,} | CSC-high: {(adata_mal.obs['CSC_class']=='CSC-high').sum():,} | Subclusters: {n_clusters}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE A — UMAP (3 panels, external legends, bold, correct title)
# ══════════════════════════════════════════════════════════════════════════════
print("Generating UMAP figure...")

fig_umap, axes = plt.subplots(1, 3, figsize=(20, 6.5))
fig_umap.patch.set_facecolor('white')

# ── Panel 1: CSC Score ──
ax = axes[0]
sc.pl.umap(adata_mal, color='CSC_score', ax=ax, show=False,
           color_map='RdBu_r', frameon=True, colorbar_loc=None)
ax.set_title('CSC Score', fontsize=14, fontweight='bold', pad=8)
ax.set_xlabel('UMAP 1', fontsize=11, fontweight='bold')
ax.set_ylabel('UMAP 2', fontsize=11, fontweight='bold')
ax.tick_params(labelsize=9)
# Add colorbar below
sm = plt.cm.ScalarMappable(cmap='RdBu_r',
    norm=plt.Normalize(adata_mal.obs['CSC_score'].min(),
                       adata_mal.obs['CSC_score'].max()))
sm.set_array([])
cbar = fig_umap.colorbar(sm, ax=ax, orientation='horizontal',
                          fraction=0.046, pad=0.08, shrink=0.8)
cbar.set_label('CSC Score', fontsize=10, fontweight='bold')
cbar.ax.tick_params(labelsize=8)

# ── Panel 2: CSC Class (external legend, large patches) ──
ax = axes[1]
palette_csc = {'CSC-high': '#B2182B', 'CSC-low': '#4393C3'}
sc.pl.umap(adata_mal, color='CSC_class', ax=ax, show=False,
           palette=palette_csc, frameon=True, legend_loc=None)
ax.set_title('CSC Classification', fontsize=14, fontweight='bold', pad=8)
ax.set_xlabel('UMAP 1', fontsize=11, fontweight='bold')
ax.set_ylabel('UMAP 2', fontsize=11, fontweight='bold')
ax.tick_params(labelsize=9)
n_high = (adata_mal.obs['CSC_class'] == 'CSC-high').sum()
n_low  = (adata_mal.obs['CSC_class'] == 'CSC-low').sum()
legend_patches = [
    Patch(color='#B2182B', label=f'CSC-high  (n={n_high:,})'),
    Patch(color='#4393C3', label=f'CSC-low   (n={n_low:,})'),
]
ax.legend(handles=legend_patches, loc='lower right', fontsize=10,
          framealpha=0.9, edgecolor='gray',
          prop={'weight': 'bold', 'size': 10})

# ── Panel 3: Leiden subclusters (external legend, 2-col) ──
ax = axes[2]
sc.pl.umap(adata_mal, color='leiden_csc', ax=ax, show=False,
           frameon=True, legend_loc=None)
ax.set_title(f'Leiden Subclusters (res=0.8)', fontsize=14, fontweight='bold', pad=8)
ax.set_xlabel('UMAP 1', fontsize=11, fontweight='bold')
ax.set_ylabel('UMAP 2', fontsize=11, fontweight='bold')
ax.tick_params(labelsize=9)

# Build Leiden palette from existing colors in the plot
cats = adata_mal.obs['leiden_csc'].cat.categories.tolist()
colors = adata_mal.uns.get('leiden_csc_colors',
    plt.cm.tab20.colors[:len(cats)])
leiden_patches = [Patch(color=colors[i], label=f'Cluster {c}')
                  for i, c in enumerate(cats)]
ax.legend(handles=leiden_patches, loc='lower right', fontsize=8,
          ncol=2, framealpha=0.9, edgecolor='gray',
          prop={'weight': 'bold', 'size': 8})

# ── Suptitle (corrected) ──
fig_umap.suptitle(
    f'CSC Subpopulation Analysis — Malignant Cells\n'
    f'n = {N_MAL:,} malignant cells · {n_clusters} Leiden subclusters',
    fontsize=15, fontweight='bold', y=1.03
)
plt.tight_layout(rect=[0, 0, 1, 1])
path_umap = f'{OUT_DIR}Fig_CSC_UMAP.png'
fig_umap.savefig(path_umap, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  ✓ UMAP saved")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE B — DOTPLOT (bold text, title closer to plot)
# ══════════════════════════════════════════════════════════════════════════════
print("Generating Dotplot figure...")

fig_dot, ax_dot = plt.subplots(figsize=(11, 5))
fig_dot.patch.set_facecolor('white')

# Use savefig directly via scanpy's dotplot
plt.close('all')
sc.pl.dotplot(
    adata_mal, var_names=top_markers,
    groupby='CSC_class',
    show=False,
    color_map='RdBu_r',
    standard_scale='var',
    title='Top CSC-high Markers vs CSC-low',
    save='_csc_tmp.png'
)
plt.close('all')

# Now post-process: reload and add bold text via PIL annotation
import shutil
tmp_src = str(sc.settings.figdir) + 'dotplot__csc_tmp.png'
if not os.path.exists(tmp_src):
    # scanpy saves to figdir
    candidates = [f for f in os.listdir(sc.settings.figdir) if 'csc_tmp' in f]
    tmp_src = os.path.join(sc.settings.figdir, candidates[0]) if candidates else None

# Alternative: generate via figure API manually with matplotlib
plt.close('all')
fig_dot2 = plt.figure(figsize=(12, 5), facecolor='white')
ax_main = fig_dot2.add_subplot(111)

# Build dotplot data manually for full bold control
groups = ['CSC-high', 'CSC-low']
n_genes = len(top_markers)
dot_sizes, dot_colors = [], []

for g_idx, gene in enumerate(top_markers):
    row_sizes, row_colors = [], []
    if gene not in var_names:
        continue
    for grp in groups:
        mask = adata_mal.obs['CSC_class'] == grp
        cells = adata_mal[mask]
        expr = cells[:, gene].X.toarray().flatten() if hasattr(cells.X, 'toarray') \
               else cells[:, gene].X.flatten()
        pct_exp  = (expr > 0).mean() * 100
        mean_exp = expr[expr > 0].mean() if (expr > 0).any() else 0
        row_sizes.append(pct_exp)
        row_colors.append(mean_exp)
    dot_sizes.append(row_sizes)
    dot_colors.append(row_colors)

dot_sizes  = np.array(dot_sizes)
dot_colors = np.array(dot_colors)

# Normalize colors per gene (standard_scale='var')
col_min = dot_colors.min(axis=1, keepdims=True)
col_max = dot_colors.max(axis=1, keepdims=True)
dot_colors_norm = (dot_colors - col_min) / (col_max - col_min + 1e-9)

cmap_dot = plt.cm.RdBu_r  # reversed: high expression (1.0) → red, low (0.0) → blue
# Distance between groups: 0.4 ≈ 4/10 of original unit spacing
x_positions = np.array([0, 0.4])
y_positions = np.arange(len(top_markers))

for g_idx in range(len(top_markers)):
    for grp_idx in range(len(groups)):
        size  = dot_sizes[g_idx, grp_idx]
        # Rescale [0,1] → [0.15, 0.85] to boost saturation and avoid pale midtones
        rescaled = 0.15 + dot_colors_norm[g_idx, grp_idx] * 0.70
        color = cmap_dot(rescaled)
        ax_main.scatter(x_positions[grp_idx], g_idx,   # use compressed x_positions
                        s=size * 5,          # scale dot to % expressed
                        c=[color],
                        edgecolors='gray', linewidths=0.4,
                        zorder=3)

ax_main.set_xticks(x_positions)
ax_main.set_xticklabels(groups, fontsize=12, fontweight='bold')
ax_main.set_xlim(x_positions[0] - 0.3, x_positions[-1] + 0.3)  # tight x limits
ax_main.set_yticks(y_positions)
ax_main.set_yticklabels(top_markers, fontsize=12, fontweight='bold')
ax_main.set_xlabel('CSC Class', fontsize=12, fontweight='bold')
ax_main.set_ylabel('Gene', fontsize=12, fontweight='bold')
ax_main.grid(axis='both', linestyle='--', alpha=0.3, zorder=0)
ax_main.set_facecolor('white')

# Colorbar
sm_dot = plt.cm.ScalarMappable(cmap=cmap_dot,
                                norm=plt.Normalize(0, 1))
sm_dot.set_array([])
cbar_dot = fig_dot2.colorbar(sm_dot, ax=ax_main, shrink=0.6, pad=0.02)
cbar_dot.set_label('Scaled Mean Expression', fontsize=10, fontweight='bold')
cbar_dot.ax.tick_params(labelsize=9)
for t in cbar_dot.ax.get_yticklabels():
    t.set_fontweight('bold')

# Dot size legend
for pct in [25, 50, 75, 100]:
    ax_main.scatter([], [], s=pct * 5, c='gray', alpha=0.6,
                    label=f'{pct}% expressed')
leg = ax_main.legend(title='% Cells Expressing',
                     title_fontsize=10, fontsize=9,
                     loc='upper left', bbox_to_anchor=(1.12, 1.0),
                     framealpha=0.9, borderaxespad=0,
                     labelspacing=1.8,   # taller spacing between items
                     handletextpad=1.0)
leg.get_title().set_fontweight('bold')
for t in leg.get_texts():
    t.set_fontweight('bold')

# Title close to plot (pad=2)
ax_main.set_title('Top CSC-high Markers vs CSC-low',
                  fontsize=14, fontweight='bold', pad=2)

fig_dot2.tight_layout()
path_dot = f'{OUT_DIR}Fig_CSC_dotplot.png'
fig_dot2.savefig(path_dot, dpi=300, bbox_inches='tight', facecolor='white')
plt.close('all')
print(f"  ✓ Dotplot saved")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE C — METABOLIC-IMMUNE HEATMAP (no overlap, improved aesthetics)
# ══════════════════════════════════════════════════════════════════════════════
print("Generating Metabolic-Immune figure...")

gene_panels = {
    'OXPHOS\n(Metabolic\nActivation)': ['NDUFA1','COX7C','ATP5F1A','SDHA','FH'],
    'Immune\nSilencing':               ['CD3E','GZMK','NCAM1','HLA-A','PDCD1'],
    'Wnt\nStemness':                   ['MYC','CCND1','AXIN2'],
    'Hedgehog\nStemness':              ['GLI1','PTCH1'],
}

# Compute mean expression per group
results = {}
for panel_name, genes in gene_panels.items():
    present = [g for g in genes if g in var_names]
    if not present:
        continue
    means = {}
    for grp in ['CSC-high', 'CSC-low']:
        mask = adata_mal.obs['CSC_class'] == grp
        cells = adata_mal[mask]
        expr = cells[:, present].X.toarray() if hasattr(cells.X, 'toarray') else cells[:, present].X
        means[grp] = np.mean(expr, axis=0)
    results[panel_name] = {'genes': present, 'CSC-high': means['CSC-high'], 'CSC-low': means['CSC-low']}

n_panels = len(results)
# Wide figure: each panel gets enough room, extra right margin for log2FC annotation
fig_met = plt.figure(figsize=(5 * n_panels + 1, 7), facecolor='white')
gs = gridspec.GridSpec(1, n_panels + 1,
                       width_ratios=[4] * n_panels + [0.5],
                       wspace=0.6)

axes_met = [fig_met.add_subplot(gs[0, i]) for i in range(n_panels)]
cbar_ax  = fig_met.add_subplot(gs[0, n_panels])

vmin, vmax = -2.5, 2.5
norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
cmap = matplotlib.cm.RdBu_r
im_last = None

for ax, (panel_name, data) in zip(axes_met, results.items()):
    genes   = data['genes']
    hi_exp  = data['CSC-high']
    lo_exp  = data['CSC-low']
    fc      = np.log2((hi_exp + 1e-9) / (lo_exp + 1e-9))

    mat      = np.column_stack([hi_exp, lo_exp])
    mat_norm = (mat - mat.mean(axis=1, keepdims=True)) / (mat.std(axis=1, keepdims=True) + 1e-9)
    mat_norm = np.clip(mat_norm, vmin, vmax)

    im = ax.imshow(mat_norm, cmap=cmap, norm=norm, aspect='auto')
    im_last = im

    # X axis
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['CSC-high', 'CSC-low'],
                       fontsize=10, fontweight='bold', rotation=30, ha='right')
    # Y axis
    ax.set_yticks(range(len(genes)))
    ax.set_yticklabels(genes, fontsize=10, fontweight='bold')
    ax.tick_params(length=0)

    # Panel title
    ax.set_title(panel_name, fontsize=11, fontweight='bold', pad=10,
                 linespacing=1.3)

    # log2FC annotations — RIGHT side OUTSIDE the heatmap, no overlap
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(range(len(genes)))
    fc_labels = [f'{v:+.1f}' for v in fc]
    ax2.set_yticklabels(fc_labels, fontsize=9, fontweight='bold',
                        color='#333333')
    ax2.tick_params(length=0, pad=3)
    ax2.set_ylabel(r'$\log_2$FC' + '\n(high/low)', fontsize=9, fontweight='bold',
                   labelpad=6, color='#333333')
    ax2.spines['right'].set_visible(True)
    ax2.spines['right'].set_color('#aaaaaa')

    # Cell value overlay (optional — expression values as text)
    for row in range(len(genes)):
        for col, (val, label) in enumerate(zip(mat_norm[row], ['H','L'])):
            text_color = 'white' if abs(val) > 1.2 else 'black'
            ax.text(col, row, f'{mat[row, col]:.2f}',
                    ha='center', va='center',
                    fontsize=7.5, fontweight='bold', color=text_color)

    # Remove ALL grid lines (major and minor, both axes)
    ax.grid(which='both', visible=False)
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())

    # Separator lines between rows (white, not a grid)
    for row in range(len(genes) - 1):
        ax.axhline(row + 0.5, color='white', linewidth=1.5)

# Shared colorbar — in its own axis, no overlap
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cb = fig_met.colorbar(sm, cax=cbar_ax)
cb.set_label('Scaled\nExpression', fontsize=10, fontweight='bold', labelpad=6)
cb.ax.tick_params(labelsize=9)
for t in cb.ax.get_yticklabels():
    t.set_fontweight('bold')

fig_met.suptitle(
    'Metabolic · Immune · Stemness Profile — CSC-high vs CSC-low\n'
    f'Malignant Cells (n = {N_MAL:,})',
    fontsize=13, fontweight='bold', y=1.04
)

path_met = f'{OUT_DIR}Fig_CSC_metabolic_immune.png'
fig_met.savefig(path_met, dpi=300, bbox_inches='tight', facecolor='white')
plt.close('all')
print(f"  ✓ Metabolic-Immune saved")

# ══════════════════════════════════════════════════════════════════════════════
# PANEL ASSEMBLY — 3 rows × 1 column (UMAP → Dotplot → Metabolic)
# ══════════════════════════════════════════════════════════════════════════════
print("Assembling final panel (3 rows × 1 col)...")

imgs = {
    'A': Image.open(path_umap),
    'B': Image.open(path_dot),
    'C': Image.open(path_met),
}

# Normalise widths to the widest panel, preserve aspect ratios
max_w = max(img.width for img in imgs.values())
resized = {}
for label, img in imgs.items():
    scale  = max_w / img.width
    new_h  = int(img.height * scale)
    resized[label] = img.resize((max_w, new_h), Image.LANCZOS)

# DPI used when saving individual panels (300), convert px → inches
DPI = 300
panel_heights = [resized[k].height for k in ['A', 'B', 'C']]
total_h_px    = sum(panel_heights) + 60 * 3   # 60px padding between panels

fig_panel = plt.figure(figsize=(max_w / DPI, total_h_px / DPI),
                        facecolor='white', dpi=DPI)

label_kwargs = dict(fontsize=18, fontweight='bold', va='top', ha='left',
                    transform=fig_panel.transFigure)

y_cursor = 1.0   # top → bottom
for label, key in [('A', 'A'), ('B', 'B'), ('C', 'C')]:
    img    = resized[key]
    h_frac = img.height / total_h_px
    ax_img = fig_panel.add_axes([0.0, y_cursor - h_frac, 1.0, h_frac])
    ax_img.imshow(np.array(img))
    ax_img.axis('off')
    # Bold panel label
    ax_img.text(0.01, 1.0, label, transform=ax_img.transAxes,
                fontsize=20, fontweight='bold', va='top', ha='left',
                color='black')
    y_cursor -= h_frac + (60 / total_h_px)   # gap between panels

path_panel = f'{OUT_DIR}Fig_CSC_panel_FINAL.png'
fig_panel.savefig(path_panel, dpi=DPI, bbox_inches='tight', facecolor='white')
plt.close('all')

size_mb = os.path.getsize(path_panel) / 1e6
print(f"  ✓ Final panel saved: {path_panel} ({size_mb:.1f} MB)")

print("\n✅ ALL CSC FIGURES REVISED AND ASSEMBLED")
print(f"   A) {path_umap}")
print(f"   B) {path_dot}")
print(f"   C) {path_met}")
print(f"   PANEL) {path_panel}")
