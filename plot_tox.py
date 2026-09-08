import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.spatial.distance import pdist

# ── Aesthetics ────────────────────────────────────────────────────────────────
mpl.rcParams.update({
    'font.family':       'DejaVu Sans',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.linewidth':    0.8,
    'xtick.major.width': 0.7,
    'ytick.major.width': 0.7,
    'xtick.labelsize':   10,
    'ytick.labelsize':   10,
    'axes.labelsize':    10,
    'axes.titlesize':    10,
    'axes.titleweight':  'bold',
    'figure.dpi':        500,
    'savefig.dpi':       500,
})

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv('toxicity_ppts.csv', encoding='utf-8-sig')
df = df.iloc[:40]   # TQ_D1–TQ_D40

# ── Partition columns ─────────────────────────────────────────────────────────
ECOTOX   = ['BCF', 'IGC50', 'LC50DM', 'LC50FM']
PROB_TOX = ['hERG', 'hERG-10um', 'DILI', 'Ames', 'ROA', 'FDAMDD',
            'SkinSen', 'Carcinogenicity', 'EC', 'EI', 'Respiratory',
            'H-HT', 'Neurotoxicity-DI', 'Ototoxicity', 'Hematotoxicity',
            'Nephrotoxicity-DI', 'Genotoxicity']
NUCLEAR  = ['NR-AhR', 'NR-AR', 'NR-AR-LBD', 'NR-Aromatase',
            'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma']
STRESS   = ['SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
CYTOTOX  = ['RPMI-8226', 'A549', 'HEK293']
CATEG    = ['NonBiodegradable', 'NonGenotoxic_Carcinogenicity', 'LD50_oral']

# All probability columns (0–1 range) for heatmap
PROB_ALL = PROB_TOX + NUCLEAR + STRESS + CYTOTOX

# ── Pretty labels ─────────────────────────────────────────────────────────────
pretty = {
    'hERG': 'hERG', 'hERG-10um': 'hERG (10 µM)', 'DILI': 'DILI',
    'Ames': 'Ames mutagenicity', 'ROA': 'Reactive oxygen', 'FDAMDD': 'FDA max daily dose',
    'SkinSen': 'Skin sensitization', 'Carcinogenicity': 'Carcinogenicity',
    'EC': 'Eye corrosion', 'EI': 'Eye irritation', 'Respiratory': 'Respiratory tox.',
    'H-HT': 'Human hepatotox.', 'Neurotoxicity-DI': 'Neurotoxicity',
    'Ototoxicity': 'Ototoxicity', 'Hematotoxicity': 'Hematotoxicity',
    'Nephrotoxicity-DI': 'Nephrotoxicity', 'Genotoxicity': 'Genotoxicity',
    'NR-AhR': 'NR-AhR', 'NR-AR': 'NR-AR', 'NR-AR-LBD': 'NR-AR-LBD',
    'NR-Aromatase': 'NR-Aromatase', 'NR-ER': 'NR-ER', 'NR-ER-LBD': 'NR-ER-LBD',
    'NR-PPAR-gamma': 'NR-PPARγ',
    'SR-ARE': 'SR-ARE', 'SR-ATAD5': 'SR-ATAD5', 'SR-HSE': 'SR-HSE',
    'SR-MMP': 'SR-MMP', 'SR-p53': 'SR-p53',
    'RPMI-8226': 'RPMI-8226', 'A549': 'A549', 'HEK293': 'HEK293',
    'BCF': 'BCF (log)', 'IGC50': 'IGC₅₀ (log)', 'LC50DM': 'LC₅₀DM (log)',
    'LC50FM': 'LC₅₀FM (log)',
}

# ── Custom diverging-ish colormap (white→orange→red) ────────────────────────
tox_cmap = LinearSegmentedColormap.from_list(
    'tox', ['#f0f4ff', '#ffe599', '#f4a261', '#e63946', '#7b1a2a'])

# ── Hierarchical clustering of compounds on PROB_ALL ─────────────────────────
mat = df[PROB_ALL].values
Z   = linkage(pdist(mat, 'euclidean'), method='ward')
order = leaves_list(Z)
df_ordered = df.iloc[order].reset_index(drop=True)
compound_labels = df_ordered['Compound'].tolist()

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(12, 12), facecolor='white')
gs  = gridspec.GridSpec(
    3, 2,
    figure=fig,
    height_ratios=[2.6, 1.0, 1.0],
    width_ratios=[1.5, 1],
    hspace=0.40,
    wspace=0.38,
)

# ── Panel A: Heatmap (all probability endpoints × 40 compounds) ──────────────
ax_heat = fig.add_subplot(gs[0, :])

heat_data = df_ordered[PROB_ALL].T
ylabels   = [pretty.get(c, c) for c in PROB_ALL]

# Section dividers
section_sizes = [len(PROB_TOX), len(NUCLEAR), len(STRESS), len(CYTOTOX)]
section_labels = ['Organ / Systemic Toxicity', 'Nuclear Receptor Signalling',
                  'Stress Response Pathways', 'Cytotoxicity']
section_colors = ['#264653', '#2a9d8f', '#e9c46a', '#e76f51']

im = ax_heat.imshow(heat_data.values, aspect='auto', cmap=tox_cmap,
                    vmin=0, vmax=1, interpolation='nearest')

# y-tick labels
ax_heat.set_yticks(range(len(PROB_ALL)))
ax_heat.set_yticklabels(ylabels, fontsize=10)

# x-tick labels
ax_heat.set_xticks(range(len(compound_labels)))
ax_heat.set_xticklabels(compound_labels, rotation=90, fontsize=10, ha='center')

# Section color bar on left side
#cumulative = 0
#for size, lbl, col in zip(section_sizes, section_labels, section_colors):
 #   ax_heat.axhline(cumulative - 0.5, color='white', linewidth=1.8)
#    mid = cumulative + size / 2 - 0.5
#    ax_heat.annotate(lbl, xy=(-0.5, mid), xycoords=('data','data'),
#                     xytext=(-7.5, 0), textcoords='offset points',
#                     ha='right', va='center', fontsize=10, fontweight='bold',
#                     color=col, rotation=90,
 #                    annotation_clip=False)
  #  cumulative += size

# Colorbar
cbar = fig.colorbar(im, ax=ax_heat, fraction=0.012, pad=0.01, shrink=0.85)
cbar.set_label('Probability score', fontsize=10)
cbar.ax.tick_params(labelsize=7)

# Threshold line at 0.5
cbar.ax.axhline(0.5, color='black', linewidth=1.2, linestyle='--')
cbar.ax.text(2.0, 0.5, '0.5', va='center', fontsize=10, transform=cbar.ax.transData)

ax_heat.set_title('a   Toxicity Probability Scores — All Endpoints (Hierarchically Clustered Compounds)',
                  loc='left', pad=8, color='#1a1a2e')
ax_heat.set_xlabel('Compound (ward-linkage cluster order)', labelpad=6)

# ── Panel B: Ecotoxicity box + strip chart ────────────────────────────────────
ax_eco = fig.add_subplot(gs[1, 0])
eco_long = df[ECOTOX + ['Compound']].melt(id_vars='Compound',
                                           var_name='Endpoint', value_name='Value')
eco_long['Endpoint'] = eco_long['Endpoint'].map(pretty)

palette_eco = {'BCF (log)': '#457b9d', 'IGC₅₀ (log)': '#1d3557',
               'LC₅₀DM (log)': '#e63946', 'LC₅₀FM (log)': '#a8dadc'}

sns.boxplot(data=eco_long, x='Endpoint', y='Value', palette=palette_eco,
            width=0.45, linewidth=0.9, fliersize=0,
            ax=ax_eco, order=list(palette_eco.keys()))
sns.stripplot(data=eco_long, x='Endpoint', y='Value', palette=palette_eco,
              size=3.5, alpha=0.65, jitter=True,
              ax=ax_eco, order=list(palette_eco.keys()), zorder=3)

ax_eco.set_xlabel('')
ax_eco.set_ylabel('log-scale value')
ax_eco.set_title('b')
ax_eco.tick_params(axis='x', labelsize=8)

# ── Panel C: Grouped mean bar chart — organ/systemic tox ─────────────────────
ax_bar = fig.add_subplot(gs[1, 1])
means = df[PROB_TOX].mean().sort_values(ascending=False)
errs  = df[PROB_TOX].sem()

bar_colors = ['#e63946' if v >= 0.5 else '#457b9d' for v in means.values]
bars = ax_bar.barh([pretty.get(c, c) for c in means.index], means.values,
                    xerr=errs[means.index].values,
                    color=bar_colors, edgecolor='white', linewidth=0.5,
                    error_kw=dict(ecolor='#444', elinewidth=0.7, capsize=2))
ax_bar.axvline(0.5, color='#e63946', linewidth=1.2, linestyle='--', alpha=0.7,
               label='0.5 threshold')
ax_bar.set_xlim(0, 1.05)
ax_bar.set_xlabel('Mean probability score (± SEM)')
ax_bar.set_title('c   Systemic Toxicity: Mean Scores')
ax_bar.legend(fontsize=10, loc='lower right')
ax_bar.invert_yaxis()
ax_bar.tick_params(axis='y', labelsize=7.2)

red_patch  = mpatches.Patch(color='#e63946', label='Mean ≥ 0.5 (concern)')
blue_patch = mpatches.Patch(color='#457b9d', label='Mean < 0.5 (low concern)')
ax_bar.legend(handles=[red_patch, blue_patch], fontsize=10, loc='lower right',
              framealpha=0.8)

# ── Panel D: Nuclear receptor + stress response violin ───────────────────────
ax_vio = fig.add_subplot(gs[2, 0])
nr_sr = NUCLEAR + STRESS
nr_sr_long = df[nr_sr + ['Compound']].melt(id_vars='Compound',
                                             var_name='Endpoint', value_name='Score')
nr_sr_long['Group'] = nr_sr_long['Endpoint'].apply(
    lambda x: 'Nuclear Receptor' if x in NUCLEAR else 'Stress Response')
nr_sr_long['Endpoint'] = nr_sr_long['Endpoint'].map(pretty)

palette_vio = {'Nuclear Receptor': '#2a9d8f', 'Stress Response': '#e9c46a'}
order_vio = [pretty.get(c,c) for c in nr_sr]
sns.violinplot(data=nr_sr_long, x='Endpoint', y='Score',
               hue='Group', palette=palette_vio,
               inner='box', linewidth=0.8, cut=0,
               ax=ax_vio, order=order_vio, legend=False)

ax_vio.axhline(0.5, color='#e63946', linewidth=1.0, linestyle='--', alpha=0.7)
ax_vio.set_xlabel('')
ax_vio.set_ylabel('Probability score')
ax_vio.set_title('d   Nuclear Receptor & Stress Response Pathways')
ax_vio.tick_params(axis='x', rotation=45, labelsize=7.5)

handles_vio = [mpatches.Patch(color=v, label=k) for k,v in palette_vio.items()]
ax_vio.legend(handles=handles_vio, fontsize=10, loc='upper right', framealpha=0.85)

# ── Panel E: Categorical summary stacked bar ──────────────────────────────────
ax_cat = fig.add_subplot(gs[2, 1])

cat_labels = {
    'NonBiodegradable': 'Non-biodegradable\n(class)',
    'NonGenotoxic_Carcinogenicity': 'Non-genotoxic\nCarcinogenicity (class)',
    'LD50_oral': 'LD₅₀ oral\n(class)',
}
colors_cat = ['#264653', '#2a9d8f', '#e9c46a', '#f4a261', '#e63946',
              '#6d6875', '#b5838d', '#c9ada7']

for ax_pos, (col, lbl) in enumerate(cat_labels.items()):
    vc = df[col].value_counts().sort_index()
    bottom = 0
    for vi, (cat_val, cnt) in enumerate(vc.items()):
        ax_cat.bar(ax_pos, cnt, bottom=bottom, color=colors_cat[vi % len(colors_cat)],
                   edgecolor='white', linewidth=0.5, width=0.55)
        if cnt >= 2:
            ax_cat.text(ax_pos, bottom + cnt/2, str(int(cnt)),
                        ha='center', va='center', fontsize=10, color='white',
                        fontweight='bold')
        bottom += cnt

ax_cat.set_xticks(range(len(cat_labels)))
ax_cat.set_xticklabels(list(cat_labels.values()), fontsize=8.5)
ax_cat.set_ylabel('Number of compounds')
ax_cat.set_title('e   Categorical Safety Flags')
ax_cat.set_xlim(-0.5, len(cat_labels) - 0.5)

# ── Supertitle ────────────────────────────────────────────────────────────────
#fig.suptitle(
 #   'ADMETLab 3.0 Toxicity Profile — Thymoquinone Derivatives (TQ_D1–TQ_D40)',
  #  fontsize=14, fontweight='bold', y=0.995, color='#0d0d2b'
#)

plt.savefig('toxicity_tq_derivatives.png',
            bbox_inches='tight', facecolor='white')
plt.savefig('toxicity_tq_derivatives.pdf',
            bbox_inches='tight', facecolor='white')
print("Saved.")
