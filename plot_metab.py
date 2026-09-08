import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------
# Input file
# --------------------------------------------------
input_file = "all_admet_tq_derivative_data.csv"

# --------------------------------------------------
# Read data
# --------------------------------------------------
df = pd.read_csv(input_file)

# --------------------------------------------------
# Compound names
# Change "Compound" to your compound-name column
# --------------------------------------------------
compound_col = "Compound"

# --------------------------------------------------
# Metabolism descriptors
# --------------------------------------------------
cols = [
    "CYP1A2-inh",
    "CYP1A2-sub",
    "CYP2C19-inh",
    "CYP2C19-sub",
    "CYP2C9-inh",
    "CYP2C9-sub",
    "CYP2D6-inh",
    "CYP2D6-sub",
    "CYP3A4-inh",
    "CYP3A4-sub",
    "CYP2B6-inh",
    "CYP2B6-sub",
    "CYP2C8-inh",
    "HLM-human"
]

heatmap = df[cols].astype(float)

# --------------------------------------------------
# Plot
# --------------------------------------------------
fig, ax = plt.subplots(figsize=(6,6))

im = ax.imshow(
    heatmap,
    aspect='auto',
    cmap='RdYlBu_r',
    interpolation='nearest'
)

# --------------------------------------------------
# Labels
# --------------------------------------------------
ax.set_xticks(np.arange(len(cols)))
ax.set_xticklabels(cols, rotation=60, ha="right", fontsize=10)

ax.set_ylabel("Thymoquinone Derivatives (TQ_D)", fontsize=14, weight='bold')
ax.set_xlabel("Metabolic Descriptors", fontsize=14, weight='bold')

ax.set_yticks(np.arange(len(df)))
ax.set_yticklabels(df[compound_col], fontsize=9)

# --------------------------------------------------
# Colorbar
# --------------------------------------------------
cbar = plt.colorbar(im)

cbar.set_label(
    "Predicted Probability",
    fontsize=11
)

# --------------------------------------------------
# Layout
# --------------------------------------------------
#plt.title(
 #   "ADMETLab 3.0 Metabolic Profile",
  #  fontsize=14,
   # weight='bold'
#)

plt.tight_layout()

plt.savefig(
    "ADMET_Metabolism_Heatmap.png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()
