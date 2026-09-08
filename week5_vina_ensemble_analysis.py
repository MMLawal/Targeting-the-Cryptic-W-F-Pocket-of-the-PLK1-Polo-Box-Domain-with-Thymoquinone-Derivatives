#!/usr/bin/env python3
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# USER SETTINGS
# ============================================================
INPUT_DIR = "docking_results"
N_ENSEMBLES = 50
N_LIGANDS = 41
TOP_N_ENRICHMENT = 10
OUTPUT_DIR = "Week5_Vina_Ensemble_Analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# LIGAND MAPPING
# Lig001-Lig040 -> TQ_D1-TQ_D40
# Lig041 -> TQ (parent thymoquinone)
# ============================================================
def ligand_label(ligand_number):
    if ligand_number == 41:
        return "TQ"
    return f"TQ_D{ligand_number}"

# ============================================================
# FILENAME PARSER
# ============================================================
def parse_vina_filename(filename):
    match = re.match(
        r"^ensemble_(\d+)_Lig(\d+)\.log$",
        filename,
        re.IGNORECASE
    )
    if match is None:
        return None, None
    return int(match.group(1)), int(match.group(2))

# ============================================================
# VINA LOG PARSER
# ============================================================
def parse_vina_log(log_file):
    # Standard AutoDock Vina result rows:
    # mode  affinity  rmsd_l.b.  rmsd_u.b.
    affinities = []

    with open(log_file, "r", errors="ignore") as f:
        for line in f:
            match = re.match(
                r"^\s*(\d+)\s+"
                r"(-?\d+(?:\.\d+)?)\s+"
                r"(-?\d+(?:\.\d+)?)\s+"
                r"(-?\d+(?:\.\d+)?)\s*$",
                line.strip()
            )
            if match:
                affinities.append(float(match.group(2)))

    if not affinities:
        return np.nan

    # More negative = better
    return min(affinities)

# ============================================================
# FIND LOG FILES
# ============================================================
if not os.path.isdir(INPUT_DIR):
    raise FileNotFoundError(
        f"Cannot find '{INPUT_DIR}'. "
        "Place this script in the directory containing docking_results/."
    )

log_files = [
    f for f in os.listdir(INPUT_DIR)
    if re.match(
        r"^ensemble_\d+_Lig\d+\.log$",
        f,
        re.IGNORECASE
    )
]

log_files = sorted(
    log_files,
    key=lambda x: (
        parse_vina_filename(x)[0],
        parse_vina_filename(x)[1]
    )
)

expected = N_ENSEMBLES * N_LIGANDS

print("=" * 80)
print("AUTODOCK VINA WEEK 5 ENSEMBLE ANALYSIS")
print("=" * 80)
print(f"Log files detected : {len(log_files)}")
print(f"Expected            : {expected}")

# ============================================================
# READ ALL RESULTS
# ============================================================
records = []

for filename in log_files:

    ensemble, ligand_number = parse_vina_filename(filename)

    if ensemble is None:
        continue

    if not (1 <= ensemble <= N_ENSEMBLES):
        continue

    if not (1 <= ligand_number <= N_LIGANDS):
        continue

    path = os.path.join(INPUT_DIR, filename)

    records.append({
        "Ensemble": ensemble,
        "Ligand_Number": ligand_number,
        "TQ_D": ligand_label(ligand_number),
        "Vina_Affinity": parse_vina_log(path),
        "Log_File": filename
    })

if not records:
    raise RuntimeError("No Vina log files were successfully parsed.")

combined = pd.DataFrame(records)

# Explicit column order prevents TQ_D KeyErrors later
combined = combined[
    [
        "Ensemble",
        "Ligand_Number",
        "TQ_D",
        "Vina_Affinity",
        "Log_File"
    ]
]

combined = combined.sort_values(
    ["Ensemble", "Ligand_Number"]
)

combined.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "All_Vina_Ensemble_Docking_Results.csv"
    ),
    index=False
)

# ============================================================
# VALIDATION
# ============================================================
print(f"Parsed records       : {len(combined)}")
print(f"Unique ensembles     : {combined['Ensemble'].nunique()}")
print(f"Unique compounds     : {combined['TQ_D'].nunique()}")
print(
    f"Missing Vina scores  : "
    f"{combined['Vina_Affinity'].isna().sum()}"
)

expected_pairs = {
    (e, l)
    for e in range(1, N_ENSEMBLES + 1)
    for l in range(1, N_LIGANDS + 1)
}

observed_pairs = set(
    zip(
        combined["Ensemble"],
        combined["Ligand_Number"]
    )
)

missing_pairs = sorted(expected_pairs - observed_pairs)

if missing_pairs:
    print(f"WARNING: {len(missing_pairs)} ensemble/ligand pairs missing.")
    pd.DataFrame(
        missing_pairs,
        columns=["Ensemble", "Ligand_Number"]
    ).to_csv(
        os.path.join(
            OUTPUT_DIR,
            "Missing_Ensemble_Ligand_Combinations.csv"
        ),
        index=False
    )

# ============================================================
# DOCKING MATRIX: 41 compounds x 50 ensembles
# ============================================================
docking_matrix = combined.pivot_table(
    index="TQ_D",
    columns="Ensemble",
    values="Vina_Affinity",
    aggfunc="min"
)

ligand_order = [
    f"TQ_D{i}" for i in range(1, 41)
] + ["TQ"]

docking_matrix = docking_matrix.reindex(
    index=ligand_order,
    columns=range(1, N_ENSEMBLES + 1)
)

docking_matrix.columns = [
    f"Ensemble_{int(x)}"
    for x in docking_matrix.columns
]

docking_matrix.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "Docking_Matrix_41x50.csv"
    )
)

# ============================================================
# SUMMARY STATISTICS
# ============================================================
summary = pd.DataFrame(index=docking_matrix.index)
summary.index.name = "TQ_D"

summary["N_Ensembles"] = docking_matrix.notna().sum(axis=1)
summary["Mean_Score"] = docking_matrix.mean(axis=1)
summary["Median_Score"] = docking_matrix.median(axis=1)
summary["Best_Score"] = docking_matrix.min(axis=1)
summary["Worst_Score"] = docking_matrix.max(axis=1)
summary["SD_Score"] = docking_matrix.std(axis=1)
summary["Score_Range"] = (
    summary["Worst_Score"] - summary["Best_Score"]
)

# ============================================================
# RANK MATRIX
# ============================================================
rank_matrix = docking_matrix.rank(
    axis=0,
    method="min",
    ascending=True
)

rank_matrix.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "Rank_Matrix_41x50.csv"
    )
)

summary["Average_Rank"] = rank_matrix.mean(axis=1)
summary["Median_Rank"] = rank_matrix.median(axis=1)

summary["Best_Score_Rank"] = (
    summary["Best_Score"].rank(
        method="min",
        ascending=True
    )
)

# ============================================================
# Z-SCORE RANKING
# ============================================================
mean_sd = summary["Mean_Score"].std()

if mean_sd == 0 or pd.isna(mean_sd):
    summary["Mean_Score_Z"] = 0.0
else:
    summary["Mean_Score_Z"] = (
        summary["Mean_Score"]
        - summary["Mean_Score"].mean()
    ) / mean_sd

summary["Z_Score_Rank"] = (
    summary["Mean_Score_Z"].rank(
        method="min",
        ascending=True
    )
)

avg_rank_sd = summary["Average_Rank"].std()

if avg_rank_sd == 0 or pd.isna(avg_rank_sd):
    summary["Average_Rank_Z"] = 0.0
else:
    summary["Average_Rank_Z"] = (
        summary["Average_Rank"]
        - summary["Average_Rank"].mean()
    ) / avg_rank_sd

summary["Z_Rank_Rank"] = (
    summary["Average_Rank_Z"].rank(
        method="min",
        ascending=True
    )
)

# ============================================================
# TOP-10 ENRICHMENT
# ============================================================
top10_counts = (
    rank_matrix.le(TOP_N_ENRICHMENT).sum(axis=1)
)

summary["Top10_Count"] = top10_counts

summary["Top10_Enrichment_Frequency"] = (
    top10_counts
    / rank_matrix.notna().sum(axis=1)
)

summary["Top10_Enrichment_Percent"] = (
    summary["Top10_Enrichment_Frequency"] * 100
)

summary["Enrichment_Rank"] = (
    summary["Top10_Enrichment_Frequency"].rank(
        method="min",
        ascending=False
    )
)

# ============================================================
# CONSENSUS RANK
# ============================================================
summary["Consensus_Rank_Score"] = (
    summary["Average_Rank"]
    + summary["Best_Score_Rank"]
    + summary["Z_Score_Rank"]
    + summary["Enrichment_Rank"]
) / 4

summary["Consensus_Rank"] = (
    summary["Consensus_Rank_Score"].rank(
        method="min",
        ascending=True
    )
)

# Keep TQ_D as a real column
summary = summary.reset_index()

summary = summary.sort_values(
    ["Consensus_Rank", "Mean_Score"],
    ascending=[True, True]
)

summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "TQ_Consensus_Ranking_41compounds.csv"
    ),
    index=False
)

# ============================================================
# TOP 10
# ============================================================
top10 = summary.head(10).copy()

top10.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "Top10_TQ_Consensus.csv"
    ),
    index=False
)

print("\n" + "=" * 110)
print("TOP 10 TQ COMPOUNDS — VINA ENSEMBLE CONSENSUS")
print("=" * 110)

display_columns = [
    "Consensus_Rank",
    "TQ_D",
    "N_Ensembles",
    "Mean_Score",
    "Median_Score",
    "Best_Score",
    "SD_Score",
    "Average_Rank",
    "Best_Score_Rank",
    "Z_Score_Rank",
    "Top10_Count",
    "Top10_Enrichment_Percent"
]

print(
    top10[display_columns]
    .round(3)
    .to_string(index=False)
)

# ============================================================
# HEATMAP: DOCKING SCORES
# ============================================================
fig, ax = plt.subplots(figsize=(10, 8))

im = ax.imshow(
    docking_matrix.values,
    aspect="auto",
    interpolation="nearest"
)

ax.set_xticks(np.arange(len(docking_matrix.columns)))
ax.set_xticklabels(
    docking_matrix.columns,
    rotation=90,
    fontsize=8
)

ax.set_yticks(np.arange(len(docking_matrix.index)))
ax.set_yticklabels(
    docking_matrix.index,
    fontsize=8
)

ax.set_xlabel("PLK1-PBD Ensemble")
ax.set_ylabel("Thymoquinone Compound")
ax.set_title(
    "Ensemble Docking Matrix",
    fontsize=15,
    weight="bold"
)

cbar = plt.colorbar(im)
cbar.set_label("Vina Affinity (kcal/mol)")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "Vina_Docking_Matrix_Heatmap.png"
    ),
    dpi=600,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# HEATMAP: RANKS
# ============================================================
fig, ax = plt.subplots(figsize=(10, 8))

im = ax.imshow(
    rank_matrix.values,
    aspect="auto",
    interpolation="nearest"
)

ax.set_xticks(np.arange(len(rank_matrix.columns)))
ax.set_xticklabels(
    rank_matrix.columns,
    rotation=90,
    fontsize=8
)

ax.set_yticks(np.arange(len(rank_matrix.index)))
ax.set_yticklabels(
    rank_matrix.index,
    fontsize=8
)

ax.set_xlabel("PLK1-PBD Ensemble")
ax.set_ylabel("Thymoquinone Compound")
ax.set_title(
    "Ensemble Rank Matrix",
    fontsize=15,
    weight="bold"
)

cbar = plt.colorbar(im)
cbar.set_label("Vina Rank")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "Vina_Rank_Matrix_Heatmap.png"
    ),
    dpi=600,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# ENRICHMENT PLOT
# ============================================================
enrichment_plot = summary.sort_values(
    "Top10_Enrichment_Percent",
    ascending=True
)

fig, ax = plt.subplots(figsize=(5, 7))

ax.barh(
    enrichment_plot["TQ_D"],
    enrichment_plot["Top10_Enrichment_Percent"]
)

ax.set_xlabel("Top-10 Enrichment Frequency (%)")
ax.set_ylabel("Compound")
ax.set_title(
    "Top-10 Enrichment Across PLK1-PBD Ensembles"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "Vina_Enrichment_Frequency.png"
    ),
    dpi=600,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# LIGAND MAPPING
# ============================================================
mapping = pd.DataFrame({
    "Ligand_Number": range(1, 42),
    "Vina_Ligand_ID": [
        f"Lig{i:03d}" for i in range(1, 42)
    ],
    "Compound": [
        ligand_label(i) for i in range(1, 42)
    ]
})

mapping.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "Vina_Ligand_Mapping.csv"
    ),
    index=False
)

print("\n" + "=" * 80)
print("VINA WEEK 5 ANALYSIS COMPLETE")
print("=" * 80)
print(f"Ensembles processed : {combined['Ensemble'].nunique()}")
print(f"Compounds detected  : {combined['TQ_D'].nunique()}")
print(f"Docking records     : {len(combined)}")
print("\nMapping:")
print("  Lig001-Lig040 -> TQ_D1-TQ_D40")
print("  Lig041        -> TQ")
print(f"\nResults saved in: {OUTPUT_DIR}/")
