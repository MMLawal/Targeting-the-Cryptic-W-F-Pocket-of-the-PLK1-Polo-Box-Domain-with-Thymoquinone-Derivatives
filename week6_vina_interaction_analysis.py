#!/usr/bin/env python3

"""
Week 6 — AutoDock Vina Interaction Analysis
PLK1-PBD cryptic W–F pocket

RECEPTORS:
    ./50_plk1-pbd_ensembles_W-F/
        ensemble_001.pdbqt
        ...
        ensemble_050.pdbqt

DOCKED POSES:
    ./docking_results/
        ensemble_001_Lig001.pdbqt
        ...
        ensemble_050_Lig041.pdbqt

Ligand mapping:
    Lig001 -> TQ_D1
    ...
    Lig040 -> TQ_D40
    Lig041 -> TQ (parent thymoquinone)

IMPORTANT:
    Each Vina ligand PDBQT contains multiple MODEL records.
    Only MODEL 1 is analyzed because it is the best-ranked Vina pose.

The analysis is a proximity-based interaction screen:
    • H-bond: donor/acceptor atom pairs <= 3.5 Å
    • Hydrophobic: ligand carbon / hydrophobic protein atoms <= 4.0 Å
    • Pi-aromatic: aromatic ligand / aromatic protein atoms <= 5.5 Å

These are suitable for ensemble-level screening and student analysis,
but should be treated as interaction fingerprints rather than rigorous
geometry-based hydrogen-bond or pi-stacking assignments.

W–F pocket residues (renumbered ensemble numbering):
    36 ILE
    37 PRO
    40 TRP
    58 CYS
    122 ARG
    123 ALA
    124 GLY
    125 ALA
    126 ASN
    127 ILE
    128 THR
    129 PRO
    181 ?
    183 ILE
    189 PHE

Key anchors:
    Trp40  = Trp410 in the original PLK1-PBD numbering
    Phe189  = Phe559 in the original PLK1-PBD numbering

If your receptor PDBQT uses original numbering instead, change
TRP_RESIDUE, PHE_RESIDUE and WF_RESIDUES below.
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# USER SETTINGS
# ============================================================

REC_DIR = "./50_plk1-pbd_ensembles_W-F"
OUT_DIR = "./docking_results"

OUTPUT_DIR = "./Week6_Vina_Interaction_Atlas"

N_ENSEMBLES = 50
N_LIGANDS = 41

# Only the best Vina pose
TOP_MODEL = 1

# Distance cutoffs in Angstrom
HBOND_DISTANCE = 3.5
HYDROPHOBIC_DISTANCE = 4.0
PI_DISTANCE = 5.5

# ------------------------------------------------------------
# W–F pocket residues in the renumbered PBD ensembles
# ------------------------------------------------------------

TRP_RESIDUE = 40
PHE_RESIDUE = 189

WF_RESIDUES = [
    36, 37, 40, 58,
    122, 123, 124, 125,
    126, 127, 128, 129,
    181, 183, 189
]

# Original PLK1 numbering:
# Trp410 -> ensemble residue 40
# Phe559 -> ensemble residue 189

ORIGINAL_NUMBERING = {
    40: 410,
    189: 559
}

# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# RESIDUE CLASSIFICATION
# ============================================================

HYDROPHOBIC_RESIDUES = {
    "ALA", "VAL", "LEU", "ILE",
    "MET", "PRO", "PHE", "TRP",
    "TYR"
}

AROMATIC_RESIDUES = {
    "PHE", "TRP", "TYR", "HIS"
}

STANDARD_RESIDUES = {
    "ALA", "ARG", "ASN", "ASP",
    "CYS", "GLN", "GLU", "GLY",
    "HIS", "ILE", "LEU", "LYS",
    "MET", "PHE", "PRO", "SER",
    "THR", "TRP", "TYR", "VAL"
}


# ============================================================
# LIGAND MAPPING
# ============================================================

def ligand_label(ligand_number):

    if ligand_number == 41:
        return "TQ"

    return f"TQ_D{ligand_number}"


# ============================================================
# PARSE DOCKING FILENAME
# ============================================================

def parse_ligand_filename(filename):

    """
    Example:
        ensemble_001_Lig001.pdbqt
        ensemble_050_Lig041.pdbqt
    """

    match = re.match(
        r"^ensemble_(\d+)_Lig(\d+)\.pdbqt$",
        filename,
        re.IGNORECASE
    )

    if match is None:
        return None, None

    return (
        int(match.group(1)),
        int(match.group(2))
    )


# ============================================================
# PDBQT ATOM PARSER
# ============================================================

def parse_pdbqt_atom_line(line):

    """
    Parse an ATOM/HETATM line from PDBQT.

    Coordinates are taken from standard PDB columns.
    The final PDBQT atom-type field is also retained.
    """

    if not (
        line.startswith("ATOM")
        or line.startswith("HETATM")
    ):
        return None

    try:

        atom_name = line[12:16].strip()
        residue_name = line[17:20].strip()
        chain = line[21:22].strip()

        # PDBQT residue number
        residue_number = int(
            line[22:26].strip()
        )

        x = float(
            line[30:38].strip()
        )

        y = float(
            line[38:46].strip()
        )

        z = float(
            line[46:54].strip()
        )

        # PDBQT charge field
        charge = None

        if len(line) >= 70:

            try:
                charge = float(
                    line[70:76].strip()
                )
            except ValueError:
                charge = None

        # PDBQT atom type is normally the last token
        tokens = line.split()

        atom_type = (
            tokens[-1].upper()
            if tokens
            else ""
        )

        # Infer element from atom name first,
        # then PDBQT atom type.
        element = infer_element(
            atom_name,
            atom_type
        )

        return {
            "atom": atom_name,
            "resname": residue_name,
            "residue": residue_number,
            "chain": chain,
            "x": x,
            "y": y,
            "z": z,
            "charge": charge,
            "atom_type": atom_type,
            "element": element
        }

    except (ValueError, IndexError):

        return None


# ============================================================
# INFER ELEMENT
# ============================================================

def infer_element(atom_name, atom_type):

    """
    Infer chemical element from PDBQT atom type / atom name.
    """

    atom_type = str(atom_type).upper()
    atom_name = str(atom_name).strip().upper()

    # Common AutoDock/Vina atom types
    if atom_type.startswith("CL"):
        return "Cl"

    if atom_type.startswith("BR"):
        return "Br"

    if atom_type.startswith("SI"):
        return "Si"

    if atom_type.startswith("NA"):
        return "N"

    if atom_type.startswith("OA"):
        return "O"

    if atom_type.startswith("SA"):
        return "S"

    if atom_type.startswith("HD"):
        return "H"

    if atom_type.startswith("A"):
        return "C"

    for element in [
        "Cl", "Br", "Si",
        "Na", "Mg", "Ca",
        "Fe", "Zn", "P", "S",
        "O", "N", "C", "H"
    ]:

        if atom_name.startswith(
            element.upper()
        ):
            return element

    return atom_name[0] if atom_name else ""


# ============================================================
# READ RECEPTOR PDBQT
# ============================================================

def read_receptor_pdbqt(pdbqt_file):

    """
    Read receptor PDBQT.

    Receptor files are expected to contain a single structure.
    """

    atoms = []

    with open(
        pdbqt_file,
        "r",
        errors="ignore"
    ) as f:

        for line in f:

            atom = parse_pdbqt_atom_line(
                line
            )

            if atom is None:
                continue

            if atom["resname"] not in STANDARD_RESIDUES:
                continue

            if atom["element"] == "H":
                continue

            atoms.append(atom)

    return pd.DataFrame(atoms)


# ============================================================
# READ ONLY MODEL 1 OF VINA LIGAND PDBQT
# ============================================================

def read_top_vina_model(pdbqt_file):

    """
    Extract MODEL 1 only.

    Vina PDBQT may contain:
        MODEL 1
        ...
        ENDMDL
        MODEL 2
        ...

    MODEL 1 is the best-ranked pose.
    """

    atoms = []

    in_model = False
    model_found = False

    with open(
        pdbqt_file,
        "r",
        errors="ignore"
    ) as f:

        for line in f:

            stripped = line.strip()

            if stripped.upper().startswith("MODEL"):

                model_number_text = (
                    stripped.split()[-1]
                )

                try:
                    model_number = int(
                        model_number_text
                    )
                except ValueError:
                    model_number = None

                if model_number == TOP_MODEL:

                    in_model = True
                    model_found = True
                    continue

                if model_found:

                    # We have reached another model
                    break

            if (
                in_model
                and stripped.upper() == "ENDMDL"
            ):

                break

            if in_model:

                atom = parse_pdbqt_atom_line(
                    line
                )

                if atom is not None:

                    if atom["element"] != "H":

                        atoms.append(atom)

    # --------------------------------------------------------
    # Some PDBQT files may contain only one pose and no MODEL
    # records. In that case read all atoms.
    # --------------------------------------------------------

    if not model_found:

        atoms = []

        with open(
            pdbqt_file,
            "r",
            errors="ignore"
        ) as f:

            for line in f:

                atom = parse_pdbqt_atom_line(
                    line
                )

                if atom is not None:

                    if atom["element"] != "H":

                        atoms.append(atom)

    return pd.DataFrame(atoms)


# ============================================================
# DISTANCE
# ============================================================

def distance(a, b):

    return np.sqrt(
        (a["x"] - b["x"]) ** 2
        + (a["y"] - b["y"]) ** 2
        + (a["z"] - b["z"]) ** 2
    )


# ============================================================
# HYDROGEN-BOND PROXIMITY SCREEN
# ============================================================

def identify_hbond(
    ligand_atoms,
    protein_atoms
):

    """
    Approximate hydrogen-bond screening.

    Because Vina PDBQT files normally contain no explicit H atoms,
    donor/acceptor assignment is inferred from heteroatom types.

    A contact is counted when a plausible donor/acceptor atom pair
    is <= 3.5 Å.

    This is a screening criterion, not a geometry-complete
    hydrogen-bond definition.
    """

    ligand_hetero = ligand_atoms[
        ligand_atoms["element"].isin(
            ["N", "O", "S"]
        )
    ]

    protein_hetero = protein_atoms[
        protein_atoms["element"].isin(
            ["N", "O", "S"]
        )
    ]

    contacts = []

    for _, lig in ligand_hetero.iterrows():

        for _, prot in protein_hetero.iterrows():

            d = distance(
                lig,
                prot
            )

            if d > HBOND_DISTANCE:
                continue

            # Avoid calling two identical atom types automatically
            # hydrogen bonds in cases where both are unlikely.
            # N/O/S proximity is retained as a broad screen.
            contacts.append(
                {
                    "interaction": "H-bond",
                    "direction": "Ligand-Protein",
                    "ligand_atom": lig["atom"],
                    "ligand_element": lig["element"],
                    "protein_residue": prot["residue"],
                    "protein_resname": prot["resname"],
                    "protein_atom": prot["atom"],
                    "distance": round(d, 3)
                }
            )

    return contacts


# ============================================================
# HYDROPHOBIC CONTACTS
# ============================================================

def identify_hydrophobic_contacts(
    ligand_atoms,
    protein_atoms
):

    """
    Approximate hydrophobic contact:
        ligand carbon
        +
        hydrophobic amino-acid residue
        +
        distance <= 4.0 Å
    """

    ligand_carbon = ligand_atoms[
        ligand_atoms["element"] == "C"
    ]

    protein_hydrophobic = protein_atoms[
        protein_atoms["resname"].isin(
            HYDROPHOBIC_RESIDUES
        )
    ]

    contacts = []

    for _, lig in ligand_carbon.iterrows():

        for _, prot in protein_hydrophobic.iterrows():

            d = distance(
                lig,
                prot
            )

            if d <= HYDROPHOBIC_DISTANCE:

                contacts.append(
                    {
                        "interaction": "Hydrophobic",
                        "direction": "Ligand-Protein",
                        "ligand_atom": lig["atom"],
                        "ligand_element": lig["element"],
                        "protein_residue": prot["residue"],
                        "protein_resname": prot["resname"],
                        "protein_atom": prot["atom"],
                        "distance": round(d, 3)
                    }
                )

    return contacts


# ============================================================
# PI-AROMATIC PROXIMITY SCREEN
# ============================================================

def identify_pi_interactions(
    ligand_atoms,
    protein_atoms
):

    """
    Approximate aromatic interaction.

    AutoDock Vina PDBQT does not always retain complete aromatic
    ring topology in a convenient form. Therefore this analysis
    uses aromatic ligand atoms (PDBQT atom type A or aromatic
    carbon) near aromatic protein residues.

    This should be interpreted as an aromatic-contact screen,
    not a rigorous centroid/angle-based pi-stacking classifier.
    """

    # AutoDock aromatic carbon type is generally "A"
    aromatic_ligand = ligand_atoms[
        (
            ligand_atoms["atom_type"]
            .str.upper()
            .eq("A")
        )
        |
        (
            ligand_atoms["atom_type"]
            .str.upper()
            .str.startswith("A")
        )
    ]

    aromatic_protein = protein_atoms[
        protein_atoms["resname"].isin(
            AROMATIC_RESIDUES
        )
    ]

    contacts = []

    for _, lig in aromatic_ligand.iterrows():

        for _, prot in aromatic_protein.iterrows():

            d = distance(
                lig,
                prot
            )

            if d <= PI_DISTANCE:

                contacts.append(
                    {
                        "interaction": "Pi-Aromatic",
                        "direction": "Ligand-Protein",
                        "ligand_atom": lig["atom"],
                        "ligand_element": lig["element"],
                        "protein_residue": prot["residue"],
                        "protein_resname": prot["resname"],
                        "protein_atom": prot["atom"],
                        "distance": round(d, 3)
                    }
                )

    return contacts


# ============================================================
# PROCESS ONE ENSEMBLE/LIGAND SYSTEM
# ============================================================

def analyze_system(
    ensemble,
    ligand_number,
    receptor_file,
    ligand_file
):

    receptor_atoms = read_receptor_pdbqt(
        receptor_file
    )

    ligand_atoms = read_top_vina_model(
        ligand_file
    )

    if receptor_atoms.empty:
        raise ValueError(
            "No receptor atoms were read."
        )

    if ligand_atoms.empty:
        raise ValueError(
            "No ligand atoms were read from MODEL 1."
        )

    hbonds = identify_hbond(
        ligand_atoms,
        receptor_atoms
    )

    hydrophobic = identify_hydrophobic_contacts(
        ligand_atoms,
        receptor_atoms
    )

    pi_contacts = identify_pi_interactions(
        ligand_atoms,
        receptor_atoms
    )

    interactions = (
        hbonds
        + hydrophobic
        + pi_contacts
    )

    ligand_name = ligand_label(
        ligand_number
    )

    for contact in interactions:

        contact["Ensemble"] = ensemble
        contact["Ligand_Number"] = ligand_number
        contact["TQ_D"] = ligand_name
        contact["Receptor_File"] = os.path.basename(
            receptor_file
        )
        contact["Ligand_File"] = os.path.basename(
            ligand_file
        )

    return interactions


# ============================================================
# FIND AND PROCESS ALL VINA DOCKINGS
# ============================================================

print("=" * 90)
print("WEEK 6 — AUTODOCK VINA PLK1-PBD W–F POCKET INTERACTION ATLAS")
print("=" * 90)

print(
    f"\nReceptor directory: {REC_DIR}"
)

print(
    f"Docking directory : {OUT_DIR}"
)

print(
    f"Output directory  : {OUTPUT_DIR}"
)

if not os.path.isdir(REC_DIR):

    raise FileNotFoundError(
        f"Receptor directory not found: {REC_DIR}"
    )

if not os.path.isdir(OUT_DIR):

    raise FileNotFoundError(
        f"Docking directory not found: {OUT_DIR}"
    )


all_interactions = []

system_summary = []

missing_systems = []

error_systems = []


for ensemble in range(
    1,
    N_ENSEMBLES + 1
):

    receptor_file = os.path.join(
        REC_DIR,
        f"ensemble_{ensemble:03d}.pdbqt"
    )

    if not os.path.isfile(receptor_file):

        print(
            f"WARNING: Missing receptor: "
            f"{receptor_file}"
        )

        continue

    for ligand_number in range(
        1,
        N_LIGANDS + 1
    ):

        ligand_file = os.path.join(
            OUT_DIR,
            f"ensemble_{ensemble:03d}_Lig{ligand_number:03d}.pdbqt"
        )

        ligand_name = ligand_label(
            ligand_number
        )

        if not os.path.isfile(ligand_file):

            missing_systems.append(
                {
                    "Ensemble": ensemble,
                    "Ligand_Number": ligand_number,
                    "TQ_D": ligand_name,
                    "Missing_File": ligand_file
                }
            )

            continue

        try:

            interactions = analyze_system(
                ensemble,
                ligand_number,
                receptor_file,
                ligand_file
            )

            all_interactions.extend(
                interactions
            )

            # Count interaction types
            n_hbond = sum(
                x["interaction"] == "H-bond"
                for x in interactions
            )

            n_hydrophobic = sum(
                x["interaction"] == "Hydrophobic"
                for x in interactions
            )

            n_pi = sum(
                x["interaction"] == "Pi-Aromatic"
                for x in interactions
            )

            system_summary.append(
                {
                    "Ensemble": ensemble,
                    "Ligand_Number": ligand_number,
                    "TQ_D": ligand_name,
                    "N_Hbond": n_hbond,
                    "N_Hydrophobic": n_hydrophobic,
                    "N_Pi_Aromatic": n_pi,
                    "N_Total_Interactions": len(
                        interactions
                    )
                }
            )

        except Exception as e:

            error_systems.append(
                {
                    "Ensemble": ensemble,
                    "Ligand_Number": ligand_number,
                    "TQ_D": ligand_name,
                    "Error": str(e)
                }
            )

    print(
        f"Processed ensemble {ensemble:03d}"
    )


# ============================================================
# SAVE SYSTEM STATUS
# ============================================================

if missing_systems:

    pd.DataFrame(
        missing_systems
    ).to_csv(
        os.path.join(
            OUTPUT_DIR,
            "Missing_Vina_Systems.csv"
        ),
        index=False
    )

if error_systems:

    pd.DataFrame(
        error_systems
    ).to_csv(
        os.path.join(
            OUTPUT_DIR,
            "Vina_Interaction_Errors.csv"
        ),
        index=False
    )


# ============================================================
# RAW INTERACTION DATA
# ============================================================

if not all_interactions:

    raise RuntimeError(
        "No interactions were detected. "
        "Check receptor/ligand numbering and PDBQT formatting."
    )


interactions_df = pd.DataFrame(
    all_interactions
)


# Put metadata first
metadata_columns = [
    "Ensemble",
    "Ligand_Number",
    "TQ_D",
    "interaction",
    "direction",
    "protein_residue",
    "protein_resname",
    "protein_atom",
    "ligand_atom",
    "ligand_element",
    "distance",
    "Receptor_File",
    "Ligand_File"
]

interactions_df = interactions_df[
    [
        c for c in metadata_columns
        if c in interactions_df.columns
    ]
]


interactions_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "All_Vina_Interactions.csv"
    ),
    index=False
)


# ============================================================
# W–F POCKET FILTER
# ============================================================

wf_interactions = interactions_df[
    interactions_df[
        "protein_residue"
    ].isin(
        WF_RESIDUES
    )
].copy()


wf_interactions.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "Vina_WF_Pocket_Interactions.csv"
    ),
    index=False
)


# ============================================================
# TRP40 / PHE189 ANCHOR INTERACTIONS
# ============================================================

wf_anchor = wf_interactions[
    wf_interactions[
        "protein_residue"
    ].isin(
        [
            TRP_RESIDUE,
            PHE_RESIDUE
        ]
    )
].copy()


wf_anchor["Original_PLK1_Residue"] = (
    wf_anchor[
        "protein_residue"
    ].map(
        ORIGINAL_NUMBERING
    )
)


wf_anchor.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "Vina_Trp410_Phe559_Interactions.csv"
    ),
    index=False
)


# ============================================================
# UNIQUE CONTACTS PER ENSEMBLE
# ============================================================

unique_contacts = (
    wf_interactions[
        [
            "Ensemble",
            "TQ_D",
            "protein_residue",
            "protein_resname",
            "interaction"
        ]
    ]
    .drop_duplicates()
)


# ============================================================
# INTERACTION FREQUENCY BY COMPOUND
# ============================================================

frequency = (
    unique_contacts
    .groupby(
        [
            "TQ_D",
            "protein_residue",
            "protein_resname",
            "interaction"
        ]
    )
    .size()
    .reset_index(
        name="Ensemble_Count"
    )
)


frequency["Frequency_Percent"] = (
    frequency["Ensemble_Count"]
    / N_ENSEMBLES
    * 100
)


frequency["Original_PLK1_Residue"] = (
    frequency[
        "protein_residue"
    ].map(
        ORIGINAL_NUMBERING
    )
)


frequency = frequency.sort_values(
    [
        "TQ_D",
        "Frequency_Percent"
    ],
    ascending=[
        True,
        False
    ]
)


frequency.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "Vina_Interaction_Frequency.csv"
    ),
    index=False
)


# ============================================================
# CONSERVED INTERACTIONS
# >= 80% of ensembles
# ============================================================

conserved = frequency[
    frequency[
        "Frequency_Percent"
    ] >= 80
].copy()


conserved.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "Vina_Conserved_Interactions.csv"
    ),
    index=False
)


# ============================================================
# STRUCTURE-SPECIFIC INTERACTIONS
# <= 20% of ensembles
# ============================================================

structure_specific = frequency[
    frequency[
        "Frequency_Percent"
    ] <= 20
].copy()


structure_specific.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "Vina_Structure_Specific_Interactions.csv"
    ),
    index=False
)


# ============================================================
# RESIDUE-LEVEL FREQUENCY
# ============================================================

residue_frequency = (
    unique_contacts
    .groupby(
        [
            "protein_residue",
            "protein_resname",
            "interaction"
        ]
    )
    .size()
    .reset_index(
        name="Contact_Count"
    )
)


residue_frequency["Frequency_Percent"] = (
    residue_frequency["Contact_Count"]
    / N_ENSEMBLES
    * 100
)


residue_frequency["Original_PLK1_Residue"] = (
    residue_frequency[
        "protein_residue"
    ].map(
        ORIGINAL_NUMBERING
    )
)


residue_frequency = residue_frequency.sort_values(
    "Frequency_Percent",
    ascending=False
)


residue_frequency.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "Vina_Residue_Interaction_Frequency.csv"
    ),
    index=False
)


# ============================================================
# INTERACTION FREQUENCY MATRIX
# ============================================================

heatmap_data = frequency.pivot_table(
    index="TQ_D",
    columns=[
        "protein_residue",
        "protein_resname",
        "interaction"
    ],
    values="Frequency_Percent",
    fill_value=0
)


# Force consistent compound order
compound_order = [
    f"TQ_D{i}"
    for i in range(1, 41)
] + ["TQ"]


heatmap_data = heatmap_data.reindex(
    compound_order
)


heatmap_data.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "Vina_Interaction_Frequency_Matrix.csv"
    )
)


# ============================================================
# W–F INTERACTION HEATMAP
# ============================================================

if not heatmap_data.empty:

    fig, ax = plt.subplots(
        figsize=(18, 12)
    )

    im = ax.imshow(
        heatmap_data.values,
        aspect="auto",
        interpolation="nearest"
    )

    ax.set_yticks(
        np.arange(
            len(heatmap_data.index)
        )
    )

    ax.set_yticklabels(
        heatmap_data.index,
        fontsize=10
    )

    labels = []

    for col in heatmap_data.columns:

        residue = col[0]
        resname = col[1]
        interaction = col[2]

        original = ORIGINAL_NUMBERING.get(
            residue
        )

        if original is not None:

            labels.append(
                f"{resname}{residue}/{original}\n{interaction}"
            )

        else:

            labels.append(
                f"{resname}{residue}\n{interaction}"
            )

    ax.set_xticks(
        np.arange(
            len(labels)
        )
    )

    ax.set_xticklabels(
        labels,
        rotation=90,
        fontsize=10
    )

    ax.set_xlabel(
        "PLK1-PBD W–F Pocket Interaction"
    )

    ax.set_ylabel(
        "Thymoquinone Compound"
    )

    ax.set_title(
        "W–F Pocket Interaction Frequency",
        fontsize=15,
        weight="bold"
    )

    cbar = plt.colorbar(
        im
    )

    cbar.set_label(
        "Interaction Frequency (%)"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "Vina_WF_Interaction_Frequency_Heatmap.png"
        ),
        dpi=600,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# TRP410 / PHE559 ANCHOR HEATMAP
# ============================================================

anchor_frequency = frequency[
    frequency[
        "protein_residue"
    ].isin(
        [
            TRP_RESIDUE,
            PHE_RESIDUE
        ]
    )
].copy()


anchor_heatmap = anchor_frequency.pivot_table(
    index="TQ_D",
    columns=[
        "protein_residue",
        "protein_resname",
        "interaction"
    ],
    values="Frequency_Percent",
    fill_value=0
)


anchor_heatmap = anchor_heatmap.reindex(
    compound_order
)


if not anchor_heatmap.empty:

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    im = ax.imshow(
        anchor_heatmap.values,
        aspect="auto",
        interpolation="nearest"
    )

    ax.set_yticks(
        np.arange(
            len(anchor_heatmap.index)
        )
    )

    ax.set_yticklabels(
        anchor_heatmap.index,
        fontsize=10
    )

    labels = []

    for col in anchor_heatmap.columns:

        residue = col[0]
        resname = col[1]
        interaction = col[2]

        original = ORIGINAL_NUMBERING.get(
            residue
        )

        labels.append(
            f"{resname}{residue}/{original}\n{interaction}"
        )

    ax.set_xticks(
        np.arange(
            len(labels)
        )
    )

    ax.set_xticklabels(
        labels,
        rotation=45,
        ha="right"
    )

    ax.set_xlabel(
        "W–F Pocket Anchor"
    )

    ax.set_ylabel(
        "Thymoquinone Compound"
    )

    ax.set_title(
        "Trp410–Phe559 Interaction Frequency",
        fontsize=15,
        weight="bold"
    )

    cbar = plt.colorbar(
        im
    )

    cbar.set_label(
        "Interaction Frequency (%)"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "Vina_Trp410_Phe559_Heatmap.png"
        ),
        dpi=600,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# ANCHOR SUMMARY PER COMPOUND
# ============================================================

anchor_unique = (
    wf_anchor[
        [
            "Ensemble",
            "TQ_D",
            "protein_residue",
            "protein_resname",
            "interaction"
        ]
    ]
    .drop_duplicates()
)


anchor_summary = (
    anchor_unique
    .groupby(
        [
            "TQ_D",
            "protein_residue",
            "protein_resname",
            "interaction"
        ]
    )
    .size()
    .reset_index(
        name="Ensemble_Count"
    )
)


anchor_summary["Frequency_Percent"] = (
    anchor_summary["Ensemble_Count"]
    / N_ENSEMBLES
    * 100
)


anchor_summary["Original_PLK1_Residue"] = (
    anchor_summary[
        "protein_residue"
    ].map(
        ORIGINAL_NUMBERING
    )
)


anchor_summary = anchor_summary.sort_values(
    [
        "TQ_D",
        "Frequency_Percent"
    ],
    ascending=[
        True,
        False
    ]
)


anchor_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "Vina_Anchor_Summary_Trp410_Phe559.csv"
    ),
    index=False
)


# ============================================================
# COMPOUND-LEVEL W–F ANCHOR SCORE
# ============================================================

# Count distinct ensembles in which the compound contacts either
# Trp40/Trp410 or Phe189/Phe559.
anchor_ensemble_counts = (
    anchor_unique
    .groupby("TQ_D")["Ensemble"]
    .nunique()
)


# Count distinct ensembles contacting both anchors
both_anchor = (
    anchor_unique
    .groupby(
        [
            "TQ_D",
            "Ensemble"
        ]
    )["protein_residue"]
    .nunique()
)


both_anchor_counts = (
    both_anchor[
        both_anchor >= 2
    ]
    .groupby(level=0)
    .size()
)


anchor_score = pd.DataFrame(
    {
        "TQ_D": compound_order
    }
)


anchor_score["Anchor_Contact_Ensembles"] = (
    anchor_score["TQ_D"]
    .map(
        anchor_ensemble_counts
    )
    .fillna(0)
    .astype(int)
)


anchor_score["Anchor_Frequency_Percent"] = (
    anchor_score[
        "Anchor_Contact_Ensembles"
    ]
    / N_ENSEMBLES
    * 100
)


anchor_score["Both_Trp_Phe_Ensembles"] = (
    anchor_score["TQ_D"]
    .map(
        both_anchor_counts
    )
    .fillna(0)
    .astype(int)
)


anchor_score["Both_Trp_Phe_Frequency_Percent"] = (
    anchor_score[
        "Both_Trp_Phe_Ensembles"
    ]
    / N_ENSEMBLES
    * 100
)


anchor_score = anchor_score.sort_values(
    [
        "Both_Trp_Phe_Frequency_Percent",
        "Anchor_Frequency_Percent"
    ],
    ascending=False
)


anchor_score.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "Vina_WF_Anchor_Score_by_Compound.csv"
    ),
    index=False
)


# ============================================================
# SYSTEM SUMMARY
# ============================================================

if system_summary:

    system_summary_df = pd.DataFrame(
        system_summary
    )

    system_summary_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "Vina_System_Interaction_Summary.csv"
        ),
        index=False
    )


# ============================================================
# FINAL REPORT
# ============================================================

print("\n")
print("=" * 90)
print("WEEK 6 VINA INTERACTION ANALYSIS COMPLETE")
print("=" * 90)

print(
    f"\nSystems expected: "
    f"{N_ENSEMBLES * N_LIGANDS}"
)

print(
    f"Systems successfully analyzed: "
    f"{len(system_summary)}"
)

print(
    f"Missing systems: "
    f"{len(missing_systems)}"
)

print(
    f"Systems with errors: "
    f"{len(error_systems)}"
)

print(
    f"\nTotal detected interactions: "
    f"{len(interactions_df)}"
)

print(
    f"W–F pocket interactions: "
    f"{len(wf_interactions)}"
)

print(
    f"Trp410/Phe559 interactions: "
    f"{len(wf_anchor)}"
)

print(
    f"\nConserved W–F interactions (>=80%): "
    f"{len(conserved)}"
)

print(
    f"Structure-specific W–F interactions (<=20%): "
    f"{len(structure_specific)}"
)

print(
    "\nImportant residue mapping:"
)

print(
    "  Ensemble residue 40  = PLK1 Trp410"
)

print(
    "  Ensemble residue 189 = PLK1 Phe559"
)

print(
    f"\nResults saved in:"
    f"\n{OUTPUT_DIR}/"
)

print(
    "\nKey files:"
)

for filename in sorted(
    os.listdir(OUTPUT_DIR)
):

    print(
        "  ",
        filename
    )
