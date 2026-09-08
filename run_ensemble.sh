#!/bin/bash

LIG_DIR="./tq_ligands"
REC_DIR="./50_plk1-pbd_ensembles_W-F"
OUT_DIR="./docking_results"

#mkdir -p $OUT_DIR

for receptor in $REC_DIR/ensemble_*.pdbqt; do

    recbase=$(basename "$receptor" .pdbqt)

    for ligand in $LIG_DIR/Lig041.pdbqt; do

        ligbase=$(basename "$ligand" .pdbqt)

        outname="${OUT_DIR}/${recbase}_${ligbase}.pdbqt"
        logfile="${OUT_DIR}/${recbase}_${ligbase}.log"

        echo "Docking $ligbase into $recbase"

        vina \
            --receptor "$receptor" \
            --ligand "$ligand" \
            --center_x 18 \
            --center_y 49 \
            --center_z 43 \
            --size_x 20 \
            --size_y 21 \
            --size_z 22 \
            --exhaustiveness 12 \
            --num_modes 5 \
            --out "$outname" \
            --log "$logfile"

    done
done

