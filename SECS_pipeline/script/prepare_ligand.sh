# Based on these documentations
#     https://meeko.readthedocs.io/en/release-doc/lig_prep_basic.html
#     https://meeko.readthedocs.io/en/release-doc/cli_rec_prep.html
#     https://autodock-vina.readthedocs.io/en/latest/docking_basic.html#

# Example run command of this script:
#     sh prepare_ligand.sh "PF74" "CN(C([C@H](CC1=CC=CC=C1)NC(CC2=C(C)NC3=C2C=CC=C3)=O)=O)C4=CC=CC=C4" 
#     sh prepare_ligand.sh "CPX" "O=C1C=C(C)C=C(C2CCCCC2)N1O"
#     sh prepare_ligand.sh "HBD_test" "ON1C(NC(NCC2=CC=CC=C2)=CC1=O)=O"

#!/bin/bash
set -euo pipefail

# Usage check
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <lig_name> <lig_smiles>"
    exit 1
fi

# Set up variables
lig_name="$1"
lig_smiles="$2"
out_dir="prepared_ligands"

# Create SDF from SMILES
scrub.py $lig_smiles -o $lig_name.sdf --skip_tautomers --ph_low 5 --ph_high 9

# Make directory for the prepared ligands
#rm -r prepared_ligands
#mkdir prepared_ligands

# Create PDBQT (needed for Vina) from SDF
# Kinda janky but --multimol_outdir doesn't work if there's only one prepared ligand
# So run both to be sure?
mk_prepare_ligand.py -i $lig_name.sdf --multimol_outdir $out_dir --multimol_prefix $lig_name
mk_prepare_ligand.py -i $lig_name.sdf
mv $lig_name.sdf $out_dir

# Two cases:
#   a) Single compound in $lig_name.sdf, which then gives $lig_name.pdbqt and NO ($lig_name)_*.pdbqt
#   b) Multiple compound in $lig_name.sdf, which then gives $lig_name.pdbqt and ($lig_name)_*.pdbqt
# Kinda janky solution, but seems like $lig_name.pdbqt is always (I checked it for two cases) same as ($lig_name)_1.pdbqt
# So just overwrite latter with former
mv $lig_name.pdbqt $out_dir/"$lig_name"-1.pdbqt

