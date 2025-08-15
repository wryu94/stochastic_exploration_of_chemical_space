# Based on these documentations
# https://meeko.readthedocs.io/en/release-doc/lig_prep_basic.html
# https://meeko.readthedocs.io/en/release-doc/cli_rec_prep.html
# https://autodock-vina.readthedocs.io/en/latest/docking_basic.html#

# Example run command of this script:
#     sh prepare_receptor.sh "4XFZ" "reference/4xfz_two_chain_dry_clean.pdb"

#!/bin/bash
set -euo pipefail

# Usage check
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <lig_name> <lig_smiles>"
    exit 1
fi

# Set up variables
receptor_name="$1"
receptor_pdb_path="$2"

# Prepare receptor with box specification (specific coordinate from Schrodinger) 
# Tried using AutoDock FF, but issue with using autogrid4 command, 
# so going with Vina FF (no need for affinity map)
#mk_prepare_receptor.py -i $receptor_pdb_path -o $receptor_name -p -v \
#--box_size 20 20 20 --box_center -58.68 11.90 25.66
mk_prepare_receptor.py -i $receptor_pdb_path -o $receptor_name -p

