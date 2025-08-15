import molpert as mpt
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import MolsToGridImage
from rdkit.Chem.Descriptors import CalcMolDescriptors
from rdkit import RDLogger
from rdkit.Chem import AllChem, DataStructs

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from PIL import Image
import textwrap
from tqdm import tqdm

import sascorer

from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina

# Suppress RDKit warnings and errors
RDLogger.DisableLog('rdApp.*')

from vina import Vina
import subprocess

def propagate_chemical_space_trajectory(
    starting_smiles,            # smiles of the starting structure
    iteration,         # how many times to do the perturbation for, including the first structure
    SA_score_threshold # only get structures with SA score lower than the threshold 
):
    def sensible_structure(molecule):
    # Only accept structures that raise no flags when using RDKit sanitization
        try:
            result = Chem.SanitizeMol(molecule)
            return result == Chem.rdmolops.SanitizeFlags.SANITIZE_NONE
        except Exception as e:
            # You can log or inspect e if needed
            return False

    def SA_score(molecule):
    # Only accept structures with SA score under some threshold
    # The threshold value is kinda arbitrary (and would depend on the system)
        return sascorer.calculateScore(molecule) < SA_score_threshold
    
    # Molecule array
    molecule = Chem.MolFromSmiles(starting_smiles, sanitize=True)
    mol_array = [Chem.Mol(molecule)]

    # Set constraint to only get 'sensible' structures
    # Could add pains filter here 
    constraints = mpt.MolecularConstraints()
    constraints.SetMoleculeConstraint(
        molecule_constraint=sensible_structure
    )
    constraints.SetMoleculeConstraint(
        molecule_constraint=SA_score
    )

    # Iterate for a given number of steps 
    for i in tqdm(range(iteration-1)):
        # Build molecule on rdkit & tag them which is needed for molpert
        mpt.TagMolecule(molecule)
        
        # Perturber and random number
        perturber = mpt.MoleculePerturber()
        #perturber = mpt.MoleculePerturber(use_aromatic_bonds=False, acyclic_can_be_aromatic=False)
            # Idk what the best setting for these / especially how to deal with aromaticity 
        prng = mpt.PRNG()

        # Actually perform perturbation 
        perturbation = perturber.Perturbation(
            molecule=molecule,
            prng=prng,
            constraints=constraints
        )
        mpt.ExecutePerturbation(perturbation, molecule)

        # Sanitize
        Chem.SanitizeMol(molecule)
        
        # Add the new perturbed molecule to the array
        # Reset smiles_i to be used for next perturbation 
        mol_array.append(Chem.Mol(molecule))

    return np.array(mol_array)

def mol_movie(mols, outfile, fps=6, img_size=200):
    """
    Create an animated GIF from a list of RDKit Mol objects,
    aligning each to the previous one based on their common substructure.
    
    mols: list of RDKit Mol objects
    outfile: name of the output GIF
    fps: frames per second
    """
    # Ensure molecules have coordinates for drawing
    for mol in mols:
        AllChem.Compute2DCoords(mol)

    images = []

    for i, mol in tqdm(enumerate(mols)):
        if i > 0:
            # Find maximum common substructure
            core = Chem.MolFromSmarts(Chem.MolToSmarts(mols[i-1]))
            match1 = mols[i-1].GetSubstructMatch(core)
            match2 = mol.GetSubstructMatch(core)

            if match1 and match2:
                AllChem.AlignMol(mol, mols[i-1], atomMap=list(zip(match2, match1)))

        # Draw molecule
        img = Draw.MolToImage(mol, size=(img_size, img_size))
    
        # Add index text
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), str(i), font=ImageFont.load_default(), fill=(0, 0, 0))

        images.append(img)
        
        #img = Draw.MolToImage(mol, size=(300, 300))
        #images.append(img)

    # Save as animated GIF using PIL
    images[0].save(
        f"{outfile}.gif",
        save_all=True,
        append_images=images[1:],
        duration=int(1000/fps),  # milliseconds per frame
        loop=0
    )
    print(f"Movie saved to {outfile}.gif")

def mol_similarity(
    mol1, 
    mol2
):
    # Generate Morgan fingerprints (ECFP4 radius=2)
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius=2, nBits=2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius=2, nBits=2048)

    return DataStructs.TanimotoSimilarity(fp1, fp2)

def trajectory_quantification(
    mol_array
):
# For each trajectory, want to calculate 3 sets of quantities
# a) graph quantities: # of atoms, # of bonds, topological complexity 
# https://www.rdkit.org/docs/source/rdkit.Chem.GraphDescriptors.html
# b) Lipinski quantities: Hbond donor, Hbond acceptor, molecular mass, ClogP
# c) Fingerprint similiariy wrt starting point
# d) SA score
    rows = []

    for mol in mol_array:
        rows.append({
            # a) Graph quantities
            "NumAtoms": mol.GetNumAtoms(),
            "NumBonds": mol.GetNumBonds(),
            "BertzCT": Chem.GraphDescriptors.BertzCT(mol),
            # b) Lipinski quantities
            "HBA": Chem.Lipinski.NumHAcceptors(mol),
            "HBD": Chem.Lipinski.NumHDonors(mol),
            "MolWt": Chem.Descriptors.MolWt(mol),
            "ClogP": Chem.Descriptors.MolLogP(mol),
            # c) Fingerprint similarity w.r.t. starting point
            "FingerprintSim": mol_similarity(mol, mol_array[0]),
            # d) SA score
            "SA_Score": sascorer.calculateScore(mol),
            # Extra
            "SMILES": Chem.MolToSmiles(mol)
        })
    
    # Create DataFrame with explicit column order
    df = pd.DataFrame(rows, columns=[
        "NumAtoms", "NumBonds", "BertzCT",
        "HBA", "HBD", "MolWt", "ClogP",
        "FingerprintSim", "SA_Score", "SMILES"
    ])

    return df    

def plot_quantification_traj(
    df,
    show_initial_value
):
    fig, axs = plt.subplots(2, 3, figsize=(12, 6))  # 1 row, 3 columns
    font = 14
    
    axs[0,0].plot(df['BertzCT'])
    axs[0,0].tick_params(axis='both', labelsize=font)
    axs[0,0].set_xlabel('Perturbations',size=font);
    axs[0,0].legend(['Topological Index'],fontsize=font-4);

    axs[0,1].plot(df['HBA'])
    axs[0,1].plot(df['HBD'])
    axs[0,1].tick_params(axis='both', labelsize=font)
    axs[0,1].legend(['# HBA','# HBD'],fontsize=font-4);
    axs[0,1].set_xlabel('Perturbations',size=font);

    axs[0,2].plot(df['MolWt'])
    axs[0,2].tick_params(axis='both', labelsize=font)
    axs[0,2].set_xlabel('Perturbations',size=font);
    axs[0,2].legend(['Molecular Weight'],fontsize=font-4);

    axs[1,0].plot(df['ClogP'])
    axs[1,0].tick_params(axis='both', labelsize=font)
    axs[1,0].set_xlabel('Perturbations',size=font);
    axs[1,0].legend(['Partition Coefficient'],fontsize=font-4);

    axs[1,1].plot(df['FingerprintSim'])
    axs[1,1].tick_params(axis='both', labelsize=font)
    axs[1,1].set_xlabel('Perturbations',size=font);
    axs[1,1].legend(['Fingerprint Similarity\nwrt Starting Point'],fontsize=font-4);

    axs[1,2].plot(df['SA_Score'])
    axs[1,2].tick_params(axis='both', labelsize=font)
    axs[1,2].set_xlabel('Perturbations',size=font);
    axs[1,2].legend(['SA Score'],fontsize=font-4);

    if show_initial_value:
        axs[0,0].axhline(df.iloc[0]['BertzCT'], color='red', linestyle=':')
        axs[0,1].axhline(df.iloc[0]['HBA'], color='tab:blue', linestyle=':')
        axs[0,1].axhline(df.iloc[0]['HBD'], color='tab:orange', linestyle=':')
        axs[0,2].axhline(df.iloc[0]['MolWt'], color='red', linestyle=':')
        axs[1,0].axhline(df.iloc[0]['ClogP'], color='red', linestyle=':')
        axs[1,1].axhline(df.iloc[0]['FingerprintSim'], color='red', linestyle=':')
        axs[1,2].axhline(df.iloc[0]['SA_Score'], color='red', linestyle=':')
    
    plt.tight_layout()
    plt.show()

def plot_quantification_hist(
    df,
    show_initial_value
):
    fig, axs = plt.subplots(2, 3, figsize=(12, 6))  # 1 row, 3 columns
    font = 14
    
    axs[0,0].hist(df['BertzCT'],bins=20,density=True)
    axs[0,0].tick_params(axis='both', labelsize=font)
    axs[0,0].legend(['Topological Index'],fontsize=font-4);

    axs[0,1].hist(df['HBA'],bins=20,density=True)
    axs[0,1].hist(df['HBD'],bins=20,density=True)
    axs[0,1].tick_params(axis='both', labelsize=font)
    axs[0,1].legend(['# HBA','# HBD'],fontsize=font-4);

    axs[0,2].hist(df['MolWt'],bins=20,density=True)
    axs[0,2].tick_params(axis='both', labelsize=font)
    axs[0,2].legend(['Molecular Weight'],fontsize=font-4);

    axs[1,0].hist(df['ClogP'],bins=20,density=True)
    axs[1,0].tick_params(axis='both', labelsize=font)
    axs[1,0].legend(['Partition Coefficient'],fontsize=font-4);

    axs[1,1].hist(df['FingerprintSim'],bins=20,density=True)
    axs[1,1].tick_params(axis='both', labelsize=font)
    axs[1,1].legend(['Fingerprint Similarity\nwrt Starting Point'],fontsize=font-4);

    axs[1,2].hist(df['SA_Score'],bins=20,density=True)
    axs[1,2].tick_params(axis='both', labelsize=font)
    axs[1,2].legend(['SA Score'],fontsize=font-4);
    
    if show_initial_value:
        axs[0,0].axvline(df.iloc[0]['BertzCT'], color='red', linestyle=':')
        axs[0,1].axvline(df.iloc[0]['HBA'], color='tab:blue', linestyle=':')
        axs[0,1].axvline(df.iloc[0]['HBD'], color='tab:orange', linestyle=':')
        axs[0,2].axvline(df.iloc[0]['MolWt'], color='red', linestyle=':')
        axs[1,0].axvline(df.iloc[0]['ClogP'], color='red', linestyle=':')
        axs[1,1].axvline(df.iloc[0]['FingerprintSim'], color='red', linestyle=':')
        axs[1,2].axvline(df.iloc[0]['SA_Score'], color='red', linestyle=':')
    
    plt.tight_layout()
    plt.show()

def tanimoto_distance_matrix(fp_list):
    """Calculate distance matrix for fingerprint list"""
    dissimilarity_matrix = []
    # Notice how we are deliberately skipping the first and last items in the list
    # because we don't need to compare them against themselves
    for i in range(1, len(fp_list)):
        # Compare the current fingerprint against all the previous ones in the list
        similarities = DataStructs.BulkTanimotoSimilarity(fp_list[i], fp_list[:i])
        # Since we need a distance matrix, calculate 1-x for every element in similarity matrix
        dissimilarity_matrix.extend([1 - x for x in similarities])
    return dissimilarity_matrix

def cluster_fingerprints(fingerprints, cutoff=0.2):
    """Cluster fingerprints
    Parameters:
        fingerprints
        cutoff: threshold for the clustering
    """
    # Calculate Tanimoto distance matrix
    distance_matrix = tanimoto_distance_matrix(fingerprints)
    # Now cluster the data with the implemented Butina algorithm:
    clusters = Butina.ClusterData(distance_matrix, len(fingerprints), cutoff, isDistData=True)
    clusters = sorted(clusters, key=len, reverse=True)
    return clusters

def get_cluster_medoids(
    fingerprints, 
    clusters,
    population_threshold # Only save info for cluster with certain amount of elements innit
):
    medoids = []
    for cluster in clusters:
        if len(cluster) < population_threshold:
            continue
        else:
            # Compute average similarity to all others in the cluster
            avg_sims = []
            for idx in cluster:
                sims = DataStructs.BulkTanimotoSimilarity(fingerprints[idx], [fingerprints[i] for i in cluster if i != idx])
                avg_sims.append(np.mean(sims))
            avg_sims = np.array(avg_sims)
    
            # Select the index with the highest average similarity
            medoid_idx = cluster[np.argmax(avg_sims)]
            medoids.append(medoid_idx)
    return np.array(medoids)

def cluster_trajectory(
    mol_array
):
    # https://projects.volkamerlab.org/teachopencadd/talktorials/T005_compound_clustering.html
    # use case would be: 
    # traj_concat = np.concatenate(trajectories)           // concatenate trajectories which most likely will be in replicates
    # clusters, medoids = cluster_trajectory(traj_concat)  // get cluster and medoid info (in index of traj_concat) 
    # then traj_concat[medoids] would be the representative structures that will be docked (or sth) 
    
    # Create fingerprints for all molecules
    rdkit_gen = rdFingerprintGenerator.GetRDKitFPGenerator(maxPath=5)
    fingerprints = [rdkit_gen.GetFingerprint(mol) for mol in mol_array]

    # Cluster based on fingerprints
    clusters = cluster_fingerprints(fingerprints)
    medoids = get_cluster_medoids(fingerprints, clusters)

    return clusters, medoids

def prepare_ligand(
    lig_name,
    lig_smiles,
    prepare_ligand_script_path
):
    try:
        subprocess.run(
            ["sh", prepare_ligand_script_path, lig_name, lig_smiles],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Skipping {lig_name} due to error: {e}")

def prepare_receptor(
    receptor_name,
    receptor_pdb_path,
    prepare_receptor_script_path
):
    subprocess.run(
    ["sh", prepare_receptor_script_path, receptor_name, receptor_pdb_path],
    check=True
    );

def vina_setup_receptor(
    receptor_pdbqt,
    grid_center,
):
# https://github.com/ccsb-scripps/AutoDock-Vina/tree/develop
# https://github.com/janash/iqb-2024/blob/main/docking_single_ligand.ipynb
# https://autodock-vina.readthedocs.io/en/latest/docking_python.html
    v = Vina(sf_name='vina')
    v.set_receptor(rigid_pdbqt_filename=receptor_pdbqt)
    v.compute_vina_maps(center=grid_center, box_size=[20, 20, 20])
    return v 

def vina_dock_ligand(
    v, # vina model
    ligand_pdbqt,
    output_folder,
    output_name,
    exhaustiveness=100
):
# https://github.com/ccsb-scripps/AutoDock-Vina/tree/develop
# https://github.com/janash/iqb-2024/blob/main/docking_single_ligand.ipynb
# https://autodock-vina.readthedocs.io/en/latest/docking_python.html
    v.set_ligand_from_file(ligand_pdbqt)    
    # Dock the ligand
    v.dock(exhaustiveness=exhaustiveness)
    v.optimize()
    v.write_poses(output_folder+output_name, n_poses=5, overwrite=True)
    return v
