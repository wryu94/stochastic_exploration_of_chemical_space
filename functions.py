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

# Suppress RDKit warnings and errors
RDLogger.DisableLog('rdApp.*')

def mol_perturbation_save_molecule_w_constraint(
    smiles,            # smiles of the starting structure
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
    molecule = Chem.MolFromSmiles(smiles, sanitize=True)
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

def mol_movie(mols, outfile, fps=1):
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
        
        img = Draw.MolToImage(mol, size=(300, 300))
        images.append(img)

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

def plot_quantification(
    df
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
    
    plt.tight_layout()
    plt.show()

def plot_quantification_hist(
    df
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
    
    plt.tight_layout()
    plt.show()