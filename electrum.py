import numpy as np
import pandas as pd
import hashlib

from rdkit import Chem
from rdkit.Chem import rdmolops

metals_valence = pd.read_csv('metals_dict.csv', sep=',', header=None)
metals_dict = {}
for index, row in metals_valence.iterrows():
    key = row[0]
    value = np.array(row[1:])
    metals_dict[key] = value

def get_atom_env(mol, radius:int, atom:int) -> str:
    """
    Extracts the local chemical environment around a specified atom within a molecular structure.

    Parameters:
    ----------
    mol (RDKit Mol): The molecular structure from which the atom environment will be extracted.
    radius: The radius of the desired atom environment. Defines the number of bonds away from the central atom to be considered in the local environment.
    atom: The index of the target atom in the molecular structure. The function will extract the environment around this specific atom.

    Returns:
    ----------
    smiles: The SMILES representation of the identified substructure.

    Example Usage:
    ----------
    ```python
    mol = Chem.MolFromSmiles('C1CC(=O)NC(=O)[C@@H]1N2C(=O)C3=CC=CC=C3C2=O')
    substructure = get_atom_env(mol, radius=2, atom_index=1)
    print(substructure)
    ```
    """
    env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom)
    atom_map = {}
    mol = Chem.PathToSubmol(mol, env, atomMap=atom_map)
    smiles = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
    return smiles

def get_mol_substructs(mol, radius:int) -> list:
    """
    Generates a list of unique substructures within a molecular structure, considering atoms within a specified radius.

    Parameters:
    ----------
    mol (RDKit Mol): The molecular structure from which the substructures will be extracted.
    radius (int): The maximum radius to consider for the extraction of substructures. Defines the number of bonds away from each atom to be considered in the local environment.

    Returns:
    ----------
    list: A list of unique SMILES representations of the identified substructures.

    Example Usage:
    ----------
    ```python
    mol = Chem.MolFromSmiles('C1CC(=O)NC(=O)[C@@H]1N2C(=O)C3=CC=CC=C3C2=O')
    substructures = get_mol_substructs(mol, radius=2)
    print(substructures)
    ```
    """
    substructs = []
    for r in range(1, radius+1):
        for a in range(mol.GetNumAtoms()):
            substructs.append(get_atom_env(mol, r, a))
    return list(set(substructs))

def hash_and_fold(mol, radius:int, n_bits:int, return_mapping=False) -> np.ndarray or tuple:
    """
    Generates a folded fingerprint for a molecular structure using hashing of substructures.

    Parameters:
    ----------
    mol (RDKit Mol): The molecular structure for which the fingerprint will be generated.
    radius (int): The maximum radius to consider for the extraction of substructures. Defines the number of bonds away from each atom to be considered in the local environment.
    n_bits (int): The number of bits in the fingerprint.
    return_mapping (bool, optional): Whether to return a mapping of hash values to substructures. Defaults to False.

    Returns:
    ----------
    np.ndarray or tuple: If `return_mapping` is False, returns a binary fingerprint represented as a numpy array. If `return_mapping` is True, returns a tuple containing the fingerprint and a dictionary mapping hash values to substructures.

    Example Usage:
    ----------
    ```python
    mol = Chem.MolFromSmiles('C1CC(=O)NC(=O)[C@@H]1N2C(=O)C3=CC=CC=C3C2=O')
    fingerprint = hash_and_fold(mol, radius=2, n_bits=1024)
    print(fingerprint)
    ```
    """
    substructs = get_mol_substructs(mol, radius)
    unique_hashes = set()
    hash_to_position = {}

    for s in substructs:
        hash_val = int(hashlib.sha1(s.encode('utf-8')).hexdigest(), 16) % n_bits
        unique_hashes.add(hash_val)
        hash_to_position[hash_val] = s

    fp = np.zeros(n_bits, dtype=int)
    fp[list(unique_hashes)] = 1

    if return_mapping:
        return fp, hash_to_position
    else:
        return fp

def process_smiles(smiles:str) -> list:
    """
    Processes a string containing multiple SMILES strings separated by '.' and returns a list of RDKit Mol objects.

    Parameters:
    ----------
    smiles (str): A string containing one or more SMILES strings separated by '.'.

    Returns:
    ----------
    list: A list of RDKit Mol objects corresponding to the input SMILES strings.

    Example Usage:
    ----------
    ```python
    smiles = 'CCO.CN.CC'
    mol_list = process_smiles(smiles)
    for mol in mol_list:
        print(Chem.MolToSmiles(mol))
    ```
    """
    smiles_list = smiles.split('.')
    return [Chem.MolFromSmiles(s, sanitize=False) for s in smiles_list]

def calculate_fingerprint(smiles:str, metal:str, radius:int, n_bits:int, return_mapping=False) -> np.ndarray or tuple:
    """
    Calculates a combined fingerprint for a set of SMILES strings with a specified metal ion.

    Parameters:
    ----------
    smiles (str): A string containing one or more SMILES strings separated by '.'.
    metal (str): The SMILES representation of the metal ion to be included in the fingerprint.
    radius (int): The maximum radius to consider for the extraction of substructures. Defines the number of bonds away from each atom to be considered in the local environment.
    n_bits (int): The number of bits in the fingerprint.
    return_mapping (bool, optional): Whether to return a mapping of hash values to substructures. Defaults to False.

    Returns:
    ----------
    np.ndarray or tuple: If `return_mapping` is False, returns a combined binary fingerprint represented as a numpy array. If `return_mapping` is True, returns a tuple containing the fingerprint and a dictionary mapping hash values to substructures.

    Example Usage:
    ----------
    ```python
    smiles = 'CCO.CN.CC'
    metal = '[Fe]'
    fingerprint = calculate_fingerprint(smiles, metal, radius=2, n_bits=1024)
    print(fingerprint)
    ```
    """
    mol_list = process_smiles(smiles)
    fp_list = [hash_and_fold(m, radius, n_bits, return_mapping=return_mapping) for m in mol_list]

    if return_mapping:
        combined_mapping = {}

        for idx, (_, substructure_mapping) in enumerate(fp_list):
            for substruct_idx, substruct_smiles in substructure_mapping.items():
                combined_mapping[substruct_idx + idx * n_bits] = substruct_smiles

        fingerprint = np.append(np.sum([fps for fps, _ in fp_list], axis=0), metals_dict[metal])
        adjusted_mapping = {pos % n_bits: smiles for pos, smiles in combined_mapping.items()}

        return fingerprint, adjusted_mapping
    else:
        fingerprint = np.append(np.sum(fp_list, axis=0), metals_dict[metal])
        return fingerprint

def calculate_fingerprints(smiles_list:list, metals_list:list, radius:int, n_bits:int, return_mapping=False) -> np.ndarray or tuple:
    """
    Calculates fingerprints for a list of SMILES strings with corresponding metal ions.

    Parameters:
    ----------
    smiles_list (list): A list of strings, each containing one or more SMILES strings separated by '.'.
    metals_list (list): A list of strings, each containing the SMILES representation of the metal ion corresponding to the SMILES string in `smiles_list`.
    radius (int): The maximum radius to consider for the extraction of substructures. Defines the number of bonds away from each atom to be considered in the local environment.
    n_bits (int): The number of bits in each fingerprint.
    return_mapping (bool, optional): Whether to return a mapping of hash values to substructures. Defaults to False.

    Returns:
    ----------
    np.ndarray or tuple: If `return_mapping` is False, returns a list of combined binary fingerprints represented as numpy arrays. If `return_mapping` is True, returns a tuple containing the list of fingerprints and a dictionary mapping hash values to substructures.

    Example Usage:
    ----------
    ```python
    smiles_list = ['CCO.CN.CC', 'CCC.O=O.CN']
    metals_list = ['[Fe]', '[Cu]']
    fingerprints = calculate_fingerprints(smiles_list, metals_list, radius=2, n_bits=1024)
    print(fingerprints)
    ```
    """
    fingerprints = []
    combined_mapping = {}

    for smiles, metal in zip(smiles_list, metals_list):
        fingerprint, substruct_mapping = calculate_fingerprint(smiles, metal, radius, n_bits, return_mapping=True)

        if return_mapping:
            combined_mapping.update(substruct_mapping)

        fingerprints.append(fingerprint)

    if return_mapping:
        adjusted_mapping = {pos % n_bits: smiles for pos, smiles in combined_mapping.items()}
        return fingerprints, adjusted_mapping
    else:
        return fingerprints