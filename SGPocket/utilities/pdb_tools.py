import datetime

import numpy as np
import pandas as pd
from biopandas.mol2 import PandasMol2
from biopandas.pdb import PandasPdb


def clean_pdb(pdb_path: str,
              out_filename: str):
    """Remove HETATM in the given PDB file and remove H
       and remove atome with insertion

    Args:
        pdb_path (str): The input pdb file
        out_filename (str): Path where save the cleaned file
    """
    ppdb = PandasPdb()
    ppdb.read_pdb(pdb_path)
    df_atom_cleaned = ppdb.df['ATOM'][ppdb.df['ATOM']['element_symbol'] != 'H']

    df_atom_cleaned = df_atom_cleaned[df_atom_cleaned['insertion'] == '']
    df_atom_cleaned = df_atom_cleaned.reset_index(drop=True)
    df_atom_cleaned['atom_number'] = df_atom_cleaned.index + 1
    df_atom_cleaned['line_idx'] = df_atom_cleaned.index + 1

    ppdb.df['ATOM'] = df_atom_cleaned
    ppdb.to_pdb(path=out_filename,
                records=['ATOM'],
                gz=False,
                append_newline=True)


def keep_only_CA(pdb_path: str,
                 out_filename: str):
    """Create a new PDB file with only the C_alpha

    Args:
        pdb_path (str): Path to the input PDB
        out_filename (str): Path to the output PDB
    """
    ppdb = PandasPdb()
    ppdb.read_pdb(pdb_path)
    df_atom_cleaned = ppdb.df['ATOM'][ppdb.df['ATOM']['atom_name'] == 'CA']
    df_atom_cleaned = df_atom_cleaned.reset_index(drop=True)
    df_atom_cleaned['atom_number'] = df_atom_cleaned.index + 1
    df_atom_cleaned['line_idx'] = df_atom_cleaned.index + 1

    ppdb.df['ATOM'] = df_atom_cleaned
    ppdb.to_pdb(path=out_filename,
                records=['ATOM'],
                gz=False,
                append_newline=True)


def res_close_to_ligand(ligand_coords: np.array,
                        res_coords: np.array,
                        cutoff: float) -> bool:
    """Check if a residue is close enought (regarding a cutoff) of the ligand
        A distance between the ligand and the residue is the smallest distance
        between any pair of residue-ligand atoms (except H)

    Args:
        ligand_coords (np.array): Ligand coords
        res_coords (np.array): Residue coords
        cutoff (float): The distance cutoff

    Returns:
        bool: True if the residue is close enought of the ligand
    """
    for res_coord in res_coords:
        for lig_coord in ligand_coords:
            distance = np.linalg.norm(res_coord - lig_coord)
            if distance <= cutoff:
                return True
    return False


def extract_pocket(protein_path: str,
                   ligand_path: str,
                   pocket_out: str,
                   cutoff: float):
    """Extracts the ligand binding site

    Args:
        protein_path (str): Path to a protein PDB
        ligand_path (str): Path to a ligand MOL2
        pocket_out (str): Path to the extracted pocket/binding site
        cutoff (float): Cutoff use to select residues in binding site
    """
    now = datetime.datetime.now()
    now_str = now.strftime('%Y-%m-%d %H:%M:%S')

    pmol = PandasMol2()
    pmol.read_mol2(ligand_path)
    df_ligand = pmol.df[pmol.df['atom_type'] != 'H']

    ligand_coords = df_ligand[['x', 'y', 'z']].to_numpy()

    ppdb = PandasPdb()
    ppdb.read_pdb(protein_path)
    df_prot_grouped = ppdb.df['ATOM'].groupby(['residue_number', 'chain_id'])

    list_df_in_site = []
    for _, df in df_prot_grouped:
        res_coords = df[[
            'x_coord', 'y_coord', 'z_coord']].to_numpy()
        in_site = res_close_to_ligand(ligand_coords, res_coords, cutoff)
        if in_site:
            list_df_in_site += [df]

    df_site = pd.concat(list_df_in_site).reset_index(drop=True)
    df_site['atom_number'] = [i for i in range(df_site.shape[0])]

    ppdb.df['OTHERS'] = ppdb.df['OTHERS'][:3]
    ppdb.df['OTHERS'].loc[3] = [
        'REMARK', '    Extracted by SGPocket on {}'.format(now_str), 3]
    ppdb.df['OTHERS'].loc[4] = [
        'REMARK', '    With a cutoff of {}'.format(cutoff), 4]

    ppdb.df['ATOM'] = df_site
    ppdb.to_pdb(path=pocket_out,
                records=['OTHERS', 'ATOM'],
                append_newline=True)
