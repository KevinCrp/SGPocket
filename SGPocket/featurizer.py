import os.path as osp
import subprocess
from typing import Dict, List, Tuple, Union

import Bio
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from biopandas.pdb import PandasPdb

from SGPocket.utilities.geometry import geometric_center
from SGPocket.utilities.pdb_tools import keep_only_CA

BACKBONE = ['CA', 'C', 'O', 'N']

AA_DICT = {
    'GLY': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'ALA': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'VAL': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'LEU': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'ILE': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'MET': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'PHE': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'TRP': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'PRO': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'SER': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'THR': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'CYS': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    'TYR': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    'ASN': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    'GLN': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    'ASP': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    'GLU': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    'LYS': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    'ARG': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    'HIS': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
}


def get_voronota_edges(voronota_path: str,
                       pdb_directory_path: str,
                       protein_pbd_file_name: str) -> List[List[int]]:
    """Compute edge index using the Voronota binary
    https://gitlab.inria.fr/GruLab/s-gcn/-/blob/master/src/common/graph.py

    Args:
        voronota_path (str): Path to the Voronota binary 
        pdb_directory_path (str): Path to the directory of the PDB file
        protein_pbd_file_name (str): PDB file name (without extension)

    Returns:
        List[List[int]]: Returns the edge index
    """
    pdb_file_path = osp.join(pdb_directory_path, protein_pbd_file_name+'.pdb')
    pdb_CA_file_path = osp.join(
        pdb_directory_path, protein_pbd_file_name+'_CA.pdb')
    balls_file_path = osp.join(pdb_directory_path, 'balls.tmp')
    contacts_file_path = osp.join(pdb_directory_path, 'contacts.tmp')
    if not osp.isfile(pdb_CA_file_path):
        keep_only_CA(pdb_file_path,
                     pdb_CA_file_path)
    with open(pdb_CA_file_path) as f:
        lines = f.read()

    if not osp.isfile(balls_file_path):
        cmd = 'cat {} | {} get-balls-from-atoms-file > {}'.format(
            pdb_CA_file_path,
            voronota_path,
            balls_file_path)
        subprocess.call(cmd, stdout=subprocess.PIPE, shell=True)

    if not osp.isfile(contacts_file_path):
        cmd = 'cat {} | {} calculate-contacts > {}'.format(
            balls_file_path,
            voronota_path,
            contacts_file_path)
        subprocess.call(cmd, stdout=subprocess.PIPE, shell=True)

    edge_index = [[], []]
    with open(contacts_file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            tab_line = line.split()
            if tab_line[0] != tab_line[1]:
                edge_index[0] += [int(tab_line[0])]
                edge_index[1] += [int(tab_line[1])]
    return edge_index


def get_sperical_harmonics(protein_pdb_filepath: str,
                           sph_order: Union[str, int],
                           edge_index: List[List[int]],
                           sh_featurizer_linux_path: str) -> List[List[float]]:
    """Compute the spherical harmonics for each edge
    # Extraction de spherical harmonics from:
    # https://gitlab.inria.fr/GruLab/s-gcn/-/blob/master/src/common/graph.py

    Args:
        protein_pdb_filepath (str): Path to the protein PDB file
        sph_order (Union[str, int]): The order of computed spherical harmonics
        edge_index (List[List[int]]): The edge index
        sh_featurizer_linux_path (str): Path to the sh_featurizer_linux

    Returns:
        List[List[float]]: Returns for each edge the corresponding spherical harmonics
    """
    sph_order = str(sph_order)
    cmd = [sh_featurizer_linux_path, '-i', protein_pdb_filepath, '-p',
           sph_order]
    result = subprocess.run(cmd, stdout=subprocess.PIPE)
    sh_output = result.stdout.decode('utf-8')
    sh_lines = sh_output.split('\n')
    residues_num = int(sh_lines[0].split()[3])
    lines = sh_lines[1:]
    sph_harms = {}
    i = 0
    for res1 in range(residues_num):
        for res2 in range(residues_num):
            if res1 != res2:
                if res1 not in sph_harms.keys():
                    sph_harms[res1] = {}
                sph_harms[res1][res2] = lines[i].split()
                i += 1
    sh = []
    nb_edges = len(edge_index[0])
    for i in range(nb_edges):
        source = edge_index[0][i]
        dest = edge_index[1][i]
        # -----------------------
        # The features list contains the spherical harmonics weights for each
        # residue_1 (which is used as based) to each residue_2
        # However, in the built graph the weight residue_1 to residue_2
        # must be used as edge feature of the edge from residue_2 to residue_1
        # -----------------------
        sh_features = list(map(float, sph_harms[dest][source]))  # order^2
        sh.append(sh_features)
    return sh


def get_residues_id(pdb_paths: List[str]) -> Dict:
    """From a list of PDB, extracts all residues ID

    Args:
        pdb_paths (List[str]): List of PDB file paths

    Returns:
        Dict: A dictionnary containing all residues ID from PDB files
    """
    p = PDBParser(QUIET=1)
    dict_chain_res_ids = {}
    for pdb_path in pdb_paths:
        struct = p.get_structure("YOYO", pdb_path)
        for res in struct.get_residues():
            res_chain = res.full_id[2]
            res_id = int(res.full_id[3][1])
            res_insert = res.full_id[3][2]

            if res_chain not in dict_chain_res_ids.keys():
                dict_chain_res_ids[res_chain] = []
            dict_chain_res_ids[res_chain] += [(res_id, res_insert)]
    return dict_chain_res_ids


def get_residues_properties(struct: Bio.PDB.Structure.Structure,
                            site_res_dict: Dict) -> Tuple[pd.DataFrame, Dict]:
    """Computes residues features

    Args:
        struct (Bio.PDB.Structure.Structure): BioPython protein structure
        site_res_dict (Dict): Dictionnary indicating for each residue if it belongs
            to a binding site or not

    Returns:
        Tuple[pd.DataFrame, Dict]: Residues/nodes features, a dictionnary for
            graph nodes correspondance with residues
    """
    features = []
    coords = []
    dict_chain_res_ids = {}
    for chain in struct.get_chains():
        dict_chain_res_ids[chain.get_full_id()[2]] = {}
    res_graph_id = 0
    for res in struct.get_residues():
        res_chain = res.full_id[2]
        res_id = int(res.full_id[3][1])
        res_insert = res.full_id[3][2]
        dict_chain_res_ids[res_chain][(res_id, res_insert)] = res_graph_id

        residue_aa = AA_DICT[res.get_resname()]

        res_seq_id = res.get_id()[1]
        is_pocket = 0.0
        if site_res_dict is not None:
            if res_chain in site_res_dict.keys():
                if (res_id, res_insert) in site_res_dict[res_chain]:
                    is_pocket = 1.0
        features += [residue_aa + [res_seq_id] +
                     [is_pocket, res_chain, res_id, res_insert, res_graph_id]]
        coords += [res['CA'].coord[0],
                   res['CA'].coord[1], res['CA'].coord[2]]
        res_graph_id += 1
    fts_df = pd.DataFrame(data=features, columns=['GLY', 'ALA', 'VAL', 'LEU', 'ILE',
                                                  'MET', 'PHE', 'TRP',
                                                  'PRO', 'SER', 'THR', 'CYS',
                                                  'TYR', 'ASN', 'GLN',
                                                  'ASP', 'GLU', 'LYS', 'ARG', 'HIS',
                                                  'res_seq_id',
                                                  'is_pocket', 'res_chain',
                                                  'res_id', 'res_insert',
                                                  'res_graph_id'])

    coords = np.array(coords).reshape(-1, 3)

    fts_df = fts_df.assign(res_x=coords[:, 0],
                           res_y=coords[:, 1],
                           res_z=coords[:, 2])
    return fts_df, dict_chain_res_ids


def compute_edges(voronota_path: str,
                  pdb_directory_path: str,
                  protein_pbd_file_name: str,
                  sh_featurizer_linux_path: str,
                  sph_order: Union[str, int]) -> Tuple[List[List[int]], List[List[float]]]:
    """Computes edge index and edge attributes/features

    Args:
        voronota_path (str): Path to the Voronota bin
        pdb_directory_path (str): Directory of PDB
        protein_pbd_file_name (str): PDB filename (without extension)
        sh_featurizer_linux_path (str): Path to the SH Featurizer bin
        sph_order (Union[str, int]): Spherical Harmonics order

    Returns:
        Tuple[List[List[int]], List[List[float]]]: Edge index, edge attributes (spherical harmonics)
    """
    edge_index = get_voronota_edges(voronota_path,
                                    pdb_directory_path,
                                    protein_pbd_file_name)
    protein_path = osp.join(pdb_directory_path, protein_pbd_file_name+'.pdb')
    edge_attr = get_sperical_harmonics(protein_path,
                                       sph_order,
                                       edge_index,
                                       sh_featurizer_linux_path)
    return edge_index, edge_attr


def get_centers(site_pdb_paths: List[str]) -> List[List[float]]:
    """Compute sites centers from a list of PDB files

    Args:
        site_pdb_paths (List[str]): List of all sites PDB

    Returns:
        List[List[float]]: Sites centers
    """
    centers = []
    for site_path in site_pdb_paths:
        ppdb = PandasPdb()
        ppdb.read_pdb(site_path)
        df_CA = ppdb.df['ATOM'][ppdb.df['ATOM']['atom_name'] == 'CA']
        coords = np.array(
            df_CA[['x_coord', 'y_coord', 'z_coord']].values.tolist())
        center = geometric_center(coords)
        centers += [center]
    return np.array(centers)


def featurize(pdb_directory: str,
              protein_name: str,
              pocket_name: str,
              sph_order: Union[str, int],
              voronota_path: str,
              sh_featurizer_linux_path: str) -> Tuple[pd.DataFrame,
                                                      np.array,
                                                      List,
                                                      List]:
    """Featurize a protein and returns a dataframe containing all features

    Args:
        pdb_directory (str): Directory containing the PDB
        protein_name (str): The PDB protein name (without extension)
        pocket_name (str): The PDB pocket name (without extension)
        sph_order (Union[str, int]): The order of the spherical harmonics
        voronota_path (str): Path to the Voronota binary
        sh_featurizer_linux_path (str): Path to the sh_featurizer

    Returns:
        Tuple[pd.DataFrame, np.array, List, List]: A DataFrame with all computed features,
            the pocket center, the edge_index, the edge features/attributes
    """
    protein_path_pdb = osp.join(pdb_directory, protein_name+'.pdb')
    p = PDBParser(QUIET=1)
    struct = p.get_structure("YOYO", protein_path_pdb)

    sites_center = None
    site_res_dict = None
    if pocket_name is not None:
        pocket_path_pdb = osp.join(pdb_directory, pocket_name+'.pdb')
        site_res_dict = get_residues_id([pocket_path_pdb])
        sites_center = get_centers([pocket_path_pdb])
    fts_df, _ = get_residues_properties(
        struct=struct, site_res_dict=site_res_dict)
    edge_index, edge_attr = compute_edges(voronota_path=voronota_path,
                                          pdb_directory_path=pdb_directory,
                                          protein_pbd_file_name=protein_name,
                                          sh_featurizer_linux_path=sh_featurizer_linux_path,
                                          sph_order=sph_order)
    return fts_df, sites_center, edge_index, edge_attr
