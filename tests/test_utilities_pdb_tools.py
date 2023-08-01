import os

import numpy as np
from biopandas.pdb import PandasPdb

from SGPocket.utilities.pdb_tools import (clean_pdb, extract_pocket,
                                          keep_only_CA, res_close_to_ligand)

PATH_TO_UNCLEAN_PDB = "tests/data/raw/1a1e/1a1e_protein.pdb"
PATH_TO_LIGAND_MOL2 = "tests/data/raw/1a1e/1a1e_ligand.mol2"
PATH_TO_CLEAN_PDB = "tests/data/for_test_1a1e_protein_clean.pdb"
PATH_TO_CLEAN_CA_PDB = "tests/data/for_test_1a1e_protein_clean_ca.pdb"
PATH_TO_CLEAN_POCKET_PDB = "tests/data/for_test_1a1e_pocket.pdb"


def test_clean_pdb():
    ppdb = PandasPdb()
    ppdb.read_pdb(PATH_TO_UNCLEAN_PDB)
    nb_atom_unclean = ppdb.df['ATOM'].shape[0]
    clean_pdb(PATH_TO_UNCLEAN_PDB, PATH_TO_CLEAN_PDB)
    ppdb.read_pdb(PATH_TO_CLEAN_PDB)
    nb_atom_clean = ppdb.df['ATOM'].shape[0]
    assert (nb_atom_unclean == 3235)
    assert (nb_atom_clean == 1639)


def test_keep_only_CA():
    keep_only_CA(PATH_TO_CLEAN_PDB, PATH_TO_CLEAN_CA_PDB)
    ppdb = PandasPdb()
    ppdb.read_pdb(PATH_TO_CLEAN_CA_PDB)
    nb_atom_clean_ca = ppdb.df['ATOM'].shape[0]
    assert (nb_atom_clean_ca == 208)
    os.remove(PATH_TO_CLEAN_CA_PDB)


def test_res_close_to_ligand():
    ligand_coords = np.array([[1.0, 2.0, 3.0],
                              [3.0, 1.0, 2.0]])
    residue_coords_true = np.array([[1.0, 2.5, 3.0],
                                    [5.0, 8.0, 8.0]])
    residue_coords_false = np.array([[1.0, 12.5, 3.0],
                                     [5.0, 8.0, 8.0]])
    assert (res_close_to_ligand(ligand_coords,
                                residue_coords_true, 4.0) == True)
    assert (res_close_to_ligand(ligand_coords,
                                residue_coords_false, 4.0) == False)


def test_extract_pocket():
    extract_pocket(PATH_TO_CLEAN_PDB,
                   PATH_TO_LIGAND_MOL2,
                   PATH_TO_CLEAN_POCKET_PDB,
                   10.0)
    ppdb = PandasPdb()
    ppdb.read_pdb(PATH_TO_CLEAN_POCKET_PDB)
    nb_atom_clean_ca = ppdb.df['ATOM'].shape[0]
    assert (nb_atom_clean_ca == 391)
    os.remove(PATH_TO_CLEAN_PDB)
    os.remove(PATH_TO_CLEAN_POCKET_PDB)
