import os
import os.path as osp

from Bio.PDB import PDBParser

from SGPocket.featurizer import (compute_edges, featurize, get_centers,
                                 get_residues_id, get_residues_properties,
                                 get_sperical_harmonics, get_voronota_edges)
from SGPocket.utilities.pdb_tools import clean_pdb

PROTEIN_DIR = "tests/data"
PROTEIN_NAME = "for_test_1a1e_protein_clean"
POCKET_NAME = "for_test_1a1e_pocket_clean"

PATH_TO_UNCLEAN_PDB = "tests/data/raw/1a1e/1a1e_protein.pdb"
PATH_TO_CLEAN_PDB = "tests/data/for_test_1a1e_protein_clean.pdb"
PATH_TO_UNCLEAN_POCKET_PDB = "tests/data/raw/1a1e/1a1e_pocket.pdb"
PATH_TO_CLEAN_POCKET_PDB = "tests/data/for_test_1a1e_pocket_clean.pdb"


def test_featurize():
    clean_pdb(PATH_TO_UNCLEAN_PDB, PATH_TO_CLEAN_PDB)
    clean_pdb(PATH_TO_UNCLEAN_POCKET_PDB, PATH_TO_CLEAN_POCKET_PDB)
    fts_df, sites_center, edge_index, edge_attr = featurize(PROTEIN_DIR,
                                                            PROTEIN_NAME,
                                                            POCKET_NAME,
                                                            1,
                                                            'bin/voronota-linux',
                                                            'bin/sh-featurizer-linux')
    assert (fts_df.shape[0] == 208)
    print(sites_center)
    assert (round(sites_center[0][0], 4) == 42.1315)
    assert (round(sites_center[0][1], 4) == -0.5870)
    assert (round(sites_center[0][2], 4) == 41.7350)
    assert (len(edge_index[0]) == 545)
    assert (len(edge_attr) == 545)
    os.remove("tests/data/for_test_1a1e_protein_clean_CA.pdb")
    os.remove("tests/data/balls.tmp")
    os.remove("tests/data/contacts.tmp")
    os.remove(PATH_TO_CLEAN_PDB)
    os.remove(PATH_TO_CLEAN_POCKET_PDB)


def test_featurize_none():
    clean_pdb(PATH_TO_UNCLEAN_PDB, PATH_TO_CLEAN_PDB)
    fts_df, sites_center, edge_index, edge_attr = featurize(PROTEIN_DIR,
                                                            PROTEIN_NAME,
                                                            None,
                                                            1,
                                                            'bin/voronota-linux',
                                                            'bin/sh-featurizer-linux')
    assert (fts_df.shape[0] == 208)
    assert (sites_center is None)
    assert (len(edge_index[0]) == 545)
    assert (len(edge_attr) == 545)
    os.remove("tests/data/for_test_1a1e_protein_clean_CA.pdb")
    os.remove("tests/data/balls.tmp")
    os.remove("tests/data/contacts.tmp")
    os.remove(PATH_TO_CLEAN_PDB)


def test_get_voronota_edges():
    clean_pdb(PATH_TO_UNCLEAN_PDB, PATH_TO_CLEAN_PDB)
    edge_index = get_voronota_edges('bin/voronota-linux', PROTEIN_DIR,
                                    PROTEIN_NAME)

    assert (len(edge_index[0]) == 545)
    os.remove("tests/data/balls.tmp")
    os.remove("tests/data/contacts.tmp")
    os.remove(PATH_TO_CLEAN_PDB)


def test_get_sperical_harmonics():
    clean_pdb(PATH_TO_UNCLEAN_PDB, PATH_TO_CLEAN_PDB)
    edge_index = get_voronota_edges('bin/voronota-linux', PROTEIN_DIR,
                                    PROTEIN_NAME)
    sph = get_sperical_harmonics(
        PATH_TO_CLEAN_PDB, 2, edge_index, 'bin/sh-featurizer-linux')
    assert (len(sph) == 545)
    assert (len(sph[0]) == 4)
    os.remove("tests/data/for_test_1a1e_protein_clean_CA.pdb")
    os.remove("tests/data/balls.tmp")
    os.remove("tests/data/contacts.tmp")
    os.remove(PATH_TO_CLEAN_PDB)


def test_get_residues_id():
    clean_pdb(PATH_TO_UNCLEAN_POCKET_PDB, PATH_TO_CLEAN_POCKET_PDB)
    dict_res_id = get_residues_id([PATH_TO_CLEAN_POCKET_PDB])
    print(dict_res_id)
    assert (len(dict_res_id.keys()) == 1)
    assert (len(dict_res_id['B']) == 38)
    os.remove(PATH_TO_CLEAN_POCKET_PDB)


def test_get_residues_properties_with_site():
    clean_pdb(PATH_TO_UNCLEAN_PDB, PATH_TO_CLEAN_PDB)
    clean_pdb(PATH_TO_UNCLEAN_POCKET_PDB, PATH_TO_CLEAN_POCKET_PDB)
    protein_path_pdb = osp.join(PROTEIN_DIR, PROTEIN_NAME+'.pdb')
    p = PDBParser(QUIET=1)
    struct = p.get_structure("YOYO", protein_path_pdb)

    site_res_dict = get_residues_id([PATH_TO_CLEAN_POCKET_PDB])
    fts_df, _ = get_residues_properties(
        struct=struct, site_res_dict=site_res_dict)
    assert (fts_df.shape[0] == 208)
    os.remove(PATH_TO_CLEAN_PDB)
    os.remove(PATH_TO_CLEAN_POCKET_PDB)


def test_get_residues_properties_without_site():
    clean_pdb(PATH_TO_UNCLEAN_PDB, PATH_TO_CLEAN_PDB)
    clean_pdb(PATH_TO_UNCLEAN_POCKET_PDB, PATH_TO_CLEAN_POCKET_PDB)
    protein_path_pdb = osp.join(PROTEIN_DIR, PROTEIN_NAME+'.pdb')
    p = PDBParser(QUIET=1)
    struct = p.get_structure("YOYO", protein_path_pdb)

    site_res_dict = None
    fts_df, _ = get_residues_properties(
        struct=struct, site_res_dict=site_res_dict)
    assert (fts_df.shape[0] == 208)
    os.remove(PATH_TO_CLEAN_PDB)
    os.remove(PATH_TO_CLEAN_POCKET_PDB)


def test_compute_edges():
    clean_pdb(PATH_TO_UNCLEAN_PDB, PATH_TO_CLEAN_PDB)
    edge_index, edge_attr = compute_edges('bin/voronota-linux',
                                          PROTEIN_DIR,
                                          PROTEIN_NAME,
                                          'bin/sh-featurizer-linux',
                                          2)

    assert (len(edge_index[0]) == 545)
    assert (len(edge_attr[0]) == 4)
    os.remove("tests/data/for_test_1a1e_protein_clean_CA.pdb")
    os.remove("tests/data/balls.tmp")
    os.remove("tests/data/contacts.tmp")
    os.remove(PATH_TO_CLEAN_PDB)


def test_get_centers():
    clean_pdb(PATH_TO_UNCLEAN_POCKET_PDB, PATH_TO_CLEAN_POCKET_PDB)
    centers = get_centers([PATH_TO_CLEAN_POCKET_PDB])
    assert (round(centers[0][0], 4) == 42.1315)
    assert (round(centers[0][1], 4) == -0.587)
    assert (round(centers[0][2], 4) == 41.735)
    os.remove(PATH_TO_CLEAN_POCKET_PDB)
