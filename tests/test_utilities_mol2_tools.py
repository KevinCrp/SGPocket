import os

from biopandas.mol2 import PandasMol2

from SGPocket.utilities.mol2_tools import clean_mol2

PATH_TO_LIGAND_MOL2 = "tests/data/raw/1a1e/1a1e_ligand.mol2"
PATH_TO_LIGAND_CLEAN_MOL2 = "tests/data/for_test_1a1e_ligand_clean.mol2"


def test_clean_mol2():
    clean_mol2(PATH_TO_LIGAND_MOL2, PATH_TO_LIGAND_CLEAN_MOL2)
    pmol = PandasMol2().read_mol2(PATH_TO_LIGAND_CLEAN_MOL2)
    assert (pmol.df.shape[0] == 73)
    os.remove(PATH_TO_LIGAND_CLEAN_MOL2)
