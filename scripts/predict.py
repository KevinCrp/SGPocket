import argparse
import configparser
import os.path as osp
from typing import List, Union

import torch_geometric as pyg

from SGPocket.data import to_graph
from SGPocket.featurizer import featurize
from SGPocket.model import Model
from SGPocket.utilities.pdb_tools import clean_pdb
from SGPocket.utilities.pocket_segmentation import from_cluster_to_pdb


def predict(model: Model,
            protein_path_pdb: str,
            sph_order: Union[str, int],
            voronota_path: str,
            sh_featurizer_linux_path: str,
            threshold: float,
            dbscan_eps: float) -> List[str]:
    """Predict pockets using a SGPocket Model

    Args:
        model (Model): A SGPocket model
        protein_path_pdb (str): Path to the protein PDB
        sph_order (Union[str, int]): Spherical harmonic order
        voronota_path (str): Path to the Voronota binary file
        sh_featurizer_linux_path (str): Path to the SH featurizer binary file
        threshold (float): Threshold to classify nodes
        dbscan_eps (float): DBScan epsilon

    Returns:
        List[str]: A list of path to extracted pockets
    """
    protein_name = 'SGPocket_clean'
    pdb_dir, _ = osp.split(protein_path_pdb)
    protein_path_pdb_clean = osp.join(pdb_dir, protein_name+'.pdb')
    clean_pdb(protein_path_pdb, protein_path_pdb_clean)

    fts_df, _, edge_index, edge_attr = featurize(
        pdb_dir, protein_name, None,
        sph_order, voronota_path, sh_featurizer_linux_path)

    g = to_graph(fts_df, edge_index, edge_attr, None, True)

    batch = pyg.data.Batch.from_data_list([g])

    segmented_labels, dict_lab_freq = model.predict_and_extract(
        batch, threshold, dbscan_eps)

    pockets_files = from_cluster_to_pdb(segmented_labels, dict_lab_freq,
                                        pdb_dir+'/'+protein_name+'.pdb', g)

    return pockets_files


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-config",
                        help="Path to config file, default = config.ini",
                        type=str,
                        default='config.ini')
    parser.add_argument("-model", "-m",
                        help="Path to the model",
                        type=str,
                        required=True)
    parser.add_argument("-pdb",
                        help="Path to the protein PDB",
                        type=str,
                        required=True)
    parser.add_argument("-threshold", "-th",
                        help="Threshold used to class each node",
                        type=float,
                        required=True)
    parser.add_argument("-dbscan_eps", "-eps",
                        help="The DBScan EPS",
                        type=float,
                        required=True)
    args = parser.parse_args()
    config_file_path = args.config
    model_ckpt = args.model
    protein_path_pdb = args.pdb
    threshold = args.threshold
    dbscan_eps = args.dbscan_eps

    config = configparser.ConfigParser()
    config.read(config_file_path)
    voronota_path = config['PATHS']['voronota']
    sh_featurizer_linux_path = config['PATHS']['sh_featurizer']
    sph_order = config['GRAPHS']['spherical_harmonics_order']

    model = Model.load_from_checkpoint(model_ckpt, map_location='cpu')

    pockets_files = predict(model,
                            protein_path_pdb,
                            sph_order,
                            voronota_path,
                            sh_featurizer_linux_path,
                            threshold,
                            dbscan_eps)

    print("{} pocket(s) found".format(len(pockets_files)))
