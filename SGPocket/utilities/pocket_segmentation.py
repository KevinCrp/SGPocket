import os
import os.path as osp
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch_geometric as pyg
from biopandas.pdb import PandasPdb
from pandas.core.groupby.generic import DataFrameGroupBy
from sklearn.cluster import DBSCAN

import SGPocket.data as data
from SGPocket.utilities.geometry import geometric_center


def clusterize(y_pred: torch.Tensor,
               pos: torch.Tensor,
               th: float = 0.5,
               min_freq: int = 5,
               dbscan_eps: float = 7.0) -> Tuple[torch.Tensor, Dict]:
    """Use DBScan to clusterize nodes to produce pockets

    Args:
        y_pred (torch.Tensor): Model prediction for each node
        pos (torch.Tensor): Node position
        th (float, optional): Threshold to classify nodes. Defaults to 0.5.
        min_freq (int, optional): Minimal number of element to create a cluster. Defaults to 5.
        dbscan_eps (float, optional): DBScan epsilon. Defaults to 7.0.

    Returns:
        Tuple[torch.Tensor, Dict]: Cluster id for each node, number of node in each cluster
    """
    pred_bool = y_pred >= th
    pred_bool = pred_bool.expand(pos.shape[0], 3)
    pred_poses = pos[pred_bool].reshape((-1, 3))
    if pred_poses.shape[0] > 0:
        clust = DBSCAN(eps=dbscan_eps)
        clustering = clust.fit(pred_poses)
        labels = clustering.labels_
        unique_l, freq = np.unique(labels, return_counts=True)

        dict_label_freq = defaultdict(int)
        for l, f in zip(unique_l, freq):
            if f >= min_freq:
                dict_label_freq[l] += f
            else:
                dict_label_freq[-1] += f

        list_final_labels = []
        j = 0

        for i in pred_bool:
            if i[0].item():
                pred_label = labels[j]
                if dict_label_freq[pred_label] >= min_freq:
                    list_final_labels.append(pred_label)
                else:
                    list_final_labels.append(-1)
                j += 1
            else:
                dict_label_freq[-1] += 1
                list_final_labels.append(-1)

        segmented_labels = torch.tensor(list_final_labels)
        return segmented_labels, dict_label_freq
    return None, None


def get_dcc(segmented_labels: torch.Tensor,
            dict_lab_freq: Dict,
            pos: torch.Tensor,
            sites_center: torch.Tensor) -> Tuple[float, float]:
    """Compute DCC and Mean DCC
    DCC : For the best pocket
    Mean DCC : The average of all DCC for all pockets

    Args:
        segmented_labels (torch.Tensor): Cluster id for each node
        dict_lab_freq (Dict): Number of node in each cluster
        pos (torch.Tensor): Node position
        sites_center (torch.Tensor): Center of ground thruth binding sites

    Returns:
        Tuple[float, float]: (DCC, mean DCC)
    """
    if len(segmented_labels) == dict_lab_freq[-1]:
        # No pocket found
        return torch.tensor(1000.0), torch.tensor(1000.0)
    pred_min_dcc_list = []
    for l in dict_lab_freq.keys():
        pred_min_dcc = 1000.0
        if l != -1:
            for gt_site_center in sites_center:
                site_coords = pos[segmented_labels == l].numpy()
                if site_coords.shape[0] != 0:
                    site_center = torch.tensor(geometric_center(site_coords))
                    dcc = (site_center - gt_site_center).pow(2).sum().sqrt()
                    if dcc < pred_min_dcc:
                        pred_min_dcc = dcc
            pred_min_dcc_list += [pred_min_dcc]
    dcc_sum = 0.0
    dcc_min = 1000.0
    for dcc in pred_min_dcc_list:
        dcc_sum += dcc
        if dcc < dcc_min:
            dcc_min = dcc
    return dcc_min, dcc_sum/len(pred_min_dcc_list)


def grouped_to_dict(grouped_df: DataFrameGroupBy) -> Dict:
    """Converts a DataFrameGroupBy amino acids to a Dictionnary

    Args:
        grouped_df (DataFrameGroupBy): A DataFrameGroupBY

    Returns:
        Dict: A Dictionnary
    """
    dict_groups = {}
    for ind, df in grouped_df:
        c = ind[0]
        r = int(ind[1])
        i = ind[2]
        if i == '':
            i = ' '
        dict_groups[(c, r, i)] = df
    return dict_groups


def from_cluster_to_pdb(segmented_labels: torch.Tensor,
                        dict_lab_freq: Dict,
                        protein_path: str,
                        g: pyg.data.Data,
                        pocket_suffix: str = '') -> List[str]:
    """Converts pocket clusters to PDB files

    Args:
        segmented_labels (torch.Tensor): Cluster id for each node
        dict_lab_freq (Dict): Number of node in each cluster
        protein_path (str): Path to the base protein
        g (pyg.data.Data): The protein PyTorch Geometric graph
        pocket_suffix (str, optional): Suffix for pocket filename. Defaults to ''.

    Returns:
        List[str]: List of path of extracted pockets
    """
    list_pocket_paths = []
    dir_prot, prot_file = osp.split(protein_path)
    dir_pocket = osp.join(dir_prot, 'SGPocket_out')
    if not osp.isdir(dir_pocket):
        os.mkdir(dir_pocket)

    ppdb = PandasPdb()
    ppdb.read_pdb(protein_path)
    grouped_df = ppdb.df['ATOM'].groupby(
        ['chain_id', 'residue_number', 'insertion'])
    dict_groups = grouped_to_dict(grouped_df)

    for k in dict_lab_freq.keys():
        if k != -1:
            list_dfs = []
            indices = []
            for i in range(len(segmented_labels)):
                if segmented_labels[i].item() == k:
                    indices += [g.res_id[i]]

            for res_ind in indices:
                list_dfs += [dict_groups[(res_ind[0], res_ind[1], res_ind[2])]]
            df_pocket = pd.concat(list_dfs).reset_index(drop=True)
            df_pocket['line_idx'] = df_pocket.index+1

            pocket_path = osp.join(
                dir_pocket, 'pocket_{}{}.pdb'.format(k, pocket_suffix))
            list_pocket_paths += [pocket_path]
            ppdb.df["ATOM"] = df_pocket
            ppdb.df["OTHERS"] = pd.DataFrame({'record_name': 'REMARK ',
                                              'entry': ' 1  Pocket extracted by SGPocket',
                                              'line_idx': 1},
                                             index=[0])

            ppdb.to_pdb(path=pocket_path,
                        records=['OTHERS', 'ATOM'],
                        gz=False,
                        append_newline=True)
    return list_pocket_paths
