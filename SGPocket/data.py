import configparser
import multiprocessing as mp
import os
import os.path as osp
from dataclasses import dataclass, field
from typing import List, Union

import pandas as pd
import pytorch_lightning as pl
import torch
import torch_geometric as pyg

from SGPocket.featurizer import featurize
from SGPocket.utilities.mol2_tools import clean_mol2
from SGPocket.utilities.pdb_tools import clean_pdb, extract_pocket


def to_graph(nodes_df: pd.DataFrame,
             edge_index: List,
             edge_attr: List,
             sites_center: List,
             add_res_id: bool = False) -> pyg.data.Data:
    """Transforms a dataframe of nodes (and nodes features) and edge_index into 
       a graph

    Args:
        nodes_df (pd.DataFrame): A Dataframe of node features
        edge_index (List): The edge index
        edge_attr (List): The edge attributes
        sites_center (List): List of all site centers
        add_res_id (bool, optional): Used to add res_id in graph. Defaults to False.

    Returns:
        pyg.data.Data: The pyg graph
    """
    list_node_fts = nodes_df[['GLY', 'ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PHE', 'TRP',
                              'PRO', 'SER', 'THR', 'CYS', 'TYR', 'ASN', 'GLN',
                              'ASP', 'GLU', 'LYS', 'ARG', 'HIS']].values.tolist()
    t_node_fts = torch.tensor(list_node_fts).to(torch.float)
    list_y = nodes_df[['is_pocket']].values.tolist()
    list_node_pos = nodes_df[['res_x', 'res_y', 'res_z']].values.tolist()
    graph = pyg.data.Data(x=t_node_fts,
                          pos=torch.tensor(list_node_pos),
                          edge_index=torch.tensor(edge_index),
                          edge_attr=torch.tensor(edge_attr).float(),
                          y=torch.tensor(list_y))
    if sites_center is not None:
        graph.sites_center = torch.tensor(sites_center)
    if add_res_id:
        res_id = nodes_df[['res_chain', 'res_id',
                           'res_insert']].values.tolist()
        graph.res_id = res_id
    return graph


def process_graph(filename: str,
                  raw_dir: str,
                  pdb_id: str,
                  sph_order: Union[int, str],
                  pocket_cutoff: float,
                  voronota_path: str,
                  sh_featurizer_linux_path: str,
                  protein_name: str = '{}_protein',
                  pocket_name: str = '{}_pocket',
                  protein_name_clean: str = '{}_protein_clean',
                  pocket_name_clean: str = '{}_pocket_clean',
                  ligand_name: str = '{}_ligand.mol2',
                  ligand_name_clean: str = '{}_ligand_clean.mol2'
                  ) -> str:
    """Create a graph from a PDB

    Args:
        filename (str): The created graph filename
        raw_dir (str): The raw directory
        pdb_id (str): The PDB id
        sph_order (Union[int, str]): The spherical harmonics order
        pocket_cutoff (float): The pocket distance cutoff for groundtruth
        protein_name (str, optional): The protein filename. Defaults to '{}_protein'.
        pocket_name (str, optional): The pocket filename. Defaults to '{}_pocket'.
        protein_name_clean (str, optional): The protein filename cleaned. Defaults to '{}_protein_clean'.
        pocket_name_clean (str, optional): The protein filename cleaned. Defaults to '{}_pocket_clean'.
        ligand_name (str, optional): The ligand filename. Defaults to '{}_ligand.mol2'.
        ligand_name_clean (str, optional): The ligand filename cleaned. Defaults to '{}_ligand_clean.mol2'.

    Returns:
        str: The used filename
    """
    if not osp.isfile(filename):
        data_pdb_dir = osp.join(raw_dir, pdb_id)
        protein_name = protein_name.format(pdb_id)
        protein_name_clean = protein_name_clean.format(pdb_id)
        protein_path_pdb = osp.join(data_pdb_dir, protein_name)
        protein_path_pdb_clean = osp.join(data_pdb_dir, protein_name_clean)

        pocket_name = pocket_name.format(pdb_id)
        pocket_name_clean = pocket_name_clean.format(pdb_id)
        pocket_path_pdb = osp.join(data_pdb_dir, pocket_name)
        pocket_path_pdb_clean = osp.join(data_pdb_dir, pocket_name_clean)

        ligand_name = ligand_name.format(pdb_id)
        ligand_name_clean = ligand_name_clean.format(pdb_id)
        ligand_path_mol2 = osp.join(data_pdb_dir, ligand_name)
        ligand_path_mol2_clean = osp.join(data_pdb_dir, ligand_name_clean)
        if not osp.isfile(protein_path_pdb_clean+'.pdb'):
            clean_pdb(protein_path_pdb+'.pdb',
                      protein_path_pdb_clean+'.pdb')
        if not osp.isfile(ligand_path_mol2_clean):
            clean_mol2(ligand_path_mol2,
                       ligand_path_mol2_clean)
        if not osp.isfile(pocket_path_pdb+'.pdb'):
            extract_pocket(protein_path_pdb_clean+'.pdb',
                           ligand_path_mol2_clean,
                           pocket_path_pdb+'.pdb',
                           pocket_cutoff)

        if not osp.isfile(pocket_path_pdb_clean+'.pdb'):
            clean_pdb(pocket_path_pdb+'.pdb', pocket_path_pdb_clean+'.pdb')

        df, sites_center, edge_index, edge_attr = featurize(
            data_pdb_dir,
            protein_name_clean,
            pocket_name_clean,
            sph_order,
            voronota_path,
            sh_featurizer_linux_path
        )

        g = to_graph(df, edge_index, edge_attr, sites_center)
        torch.save(g, filename)

    return filename


class PDBBind_Dataset(pyg.data.InMemoryDataset):
    def __init__(self,
                 root: str,
                 stage: str,
                 sph_order: int,
                 pocket_cutoff: float,
                 voronota_path: str,
                 sh_featurizer_linux_path: str,
                 transform=None,
                 pre_transform=None):
        """Constructor

        Args:
            root (str): Path to the data directory
            stage (str): Stage: Train or val
            sph_order (int): Spherical harmonics order
            pocket_cutoff (float): Pocket cutoff for binding site ground thruth extraction
            voronota_path (str): Path to the Voronota binary file
            sh_featurizer_linux_path (str): Path to the SH featurizer binary file
            transform (_type_, optional): Defaults to None.
            pre_transform (_type_, optional): Defaults to None.
        """
        self.stage = stage
        self.sph_order = sph_order
        self.pocket_cutoff = pocket_cutoff
        self.voronota_path = voronota_path
        self.sh_featurizer_linux_path = sh_featurizer_linux_path
        self.df = pd.read_csv(
            osp.join(root, '{}.csv'.format(self.stage))).set_index('pdb_id')
        self.blacklist = pd.read_csv(
            osp.join(root, 'blacklist.csv'))['pdb_id'].to_list()
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        """
        Returns:
            List[str]: Returns paths to the raw data
        """
        filename_list = []
        for pdb_id in self.df.index:
            if pdb_id not in self.blacklist:
                filename_list.append(pdb_id)
        return filename_list

    @property
    def processed_file_names(self) -> List[str]:
        """
        Returns:
            List[str]: Path to the processed dataset
        """
        return [osp.join('{}.pt'.format(self.stage))]

    def process(self):
        """Processes raw data by building graphs and save them into the processed
        dataset file
        """
        i = 0
        print('\t', self.stage)
        pool_args = []
        for raw_path in self.raw_paths:
            filename = osp.join(self.processed_dir,
                                'TMP_{}_data_{}.pt'.format(self.stage, i))
            data_dir, pdb_id = osp.split(raw_path)
            pool_args.append((filename,
                              data_dir,
                              pdb_id,
                              self.sph_order,
                              self.pocket_cutoff,
                              self.voronota_path,
                              self.sh_featurizer_linux_path))
            i += 1
        pool = mp.Pool(mp.cpu_count())
        data_path_list = list(pool.starmap(process_graph, pool_args))
        data_list = []
        for p in data_path_list:
            data_list.append(torch.load(p))
            os.remove(p)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class BSP_Benchmark(pyg.data.InMemoryDataset):
    def __init__(self,
                 root: str,
                 sph_order: int,
                 pocket_cutoff: float,
                 voronota_path: str,
                 sh_featurizer_linux_path: str,
                 transform=None,
                 pre_transform=None):
        """Constructor

        Args:
            root (str): Path to the data directory
            sph_order (int): Spherical harmonics order
            pocket_cutoff (float): Pocket cutoff for binding site ground thruth extraction
            voronota_path (str): Path to the Voronota binary file
            sh_featurizer_linux_path (str): Path to the SH featurizer binary file
            transform (_type_, optional): Defaults to None.
            pre_transform (_type_, optional): Defaults to None.
        """
        self.stage = 'bsp_bench'
        self.sph_order = sph_order
        self.pocket_cutoff = pocket_cutoff
        self.voronota_path = voronota_path
        self.sh_featurizer_linux_path = sh_featurizer_linux_path
        self.df = pd.read_csv(
            osp.join(root, '{}.csv'.format(self.stage))).set_index('pdb_id')
        self.blacklist = pd.read_csv(
            osp.join(root, 'blacklist.csv'))['pdb_id'].to_list()
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        """
        Returns:
            List[str]: Returns filename of the raw data
        """
        filename_list = []
        for pdb_id in self.df.index:
            if pdb_id not in self.blacklist:
                filename_list.append(pdb_id)
        return filename_list

    @property
    def raw_paths(self):
        """
        Returns:
            List[str]: Returns paths to the raw data
        """
        raw_path_list = []
        for raw_filename in self.raw_file_names:
            raw_path_list += [osp.join(self.root,
                                       'BSP_Benchmark', raw_filename)]
        return raw_path_list

    @property
    def processed_file_names(self):
        """
        Returns:
            List[str]: Path to the processed dataset
        """
        return [osp.join('{}.pt'.format(self.stage))]

    def process(self):
        """Processes raw data by building graphs and save them into the processed
        dataset file
        """
        i = 0
        print('\t', self.stage)
        pool_args = []
        for raw_path in self.raw_paths:
            filename = osp.join(self.processed_dir,
                                'TMP_{}_data_{}.pt'.format(self.stage, i))
            data_dir, pdb_id = osp.split(raw_path)
            pool_args.append((filename,
                              data_dir,
                              pdb_id,
                              self.sph_order,
                              self.pocket_cutoff,
                              self.voronota_path,
                              self.sh_featurizer_linux_path,
                              'protein_without_H',
                              'site_without_H',
                              'protein_without_H_clean',
                              'site_without_H_clean',
                              'ligand.mol2',
                              'ligand_clean.mol2'))
            i += 1
        pool = mp.Pool(mp.cpu_count())
        data_path_list = list(pool.starmap(process_graph, pool_args))
        data_list = []
        for p in data_path_list:
            data_list.append(torch.load(p))
            os.remove(p)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


@dataclass
class PDBBind_DataModule(pl.LightningDataModule):
    """PyTorch Lightning Datamodule
    """
    config_file_path: str
    batch_size: int = field(default=1)  # The batch size. Defaults to 1.
    # The number of workers. Defaults to 1.
    num_workers: int = field(default=1)
    # Use persistent workers in dataloader
    persistent_workers: bool = field(default=True)
    root = ''

    def __post_init__(self):
        super().__init__()

    def prepare_data(self):
        pass

    def setup(self, stage: str = ''):
        """Setups all dataset (train, val and test)

        Args:
            stage (str, optional): Defaults to ''.
        """
        config = configparser.ConfigParser()
        config.read(self.config_file_path)
        root = config['PATHS']['data']
        voronota_path = config['PATHS']['voronota']
        sh_featurizer_linux_path = config['PATHS']['sh_featurizer']
        sph_order = config['GRAPHS']['spherical_harmonics_order']
        pocket_cutoff = config['GRAPHS']['pocket_cutoff']

        self.dt_train = PDBBind_Dataset(root=root,
                                        stage='train',
                                        sph_order=sph_order,
                                        pocket_cutoff=pocket_cutoff,
                                        voronota_path=voronota_path,
                                        sh_featurizer_linux_path=sh_featurizer_linux_path)
        self.dt_val = PDBBind_Dataset(root=root,
                                      stage='val',
                                      sph_order=sph_order,
                                      pocket_cutoff=pocket_cutoff,
                                      voronota_path=voronota_path,
                                      sh_featurizer_linux_path=sh_featurizer_linux_path)

        self.dt_test = BSP_Benchmark(root=root,
                                     sph_order=sph_order,
                                     pocket_cutoff=6.5,  # Same cutoff as scPDB
                                     voronota_path=voronota_path,
                                     sh_featurizer_linux_path=sh_featurizer_linux_path)

    def train_dataloader(self) -> pyg.data.DataLoader:
        return pyg.loader.DataLoader(self.dt_train,
                                     batch_size=self.batch_size, shuffle=True,
                                     num_workers=self.num_workers,
                                     persistent_workers=self.persistent_workers)

    def val_dataloader(self) -> pyg.data.DataLoader:
        return pyg.loader.DataLoader(self.dt_val,
                                     batch_size=self.batch_size,
                                     num_workers=self.num_workers,
                                     persistent_workers=self.persistent_workers)

    def test_dataloader(self) -> pyg.data.DataLoader:
        return pyg.loader.DataLoader(self.dt_test,
                                     batch_size=1,
                                     num_workers=self.num_workers,
                                     persistent_workers=self.persistent_workers)
