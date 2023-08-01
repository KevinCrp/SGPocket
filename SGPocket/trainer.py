import multiprocessing as mp
import os
import os.path as osp
import warnings
from typing import List, Tuple

import pytorch_lightning as pl
import torch
import torch_geometric as pyg
from pytorch_lightning.utilities import rank_zero_only
from tqdm import tqdm

from SGPocket.data import PDBBind_DataModule
from SGPocket.model import Model

warnings.filterwarnings('ignore', '.*TypedStorage is deprecated.*')


def train(nb_epochs: int,
          config_file_path: str,
          batch_size: int,
          learning_rate: float,
          dropout: float,
          dbscan_eps: float,
          threshold: float,
          non_linearity: str,
          hidden_channels: List[int] = [48, 64, 128],
          mlp_channels: List[int] = [128, 64, 48, 1],
          experiments_dir:str ='SGPocket_exp') -> Tuple[float, int]:
    """Train a model

    Args:
        nb_epochs (int): Number of epochs
        config_file_path (str): Path to the config file
        batch_size (int): Batch size
        learning_rate (float): Learning rate
        dropout (float): Dropout
        dbscan_eps (float): DBScan epsilon
        threshold (float): Threshold to classify nodes
        non_linearity (str): USed non linearity for layers
        hidden_channels (List[int], optional): SGCN hidden sizes. Defaults to [48, 64, 128].
        mlp_channels (List[int], optional): MLP sizes. Defaults to [128, 64, 48, 1].
        experiments_dir (str, optional): Model will be saved in experiments/experiments_dir directory

    Returns:
        Tuple[float, int]: DCC Success rate on BSP-Benchmark, logger version
    """
    torch.set_float32_matmul_precision("high")

    gpus = torch.cuda.device_count()
    use_gpu = gpus > 0
    accelerator = 'gpu' if use_gpu else 'cpu'
    devices = gpus if gpus > 0 else 'auto'
    exp_model_name = experiments_dir
    experiments_path = osp.join('.', 'experiments')

    if not osp.isdir(experiments_path):
        os.mkdir(experiments_path)

    logger = pl.loggers.TensorBoardLogger(
        experiments_path, name=exp_model_name)

    version_path = osp.join(
        experiments_path, exp_model_name, 'version_' + str(logger.version))

    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=version_path,
                                                       save_top_k=1,
                                                       monitor="ep_end_val/loss",
                                                       mode='min')

    early_stopping_callback = pl.callbacks.early_stopping.EarlyStopping(
        monitor="ep_end_val/loss", mode="min", patience=20)

    callbacks = [checkpoint_callback, early_stopping_callback]

    datamodule = PDBBind_DataModule(config_file_path=config_file_path,
                                    batch_size=batch_size,
                                    num_workers=mp.cpu_count(),
                                    persistent_workers=True)

    model = Model(lr=learning_rate,
                  weight_decay=1e-4,
                  dropout=dropout,
                  hidden_channels=hidden_channels,
                  mlp_channels=mlp_channels,
                  non_linearity=non_linearity)

    hparam_str = "{},{},{},{},{}".format(
        nb_epochs, batch_size, learning_rate, dropout, model.__repr__().replace("\n", "").replace(',', '/'))

    trainer = pl.Trainer(accelerator=accelerator,
                         devices=devices,
                         callbacks=callbacks,
                         max_epochs=nb_epochs,
                         logger=logger,
                         log_every_n_steps=2,
                         num_sanity_val_steps=0,
                         check_val_every_n_epoch=1)

    trainer.fit(model, datamodule)

    group = torch.distributed.group.WORLD
    best_model_path = checkpoint_callback.best_model_path
    if gpus > 1:
        list_bmp = gpus * [None]
        torch.distributed.all_gather_object(
            list_bmp, best_model_path, group=group)
        best_model_path = list_bmp[0]

    test_best_model(best_model_path, datamodule, logger)
    succes_rate = test_segmentation(best_model_path, datamodule, logger,
                                    success_cutoff=4.0,
                                    threshold=threshold, dbscan_eps=dbscan_eps)
    return succes_rate, logger.version


@rank_zero_only
def test_metrics_segmentation(segmentation_results: List,
                              success_cutoff: float) -> Tuple[float, float]:
    """Compute test segmentation metrics for

    Args:
        segmentation_results (List): Segmentation results
        success_cutoff (float): Success cotoff for DCC

    Returns:
        Tuple[float, float]: Sucess rate, Mean success rate (for all pockets)
    """
    nb_dcc_success = 0
    nb_mean_dcc_success = 0
    for dcc_min, dcc_mean, _, _ in segmentation_results:
        if dcc_min is not None:
            nb_dcc_success += 1 if dcc_min <= success_cutoff else 0
            nb_mean_dcc_success += 1 if dcc_mean <= success_cutoff else 0
    success_rate = nb_dcc_success / len(segmentation_results) * 100
    success_rate_mean = nb_mean_dcc_success / len(segmentation_results) * 100
    return success_rate, success_rate_mean


@rank_zero_only
def test_segmentation(best_model_path: str,
                      datamodule: pl.LightningDataModule,
                      logger: pl.loggers.TensorBoardLogger,
                      success_cutoff: float,
                      threshold: float,
                      dbscan_eps: float) -> float:
    """Test on pocket segmentation

    Args:
        best_model_path (str): Path to the best model
        datamodule (pl.LightningDataModule): Datamodule
        logger (pl.loggers.TensorBoardLogger): Logger
        success_cutoff (float): Success cutoff dor DCC.
        threshold (float): Threshold to classify nodes.
        dbscan_eps (float): DBScan epsilon.

    Returns:
        float: Success rate
    """
    print("Test Segmentation BSP-Benchmark")
    trained_model = Model.load_from_checkpoint(
        best_model_path, map_location=torch.device('cpu'))
    res = []
    for g in tqdm(datamodule.dt_test):
        b = pyg.data.Batch.from_data_list([g])
        res += [trained_model.predict_extract_and_assess(
            b, threshold, dbscan_eps)]
    success_rate, success_rate_mean = test_metrics_segmentation(
        res, success_cutoff)
    print('success_rate : ', success_rate, '%')
    print('success_rate_mean : ', success_rate_mean, '%')
    logger.log_metrics(
        {'bsp_bench/success_rate_{}_{}'.format(threshold, success_cutoff): torch.tensor(success_rate),
         'bsp_bench/success_rate_mean_{}_{}'.format(threshold, success_cutoff): torch.tensor(success_rate_mean)})

    return success_rate


@rank_zero_only
def test_best_model(best_model_path: str,
                    datamodule: pl.LightningDataModule,
                    logger: pl.loggers.TensorBoardLogger):
    """Test the best model

    Args:
        best_model_path (str): Path to the best model
        datamodule (pl.LightningDataModule): Datamodule
        logger (pl.loggers.TensorBoardLogger): logger
    """
    print("Test on classification metrics")
    gpus = torch.cuda.device_count()
    use_gpu = gpus > 0
    accelerator = 'gpu' if use_gpu else 'cpu'
    devices = gpus if gpus > 0 else 'auto'
    trainer = pl.Trainer(accelerator=accelerator,
                         devices=devices,
                         max_epochs=1,
                         log_every_n_steps=0,
                         num_sanity_val_steps=0,
                         logger=logger)
    trained_model = Model.load_from_checkpoint(best_model_path)
    trainer.test(trained_model, datamodule.test_dataloader())
