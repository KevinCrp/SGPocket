import os
from glob import glob
from shutil import rmtree

import yaml

from SGPocket.trainer import train

MODEL_HPARAM_PATH = "tests/model_parameters.yaml"

def test_train():
    with open(MODEL_HPARAM_PATH, 'r') as f_yaml:
        model_parameters = yaml.safe_load(f_yaml)
    nb_epochs = 1
    config = 'tests/config.ini'
    batch_size = 2
    learning_rate = 1e-3
    dropout = 0.01
    dbscan_eps = 8.0
    threshold = 0.4
    non_linearity = 'tanh'
    hidden_channels=model_parameters['hidden_channels']
    mlp_channels=model_parameters['mlp_channels']
    success_rate, version = train(nb_epochs=nb_epochs,
                                  config_file_path=config,
                                  batch_size=batch_size,
                                  learning_rate=learning_rate,
                                  dropout=dropout,
                                  dbscan_eps=dbscan_eps,
                                  threshold=threshold,
                                  non_linearity=non_linearity,
                                  hidden_channels=hidden_channels,
                                  mlp_channels=mlp_channels,)
    print('{};{}'.format(success_rate, version))
    rmtree('tests/data/processed')
    files_to_delete = glob('tests/data/BSP_Benchmark/*/*.tmp')
    files_to_delete += glob('tests/data/BSP_Benchmark/*/*clean*')
    files_to_delete += glob('tests/data/raw/*/*.tmp')
    files_to_delete += glob('tests/data/raw/*/*clean*')
    for f in files_to_delete:
        os.remove(f)

