# SGPocket

A Graph Convolutional Neural Network to predict protein binding site.

![coverage badge](tests/badges/coverage.svg)

### Installation
1. Clone this repo
2. Create a conda environment or a Docker container with provided files. Dockerfile and YAML files are provided in the `./venv` directory
3. Install with `pip install .`

### Test
- Install pytest `pip install pytest`
- Run `pytest -v`


## Usage
### To train a new model

#### 1. Download the PDBBind dataset

`./scripts/download_pdbbind.sh`

#### 2. Create the graphs
`scripts/create_graphs.py`
```bash
usage: create_graphs.py [-h] [-config CONFIG]

options:
  -h, --help      show this help message and exit
  -config CONFIG  Path to config file, default = config.ini
```

#### 3. Train
`scripts/trainer.py`
```bash
usage: trainer.py [-h] [-nb_epochs NB_EPOCHS] [-config CONFIG] -batch_size BATCH_SIZE [-learning_rate LEARNING_RATE] [-dropout DROPOUT] [-non_linearity NON_LINEARITY]
                  [-dbscan_eps DBSCAN_EPS] [-threshold THRESHOLD] [-hidden_channels HIDDEN_CHANNELS] [-mlp_channels MLP_CHANNELS]

options:
  -h, --help            show this help message and exit
  -nb_epochs NB_EPOCHS, -ep NB_EPOCHS
                        The maximum number of epochs (defaults to 100)
  -config CONFIG        Path to config file, default = config.ini
  -batch_size BATCH_SIZE, -bs BATCH_SIZE
                        Batch size
  -learning_rate LEARNING_RATE, -lr LEARNING_RATE
                        Learning Rate
  -dropout DROPOUT, -dp DROPOUT
                        Dropout
  -non_linearity NON_LINEARITY, -nl NON_LINEARITY
                        non linearity
  -dbscan_eps DBSCAN_EPS, -eps DBSCAN_EPS
                        Min distance between to AA to be considered in the same pocket
  -threshold THRESHOLD, -th THRESHOLD
                        Threshold to classify each node
  -hidden_channels HIDDEN_CHANNELS
  -mlp_channels MLP_CHANNELS
```
Models are saved in `experiments/SGPocket_exp/version_X`.

### Predict
### Predict pocket(s) for one protein
`scripts/predict.py`
```bash
usage: predict.py [-h] [-config CONFIG] -model MODEL -pdb PDB -threshold THRESHOLD -dbscan_eps DBSCAN_EPS

options:
  -h, --help            show this help message and exit
  -config CONFIG        Path to config file, default = config.ini
  -model MODEL, -m MODEL
                        Path to the model
  -pdb PDB              Path to the protein PDB
  -threshold THRESHOLD, -th THRESHOLD
                        Threshold used to class each node
  -dbscan_eps DBSCAN_EPS, -eps DBSCAN_EPS
                        The DBScan EPS
```
Extracted pockets are saved in the protein direcory in a subdirectory named `SGPocket_out`.

### Predict pocket(s) for several proteins
`scripts/predict_dataset.py`
Create a `csv` files with a protein path per line.
```bash
usage: predict_dataset.py [-h] [-config CONFIG] -model MODEL -dataset DATASET -threshold THRESHOLD -dbscan_eps DBSCAN_EPS

options:
  -h, --help            show this help message and exit
  -config CONFIG        Path to config file, default = config.ini
  -model MODEL, -m MODEL
                        Path to the model
  -dataset DATASET, -d DATASET
                        Path to the CSV dataset
  -threshold THRESHOLD, -th THRESHOLD
                        Threshold used to class each node
  -dbscan_eps DBSCAN_EPS, -eps DBSCAN_EPS
                        The DBScan EPS
```
Extracted pockets are saved in each protein direcory in a subdirectory named `SGPocket_out`.

### Sources

We use the `bin/voronata-linux` and `bin/sh-featurizer-linux` binary files provided by https://gitlab.inria.fr/GruLab/s-gcn.