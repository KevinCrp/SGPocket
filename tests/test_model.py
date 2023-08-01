import SGPocket.model as md
import yaml

MODEL_HPARAM_PATH = "tests/model_parameters.yaml"
MODEL_PATH = 'models_10A/model.ckpt'
DATA_ROOT = 'tests/data'


def test_model_init():
    with open(MODEL_HPARAM_PATH, 'r') as f_yaml:
        model_parameters = yaml.safe_load(f_yaml)

    model = md.Model(
        lr=model_parameters['lr'],
        weight_decay=model_parameters['weight_decay'],
        hidden_channels=model_parameters['hidden_channels'],
        mlp_channels=model_parameters['mlp_channels'],
        dropout=model_parameters['dropout'],
        non_linearity=model_parameters['non_linearity'])

    assert (model is not None)