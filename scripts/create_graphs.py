import argparse

from SGPocket.data import PDBBind_DataModule


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-config",
                        help="Path to config file, default = config.ini",
                        type=str,
                        default='config.ini')
    args = parser.parse_args()
    config_file_path = args.config

    dm = PDBBind_DataModule(config_file_path=config_file_path)
    dm.setup()
