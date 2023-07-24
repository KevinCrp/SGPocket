import argparse

from SGPocket.trainer import train

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-nb_epochs', '-ep',
                        type=int,
                        help='The maximum number of epochs (defaults to 100)',
                        default=100)
    parser.add_argument("-config",
                        help="Path to config file, default = config.ini",
                        type=str,
                        default='config.ini')
    parser.add_argument('-batch_size', '-bs',
                        type=int,
                        required=True,
                        help='Batch size')
    parser.add_argument('-learning_rate', '-lr',
                        type=float,
                        default=1e-3,
                        help='Learning Rate')
    parser.add_argument('-dropout', '-dp',
                        type=float,
                        default=0.0,
                        help='Dropout')
    parser.add_argument('-non_linearity', '-nl',
                        type=str,
                        help='non linearity')
    parser.add_argument('-dbscan_eps', '-eps',
                        type=float,
                        default=7.0,
                        help='Min distance between to AA to be considered in the same pocket')
    parser.add_argument('-threshold', '-th',
                        type=float,
                        default=0.4,
                        help='Threshold to classify each node')
    parser.add_argument('-hidden_channels',
                        type=str)
    parser.add_argument('-mlp_channels',
                        type=str)

    args = parser.parse_args()
    nb_epochs = args.nb_epochs
    config = args.config
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    dropout = args.dropout
    dbscan_eps = args.dbscan_eps
    threshold = args.threshold
    non_linearity = args.non_linearity
    hidden_channels = list(map(int, args.hidden_channels.split('-')))
    mlp_channels = list(map(int, args.mlp_channels.split('-')))
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
