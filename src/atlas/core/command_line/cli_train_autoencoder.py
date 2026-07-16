"""Train an autoencoder for dimensionality reduction."""

import argparse
import warnings

import atlas.active_learning.extrapolation.train_autoencoder as train_ae
import atlas.core.code_utils as atl_cut

warnings.filterwarnings('ignore')


def run_train_autoencoder():
    # Set up the logger
    atl_cut.init_logger('train_autoencoder', log_path='.')

    # Argument parser
    parser = argparse.ArgumentParser(
        description='Autoencoder for dimensionality reduction'
    )

    # Device
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help="Device to run the model on. Either 'cpu', 'cuda' or 'auto'.",
    )

    # dtype
    parser.add_argument(
        '--dtype', type=str, default='float32', help='Data type for the model'
    )

    # Model name
    parser.add_argument(
        '--model_path',
        type=str,
        default='autoencoder_model.pth',
        help='Path to save the model',
    )

    parser.add_argument(
        '--load_model',
        type=bool,
        default=False,
        help='Load the model from the model path',
    )
    parser.add_argument(
        '--rng_seed',
        type=int,
        help='Seed for the random number generator',
    )

    # Dataset path
    parser.add_argument(
        '--dataset',
        type=str,
        default='all_descriptors.npz',
        help='Path to the training dataset',
    )

    # Number of hidden layers
    parser.add_argument(
        '--l1_hidden_dim',
        type=int,
        default=256,
        help='Number of units in the first hidden layer',
    )
    parser.add_argument(
        '--l2_hidden_dim',
        type=int,
        default=32,
        help='Number of units in the second hidden layer',
    )

    # Dimensionality of the bottleneck (latent space)
    parser.add_argument(
        '--bottleneck_dim',
        type=int,
        default=2,
        help='Dimensionality of the bottleneck (latent space)',
    )

    # Number of epochs to train for
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=50,
        help='Number of epochs to train the model',
    )

    # Batch size
    parser.add_argument(
        '--batch_size', type=int, default=1024 * 4, help='Batch size for training'
    )

    parser.add_argument(
        '--patience',
        type=int,
        default=5,
        help='Early stopping patience (epochs without val loss improvement)',
    )

    # Learning rate
    parser.add_argument(
        '--lr', type=float, default=1e-3, help='Learning rate for the optimizer'
    )

    # Weight decay
    # Use weight_decay > 0 for L2 regularization
    parser.add_argument(
        '--weight_decay', type=float, default=1e-5, help='L2 regularization'
    )

    # Bias flag
    parser.add_argument(
        '--bias_flag',
        action='store_true',
        help='Flag to include bias terms in the layers',
    )

    parser.add_argument(
        '--loss',
        type=str,
        default='mse',
    )

    # Size of the validation and test sets
    parser.add_argument(
        '--train_frac',
        type=float,
        default=0.8,
        help='Fraction of the data to use for training',
    )
    parser.add_argument(
        '--valid_frac',
        type=float,
        default=0.1,
        help='Fraction of the data to use for validation',
    )
    parser.add_argument(
        '--test_frac',
        type=float,
        default=0.1,
        help='Fraction of the data to use for testing',
    )
    parser.add_argument(
        '--wandb',
        action='store_true',
        help='Whether to log metrics to wandb',
    )
    parser.add_argument(
        '--wandb_name',
        type=str,
        default='',
        help='Name of the wandb run',
    )
    parser.add_argument(
        '--wandb_project',
        type=str,
        default='',
        help='Name of the wandb project',
    )

    args = parser.parse_args()

    train_ae.run_training(args=args)
