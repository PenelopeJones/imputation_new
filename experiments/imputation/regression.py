"""
Script for training various models for the goal of data imputation.
"""
import os
import sys
sys.path.append('../../')
import warnings
import argparse

import numpy as np
import torch

from models.networks import ProbabilisticVanillaNN
from models.wrappers import RegressionWrapper
from utils.data_utils import preprocess_data, write_args, write_metrics
from utils.metric_utils import baseline_metrics_calculator
import pdb

properties_map = {'Adrenergic':5, 'Kinase':159, 'Excape':526}

def main(args):
    """
    :return:
    """
    args.task_type = 'regression'
    args.model_name = 'dnn'
    warnings.filterwarnings('ignore')
    torch.set_default_dtype(torch.float64)

    filename = args.dataname + str(args.num) + '_' + args.model_name + '_' + str(args.run_number)
    args.n_properties = properties_map[args.dataname]
    args.standardise = False

    with open('results/{}/{}/{}/{}.txt'.format(args.dataname, args.task_type, args.model_name, filename), 'a') as f:
        # Write hyperparameters to file
        write_args(f, args)
        f.flush()
        args.file = f

        # These are matrices of dimension [N_train, n_descriptors + n_properties],
        # [N_test, n_descriptors + n_properties] respectively. NaNs are imputed for missing
        # values. We will use the ECFP4 fingerprint descriptor (1024 bits, radius = 3)
        x = np.load(args.directory + args.dataname + '_x' + str(args.num) +
                        '_train_fingerprints.npy', allow_pickle=True)
        x_test = np.load(args.directory + args.dataname + '_x' + str(args.num) +
                             '_test_fingerprints.npy', allow_pickle=True)

        # Preprocess the data: standardising, applying PCA transforms and selecting relevant descriptors if desired.
        # Also converts x and x_test from numpy matrices to torch tensors.
        x, x_test, args.means, args.stds = preprocess_data(x, x_test, args.n_properties, args.pca_components, args.task_type)
        args.in_dim = x.shape[1]

        if args.model_name == 'baseline':
            f.write('\n ... predictions from baseline model.')
            # The baseline corresponds to mean imputation (with predictive uncertainty equal to
            # the standard deviation of the training set)
            r2_scores, mlls, rmses = baseline_metrics_calculator(x, args.n_properties,
                                                                 means=args.means,
                                                                 stds=args.stds)
            write_metrics(r2_scores, mlls, rmses, file=f, set_type='train')

            r2_scores, mlls, rmses = baseline_metrics_calculator(x_test, args.n_properties,
                                                                 means=args.means,
                                                                 stds=args.stds)
            write_metrics(r2_scores, mlls, rmses, file=f, set_type='test')

        else:
            f.write('\n ... building {} model'.format(args.model_name))

            # Load model
            model = RegressionWrapper(network=ProbabilisticVanillaNN(in_dim=args.in_dim, out_dim=args.n_properties,
                                                                     hidden_dims=args.hidden_dims, restrict_var=False),
                                      batch_size=args.batch_size, lr=args.lr, file=args.file)
            file_path = os.path.dirname(f.name) + '/models/{}_{}_{}_{}.dat'.format(args.dataname, args.model_name,
                                                                                   args.run_number, args.task_type)

            # If model was saved after last training procedure, reload to continue training from where you left off
            if os.path.isfile(file_path):
                model.network.load_state_dict(torch.load(file_path))

            f.write('\n ... training model. \n')

            # Train model
            model.train_model(x=x, epochs=args.epochs, epoch_print_freq=50, means=args.means,
                              stds=args.stds)

            # Save model
            torch.save(model.network.state_dict(), file_path)

            # Calculate performance metrics on the train and test datasets
            r2_scores, mlls, rmses = model.metrics_calculator(x, save=False)
            write_metrics(r2_scores, mlls, rmses, file=f, set_type='train')

            r2_scores, mlls, rmses = model.metrics_calculator(x_test, save=True)
            write_metrics(r2_scores, mlls, rmses, file=f, set_type='test')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', default='data/raw_data/',
                        help='Directory where the training and test data is stored.')
    parser.add_argument('--dataname', default='Adrenergic',
                        help='Name of dataset.')
    parser.add_argument('--run_number', type=int, default=0, help='Run number.')
    parser.add_argument('--num', type=int, default=1,
                        help='The train/test split number. 1 '
                             'for Kinase, between 1 and 5 for '
                             'Adrenergic.')
    parser.add_argument('--epochs', type=int, default=1001,
                        help='Number of training epochs.')
    parser.add_argument('--pca_components', type=int, default=100,
                        help='The number of pca components.')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Number of function samples per iteration.')
    parser.add_argument('--hidden_dims', nargs='+', type=int,
                        default=[500, 500, 500],
                        help='Dimensionality of network hidden layers.')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Optimiser learning rate.')

    args = parser.parse_args()

    main(args)
