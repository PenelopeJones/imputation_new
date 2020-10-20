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

from models.networks import BinaryClassificationNN
from models.wrappers import ClassificationWrapper
from utils.data_utils import preprocess_data, write_args, write_metrics, write_classification_metrics
from utils.metric_utils import baseline_metrics_calculator
import pdb

properties_map = {'Adrenergic':5, 'Kinase':159, 'Excape':526}

def main(args):
    """
    :return:
    """

    probe_file = 'Probes_fingerprints.npy'


    args.task_type = 'classification'
    args.model_name = 'dnn'
    warnings.filterwarnings('ignore')
    torch.set_default_dtype(torch.float64)

    filename = args.dataname + str(args.num) + '_' + args.model_name + '_' + str(args.run_number) + 'probes'
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
        x_test = np.load(args.directory + probe_file, allow_pickle=True)

        y = np.empty((x_test.shape[0], args.n_properties))

        x_test = np.concatenate((x_test, y), axis=1)
        pdb.set_trace()


        # Preprocess the data: standardising, applying PCA transforms and selecting relevant descriptors if desired.
        # Also converts x and x_test from numpy matrices to torch tensors.
        x, x_test, args.means, args.stds = preprocess_data(x, x_test, args.n_properties, args.pca_components, args.task_type)

        args.in_dim = x.shape[1]

        f.write('\n ... building {} model'.format(args.model_name))

        # Load model
        model = ClassificationWrapper(network=BinaryClassificationNN(in_dim=args.in_dim, out_dim=args.n_properties,
                                                                     hidden_dims=args.hidden_dims),
                                      batch_size=args.batch_size, lr=args.lr, file=args.file)

        file_path = os.path.dirname(args.file.name) + '/models/{}_{}_{}_{}.dat'.format(args.dataname,
                                                                                           args.model_name,
                                                                                           args.run_number,
                                                                                           args.task_type)
        # If model was saved after last training procedure, reload parameters to continue training from where you left off
        if os.path.isfile(file_path):
            model.network.load_state_dict(torch.load(file_path))
        else:
            raise Exception('No model found.')

        # Make predictions

        predictions = model.predict(x_test, save=True, means=args.means, stds=args.stds)

        pdb.set_trace()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', default='data/raw_data/',
                        help='Directory where the training and test data is stored.')
    parser.add_argument('--dataname', default='Excape',
                        help='Name of training dataset.')
    parser.add_argument('--run_number', type=int, default=1000, help='Run number.')
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
