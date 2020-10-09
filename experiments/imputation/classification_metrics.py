"""
Script for training various models for the goal of data imputation.
In particular, two variants on the neural process are implemented, both including FiLM layers
but one allowing for the possibility of also including skip connections. A competing model
(previously applied by Conduit et al.) is also implemented.
"""
import os
import sys
sys.path.append('../../')
import warnings
import argparse
import pdb
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_squared_error, roc_auc_score, f1_score, roc_curve, auc
from sklearn.metrics import average_precision_score, precision_recall_curve
from utils.metric_utils import mll, metric_ordering, confidence_curve, find_nearest
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

properties_map = {'Adrenergic':5, 'Kinase':159, 'Excape':526}

def main(args):
    """
    :return:
    """

    warnings.filterwarnings('ignore')
    torch.set_default_dtype(torch.float64)

    extra = 'single'
    extra_dir = ''

    filename = args.dataname + '_' + args.model_name + '_' + extra
    task_type = 'classification'
    run_number = 1
    batches = [0,]
    epochs = 250
    n_properties = properties_map[args.dataname]

    with open('results/{}/{}/summary/{}_ensemble.txt'.format(args.dataname, task_type, filename), 'a') as f:
        r2_scores_list = []
        mlls_list = []
        rmses_list = []
        f1_scores_list = []
        roc_aucs_list = []
        roc_aucs_binary_list = []

        metric = 'rmse'
        percentiles = np.arange(100, 4, -5)
        dir_name = os.path.dirname(f.name)
        fig_pts = '{}/{}_{}_ensemble_confidence_curve.png'.format(dir_name, filename, metric)

        metric_model_mns = []
        metric_oracle_mns = []

        for batch in batches:
            metric_models = []
            metric_oracles = []
            f1_scores = []
            r2_scores = []
            roc_aucs = []
            roc_aucs_binary = []
            rmses = []
            mlls = []

            fprs = []
            tprs = []

            for p in range(n_properties):
                mns = []
                targets = []
                for i in range(run_number):
                    filestart = '{}{}_{}_{}_{}_'.format(args.dataname, args.num, args.model_name, (batch*run_number+i), p)
                    mn = np.load('results/{}/{}/{}/{}{}mean.npy'.format(args.dataname, task_type, args.model_name, extra_dir, filestart))
                    target = np.load('results/{}/{}/{}/{}{}target.npy'.format(args.dataname, task_type, args.model_name, extra_dir, filestart))
                    mns.append(mn)
                    targets.append(target)

                # Ensemble mean, var, target
                mean = np.mean(np.array(mns), axis=0)
                target = np.mean(np.array(targets), axis=0)

                roc_aucs.append(roc_auc_score(target, mean))
                r2_scores.append(r2_score(target, mean))
                rmses.append(np.sqrt(mean_squared_error(target, mean)))

            roc_aucs_list.append(np.mean(np.array(roc_aucs)))
            r2_scores_list.append(np.mean(np.array(r2_scores)))
            rmses_list.append(np.mean(np.array(rmses)))

        r2_scores_list = np.array(r2_scores_list)
        rmses_list = np.array(rmses_list)
        roc_aucs_list = np.array(roc_aucs_list)

        f.write('\n R^2 score: {:.4f}+- {:.4f}'.format(np.mean(r2_scores_list), np.std(r2_scores_list)))
        f.write('\n RMSE: {:.4f}+- {:.4f} \n'.format(np.mean(rmses_list), np.std(rmses_list)))
        f.write('\n ROC-AUC: {:.4f}+- {:.4f} \n'.format(np.mean(roc_aucs_list), np.std(roc_aucs_list)))

        f.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', default='data/raw_data/',
                        help='Directory where the training and test data is stored.')
    parser.add_argument('--dataname', default='Adrenergic',
                        help='Name of dataset.')
    parser.add_argument('--num', type=int, default=1,
                        help='The train/test split number. 1 '
                             'for Kinase, between 1 and 5 for '
                             'Adrenergic.')
    parser.add_argument('--n_properties', type=int, default=5,
                        help='The number of properties.')
    parser.add_argument('--model_name', default='dnn',
                        help='Model to use.')
    args = parser.parse_args()

    main(args)