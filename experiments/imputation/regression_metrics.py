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
from sklearn.metrics import r2_score, mean_squared_error, roc_auc_score, f1_score
from utils.metric_utils import mll, metric_ordering, confidence_curve, find_nearest
from scipy.stats import pearsonr

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
    task_type = 'regression'
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

            for p in range(n_properties):
                mns = []
                stds = []
                targets = []
                for i in range(run_number):
                    filestart = '{}{}_{}_{}_{}_'.format(args.dataname, args.num, args.model_name, (batch*run_number+i), p)
                    mn = np.load('results/{}/{}/{}/{}{}mean.npy'.format(args.dataname, task_type, args.model_name, extra_dir, filestart))
                    std = np.load('results/{}/{}/{}/{}{}std.npy'.format(args.dataname, task_type, args.model_name, extra_dir, filestart))
                    target = np.load('results/{}/{}/{}/{}{}target.npy'.format(args.dataname, task_type, args.model_name, extra_dir, filestart))
                    mns.append(mn)
                    stds.append(std)
                    targets.append(target)

                # Ensemble mean, var, target
                mean = np.mean(np.array(mns), axis=0)
                var = np.mean(np.array(stds)**2, axis=0)
                target = np.mean(np.array(targets), axis=0)
                binary_target = np.zeros_like(target)
                binary_target[target > 6] = 1.0

                binary_mean = np.zeros_like(mean)
                binary_mean[mean>6] = 1.0
                f1_scores.append(f1_score(binary_target, binary_mean))
                try:
                    roc_aucs_binary.append(roc_auc_score(binary_target, binary_mean))
                    roc_aucs.append(roc_auc_score(binary_target, binary_mean))
                except:
                    continue
                r2_scores.append(r2_score(target, mean))
                mlls.append(mll(mean, var, target))
                rmses.append(np.sqrt(mean_squared_error(target, mean)))

                conf_percentile, metric_model, metric_oracle = metric_ordering(mean, var, target, metric)
                indices = []
                for percentile in percentiles:
                    indices.append(find_nearest(conf_percentile, percentile))
                indices = np.array(indices)

                metric_models.append(metric_model[indices])
                metric_oracles.append(metric_oracle[indices])

            f1_scores_list.append(np.mean(np.array(f1_scores)))
            roc_aucs_list.append(np.mean(np.array(roc_aucs)))
            roc_aucs_binary_list.append(np.mean(np.array(roc_aucs_binary)))
            mlls_list.append(np.mean(np.array(mlls)))
            r2_scores_list.append(np.mean(np.array(r2_scores)))
            rmses_list.append(np.mean(np.array(rmses)))
            mlls_list.append(np.mean(np.array(mlls)))

            metric_models = np.array(metric_models)
            metric_oracles = np.array(metric_oracles)

            metric_model = np.mean(metric_models, axis=0)
            metric_oracle = np.mean(metric_oracles, axis=0)

            metric_model_mns.append(metric_model)
            metric_oracle_mns.append(metric_oracle)

        r2_scores_list = np.array(r2_scores_list)
        mlls_list = np.array(mlls_list)
        rmses_list = np.array(rmses_list)
        roc_aucs_list = np.array(roc_aucs_list)
        roc_aucs_binary_list = np.array(roc_aucs_binary_list)
        f1_scores_list = np.array(f1_scores_list)


        f.write('\n R^2 score: {:.4f}+- {:.4f}'.format(np.mean(r2_scores_list), np.std(r2_scores_list)))
        f.write('\n MLL: {:.4f}+- {:.4f} \n'.format(np.mean(mlls_list), np.std(mlls_list)))
        f.write('\n RMSE: {:.4f}+- {:.4f} \n'.format(np.mean(rmses_list), np.std(rmses_list)))
        f.write('\n F^1 score: {:.4f}+- {:.4f}'.format(np.mean(f1_scores_list), np.std(f1_scores_list)))
        f.write('\n ROC-AUC: {:.4f}+- {:.4f} \n'.format(np.mean(roc_aucs_list), np.std(roc_aucs_list)))
        f.write('\n ROC-AUC (binary): {:.4f}+- {:.4f} \n'.format(np.mean(roc_aucs_binary_list), np.std(roc_aucs_binary_list)))

        f.flush()

        metric_model_mns = np.array(metric_model_mns)
        metric_model_mn = np.mean(metric_model_mns, axis=0)
        metric_model_std = np.std(metric_model_mns, axis=0)
        metric_oracle_mns = np.array(metric_oracle_mns)
        metric_oracle_mn = np.mean(metric_oracle_mns, axis=0)
        metric_oracle_std = np.std(metric_oracle_mns, axis=0)

        print(metric_model_mn)
        print(metric_model_std)

        confidence_curve(percentiles, metric_model_mn, metric_oracle_mn, fig_pts,
                         metric_model_std, metric_oracle_std, metric)


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
    parser.add_argument('--model_name', default='setofconduits',
                        help='Model to use.')
    args = parser.parse_args()

    main(args)