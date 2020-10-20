"""
Conditional Neural Process inspired imputation model.
"""
import os
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score, mean_squared_error, roc_auc_score
from sklearn.metrics import precision_recall_curve, auc

from models.networks import VanillaNN, ProbabilisticVanillaNN
from utils.metric_utils import mll, negative_log_likelihood

import pdb


class NoImpRegressionWrapper:
    def __init__(self, network, batch_size, lr, file):
        self.lr = lr
        self.batch_size = batch_size
        self.network = network
        self.n_properties = self.network.out_dim
        self.optimiser = optim.Adam(self.network.parameters(), self.lr)
        self.loss_function = negative_log_likelihood
        self.file = file
        self.dir_name = os.path.dirname(file.name)
        self.file_start = file.name[len(self.dir_name) + 1:-4]

    def sample_batch(self, x):
        # Randomly select a batch of datapoints
        batch_idx = torch.randperm(x.shape[0])[:self.batch_size]
        x_batch = x[batch_idx, ...]  # [batch_size, x.shape[1]]

        # Mask of the properties that are missing
        mask_target = torch.isnan(x_batch[:, -self.n_properties:])

        return x_batch, mask_target

    def fit(self, x):
        self.optimiser.zero_grad()

        x_batch, mask_target = self.sample_batch(x)

        mu_y, var_y = self.network.forward(x_batch[:, :-self.n_properties]) # [batch_size, n_properties]

        target = x_batch[:, -self.n_properties:][~mask_target]
        mu_y = mu_y[:, -self.n_properties:][~mask_target]
        var_y = var_y[:, -self.n_properties:][~mask_target]

        loss = self.loss_function(target, mu_y, var_y, mask_target)
        loss.backward()
        self.optimiser.step()

        return loss

    def train_model(self, x, epochs, epoch_print_freq=5, means=None, stds=None):
        running_loss = 0
        self.standardised = False
        self.means = means
        self.stds = stds

        # Train the model
        for epoch in range(1, epochs+1):
            # Forward pass through the network + backprop.
            loss = self.fit(x)

            # Keep a track of the loss (averaged over the last epoch_print_freq epochs)
            running_loss += loss.item()
            if epoch % epoch_print_freq == 0:
                mean_loss = running_loss / epoch_print_freq
                running_loss = 0
                self.file.write('Epoch: %4d, Train loss = %8.3f \n' % (epoch, mean_loss))
                self.file.flush()
        return

    def metrics_calculator(self, x, save=False):
        mask = torch.isnan(x[:, -self.n_properties:])
        r2_scores = []
        mlls = []
        rmses = []

        for p in range(0, self.n_properties, 1):
            p_idx = torch.where(~mask[:, p])[0]
            x_p = x[p_idx]

            predict_mean, predict_var = self.network.forward(x_p[:, :-self.n_properties])
            predict_mean = predict_mean[:, p].reshape(-1).detach()
            predict_std = (predict_var[:, p] ** 0.5).reshape(-1).detach()

            target = x_p[:, (-self.n_properties + p)]

            if self.standardised:
                predict_mean = (predict_mean.numpy() * self.stds[p] +
                                self.means[p])
                predict_std = predict_std.numpy() * self.stds[p]
                target = (target.numpy() * self.stds[p] +
                          self.means[p])
            else:
                predict_mean = predict_mean.numpy()
                predict_std = predict_std.numpy()
                target = target.numpy()

            r2_scores.append(r2_score(target, predict_mean))
            mlls.append(mll(predict_mean, predict_std ** 2, target))
            rmses.append(np.sqrt(mean_squared_error(target, predict_mean)))

            if save:
                path_to_save = self.dir_name + '/predictions/' + self.file_start + '_' + str(p)
                np.save(path_to_save + '_mean.npy', predict_mean)
                np.save(path_to_save + '_std.npy', predict_std)
                np.save(path_to_save + '_target.npy', target)

        return r2_scores, mlls, rmses


class ClassificationWrapper:
    def __init__(self, network, batch_size, lr, file, use_properties=True):
        self.lr = lr
        self.batch_size = batch_size
        self.use_properties = use_properties
        self.network = network
        self.n_properties = self.network.out_dim
        self.optimiser = optim.Adam(self.network.parameters(), self.lr)
        self.loss_function = nn.BCELoss()
        self.file = file
        self.dir_name = os.path.dirname(file.name)
        self.file_start = file.name[len(self.dir_name) + 1:-4]

    def sample_batch(self, x):
        # Randomly select a batch of datapoints
        batch_idx = torch.randperm(x.shape[0])[:self.batch_size]
        x_batch = x[batch_idx, ...]  # [batch_size, x.shape[1]]

        # Mask of the properties that are missing
        mask_target = torch.isnan(x_batch[:, -self.n_properties:])

        # To form the context mask we will add properties to the missing values
        mask_context = copy.deepcopy(mask_target)
        batch_properties = [torch.where(~mask_target[i, ...])[0] for i in
                            range(mask_target.shape[0])]

        for i, properties in enumerate(batch_properties):
            ps = np.random.choice(properties.numpy(),
                                  size=np.random.randint(low=0, high=properties.shape[0] + 1),
                                  replace=False)

            # add property to those being masked
            mask_context[i, ps] = True

        return x_batch, mask_target, mask_context

    def fit(self, x):

        self.optimiser.zero_grad()

        x_batch, mask_target, mask_context = self.sample_batch(x)

        input_batch = copy.deepcopy(x_batch)

        # Fill in the NaN values with mean for each column.
        if self.standardised:
            input_batch[:, -self.n_properties:][mask_context] = 0.0
        else:
            input_batch[:, -self.n_properties:][mask_context] = torch.take(self.means, torch.where(mask_context)[1])

        mu_y = self.network.forward(input_batch) # [batch_size, n_properties]

        target = x_batch[:, -self.n_properties:][~mask_target]
        mu_y = mu_y[:, -self.n_properties:][~mask_target]

        loss = self.loss_function(mu_y, target)

        loss.backward()
        self.optimiser.step()

        return loss

    def train_model(self, x, epochs, epoch_print_freq=5, means=None, stds=None):
        running_loss = 0
        self.standardised = False
        self.means = means
        self.stds = stds

        # Train the model
        for epoch in range(1, epochs+1):
            # Forward pass through the network + backprop.
            loss = self.fit(x)

            # Keep a track of the loss (averaged over the last epoch_print_freq epochs)
            running_loss += loss.item()
            if epoch % epoch_print_freq == 0:
                mean_loss = running_loss / epoch_print_freq
                running_loss = 0
                self.file.write('Epoch: %4d, Train loss = %8.3f \n' % (epoch, mean_loss))
                self.file.flush()
        return

    def predict(self, x, save=False, means=None, stds=None):
        mask = torch.isnan(x[:, -self.n_properties:])

        self.standardised = False
        self.means = means
        self.stds = stds

        if self.standardised:
            x[:, -self.n_properties:][mask] = 0.0
        else:
            x[:, -self.n_properties:][mask] = torch.take(self.means, torch.where(mask)[1])

        predict_mean = self.network.forward(x) #[n_molecules, n_targets]

        if save:
            path_to_save = self.dir_name + '/predictions/' + self.file_start + '_' + str(p)


            if self.standardised:
                predict_mean = (predict_mean.numpy() * self.stds +
                                self.means)
            else:
                predict_mean = predict_mean.numpy()

            np.save(path_to_save + '_mean.npy', predict_mean)

        return predict_mean

    def metrics_calculator(self, x, save=False):
        mask = torch.isnan(x[:, -self.n_properties:])
        r2_scores = []
        rmses = []
        roc_aucs = []


        for p in range(0, self.n_properties, 1):
            p_idx = torch.where(~mask[:, p])[0]
            x_p = x[p_idx]

            input_p = copy.deepcopy(x_p)
            #input_p[:, -self.n_properties:] -= self.means
            #input_p[:, -self.n_properties:][mask[p_idx]] = 0.0
            #input_p[:, (-self.n_properties + p)] = 0.0
            if self.standardised:
                input_p[:, -self.n_properties:][mask[p_idx]] = 0.0
                input_p[:, (-self.n_properties + p)] = 0.0
            else:
                input_p[:, -self.n_properties:][mask[p_idx]] = torch.take(self.means, torch.where(mask[p_idx])[1])
                input_p[:, (-self.n_properties + p)] = self.means[p]

            mask_p = torch.zeros_like(mask[p_idx, :]).fill_(True)
            mask_p[:, p] = False

            predict_mean = self.network.forward(input_p)
            predict_mean = predict_mean[:, p].reshape(-1).detach()

            target = x_p[:, (-self.n_properties + p)]

            if self.standardised:
                predict_mean = (predict_mean.numpy() * self.stds[p] +
                                self.means[p])
                target = (target.numpy() * self.stds[p] +
                          self.means[p])
            else:
                predict_mean = predict_mean.numpy()
                target = target.numpy()

            r2_scores.append(r2_score(target, predict_mean))
            rmses.append(np.sqrt(mean_squared_error(target, predict_mean)))

            if save:
                path_to_save = self.dir_name + '/predictions/' + self.file_start + '_' + str(p)
                np.save(path_to_save + '_mean.npy', predict_mean)
                np.save(path_to_save + '_target.npy', target)

            try:
                score = roc_auc_score(target, predict_mean)
                roc_aucs.append(score)
                """
                if score > 0.75:
                    print('Target: {} \t ROC-AUC: {}'.format(p, score))
                    precision, recall, thresholds = precision_recall_curve(target, predict_mean)
                    auc_prc = auc(recall, precision)
                    print('Precision: {}'.format(precision[0]))
                    print('Recall: {}'.format(recall[0]))
                    print('AUC-PRC: {}\\n'.format(auc_prc))
                """
            except:
                continue

        return r2_scores, rmses, roc_aucs


class RegressionWrapper:
    def __init__(self, network, batch_size, lr, file):
        self.lr = lr
        self.batch_size = batch_size
        self.network = network
        self.n_properties = self.network.out_dim
        self.optimiser = optim.Adam(self.network.parameters(), self.lr)
        self.loss_function = negative_log_likelihood
        self.file = file
        self.dir_name = os.path.dirname(file.name)
        self.file_start = file.name[len(self.dir_name) + 1:-4]

    def sample_batch(self, x):

        # Randomly select a batch of datapoints
        batch_idx = torch.randperm(x.shape[0])[:self.batch_size]
        x_batch = x[batch_idx, ...]  # [batch_size, x.shape[1]]

        # Mask of the properties that are missing
        mask_target = torch.isnan(x_batch[:, -self.n_properties:])

        # To form the context mask we will add properties to the missing values
        mask_context = copy.deepcopy(mask_target)
        batch_properties = [torch.where(~mask_target[i, ...])[0] for i in
                            range(mask_target.shape[0])]

        for i, properties in enumerate(batch_properties):
            ps = np.random.choice(properties.numpy(),
                                  size=np.random.randint(low=0, high=properties.shape[0] + 1),
                                  replace=False)

            # add property to those being masked
            mask_context[i, ps] = True

        return x_batch, mask_target, mask_context

    def fit(self, x):
        self.optimiser.zero_grad()

        x_batch, mask_target, mask_context = self.sample_batch(x)

        input_batch = copy.deepcopy(x_batch)

        #input_batch[:, -self.n_properties:] -= self.means
        #input_batch[:, -self.n_properties:][mask_context] = 0.0

        # Fill in the NaN values with mean for each column.
        if self.standardised:
            input_batch[:, -self.n_properties:][mask_context] = 0.0
        else:
            input_batch[:, -self.n_properties:][mask_context] = torch.take(self.means, torch.where(mask_context)[1])

        mu_y, var_y = self.network.forward(input_batch) # [batch_size, n_properties]

        target = x_batch[:, -self.n_properties:][~mask_target]
        mu_y = mu_y[:, -self.n_properties:][~mask_target]
        var_y = var_y[:, -self.n_properties:][~mask_target]

        loss = self.loss_function(target, mu_y, var_y, mask_target)
        loss.backward()
        self.optimiser.step()

        return loss

    def train_model(self, x, epochs, epoch_print_freq=5, means=None, stds=None):
        running_loss = 0
        self.standardised = False
        self.means = means
        self.stds = stds

        # Train the model
        for epoch in range(1, epochs+1):
            # Forward pass through the network + backprop.
            loss = self.fit(x)

            # Keep a track of the loss (averaged over the last epoch_print_freq epochs)
            running_loss += loss.item()
            if epoch % epoch_print_freq == 0:
                mean_loss = running_loss / epoch_print_freq
                running_loss = 0
                self.file.write('Epoch: %4d, Train loss = %8.3f \n' % (epoch, mean_loss))
                self.file.flush()
        return

    def metrics_calculator(self, x, save=False):
        mask = torch.isnan(x[:, -self.n_properties:])
        r2_scores = []
        mlls = []
        rmses = []

        for p in range(0, self.n_properties, 1):
            p_idx = torch.where(~mask[:, p])[0]
            x_p = x[p_idx]

            input_p = copy.deepcopy(x_p)

            if self.standardised:
                input_p[:, -self.n_properties:][mask[p_idx]] = 0.0
                input_p[:, (-self.n_properties + p)] = 0.0
            else:
                input_p[:, -self.n_properties:][mask[p_idx]] = torch.take(self.means, torch.where(mask[p_idx])[1])
                input_p[:, (-self.n_properties + p)] = self.means[p]

            mask_p = torch.zeros_like(mask[p_idx, :]).fill_(True)
            mask_p[:, p] = False

            predict_mean, predict_var = self.network.forward(input_p)
            predict_mean = predict_mean[:, p].reshape(-1).detach()
            predict_std = (predict_var[:, p] ** 0.5).reshape(-1).detach()

            target = x_p[:, (-self.n_properties + p)]

            if self.standardised:
                predict_mean = (predict_mean.numpy() * self.stds[p] +
                                self.means[p])
                predict_std = predict_std.numpy() * self.stds[p]
                target = (target.numpy() * self.stds[p] +
                          self.means[p])
            else:
                predict_mean = predict_mean.numpy()
                predict_std = predict_std.numpy()
                target = target.numpy()

            r2_scores.append(r2_score(target, predict_mean))
            mlls.append(mll(predict_mean, predict_std ** 2, target))
            rmses.append(np.sqrt(mean_squared_error(target, predict_mean)))

            if save:
                path_to_save = self.dir_name + '/predictions/' + self.file_start + '_' + str(p)
                np.save(path_to_save + '_mean.npy', predict_mean)
                np.save(path_to_save + '_std.npy', predict_std)
                np.save(path_to_save + '_target.npy', target)

        return r2_scores, mlls, rmses