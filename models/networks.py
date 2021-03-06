import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb
# BASIC NETWORKS

class BinaryClassificationNN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims, non_linearity=F.relu):
        """
        :param in_dim: (int) Dimensionality of the input.
        :param out_dim: (int) Dimensionality of the target for which a distribution is being obtained.
        :param hidden_dims: (list of ints) Architecture of the network.
        :param non_linearity: Non-linear activation function to apply after each linear transformation,
                                e.g. relu or tanh.
        :param initial_sigma:
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.network = VanillaNN(in_dim, out_dim, hidden_dims, non_linearity)

    def forward(self, x):
        """

        :param x: x: (torch tensor, (batch_size, in_dim)) Input to the network.
        :return:
        """
        assert len(x.shape) == 2, 'Input must be of shape [batch_size, in_dim].'
        out = self.network(x)

        return F.sigmoid(out)


class VanillaNN(nn.Module):
    """
    A `vanilla` neural network.
    """

    def __init__(self, in_dim, out_dim, hidden_dims, non_linearity=F.relu):
        """
        :param in_dim: (int) Dimensionality of the input.
        :param out_dim: (int) Dimensionality of the output.
        :param hidden_dims: (list of ints) Architecture of the network.
        :param non_linearity: Non-linear activation function to apply after each linear transformation,
                                e.g. relu or tanh.
        """
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.non_linearity = non_linearity

        self.layers = nn.ModuleList()

        for dim in range(len(hidden_dims) + 1):
            if dim == 0:
                self.layers.append(nn.Linear(self.in_dim, hidden_dims[dim]))
            elif dim == len(hidden_dims):
                self.layers.append(nn.Linear(hidden_dims[-1], self.out_dim))
            else:
                self.layers.append(nn.Linear(hidden_dims[dim - 1],
                                             hidden_dims[dim]))

    def forward(self, x):
        """
        :param self:
        :param x: (torch tensor, (batch_size, in_dim)) Input to the network.
        :return: (torch tensor, (batch_size, out_dim)) Output of the network.
        """
        assert len(x.shape) == 2, 'Input must be of shape [batch_size, in_dim].'

        for i in range(len(self.layers) - 1):
            x = self.non_linearity(self.layers[i](x))

        return self.layers[-1](x)


class ProbabilisticVanillaNN(nn.Module):
    """
    A `vanilla' NN whose output is the natural parameters of a normal distribution over y (as opposed to a point
    estimate of y). Variance is fixed.
    """

    def __init__(self, in_dim, out_dim, hidden_dims, non_linearity=F.relu, min_var=0.01,
                 initial_sigma=None, restrict_var=True):
        """
        :param in_dim: (int) Dimensionality of the input.
        :param out_dim: (int) Dimensionality of the target for which a distribution is being obtained.
        :param hidden_dims: (list of ints) Architecture of the network.
        :param non_linearity: Non-linear activation function to apply after each linear transformation,
                                e.g. relu or tanh.
        :param initial_sigma:
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.min_var = min_var
        self.network = VanillaNN(in_dim, 2 * out_dim, hidden_dims, non_linearity)
        self.restrict_var = restrict_var

        if initial_sigma is not None:
            self.network.layers[-1].bias.data = torch.cat([
                1e-6 * torch.randn(out_dim),
                np.log(np.exp(initial_sigma ** 0.5) - 1)
                + 1e-6 * torch.randn(out_dim)])

    def forward(self, x):
        """

        :param x: x: (torch tensor, (batch_size, in_dim)) Input to the network.
        :return:
        """
        assert len(x.shape) == 2, 'Input must be of shape [batch_size, in_dim].'

        out = self.network(x)
        mu = out[:, :self.out_dim]
        if self.restrict_var:
            var = self.min_var + (0.1 - self.min_var) * F.sigmoid(out[:, self.out_dim:])
        else:
            var = self.min_var + (1.0 - self.min_var) * F.softplus(out[:, self.out_dim:])

        return mu, var

class MultiProbabilisticVanillaNN(nn.Module):
    """

    """
    def __init__(self, in_dim, out_dim, hidden_dims, n_properties,
                 non_linearity=F.tanh, min_var=0.01, restrict_var=False):
        """
        :param input_size: An integer describing the dimensionality of the input, in this case
                           r_size, (the dimensionality of the embedding r)
        :param output_size: An integer describing the dimensionality of the output, in this case
                            output_size = x_size
        :param decoder_n_hidden: An integer describing the number of hidden layers in the neural
                                 network
        :param decoder_hidden_size: An integer describing the number of nodes in each layer of the
                                    neural network
        """

        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dims = hidden_dims
        self.n_properties = n_properties
        self.min_var = min_var
        self.non_linearity = non_linearity
        self.restrict_var = restrict_var

        self.network = nn.ModuleList()

        for i in range(self.n_properties):
            self.network.append(ProbabilisticVanillaNN(self.in_dim, self.out_dim,
                                                       self.hidden_dims, non_linearity=self.non_linearity,
                                                       min_var=self.min_var, restrict_var=self.restrict_var))

    def forward(self, x, mask):
        """

        :param mask: Torch matrix showing which of the
        :return:
        """
        mus_y = []
        vars_y = []
        for p in range(self.n_properties):
            mu_y, var_y = self.network[p].forward(x[~mask[:, p]])
            mus_y.append(mu_y)
            vars_y.append(var_y)

        return mus_y, vars_y

