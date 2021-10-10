import math

import numpy as np
import torch
import torch.nn.functional as F

from torch import nn, einsum
from torch.autograd import Variable
from torch import linalg as LA

from model.ms_tcn import MultiScale_TemporalConv as MS_TCN
from model.ms_gcn import MultiScale_GraphConv as MS_GCN
from model.ms_gcn import MultiHead_GraphConv as MH_GCN
from model.port import MORT
from einops import rearrange, repeat

from utils import set_parameter_requires_grad, get_vector_property

from model.modules import import_class, bn_init, TCN_GCN_unit, TCN_aGCN_unit, TCN_attn_unit


class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, in_channels=3,
                 drop_out=0, adaptive=True, num_set=3):
        super(Model, self).__init__()

        if graph is not None:
            Graph = import_class(graph)
            self.graph = Graph()
            A = self.graph.A # 3,25,25
        else:
            A = np.stack([np.eye(num_point)] * num_set, axis=0)

        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        base_channel = 64

        self.l1 = TCN_GCN_unit(in_channels, base_channel, A, residual=False, adaptive=adaptive)
        self.l2 = TCN_GCN_unit(base_channel, base_channel,A, adaptive=adaptive)
        self.l3 = TCN_GCN_unit(base_channel, base_channel,A, adaptive=adaptive)
        self.l4 = TCN_GCN_unit(base_channel, base_channel,A, adaptive=adaptive)
        self.l5 = TCN_GCN_unit(base_channel, base_channel*2, A, stride=2, adaptive=adaptive)
        self.l6 = TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l7 = TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l8 = TCN_GCN_unit(base_channel*2, base_channel*4, A, stride=2, adaptive=adaptive)
        self.l9 = TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)
        self.l10 = TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)
        self.fc = nn.Linear(base_channel*4, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x):
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        x = self.drop_out(x)

        return self.fc(x)


class ModelwMMD(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, in_channels=3,
                 drop_out=0, adaptive=True, num_set=3, noise_ratio=0.1):
        super(ModelwMMD, self).__init__()

        if graph is not None:
            Graph = import_class(graph)
            self.graph = Graph()
            A = self.graph.A # 3,25,25
        else:
            A = np.stack([np.eye(num_point)] * num_set, axis=0)

        base_channel = 64
        self.num_class = num_class
        self.num_point = num_point
        self.noise_ratio = noise_ratio
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        self.z_prior = nn.Parameter(torch.rand(num_class, base_channel*4))

        self.l1 = TCN_GCN_unit(in_channels, base_channel, A, residual=False, adaptive=adaptive)
        self.l2 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l3 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l4 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l5 = TCN_GCN_unit(base_channel, base_channel*2, A, stride=2, adaptive=adaptive)
        self.l6 = TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l7 = TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l8 = TCN_GCN_unit(base_channel*2, base_channel*4, A, stride=2, adaptive=adaptive)
        self.l9 = TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)
        self.l10 = TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x
        self.fc_mu = nn.Linear(base_channel*4, base_channel*4)
        self.fc_logvar = nn.Linear(base_channel*4, base_channel*4)
        self.decoder = nn.Linear(base_channel*4, num_class)

        nn.init.normal_(self.z_prior, 0, 1)
        # nn.init.orthogonal_(self.z_prior)
        nn.init.normal_(self.fc_mu.weight, 0, math.sqrt(2. / num_class))
        nn.init.normal_(self.fc_logvar.weight, 0, math.sqrt(2. / num_class))
        nn.init.normal_(self.decoder.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)

    def latent_sample(self, mu, logvar):
        if self.training:
            # the reparameterization trick
            std = logvar.mul(self.noise_ratio).exp()
            std = torch.clamp(std, max=100)
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add(mu)
        else:
            return mu

    def get_loss(self, z, y):
        y_valid = [i_cls in y for i_cls in range(self.num_class)]
        z_mean = torch.stack([z[y==i_cls].mean(dim=0) for i_cls in range(self.num_class)], dim=0)
        l2_z_mean= LA.norm(z.mean(dim=0), ord=2)
        mmd_loss = F.mse_loss(z_mean[y_valid], self.z_prior[y_valid])
        return mmd_loss, l2_z_mean, z_mean[y_valid]


    def forward(self, x, y):

        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        x = self.drop_out(x)

        z_mu = self.fc_mu(x)
        z_logvar = self.fc_logvar(x)
        z = self.latent_sample(z_mu, z_logvar)

        y_hat = self.decoder(z)
        mmd_loss, l2_z_mean, z_mean = self.get_loss(z, y)

        return y_hat, mmd_loss, l2_z_mean, z_mean
