import math

import numpy as np
import torch

from torch import nn, einsum
from torch.autograd import Variable

from model.ms_tcn import MultiScale_TemporalConv as mstcn
from model.port import MORT
from einops import rearrange, repeat

from utils import set_parameter_requires_grad

from modules import import_class, bn_init, TCN_GCN_unit, TCN_aGCN_unit, TCN_attn_unit


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

class ModelwATTN(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, in_channels=3,
                 drop_out=0, adaptive=True, num_set=3):
        super(ModelwATTN, self).__init__()

        A = np.stack([np.eye(num_point)] * num_set, axis=0)
        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        base_channel = 64

        self.l1 = TCN_attn_unit(in_channels, base_channel, A, residual=False, adaptive=adaptive)
        self.l2 = TCN_attn_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l3 = TCN_attn_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l4 = TCN_attn_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l5 = TCN_attn_unit(base_channel, base_channel*2, A, stride=2, adaptive=adaptive)
        self.l6 = TCN_attn_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l7 = TCN_attn_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l8 = TCN_attn_unit(base_channel*2, base_channel*4, A, stride=2, adaptive=adaptive)
        self.l9 = TCN_attn_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)
        self.l10 = TCN_attn_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)
        self.fc = nn.Linear(base_channel*4, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x):
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V *  C, T)
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

class ModelwA(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, in_channels=3,
                 drop_out=0, adaptive=True, num_set=3):
        super(ModelwA, self).__init__()

        A = np.stack([np.eye(num_point)] * num_set, axis=0)


        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        base_channel = 64

        self.l1 = TCN_aGCN_unit(in_channels, base_channel, A, residual=False, adaptive=adaptive)
        self.l2 = TCN_aGCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l3 = TCN_aGCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l4 = TCN_aGCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l5 = TCN_aGCN_unit(base_channel, base_channel*2, A, stride=2, adaptive=adaptive)
        self.l6 = TCN_aGCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l7 = TCN_aGCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l8 = TCN_aGCN_unit(base_channel*2, base_channel*4, A, stride=2, adaptive=adaptive)
        self.l9 = TCN_aGCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)
        self.l10 = TCN_aGCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)
        self.fc = nn.Linear(base_channel*4, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x):
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V *  C, T)
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


class ModelwP(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, in_channels=3,
                 drop_out=0, adaptive=True,
                 embd_dim=64, n_layers=6, n_heads=8, pretrain_weight=None, freeze_port=True):
        super(ModelwP, self).__init__()

        A = np.stack([np.eye(num_point)] * 3, axis=0)
        self.embd_dim = embd_dim
        self.freeze_port = freeze_port
        self.n_heads = n_heads
        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * embd_dim* num_point)
        base_channel = 64

        self.l1 = TCN_aGCN_unit(embd_dim, base_channel, A, residual=False, adaptive=adaptive)
        self.l2 = TCN_aGCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l3 = TCN_aGCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l4 = TCN_aGCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l5 = TCN_aGCN_unit(base_channel, base_channel*2, A, stride=2, adaptive=adaptive)
        self.l6 = TCN_aGCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l7 = TCN_aGCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l8 = TCN_aGCN_unit(base_channel*2, base_channel*4, A, stride=2, adaptive=adaptive)
        self.l9 = TCN_aGCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)
        self.l10 = TCN_aGCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)
        self.fc = nn.Linear(base_channel*4, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

        self.port= MORT(
            channels=in_channels,
            dim=embd_dim,
            mlp_dim=embd_dim*2,
            depth=n_layers,
            heads=n_heads,
            dropout=drop_out,
            emb_dropout=drop_out
        )

        if pretrain_weight is not None:
            port_pretrained = torch.load(pretrain_weight)
            self.port.load_state_dict(port_pretrained)

        if freeze_port:
            set_parameter_requires_grad(self.port, feature_extracting=True)
        self.layernorm = nn.LayerNorm(embd_dim, eps=1e-12)


    def forward(self, x):
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 2, 3, 1).contiguous().view(N*M*T, V, C)

        if self.freeze_port:
            self.port.eval()

        embd, attns, hidden_states = self.port(x)
        output = torch.stack(hidden_states, dim=0).sum(dim=0)
        joint_embd = self.layernorm(output)
        joint_embd = joint_embd.view(N, M, T, V, self.embd_dim).permute(0,1,3,4,2).contiguous()
        joint_embd = joint_embd.view(N, M*V*self.embd_dim, T)
        x = self.data_bn(joint_embd)
        x = x.view(N*M, V, self.embd_dim, T).permute(0, 2, 3, 1).contiguous()
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


class ModelwR(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, in_channels=3,
                 drop_out=0, adaptive=True, num_set=3):
        super(ModelwR, self).__init__()

        A = np.stack([np.eye(num_point)] * num_set, axis=0)
        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        self.trans_mat = nn.Parameter(torch.eye(num_point))
        base_channel = 64

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
        self.fc = nn.Linear(base_channel*4, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x):
        N, C, T, V, M = x.size()
        x = rearrange(x, 'n c t v m -> (n m t) v c')
        x = self.trans_mat @ x
        x = rearrange(x, '(n m t) v c -> n (m v c) t', n=N, m=M, t=T)
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

class ModelwVAE(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, in_channels=3,
                 drop_out=0, adaptive=True, num_set=3):
        super(ModelwVAE, self).__init__()

        A = np.stack([np.eye(num_point)] * num_set, axis=0)
        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        base_channel = 64

        self.l1 = TCN_aGCN_unit(in_channels, base_channel, A, residual=False, adaptive=adaptive)
        self.l2 = TCN_aGCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l3 = TCN_aGCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l4 = TCN_aGCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l5 = TCN_aGCN_unit(base_channel, base_channel*2, A, stride=2, adaptive=adaptive)
        self.l6 = TCN_aGCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l7 = TCN_aGCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l8 = TCN_aGCN_unit(base_channel*2, base_channel*4, A, stride=2, adaptive=adaptive)
        self.l9 = TCN_aGCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)
        self.l10 = TCN_aGCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)
        self.fc = nn.Linear(base_channel*4, base_channel*4)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x
        self.fc_mu = nn.Linear(base_channel*4, base_channel*4)
        self.fc_logvar = nn.Linear(base_channel*4, base_channel*4)
        self.decoder = nn.Linear(base_channel*4, num_class)

    def latent_sample(self, mu, logvar):
        if self.training:
            # the reparameterization trick
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

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
        x = self.fc(x)

        latent_mu = self.fc_mu(x)
        latent_logvar = self.fc_logvar(x)
        latent = self.latent_sample(latent_mu, latent_logvar)

        y = self.decoder(latent)
        return y


class ModelTest(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, in_channels=3,
                 drop_out=0, adaptive=True, num_set=3):
        super(Model, self).__init__()

        Graph = import_class(graph)
        self.graph = Graph()

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

