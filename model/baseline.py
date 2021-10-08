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

from utils import set_parameter_requires_grad

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
                 drop_out=0, adaptive=True, num_set=3, noise_ratio=0.1):
        super(ModelwVAE, self).__init__()

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
        self.decoder = nn.Sequential(
            nn.Linear(base_channel*4, base_channel*2),
            nn.BatchNorm1d(base_channel*2),
            nn.ReLU(),
            nn.Linear(base_channel*2, num_class)
        )

        nn.init.normal_(self.z_prior, 0, 0.1)
        nn.init.normal_(self.fc_mu.weight, 0, math.sqrt(2. / num_class))
        nn.init.normal_(self.fc_logvar.weight, 0, math.sqrt(2. / num_class))
        # nn.init.normal_(self.decoder[0].weight, 0, math.sqrt(2. / num_class))
        # nn.init.normal_(self.decoder[3].weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)

    def latent_sample(self, mu, logvar):
        if self.training:
            std = torch.exp(self.noise_ratio * logvar)
            eps = torch.randn_like(std).normal_()
            return eps * std + mu
        else:
            return mu

    def mmd_loss(self, z, y, lam=0.1):
        y_valid = [i_cls in y for i_cls in range(self.num_class)]
        z_mean = torch.stack([z[y==i_cls].mean(dim=0) for i_cls in range(self.num_class)], dim=0)
        l2norm = LA.norm(z_mean[y_valid], ord=2, dim=1).mean()
        loss = F.mse_loss(z_mean[y_valid], self.z_prior[y_valid]) + lam * l2norm
        return loss


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

        import ipdb; ipdb.set_trace()
        z_mu = self.fc_mu(x)
        z_logvar = self.fc_logvar(x)
        z = self.latent_sample(z_mu, z_logvar)

        y_hat = self.decoder(z)
        mmd_loss = self.mmd_loss(z, y)
        return y_hat, mmd_loss


class MSG3D(nn.Module):
    def __init__(self,
                 num_class,
                 num_point,
                 num_person,
                 num_gcn_scales,
                 graph,
                 in_channels=3):
        super(MSG3D, self).__init__()

        Graph = import_class(graph)
        A_binary = Graph().A_binary + np.eye(num_point)

        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        # channels
        c1 = 96
        c2 = c1 * 2     # 192
        c3 = c2 * 2     # 384

        # r=3 STGC blocks
        self.sgcn1 = nn.Sequential(
            MS_GCN(num_gcn_scales, 3, c1, A_binary, disentangled_agg=True),
            MS_TCN(c1, c1),
            MS_TCN(c1, c1))
        self.sgcn1[-1].act = nn.Identity()
        self.tcn1 = MS_TCN(c1, c1)

        self.sgcn2 = nn.Sequential(
            MS_GCN(num_gcn_scales, c1, c1, A_binary, disentangled_agg=True),
            MS_TCN(c1, c2, stride=2),
            MS_TCN(c2, c2))
        self.sgcn2[-1].act = nn.Identity()
        self.tcn2 = MS_TCN(c2, c2)

        self.sgcn3 = nn.Sequential(
            MS_GCN(num_gcn_scales, c2, c2, A_binary, disentangled_agg=True),
            MS_TCN(c2, c3, stride=2),
            MS_TCN(c3, c3))
        self.sgcn3[-1].act = nn.Identity()
        self.tcn3 = MS_TCN(c3, c3)

        self.fc = nn.Linear(c3, num_class)

    def forward(self, x):
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N * M, V, C, T).permute(0,2,3,1).contiguous()

        # Apply activation to the sum of the pathways
        x = F.relu(self.sgcn1(x), inplace=True)
        x = self.tcn1(x)

        x = F.relu(self.sgcn2(x), inplace=True)
        x = self.tcn2(x)

        x = F.relu(self.sgcn3(x), inplace=True)
        x = self.tcn3(x)

        out = x
        out_channels = out.size(1)
        out = out.view(N, M, out_channels, -1)
        out = out.mean(3)   # Global Average Pooling (Spatial+Temporal)
        out = out.mean(1)   # Average pool number of bodies in the sequence

        out = self.fc(out)
        return out


class Test1(nn.Module):
    def __init__(self,
                 num_class,
                 num_point,
                 num_person,
                 graph,
                 use_bone,
                 in_channels=3):
        super(Test1, self).__init__()

        Graph = import_class(graph)()

        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        # channels
        c1 = 96
        c2 = c1 * 2     # 192
        c3 = c2 * 2     # 384

        # r=3 STGC blocks
        self.sgcn1 = nn.Sequential(
            MH_GCN(3, c1, Graph, bone=use_bone),
            MS_TCN(c1, c1),
            MS_TCN(c1, c1))
        self.sgcn1[-1].act = nn.Identity()
        self.tcn1 = MS_TCN(c1, c1)

        self.sgcn2 = nn.Sequential(
            MH_GCN(c1, c1, Graph, bone=use_bone),
            MS_TCN(c1, c2, stride=2),
            MS_TCN(c2, c2))
        self.sgcn2[-1].act = nn.Identity()
        self.tcn2 = MS_TCN(c2, c2)

        self.sgcn3 = nn.Sequential(
            MH_GCN(c2, c2, Graph, bone=use_bone),
            MS_TCN(c2, c3, stride=2),
            MS_TCN(c3, c3))
        self.sgcn3[-1].act = nn.Identity()
        self.tcn3 = MS_TCN(c3, c3)

        self.fc = nn.Linear(c3, num_class)

    def forward(self, x):
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N * M, V, C, T).permute(0,2,3,1).contiguous()

        # Apply activation to the sum of the pathways
        x = F.relu(self.sgcn1(x), inplace=True)
        x = self.tcn1(x)

        x = F.relu(self.sgcn2(x), inplace=True)
        x = self.tcn2(x)

        x = F.relu(self.sgcn3(x), inplace=True)
        x = self.tcn3(x)

        out = x
        out_channels = out.size(1)
        out = out.view(N, M, out_channels, -1)
        out = out.mean(3)   # Global Average Pooling (Spatial+Temporal)
        out = out.mean(1)   # Average pool number of bodies in the sequence

        out = self.fc(out)
        return out


class Test2(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, in_channels=3,
                 drop_out=0, adaptive=False, num_set=3):
        super(Test2, self).__init__()

        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        A = self.get_A(graph, 0)
        base_channel = 64

        self.l1 = TCN_GCN_unit(in_channels, base_channel, self.get_A(graph, 0), residual=False, adaptive=adaptive)
        self.l2 = TCN_GCN_unit(base_channel, base_channel, self.get_A(graph, 1), adaptive=adaptive)
        self.l3 = TCN_GCN_unit(base_channel, base_channel, self.get_A(graph, 2), adaptive=adaptive)
        self.l4 = TCN_GCN_unit(base_channel, base_channel, self.get_A(graph, 3), adaptive=adaptive)
        self.l5 = TCN_GCN_unit(base_channel, base_channel*2, self.get_A(graph, 3), stride=2, adaptive=adaptive)
        self.l6 = TCN_GCN_unit(base_channel*2, base_channel*2, self.get_A(graph, 4), adaptive=adaptive)
        self.l7 = TCN_GCN_unit(base_channel*2, base_channel*2, self.get_A(graph, 5), adaptive=adaptive)
        self.l8 = TCN_GCN_unit(base_channel*2, base_channel*4, self.get_A(graph, 5), stride=2, adaptive=adaptive)
        self.l9 = TCN_GCN_unit(base_channel*4, base_channel*4, self.get_A(graph, 6), adaptive=adaptive)
        self.l10 = TCN_GCN_unit(base_channel*4, base_channel*4, self.get_A(graph, 7), adaptive=adaptive)
        self.fc = nn.Linear(base_channel*4, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def get_A(self, graph, i):
        Graph = import_class(graph)()
        A_outward = Graph.A_outward_binary
        A_binary = Graph.A_binary
        I = np.eye(Graph.num_node)
        left = A_binary@(I - np.linalg.matrix_power(A_outward, 7-i))
        right = A_binary@(I - np.linalg.matrix_power(A_outward.T, 7-i))
        return np.stack([left, A_binary, right])

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

