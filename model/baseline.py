import math

import numpy as np
import torch

from torch import nn, einsum
from torch.autograd import Variable

from model.ms_tcn import MultiScale_TemporalConv as mstcn
from model.port import MORT
from einops import rearrange, repeat

from utils import set_parameter_requires_grad


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)

def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x

class SelfAttention(nn.Module):
    def __init__(self, in_channels, hidden_dim, n_heads):
        super(SelfAttention, self).__init__()
        self.scale = hidden_dim ** -0.5
        inner_dim = hidden_dim * n_heads
        self.to_qk = nn.Conv2d(in_channels, inner_dim*2, 1)
        self.n_heads = n_heads

    def forward(self, x):
        y = self.to_qk(x)
        qk = y.chunk(2, dim=1)
        q, k = map(lambda t: rearrange(t, 'b (h d) t v -> (b t) h v d', h=self.n_heads), qk)

        # attention
        dots = einsum('b h i d, b h j d -> b h i j', q, k)*self.scale
        attn = dots.softmax(dim=-1).float()
        return attn

class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, adaptive=True):
        super(unit_gcn, self).__init__()
        self.out_c = out_channels
        self.in_c = in_channels
        self.num_subset = A.shape[0]
        self.adaptive = adaptive
        if adaptive:
            self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)), requires_grad=True)
        else:
            self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)

        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)

    def L2_norm(self, A):
        # A:N,V,V
        A_norm = torch.norm(A, 2, dim=1, keepdim=True) + 1e-4  # N,1,V
        A = A / A_norm
        return A

    def forward(self, x):
        N, C, T, V = x.size()

        y = None
        if self.adaptive:
            A = self.PA
            A = self.L2_norm(A)
        else:
            A = self.A.cuda(x.get_device())
        for i in range(self.num_subset):

            A1 = A[i]
            A2 = x.view(N, C * T, V)
            z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)

        return y

class unit_agcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, adaptive=True, use_port=False):
        super(unit_agcn, self).__init__()
        self.out_c = out_channels
        self.in_c = in_channels
        self.num_subset = A.shape[0]
        self.adaptive = adaptive
        if adaptive:
            self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)), requires_grad=True)
        else:
            self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)

        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.gelu = nn.GELU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)

        if in_channels == 3:
            rel_channels = 8
        else:
            rel_channels = in_channels //  8
        self.attn = SelfAttention(in_channels, rel_channels, self.num_subset)


    def forward(self, x, attn=None):
        N, C, T, V = x.size()

        y = None
        if self.adaptive:
            A = self.PA
        else:
            A = self.A.cuda(x.get_device())

        if attn is None:
            attn = self.attn(x)
        A = A.unsqueeze(0)*attn
        for i in range(self.num_subset):
            A1 = A[:, i, :, :] # (nt)vv
            A2 = rearrange(x, 'n c t v -> (n t) v c')
            z = A1@A2
            z = rearrange(z, '(n t) v c-> n c t v', t=T).contiguous()
            z = self.conv_d[i](z)
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        y = self.gelu(y)

        return y


class unit_attn(nn.Module):
    def __init__(self, in_channels, out_channels, A, adaptive=True, dropout=0):
        super(unit_attn, self).__init__()
        self.out_c = out_channels
        self.in_c = in_channels
        self.n_heads= A.shape[0]
        self.adaptive = adaptive
        # if adaptive:
            # self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)), requires_grad=True)
        # else:
            # self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        if in_channels == 3:
            dim_head = 4
        else:
            dim_head = in_channels //  4

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.scale = dim_head ** -0.5
        inner_dim = dim_head * self.n_heads
        self.to_qk = nn.Conv2d(in_channels, inner_dim*2, 1)
        self.to_v = nn.Conv2d(in_channels, inner_dim, 1)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, out_channels),
            nn.Dropout(dropout)
        )
        self.layer_norm = nn.LayerNorm(out_channels)
        self.relu = nn.GELU(inplace=True)


    def forward(self, x):
        N, C, T, V = x.size()
        qk = self.to_qk(x)
        v = self.to_v(x)
        qk = qk.chunk(2, dim=1)
        q, k = map(lambda t: rearrange(t, 'b (h d) t v -> (b t) h v d', h=self.n_heads), qk)
        v = rearrange(v, 'b (h d) t v -> (b t) h v d', h=self.n_heads)


        # attention
        dots = einsum('b h i d, b h j d -> b h i j', q, k)*self.scale
        attn = dots.softmax(dim=-1).float()
        out = einsum('b h i j, b h j d -> b h i d', attn, v.float())
        out = rearrange(out, '(b t) h n d -> b t n (h d)', t=T)
        out = self.to_out(out)
        out = self.layer_norm(out)
        out = rearrange(out, 'n t v c -> n c t v').contiguous()
        out += self.down(x)
        out = self.relu(out)
        return out

class TCN_attn_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, adaptive=True):
        super(TCN_attn_unit, self).__init__()
        self.agcn = unit_attn(in_channels, out_channels, A, adaptive=adaptive)
        self.tcn = mstcn(out_channels, out_channels, kernel_size=5, stride=stride,
                         dilations=[1, 2], residual=False)

        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        y = self.relu(self.tcn(self.agcn(x)) + self.residual(x))
        return y


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, adaptive=True):
        super(TCN_GCN_unit, self).__init__()
        self.agcn = unit_gcn(in_channels, out_channels, A, adaptive=adaptive)
        self.tcn = mstcn(out_channels, out_channels, kernel_size=5, stride=stride,
                         dilations=[1, 2], residual=False)

        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        y = self.relu(self.tcn(self.agcn(x)) + self.residual(x))
        return y


class TCN_aGCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, adaptive=True, use_port=False):
        super(TCN_aGCN_unit, self).__init__()
        self.agcn = unit_agcn(in_channels, out_channels, A, adaptive=adaptive, use_port=use_port)
        self.tcn = mstcn(out_channels, out_channels, kernel_size=5, stride=stride,
                         dilations=[1, 2], residual=False)

        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x, attn=None):
        y = self.relu(self.tcn(self.agcn(x, attn)) + self.residual(x))
        return y


class Model(nn.Module):
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

class ModelwV(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, in_channels=3,
                 drop_out=0, adaptive=True, num_set=3):
        super(ModelwV, self).__init__()

        Graph = import_class(graph)
        self.graph = Graph()
        A_outward = Graph().A_outward_binary
        self.A_vector = torch.eye(num_point) - A_outward

        A = np.stack([np.eye(num_point)] * num_set, axis=0)
        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
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

        x = x.permute(0, 4, 2, 3, 1).contiguous().view(N*M*T, V, C)
        x = self.A_vector.to(x.device).expand(N*M*T, -1, -1) @ x
        x = x.view(N, M, T, V, C)
        x = x.permute(0, 1, 3, 4, 2).contiguous().view(N, M * V *  C, T)
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

        Graph = import_class(graph)
        self.graph = Graph()
        A_outward = Graph().A_outward_binary
        self.A_vector = torch.eye(num_point) - A_outward

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

        x = x.permute(0, 4, 2, 3, 1).contiguous().view(N*M*T, V, C)
        x = self.A_vector.to(x.device).expand(N*M*T, -1, -1) @ x
        x = x.view(N, M, T, V, C)
        x = x.permute(0, 1, 3, 4, 2).contiguous().view(N, M * V *  C, T)
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

        Graph = import_class(graph)
        self.graph = Graph()
        A_outward = Graph().A_outward_binary
        self.A_vector = torch.eye(num_point) - A_outward

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

        x = x.permute(0, 4, 2, 3, 1).contiguous().view(N*M*T, V, C)
        x = self.A_vector.to(x.device).expand(N*M*T, -1, -1) @ x
        x = x.view(N, M, T, V, C)
        x = x.permute(0, 1, 3, 4, 2).contiguous().view(N, M * V *  C, T)
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

        Graph = import_class(graph)
        self.graph = Graph()
        A_outward = Graph().A_outward_binary
        self.A_vector = torch.eye(num_point) - A_outward

        A = np.stack([np.eye(num_point)] * n_heads, axis=0)
        self.freeze_port = freeze_port
        self.n_heads = n_heads
        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        base_channel = 64

        self.l1 = TCN_aGCN_unit(in_channels, base_channel, A, residual=False, adaptive=adaptive, use_port=True)
        self.l2 = TCN_aGCN_unit(base_channel, base_channel, A, adaptive=adaptive, use_port=True)
        self.l3 = TCN_aGCN_unit(base_channel, base_channel, A, adaptive=adaptive, use_port=True)
        self.l4 = TCN_aGCN_unit(base_channel, base_channel, A, adaptive=adaptive, use_port=True)
        self.l5 = TCN_aGCN_unit(base_channel, base_channel*2, A, stride=2, adaptive=adaptive, use_port=True)
        self.l6 = TCN_aGCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive, use_port=True)
        self.l7 = TCN_aGCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive, use_port=True)
        self.l8 = TCN_aGCN_unit(base_channel*2, base_channel*4, A, stride=2, adaptive=adaptive, use_port=True)
        self.l9 = TCN_aGCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive, use_port=True)
        self.l10 = TCN_aGCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive, use_port=True)
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


    def forward(self, x):
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 2, 3, 1).contiguous().view(N*M*T, V, C)
        x = self.A_vector.to(x.device).expand(N*M*T, -1, -1) @ x

        if self.freeze_port:
            self.port.eval()

        _, attns, _ = self.port(x)
        attn = attns[-2]

        x = x.view(N, M, T, V, C)
        x = x.permute(0, 1, 3, 4, 2).contiguous().view(N, M * V *  C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        x = self.l1(x, attn=attn)
        x = self.l2(x, attn=attn)
        x = self.l3(x, attn=attn)
        x = self.l4(x, attn=attn)
        x = self.l5(x, attn=attn)
        attn = attn.view(N*M, 2, T//2, self.n_heads, V, V)
        attn = attn.mean(dim=1).view(N*M*T//2, self.n_heads, V, V)
        x = self.l6(x, attn=attn)
        x = self.l7(x, attn=attn)
        x = self.l8(x, attn=attn)
        attn = attn.view(N*M, 2, T//4, self.n_heads, V, V)
        attn = attn.mean(dim=1).view(N*M*T//4, self.n_heads, V, V)
        x = self.l9(x, attn=attn)
        x = self.l10(x, attn=attn)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        x = self.drop_out(x)

        return self.fc(x)
