import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class MPMHead(nn.Module):
    def __init__(self, dim, proj_dim):
        super().__init__()
        self.dense = nn.Linear(dim, dim)
        self.LayerNorm = nn.LayerNorm(dim, eps=1e-12)
        self.activation = nn.GELU()
        self.decoder = nn.Linear(dim, proj_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(proj_dim))
        self.decoder.bias = self.bias

    def forward(self, x):
        x = self.dense(x)
        x = self.LayerNorm(x)
        x = self.activation(x)
        x = self.decoder(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., reattn=False):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Conv1d(dim, inner_dim*3, 1)
        self.reattn = reattn
        if self.reattn:
            self.reattn_weights = nn.Parameter(torch.randn(heads, heads))
            self.reattn_norm = nn.Sequential(
                Rearrange('b h i j -> b i j h'),
                nn.LayerNorm(heads),
                Rearrange('b i j h -> b h i j')
            )
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        x = x.permute(0, 2, 1).contiguous()
        x = self.to_qkv(x).permute(0, 2, 1).contiguous()
        qkv = x.chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        # attention
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1).float()

        if self.reattn:
            # re-attention
            attn = einsum('b h i j, h g -> b g i j', attn, self.reattn_weights)
            attn = self.reattn_norm(attn)

        # aggregate and out
        out = einsum('b h i j, b h j d -> b h i d', attn, v.float())
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out, attn

class TransformerEncoder(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.self_attn = PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))
        self.ffn = PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))

    def forward(self, x):
        res = x
        x, attn_weight = self.self_attn(x)
        x = x + res
        x = self.ffn(x) + x
        return x, attn_weight

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoder(dim, heads, dim_head, mlp_dim, dropout) for _ in range(depth)
        ])
    def forward(self, x):
        hidden_states = []
        attn_weights = []
        for trans_enc in self.layers:
            x, attn_weight = trans_enc(x)
            hidden_states.append(x)
            attn_weights.append(attn_weight)
        return x, attn_weights, hidden_states

class SpatialFeatureAggregator(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout):
        super().__init__()
        self.ffn = PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
        self.to_out = nn.Sequential(
            nn.Linear(dim*heads, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, attn_weights):
        out = einsum('b h i n, b i d -> b h n d', attn_weights, x.float())
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        out = self.ffn(out) + out
        return out

class MORT(nn.Module):
    def __init__(self, *, dim, depth, heads, mlp_dim, channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        self.num_joints = 25

        self.to_joint_embedding = nn.Linear(channels, dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_joints, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        # self.to_latent = nn.Identity()
        self.mpm_head = MPMHead(dim, channels)

    def forward(self, skel):
        x = self.to_joint_embedding(skel)
        x += self.pos_embedding[:, :self.num_joints]
        x = self.dropout(x)
        x, attns, hidden_states = self.transformer(x)
        proj = self.mpm_head(x)
        return proj, attns, hidden_states

