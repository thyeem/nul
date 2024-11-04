import math

import torch
from foc import *
from torch import nn
from torch.nn import functional as F


class Decoder(nn.Module):
    """Decoder for decode-only transformer"""

    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(
            [Block(config) for _ in range(config.num_layers)],
        )
        self.ln = LayerNorm(config.size_embed, bias=config.bias)

    def forward(self, mask, x):
        return cf_(
            self.dropout,
            cf_(*[f_(layer, mask) for layer in self.layers]),
            self.ln,
        )(x)


class Block(nn.Module):
    """Decode block for  decode-only transformer
    +------------+----------------------------------------+
    | post-norm  |  x = LayerNorm(SelfAttention(x)) + x)  |
    |            |  o = LayerNorm(FFN(x)) + x)            |
    +------------+----------------------------------------+
    | pre-norm   |  x = SelfAttention(LayerNorm(x))) + x  |
    |            |  o = FFN(LayerNorm(x))) + x            |
    +------------+----------------------------------------+-

    One can easily convert the post-norm architecture (suggested by Vaswani) into
    the pre-norm architecture by changing the postion of 'layer_norm'

    In 'sublayer' method:
        cf_(_ + x, core, layer_norm)(x)    # pre-norm
        cf_(layer_norm, _ + x, core)(x)    # post-norm
    """

    def __init__(self, config):
        super().__init__()
        self.attn = SelfAttention(config)
        self.ln_1 = LayerNorm(config.size_embed, bias=config.bias)
        self.mlp = MLP(config)
        self.ln_2 = LayerNorm(config.size_embed, bias=config.bias)

    @staticmethod
    def sublayer(core, layer_norm, x):
        return cf_(
            layer_norm,  # layer-norm
            _ + x,  # residual connection
            core,  # core-fn
        )(x)

    def forward(self, mask, x):
        return cf_(
            f_(
                self.sublayer,  # sublayer-2
                self.mlp,  # feedforward net
                self.ln_2,  # layer-norm-sublayer-2
            ),
            f_(
                self.sublayer,  # sublayer-1
                f_(self.attn, mask),  # self-attention
                self.ln_1,  # layer-norm-sublayer-1
            ),
        )(x)


class LayerNorm(nn.Module):
    """Layer normalization with an optional bias"""

    def __init__(self, size, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.bias = nn.Parameter(torch.zeros(size)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias)


class MLP(nn.Module):
    """Feed-Forward Network (FFN) with non-linear activation fn"""

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(
            config.size_embed,
            config.ratio_ffn * config.size_embed,
            bias=config.bias,
        )
        self.c_proj = nn.Linear(
            config.ratio_ffn * config.size_embed,
            config.size_embed,
            bias=config.bias,
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return cf_(
            self.dropout,
            self.c_proj,
            F.gelu,
            self.c_fc,
        )(x)


class SelfAttention(nn.Module):
    """Causal Self-attention"""

    def __init__(self, config):
        super().__init__()
        guard(
            config.size_embed % config.num_heads == 0,
            f"Error, embedding size (got {config.size_embed}) "
            f"must be divisible by the number of heads (got {config.num_heads})",
        )
        self.config = config
        self.c_attn = nn.Linear(
            config.size_embed,
            3 * config.size_embed,
            bias=config.bias,
        )
        self.dropout_attn = nn.Dropout(config.dropout)
        self.c_proj = nn.Linear(
            config.size_embed,
            config.size_embed,
            bias=config.bias,
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, mask, x):
        B, S, E = x.size()  # size_batch, sequence length, size_embed
        N, H = self.config.num_heads, E // self.config.num_heads  # E == (N * H)

        q, k, v = self.c_attn(x).split(self.config.size_embed, dim=2)
        q = q.view(B, S, N, H).transpose(1, 2)  # (B,N,S,H)
        k = k.view(B, S, N, H).transpose(1, 2)  # (B,N,S,H)
        v = v.view(B, S, N, H).transpose(1, 2)  # (B,N,S,H)

        # Attention(Q,K,V)
        #   = softmax( Q*K^T / sqrt(d_k) ) * V
        #         // q*k^T: (B,N,S,H) x (B,N,H,S) -> (B,N,S,S)
        #   = attention-prob-matrix * V
        #         // prob @ v: (B,N,S,S) x (B,N,S,H) -> (B,N,S,H)
        #   = attention-weighted value (attention score)
        return cf_(
            self.dropout,  # dropout of layer's output
            self.c_proj,  # linear projection
            g_(_.view)(B, S, E),  # (B,S,N,H) -> (B,S,E)
            torch.Tensor.contiguous,  # contiguos in-memory tensor
            g_(_.transpose)(1, 2),  # (B,S,N,H)
            _ @ v,  # (B,N,S,S) x (B,N,S,H) -> (B,N,S,H)
            self.dropout_attn,  # attention dropout
            f_(F.softmax, dim=-1),  # softmax
            g_(_.masked_fill)(mask == 0, float("-inf")),  # no-look-ahead
            _ / math.sqrt(k.size(-1)),  # / sqrt(d_k)
            _ @ k.transpose(-2, -1),  # Q @ K^T -> (B,N,S,S)
        )(q)
