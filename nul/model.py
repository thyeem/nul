import math

import torch
from foc import *
from torch import nn
from torch.nn import functional as F

from .utils import attention_mask, len_cached_seq


class Transformer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.wte = nn.Embedding(config.size_vocab, config.size_embed)
        self.wpe = nn.Embedding(config.size_block, config.size_embed)
        self.decoder = Decoder(config)
        self.ln = LayerNorm(config.size_embed, bias=config.bias)

    def forward(self, x, cached=None, ipad=0):
        B, S = x.size()
        C = len_cached_seq(cached)
        mask = attention_mask(x, C=C, ipad=ipad)  # (B, 1, S, L)
        pos = (
            self.wpe(torch.arange(C, C + S, dtype=torch.long, device=x.device))
            .unsqueeze(0)  # (1, S, E)
            .expand(B, S, -1)  # (B, S, E)
        )
        return cf_(
            self.ln if cached is None else bimap(self.ln, id),  # (B, S, E)
            f_(self.decoder, mask, cached=cached),  # (B, S, E)
            _ + pos,  # W_p[:,t]: (B, S, E) + (B, S, E)
            self.wte,  # W_e[:,x(t)]: (B, S) -> (B, S, E)
        )(x)


class Decoder(nn.Module):
    """Decoder for decode-only transformer"""

    def __init__(self, config):
        super().__init__()
        self.ln = LayerNorm(config.size_embed, bias=config.bias)
        self.layers = nn.ModuleList(
            [Block(config) for _ in range(config.num_layers)],
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, mask, x, cached=None):
        def go(acc, ilayer):
            x, cached = acc
            i, layer = ilayer
            if cached is None:
                return layer(mask, x)
            else:
                x, o = layer(mask, x, cached=cached)
                if len(cached) < len(self.layers):
                    cached.append(o)
                else:
                    cached[i] = o
                return x, cached

        return cf_(
            self.dropout if cached is None else bimap(self.dropout, id),
            # (
            # lambda x: foldl(
            # go,
            # x if cached is None else (x, cached),
            # enumerate(self.layers),
            # )
            # ),
            # (lambda x: foldl(go, x, enumeratel(self.layers))),
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

    def forward(self, mask, x, cached=None):
        def sublayer(core, layer_norm):
            def go(x):
                return cf_(
                    layer_norm if cached is None else bimap(layer_norm, id),
                    _ + x if cached is None else bimap(_ + fst(x), id),  # residual
                    core,  # core-fn
                )(x)

            return go

        # attention = f_(self.attn, mask, cached=cached)
        return cf_(
            sublayer(  # feedforward network sub-layer
                self.mlp if cached is None else bimap(self.mlp, id),
                self.ln_2,
            ),
            sublayer(  # causal self attention sub-layer
                # attention if cached is None else bimap(attention, id),
                f_(self.attn, mask, cached=cached),
                self.ln_1,
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

    def forward(self, mask, x, cached=None):
        B, S, E = x.size()  # size_batch, sequence length, size_embed
        N, H = self.config.num_heads, E // self.config.num_heads  # E == (N * H)

        q, k, v = self.c_attn(x).split(self.config.size_embed, dim=2)
        q = q.view(B, S, N, H).transpose(1, 2)  # (B, N, S, H)
        k = k.view(B, S, N, H).transpose(1, 2)  # (B, N, S, H)
        v = v.view(B, S, N, H).transpose(1, 2)  # (B, N, S, H)

        if cached:
            k = torch.cat([fst(cached), k], dim=2)
            v = torch.cat([snd(cached), v], dim=2)

        # Attention(Q,K,V)
        #   = softmax( Q*K^T / sqrt(d_k) ) * V
        #         // q*k^T: (B, N, S, H) x (B, N, H, S) -> (B, N, S, S)
        #   = attention-prob-matrix * V
        #         // prob @ v: (B, N, S, S) x (B, N, S, H) -> (B, N, S, H)
        #   = attention-weighted value (attention score)
        return cf_(
            id if cached is None else lambda x: (x, (k, v)),
            self.dropout,  # dropout of layer's output
            self.c_proj,  # linear projection
            ob(_.view)(B, S, E),  # (B, S, N, H) -> (B, S, E)
            torch.Tensor.contiguous,  # contiguos in-memory tensor
            ob(_.transpose)(1, 2),  # (B, S, N, H)
            _ @ v,  # (B, N, S, S) x (B, N, S, H) -> (B, N, S, H)
            self.dropout_attn,  # attention dropout
            f_(F.softmax, dim=-1),  # softmax
            ob(_.masked_fill)(mask == 0, float("-inf")),  # no-look-ahead
            _ / math.sqrt(H),  # / sqrt(d_k)
            _ @ k.transpose(-2, -1),  # Q @ K^T -> (B, N, S, S)
        )(q)
