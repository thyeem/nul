import math

import torch
from foc import *
from torch import nn
from torch.nn import functional as F

from .utils import attention_mask, len_cached_seq


class X:
    __slots__ = ("x", "cached", "cache", "mask")

    def __init__(self, x, cached=None, cache=True, mask=None):
        self.x = x
        self.cached = cached or []
        self.cache = cache
        self.mask = mask

    def __repr__(self):
        return (
            f"X({self.x}, cached={self.cached}, cache={self.cache}, mask={self.mask})"
        )


def with_x(f):
    def go(x):
        return X(f(x.x), cached=x.cached, cache=x.cache, mask=x.mask)

    return go


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wte = nn.Embedding(config.size_vocab, config.size_embed)
        self.wpe = nn.Embedding(config.size_block, config.size_embed)
        self.decoder = Decoder(config)
        self.ln = LayerNorm(config.size_embed, bias=config.bias)

    def forward(self, x, ipad=1):
        B, S = x.x.size()
        C = len_cached_seq(x.cached)
        x.mask = attention_mask(x.x, C=C, ipad=ipad)  # (B, 1, S, L)
        pos = (
            self.wpe(torch.arange(C, C + S, dtype=torch.long, device=x.x.device))
            .unsqueeze(0)  # (1, S, E)
            .expand(B, S, -1)  # (B, S, E)
        )
        return cf_(
            with_x(self.ln),  # (B, S, E)
            self.decoder,  # (B, S, E)
            with_x(_ + pos),  # _ + W_p[:, t] -> (B, S, E)
            with_x(self.wte),  # W_e[:, x(t)] -> (B, S, E)
        )(x)


class Decoder(nn.Module):
    """Decoder for decode-only transformer"""

    def __init__(self, config):
        super().__init__()
        Layer = MUX if config.size_mux > 1 else Block
        self.ln = LayerNorm(config.size_embed, bias=config.bias)
        self.layers = nn.ModuleList(
            [Layer(config) for _ in range(config.num_layers)],
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        def go(x, ilayer):
            i, layer = ilayer
            if not x.cache or isinstance(layer, MUX):
                return layer(x)
            if len(x.cached) < len(self.layers):
                x.cache = True
                x = layer(x)
                x.cached.append(x.cache)
            else:
                x.cache = x.cached[i]
                x = layer(x)
                x.cached[i] = x.cache
            return x

        return cf_(
            with_x(self.dropout),
            (lambda x: foldl(go, x, enumeratel(self.layers))),
            with_x(self.ln),
        )(x)


class Router(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proj = nn.Linear(config.size_embed, config.size_mux)

    def forward(self, x):
        z = x.mean(dim=1)
        logits = self.proj(z)
        gumbel = -torch.empty_like(logits).exponential_().log()
        y = F.softmax(logits + gumbel, dim=-1)
        y_hard = torch.zeros_like(y)
        y_hard.scatter_(1, y.argmax(dim=-1, keepdim=True), 1.0)
        y = (y_hard - y).detach() + y
        return y


class MUX(nn.Module):
    """Block Multiplexer"""

    def __init__(self, config):
        super().__init__()
        self.ln = LayerNorm(config.size_embed, bias=config.bias)
        self.blocks = nn.ModuleList(
            [Block(config) for _ in range(config.size_mux)],
        )
        self.router = Router(config)

    def forward(self, x):
        B, _, _ = x.x.size()
        imax = self.router(x.x).argmax(dim=-1)
        o = []
        for i in range(B):
            o.append(
                self.blocks[imax[i]](
                    X(x[i : i + 1], mask=x.mask[i : i + 1], cache=False),
                ).x
            )
        o = self.ln(torch.cat(o, dim=0))
        return X(o, cached=x.cached, cache=x.cache, mask=x.mask)


class Block(nn.Module):
    """Decode block for decode-only transformer
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
        cf_(residual, core, layer_norm)(x)    # pre-norm
        cf_(layer_norm, residual, core)(x)    # post-norm
    """

    def __init__(self, config):
        super().__init__()
        self.attn = SelfAttention(config)
        self.dropout_1 = nn.Dropout(config.dropout)
        self.ln_1 = LayerNorm(config.size_embed, bias=config.bias)
        self.mlp = MLP(config)
        self.dropout_2 = nn.Dropout(config.dropout)
        self.ln_2 = LayerNorm(config.size_embed, bias=config.bias)

    def forward(self, x):
        def sublayer(core, dropout, layer_norm):
            def go(x):
                return cf_(
                    with_x(layer_norm),  # layer-normalization
                    with_x(dropout),  # dropout
                    with_x(_ + x.x),  # residual (skip-connection)
                    core,  # core-fn
                )(x)

            return go

        return cf_(
            sublayer(  # 2nd sub-layer: feedforward network
                self.mlp,
                self.dropout_2,
                self.ln_2,
            ),
            sublayer(  # 1st sub-layer: causal self attention
                self.attn,
                self.dropout_1,
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
            config.size_embed * config.ratio_ffn,
            bias=config.bias,
        )
        self.c_proj = nn.Linear(
            config.size_embed * config.ratio_ffn,
            config.size_embed,
            bias=config.bias,
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return with_x(
            cf_(
                self.dropout,
                self.c_proj,
                F.gelu,
                self.c_fc,
            )
        )(x)


class SelfAttention(nn.Module):
    """Causal Self-attention"""

    def __init__(self, config):
        super().__init__()
        guard(
            config.size_embed % config.num_heads == 0,
            f"Error, embedding size (got {config.size_embed}) "
            f"must be divisible by the number of heads (got {config.num_heads})",
            e=SystemExit,
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

    def forward(self, x):
        B, S, E = x.x.size()  # size_batch, sequence length, size_embed
        N, H = self.config.num_heads, E // self.config.num_heads  # E == (N * H)

        q, k, v = self.c_attn(x.x).split(self.config.size_embed, dim=2)
        q = q.view(B, S, N, H).transpose(1, 2)  # (B, N, S, H)
        k = k.view(B, S, N, H).transpose(1, 2)  # (B, N, S, H)
        v = v.view(B, S, N, H).transpose(1, 2)  # (B, N, S, H)

        if x.cache:
            if isinstance(x.cache, tuple):
                _k, _v = x.cache
                k, v = torch.cat([_k, k], dim=2), torch.cat([_v, v], dim=2)
            x.cache = (k, v)

        # Attention(Q, K, V)
        #   = softmax( Q*K^T / sqrt(d_k) ) * V
        #         // q*k^T: (B, N, S, H) x (B, N, H, S) -> (B, N, S, S)
        #   = attention-prob-matrix * V
        #         // prob @ v: (B, N, S, S) x (B, N, S, H) -> (B, N, S, H)
        #   = attention-weighted value (attention score)
        o = cf_(
            self.dropout,  # dropout of layer's output
            self.c_proj,  # linear projection
            ob(_.view)(B, S, E),  # (B, S, N, H) -> (B, S, E)
            torch.Tensor.contiguous,  # contiguos in-memory tensor
            ob(_.transpose)(1, 2),  # (B, S, N, H)
            _ @ v,  # (B, N, S, S) x (B, N, S, H) -> (B, N, S, H)
            self.dropout_attn,  # attention dropout
            f_(F.softmax, dim=-1),  # softmax
            ob(_.masked_fill)(x.mask == 0, float("-inf")),  # no-look-ahead
            _ / math.sqrt(H),  # / sqrt(d_k)
            _ @ k.transpose(-2, -1),  # Q @ K^T -> (B, N, S, S)
        )(q)
        return X(o, cached=x.cached, cache=x.cache, mask=x.mask)
