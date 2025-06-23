import math

import torch
from foc import *
from torch import nn
from torch.nn import functional as F

from .utils import attention_mask, len_cached_seq


def with_cache(f):
    def go(x):
        if isinstance(x, tuple):
            return first(f, x)
        else:
            return f(x)

    return go


class Transformer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.wte = nn.Embedding(config.size_vocab, config.size_embed)
        self.wpe = nn.Embedding(config.size_block, config.size_embed)
        self.decoder = Decoder(config)
        self.ln = LayerNorm(config.size_embed, bias=config.bias)

    def forward(self, x, ipad=0):
        use_cache = isinstance(x, tuple)
        if use_cache:
            x, cached = x
            cached = cached or []
        else:
            cached = None
        B, S = x.size()
        C = len_cached_seq(cached)
        mask = attention_mask(x, C=C, ipad=ipad)  # (B, 1, S, L)
        pos = (
            self.wpe(torch.arange(C, C + S, dtype=torch.long, device=x.device))
            .unsqueeze(0)  # (1, S, E)
            .expand(B, S, -1)  # (B, S, E)
        )
        x = (x, cached) if use_cache else x  # (B, S), [KV-cache]?
        return cf_(
            with_cache(self.ln),  # (B, S, E), [KV-cache]?
            f_(self.decoder, mask),  # (B, S, E), [KV-cache]?
            with_cache(_ + pos),  # _ + W_p[:, t] -> (B, S, E), [KV-cache]?
            with_cache(self.wte),  # W_e[:, x(t)] -> (B, S, E), [KV-cache]?
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

    def forward(self, mask, x):
        use_cache = isinstance(x, tuple)

        def go(x, ilayer):
            i, layer = ilayer
            if not use_cache:
                return layer(mask, x)
            x, cached = x
            if len(cached) < len(self.layers):
                x, o = layer(mask, (x, None))
                cached.append(o)
            else:
                x, o = layer(mask, (x, cached[i]))
                cached[i] = o
            return x, cached

        return cf_(
            with_cache(self.dropout),
            (lambda x: foldl(go, x, enumeratel(self.layers))),
            with_cache(self.ln),
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
        cf_(residual, core, layer_norm)(x)    # pre-norm
        cf_(layer_norm, residual, core)(x)    # post-norm
    """

    def __init__(self, config):
        super().__init__()
        self.attn = SelfAttention(config)
        self.ln_1 = LayerNorm(config.size_embed, bias=config.bias)
        self.mlp = MLP(config)
        self.ln_2 = LayerNorm(config.size_embed, bias=config.bias)

    def forward(self, mask, x):
        def sublayer(core, layer_norm):
            def residual(x):
                return first(_ + fst(x)) if isinstance(x, tuple) else _ + x

            def go(x):
                return cf_(
                    layer_norm,  # layer-normalization
                    residual(x),  # resiaul network
                    core,  # core-fn
                )(x)

            return go

        return cf_(
            sublayer(  # 2nd sub-layer: feedforward network
                self.mlp,
                self.ln_2,
            ),
            sublayer(  # 1st sub-layer: causal self attention
                f_(self.attn, mask),
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
        return with_cache(
            lambda x: F.layer_norm(x, self.weight.shape, self.weight, self.bias)
        )(x)


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
        return with_cache(
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
        """forward '(x, (..))' instead of 'x' when to use KV-cache"""
        use_cache = isinstance(x, tuple)
        if use_cache:
            x, cached = x

        B, S, E = x.size()  # size_batch, sequence length, size_embed
        N, H = self.config.num_heads, E // self.config.num_heads  # E == (N * H)

        q, k, v = self.c_attn(x).split(self.config.size_embed, dim=2)
        q = q.view(B, S, N, H).transpose(1, 2)  # (B, N, S, H)
        k = k.view(B, S, N, H).transpose(1, 2)  # (B, N, S, H)
        v = v.view(B, S, N, H).transpose(1, 2)  # (B, N, S, H)

        if use_cache:
            if cached:
                k = torch.cat([fst(cached), k], dim=2)
                v = torch.cat([snd(cached), v], dim=2)
            cached = (k, v)

        # Attention(Q, K, V)
        #   = softmax( Q*K^T / sqrt(d_k) ) * V
        #         // q*k^T: (B, N, S, H) x (B, N, H, S) -> (B, N, S, S)
        #   = attention-prob-matrix * V
        #         // prob @ v: (B, N, S, S) x (B, N, S, H) -> (B, N, S, H)
        #   = attention-weighted value (attention score)
        return cf_(
            (lambda x: (x, cached)) if use_cache else id,
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
