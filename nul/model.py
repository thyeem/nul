import math

import torch
from foc import *
from torch import nn
from torch.nn import functional as F

from .utils import attention_mask


class Transformer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.wte = nn.Embedding(config.size_vocab, config.size_embed)
        self.wpe = nn.Embedding(config.size_block, config.size_embed)
        self.decoder = Decoder(config)
        self.ln = LayerNorm(config.size_embed, bias=config.bias)

    def forward(self, x, past_kv=None, use_cache=False, ipad=0):
        B, S = x.size()
        # past_kv = [(k=(B, N, S', H), v=(B, N, S', H)), ...]
        size_kv = 0 if past_kv is None else past_kv[0][0].size(2)  # S'
        mask = attention_mask(x, size_kv=size_kv, ipad=ipad)
        pos = (
            self.wpe(
                torch.arange(size_kv, size_kv + S, dtype=torch.long, device=x.device)
            )
            .unsqueeze(0)  # (1, S, E)
            .expand(B, S, -1)  # (B, S, E)
        )
        return cf_(
            bimap(self.ln, id) if use_cache else self.ln,  # (B, S, E)
            f_(self.decoder, mask, past_kv=past_kv, use_cache=use_cache),  # (B, S, E)
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

    def forward(self, mask, x, past_kv=None, use_cache=False):
        def go(x):
            cache = []
            for i, layer in enumerate(self.layers):
                x, o = layer(
                    mask,
                    x,
                    past_kv=None if past_kv is None else past_kv[i],
                    use_cache=True,
                )
                if use_cache:
                    cache.append(o)
            return (x, cache) if use_cache else x

        return cf_(
            bimap(self.dropout, id) if use_cache else self.dropout,
            go,
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

    def forward(self, mask, x, past_kv=None, use_cache=False):
        def sublayer(core, layer_norm):
            def go(x):
                return cf_(
                    bimap(layer_norm, id) if use_cache else layer_norm,
                    bimap(_ + fst(x), id) if use_cache else _ + x,  # residual
                    core,  # core-fn
                )(x)

            return go

        return cf_(
            sublayer(  # feedforward network sub-layer
                bimap(self.mlp, id) if use_cache else self.mlp,
                self.ln_2,
            ),
            sublayer(  # causal self attention sub-layer
                f_(self.attn, mask, past_kv=past_kv, use_cache=use_cache),
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

    def forward(self, mask, x, past_kv=None, use_cache=False):
        B, S, E = x.size()  # size_batch, sequence length, size_embed
        N, H = self.config.num_heads, E // self.config.num_heads  # E == (N * H)

        q, k, v = self.c_attn(x).split(self.config.size_embed, dim=2)
        q = q.view(B, S, N, H).transpose(1, 2)  # (B, N, S, H)
        k = k.view(B, S, N, H).transpose(1, 2)  # (B, N, S, H)
        v = v.view(B, S, N, H).transpose(1, 2)  # (B, N, S, H)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        # Attention(Q,K,V)
        #   = softmax( Q*K^T / sqrt(d_k) ) * V
        #         // q*k^T: (B, N, S, H) x (B, N, H, S) -> (B, N, S, S)
        #   = attention-prob-matrix * V
        #         // prob @ v: (B, N, S, S) x (B, N, S, H) -> (B, N, S, H)
        #   = attention-weighted value (attention score)
        return cf_(
            (lambda x: (x, (k, v))) if use_cache else id,
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
