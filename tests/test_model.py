import math

import torch
from nul import *
from torch.nn import functional as F

V = 777
B = 2
S = 8
E = 6
N = 2
H = 2

conf = nulconf(
    size_vocab=V,
    size_embed=E,
    size_block=S,
    num_layers=N,
    num_heads=H,
)

# -----------------------------------------------

x = torch.randint(V, (B, 2))
print("\ntokens")
print(x)

# -----------------------------------------------

mask = attention_mask(x)
print("\nattention mask")
print(mask)

# -----------------------------------------------

y = torch.randn(B, 2, E)
print("\nIN")
print(y)

# -----------------------------------------------

print("\nMLP")
M = MLP(conf)
M.eval()
print(M)


def mlp(x, cache=False):
    if cache:
        x, q = x
    x = M.c_fc(x)
    x = F.gelu(x)
    x = M.c_proj(x)
    if cache:
        return x, q
    return x


a = M(y)
print(a)
o = mlp(y)
assert torch.equal(o, a), "MLP"

a = M((y, "cache"))
print(a)
o = mlp((y, "cache"), cache=True)
assert torch.equal(o[0], a[0]), "ML-cache"

# -----------------------------------------------

print("\nSelf Attention")
A = SelfAttention(conf)
A.eval()
print(A)


def sa(mask, x, cache=False):
    B, S, E = x.size()
    N, H = A.config.num_heads, E // A.config.num_heads

    q, k, v = A.c_attn(x).split(A.config.size_embed, dim=2)
    q = q.view(B, S, N, H).transpose(1, 2)
    k = k.view(B, S, N, H).transpose(1, 2)
    v = v.view(B, S, N, H).transpose(1, 2)

    x = q @ k.transpose(-2, -1)
    x = x / math.sqrt(H)
    x = x.masked_fill(mask == 0, float("-inf"))
    x = F.softmax(x, dim=-1)
    x = x @ v
    x = x.transpose(1, 2)
    x = torch.Tensor.contiguous(x)
    x = x.view(B, S, E)
    x = A.c_proj(x)
    return x


a = A(mask, y)
print(a)
o = sa(mask, y)
assert torch.equal(o, a), "Self-Attention"

a, (k, v) = A(mask, (y, None))
print(k.shape, v.shape)

# -----------------------------------------------

print("\nLayerNorm")
L = LayerNorm(size=conf.size_embed, bias=conf.bias)
L.eval()
print(L(a))
cached = (k, v)
assert torch.equal(L(a), L((a, cached))[0])


# -----------------------------------------------

print("\nBlock")
B = Block(conf)
B.eval()
a = B(mask, y)
print(a)
o, cached = B(mask, (y, []))
assert torch.equal(a, o)

# -----------------------------------------------

print("\nDecoder")
D = Decoder(conf)
D.eval()
a = D(mask, y)
print(a)

o, cached = D(mask, (y, []))
k, v = cached[0]
print(k.shape)
print(v.shape)
assert torch.equal(a, o)

# -----------------------------------------------

print("\nTransformer")
T = Transformer(conf)
T.eval()

a = T(x)
print(a)

o, cached = T((x, None))
assert torch.equal(a, o)
assert conf.num_layers == len(cached)
C = len_cached_seq(cached)
print(f"Cache length: {C}")

o, cached = T((x[:, -1:], cached))
assert conf.num_layers == len(cached)
print(f"Out shape: {o.shape}")
C = len_cached_seq(cached)
print(f"Cache length: {C}")
m = attention_mask(x[:, -1:], C=C)
print(f"Mask shape: {m.shape}")

o, cached = T((x[:, -1:], cached))
assert conf.num_layers == len(cached)
print(f"Out shape: {o.shape}")
C = len_cached_seq(cached)
print(f"Cache length: {C}")
