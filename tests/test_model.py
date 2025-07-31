import math

import torch
from foc import *
from nul import *
from torch.nn import functional as F

V = 777
L = 3
B = 2
W = 8
S = 4
E = 6
N = 2

conf = nulconf(
    size_vocab=V,
    size_embed=E,
    size_block=W,
    size_mux=1,
    num_layers=L,
    num_heads=N,
)

# -----------------------------------------------

x = torch.randint(V, (B, S))
print("\ntokens")
print(x)

# -----------------------------------------------

mask = attention_mask(x)
print("\nattention mask")
print(mask)

# -----------------------------------------------

y = torch.randn(B, S, E)
print("\nIN")
print(y)

# -----------------------------------------------

print("\nMLP")
M = MLP(conf)
M.eval()
print(M)


def mlp(x):
    x = M.c_fc(x)
    x = F.gelu(x)
    x = M.c_proj(x)
    return x


a = M(X(y))
print(a.x)
o = mlp(y)
assert torch.equal(o, a.x), "MLP"

# -----------------------------------------------

print("\nSelf Attention")
A = SelfAttention(conf)
A.eval()
print(A)


def sa(mask, x):
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


a = A(X(y, mask=mask, cache=True))
b = A(X(y, mask=mask, cache=False))
o = sa(mask, y)
print(a.x)
print(f"(B, S, E) = {a.x.shape}")
assert torch.equal(o, a.x), "Self-Attention"
assert torch.equal(a.x, b.x), "Self-Attention W/WO cache"

k, v = a.cache
print(k)
print(v)
print(f"(B, N, S, H) = {k.shape}")
print(f"(B, N, S, H) = {v.shape}")

# -----------------------------------------------

print("\nLayerNorm")
Q = LayerNorm(size=conf.size_embed, bias=conf.bias)
Q.eval()
ln = Q(a.x)
print(ln)
print(f"(B, S, E) = {ln.shape}")

# -----------------------------------------------

print("\nBlock")
B = Block(conf)
B.eval()
a = B(X(y, mask=mask, cache=True))
b = B(X(y, mask=mask, cache=False))
print(a.x)
assert torch.equal(a.x, b.x)
print(f"(B, S, E) = {a.x.shape}")

k, v = a.cache
print(k)
print(v)
print(f"(B, N, S, H) = {k.shape}")
print(f"(B, N, S, H) = {v.shape}")

# -----------------------------------------------

print("\nDecoder")
D = Decoder(conf)
D.eval()
a = D(X(y, mask=mask, cache=True))
b = D(X(y, mask=mask, cache=False))
print(a.x)
assert torch.equal(a.x, b.x)
print(f"(B, S, E) = {a.x.shape}")

print(f"Length of cached = {len(a.cached)}")
assert len(a.cached) == L, "Block, cached"

for i in range(L):
    k, v = a.cached[i]
    print(f"k[{i}]: (B, N, S, H) = {k.shape}")
    print(f"v[{i}]: (B, N, S, H) = {v.shape}")

# -----------------------------------------------

print("\nTransformer")
T = Transformer(conf)
T.eval()

a = T(X(x, mask=mask, cache=True))
b = T(X(x, mask=mask, cache=False))
print(a.x)
assert torch.equal(a.x, b.x)
print(f"(B, S, E) = {a.x.shape}")

print(f"Length of cached = {len(a.cached)}")
assert len(a.cached) == L, "Transformer, cached"

# -----------------------------------------------

print("Transformer-Level Cache Test")

for _ in range(4):
    C = len_cached_seq(a.cache)
    print(f"Sequence length of cached = {C}")
    o = x[:, -1:]
    print(o)
    mask = attention_mask(o, C=C)
    print(f"mask, {mask.shape}")

    a = T(X(o, mask=mask, cached=a.cached, cache=True))
    C = len_cached_seq(a.cache)
    print(f"Sequence length of cached = {C}")
    assert len(a.cached) == L, "wrong cached length"
    assert C == L, "wrong cached length"
    print(f"(B, S, E) = {a.x.shape}")
    for i in range(L):
        k, v = a.cached[i]
        print(f"k[{i}]: (B, N, S, H) = {k.shape}")
        print(f"v[{i}]: (B, N, S, H) = {v.shape}")
