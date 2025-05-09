import torch
from foc import *

from nul import *

conf = nulconf(
    num_heads=2,
    num_layers=2,
    size_block=5,
    size_embed=4,
    size_batch=2,
    bias=True,
)
md = nul.new(conf=conf)

x = torch.randint(
    conf.size_vocab,
    (conf.size_batch, conf.size_block),
)  # (B, S)
y = torch.randn(
    conf.size_batch,
    conf.size_block,
    conf.size_embed,
)  # (B, S, E)

print(x)
print(y)

mask = attention_mask(x)
print(mask)

print(cyan("Self-Attention"))
sa = SelfAttention(conf)
print(sa)

o = sa(mask, y)
print(o)

o, (k, v) = sa(mask, y, cached=[])
print(o)
print(k)
print(v)

print(cyan("MLP"))
mlp = MLP(conf)
o = mlp(y)
print(o)

print(cyan("LayerNorm"))
ln = LayerNorm(conf.size_embed, True)
o = ln(y)
print(o)

print(cyan("Block"))
block = Block(conf)
o = block(mask, y)
print(o)

o, cached = block(mask, y, cached=[])
print(o)
print(cached)

print(cyan("Decoder"))
dec = Decoder(conf)
o = dec(mask, y)
print(o)
error()

o, cached = dec(mask, y, cached=[])
print(o)
print(cached)

o, cached = dec(
    attention_mask(x, C=len_cached_seq(cached)),
    y,
    cached=cached,
)
print(o)
print(cached)

print(cyan("Transformer"))
tf = Transformer(conf)
o = tf(x)
print(o)

o, cached = tf(x, cached=[])
print(purple("o"))
print(o)
print(purple("cached"))
print(cached)

print(cyan("Text generation"))
i = md.to_ids("아빠")
logits, cached = md(i, cached=[])
y = infer(logits[:, -1, :], 1, 100, 0.9)
print(i)
print(logits)
print(y)

i = torch.cat((i, y), dim=1)
logits, cached = md(i[:, -1:], cached=cached)
print(i)
print(logits)
print(cached)
print(md.chat("아빠"))
