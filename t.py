import torch
from foc import *

from nul import *

conf = nulconf(
    num_heads=2,
    num_layers=3,
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

print(cyan("MLP"))
mlp = MLP(conf)
o = mlp(y)
print(o)

o, (k, v) = mlp((y, (1, 2)))
print(o)
print(k)
print(v)

print(cyan("LayerNorm"))
ln = LayerNorm(conf.size_embed, True)
o = ln(y)
print(o)

o, (k, v) = ln((y, (1, 2)))
print(o)
print(k)
print(v)

print(cyan("Self-Attention"))
sa = SelfAttention(conf)
print(sa)

mask = attention_mask(x)
print(mask)

o = sa(mask, y)
print(o)

o, (k, v) = sa(mask, (y, ()))
print(o)
print(k)
print(v)

print(cyan("Block"))
block = Block(conf)
o = block(mask, y)
print(o)

o, (k, v) = block(mask, (y, ()))
print(o)
print(k)
print(v)

print(cyan("Decoder"))
dec = Decoder(conf)
o = dec(mask, y)
print(o)

o, cached = dec(mask, (y, []))
print(blue("out"))
print(o)
for i in range(len(dec.layers)):
    print(blue(f"Layer{i+1}"))
    print(cached[i])

error()

o, cached = dec(attention_mask(x, C=len_cached_seq(cached)), y)
print(o)
print(cached)

print(cyan("Transformer"))
tf = Transformer(conf)
o = tf(x)
print(o)

o, cached = tf(x)
print(purple("o"))
print(o)
print(purple("cached"))
print(cached)

print(cyan("Text generation"))
i = md.to_ids("아빠")
logits, cached = md(i)
y = infer(logits[:, -1, :], 1, 100, 0.9)
print(i)
print(logits)
print(y)

i = torch.cat((i, y), dim=1)
logits, cached = md((i[:, -1:], cached))
print(i)
print(logits)
print(cached)
print(md.chat("아빠"))
