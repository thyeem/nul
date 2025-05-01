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

o = sa(mask, y, use_cache=False)
print(o)

o, (k, v) = sa(mask, y, use_cache=True)
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

o, past_kv = block(mask, y, use_cache=True)
print(o)
print(past_kv)

print(cyan("Decoder"))
dec = Decoder(conf)
o = dec(mask, y, use_cache=False)
print(o)

o, past_kv = dec(mask, y, use_cache=True)
print(o)
print(past_kv)

o, past_kv = dec(
    attention_mask(x, size_kv=past_kv[0][0].size(2)),
    y,
    past_kv=past_kv,
    use_cache=True,
)
print(o)
print(past_kv)

print(cyan("Transformer"))
tf = Transformer(conf)
o = tf(x, use_cache=False)
print(o)

o, past_kv = tf(x, use_cache=True)
print(purple("o"))
print(o)
print(purple("past_kv"))
print(past_kv)

print(cyan("Text generation"))
i = md.to_ids("아빠")
logits, past_kv = md(i, use_cache=True)
y = infer(logits[:, -1, :], 1, 100, 0.9)
print(i)
print(logits)
print(y)

i = torch.cat((i, y), dim=1)
logits, past_kv = md(i[:, -1:], past_kv=past_kv, use_cache=True)
print(i)
print(logits)
print(past_kv)
print(md.chat("아빠"))
