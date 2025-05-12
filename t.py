import torch
from foc import *

from nul import *

conf = nulconf(
    num_heads=2,
    num_layers=3,
    size_block=5,
    size_embed=6,
    size_batch=1,
    bias=True,
)
md = nul.new(conf=conf)

x = torch.randint(
    conf.size_vocab,
    (conf.size_batch, 1),
)  # (B, S)
y = torch.randn(
    conf.size_batch,
    1,
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
    print(blue(f"Layer{i+1} K"))
    print(cached[i][0])
    print(blue(f"Layer{i+1} V"))
    print(cached[i][1])

mask = attention_mask(x, C=len_cached_seq(cached))
print(blue("mask"))
print(mask)

o, cached = dec(mask, (y, cached))
print(o)
print(blue("cached"))
for i in range(len(dec.layers)):
    print(blue(f"Layer{i+1} K"))
    print(cached[i][0])
    print(blue(f"Layer{i+1} V"))
    print(cached[i][1])

print(cyan("Transformer"))
tf = Transformer(conf)
o = tf(x)
print(o)

o, cached = tf((x, []))
print(blue("out"))
print(o)
for i in range(len(tf.decoder.layers)):
    print(blue(f"Layer{i+1} K"))
    print(cached[i][0])
    print(blue(f"Layer{i+1} V"))
    print(cached[i][1])

o, cached = tf((x, cached))
print(blue("out"))
print(o)
for i in range(len(tf.decoder.layers)):
    print(blue(f"Layer{i+1} K"))
    print(cached[i][0])
    print(blue(f"Layer{i+1} V"))
    print(cached[i][1])


print(cyan("Text generation"))
j = "아빠"
logits, cached = md((md.to_ids(j), []))
y = infer(logits[:, -1, :], 1.0, 100, 0.9)
for i in range(len(md.transformer.decoder.layers)):
    print(blue(f"Layer{i+1} K"))
    print(cached[i][0])
    print(blue(f"Layer{i+1} V"))
    print(cached[i][1])
print(j, md.to_ids(j))
print(logits)
print(md.from_ids(y), y)

y = torch.cat((md.to_ids(j), y), dim=1)
logits, cached = md((y[:, -1:], cached))
q = infer(logits[:, -1, :], 1.0, 100, 0.9)
for i in range(len(md.transformer.decoder.layers)):
    print(blue(f"Layer{i+1} K"))
    print(cached[i][0])
    print(blue(f"Layer{i+1} V"))
    print(cached[i][1])
print(md.from_ids(y), y)
print(logits)
print(md.from_ids(q), q)

print(cyan("Text generation (with KV-cache)"))
print(md.chat(j, use_cache=True))

print(cyan("Text generation (no KV-cache)"))
print(md.chat(j, use_cache=False))
