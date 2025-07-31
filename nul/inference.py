import torch
from foc import *
from torch.nn import functional as F

from .model import X
from .utils import *


@torch.no_grad()
def process(model, size_seq, temperature, k, p, stopper, x):
    """Autoregressive generation process"""
    model.eval()
    count = 0
    o = X(x)
    for _ in range(size_seq):
        o = model(o)
        logits = o.x[:, -1, :]
        y = infer(logits, temperature, k, p)
        if stopper and y.item() == stopper:  # early-stop condition
            break
        x = torch.cat((x, y), dim=1)
        if o.cached:
            o = X(y, cached=o.cached)
        else:
            o = X(x)
        count += 1
    return x[..., -count:]


@torch.no_grad()
def infer(logits, temperature, k, p):
    """decode by sampling: pick the next token from a probability dist"""
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)
    logits /= temperature
    probs = F.softmax(
        cf_(  # 0 means that the filter is off
            f_(topp, p) if p != 0 else id,
            f_(topk, k) if k != 0 else id,
        )(logits),
        dim=-1,
    )
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def topk(k, logits):
    """top-k filter: keep the top 'k' tokens with higher probability"""
    logits[
        logits
        < fst(
            torch.topk(logits, min(k, logits.size(-1))),
        )[:, [-1]]
    ] = -float("inf")
    return logits


@torch.no_grad()
def topp(p, logits):
    """nucleus filter: keep the top tokens LT cumulative probability of 'p'"""
    lg, ix = torch.sort(logits, descending=True)
    probs = torch.cumsum(F.softmax(lg, dim=-1), dim=-1)
    rm = probs > p
    rm[..., :1] = 0
    rm[..., 1:] = rm[..., :-1].clone()
    rm[..., 0] = 0
    logits[rm.scatter(1, ix, rm)] = -float("inf")
    return logits
