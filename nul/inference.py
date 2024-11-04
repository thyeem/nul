import torch
from foc import *
from torch.nn import functional as F

from .utils import *


def process(model, method, size_block, size_seq, stopper, x, stat=False):
    """Autoregressive generation process"""

    @torch.no_grad()
    def do(x, probs):
        model.eval()
        for _ in range(size_seq):
            logits = model(cutoff(x, size_block))[:, -1, :]
            y, prob = method(logits)  # autoregressive text-gen method
            x = torch.cat((x, y), dim=1)
            probs.append(prob)
            if stopper and y.item() == stopper:  # early-stop condition
                break
        return x, probs

    r = x.size(1)
    text, probs = do(x, [])
    i, o = text[:, :r], text[:, r:]
    if stat:
        return qual(i, o, probs, model.cosim)
    else:
        return o


@torch.no_grad()
def decode_greedy(logits):
    """decode by greedy method"""
    probs, ix = F.softmax(logits, dim=-1).topk(1)
    return ix, probs.item()


@torch.no_grad()
def decode_sample(t, k, p, logits):
    """decode by sampling: pick the next token from a probability dist"""
    probs = F.softmax(
        adjust_logits(t, k, p, logits),
        dim=-1,
    )
    ix = torch.multinomial(probs, num_samples=1)
    return ix, probs.gather(-1, ix).item()


@torch.no_grad()
def adjust_logits(t, k, p, logits):
    return cf_(  # 0 means that the filter is off
        f_(top_p, p) if p != 0 else id,
        f_(top_k, k) if k != 0 else id,
    )(logits / t)


@torch.no_grad()
def top_k(k, logits):
    """top_k filter: keep the top 'k' tokens with higher probability"""
    logits[
        logits
        < fst(
            torch.topk(logits, min(k, logits.size(-1))),
        )[:, [-1]]
    ] = -float("inf")
    return logits


@torch.no_grad()
def top_p(p, logits):
    """nucleus filter: keep the top tokens LT cumulative probability of 'p'"""
    lg, ix = torch.sort(logits, descending=True)
    probs = torch.cumsum(F.softmax(lg, dim=-1), dim=-1)
    rm = probs > p
    rm[..., :1] = 0
    rm[..., 1:] = rm[..., :-1].clone()
    rm[..., 0] = 0
    logits[rm.scatter(1, ix, rm)] = -float("inf")
    return logits
