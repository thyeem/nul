import math
import re
from collections import Counter

import numpy as np
import torch
from foc import *
from ouch import *
from tokenizers import ByteLevelBPETokenizer, CharBPETokenizer, Tokenizer
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


# -----------
# jamo-tools
# -----------
def charset():
    base = [
        flat(0x0009, 0x000A, 0x000D, 0x0020),  # whitespaces
        seq(0x0021, 0x007E),  # sane ASCII
        seq(0x3131, 0x3163),  # jamotable (U+3131(ㄱ), .., U+3163(ㅣ))
        seq(0x03B1, 0x03C9),  # Greek lower
        flat(seq(0x0391, 0x03A1), seq(0x03A3, 0x03A9)),  # Greek upper
        seq(0x2013, 0x2015),  # dashes
        [0x2022, 0x2026, 0x00A7, 0x00B0, 0x2030],  # bullet, ellipsis,...
        [0x20AC, 0x00A3, 0x00A5, 0x20BD, 0x20B9],  # euro, pound, yen
        [0x2122, 0x00A9, 0x00AE, 0x20BF],  # circled symbols
        seq(0x2018, 0x201F),  # quotation marks, low-9 quotation marks
        seq(0x21D0, 0x21D4),  # double arrows
        seq(0x2713, 0x2718),  # check marks and cross marks
        flat(seq(0x2190, 0x2193), 0x21AA),  # arrows
        seq(0x00B1, 0x00B3),  # superscripts
        seq(0x00BC, 0x00BE),  # fractions
        seq(0x25A0, 0x25A1),  # white and black squares
        seq(0x25CB, 0x25CF),  # white and black circles
        seq(0x2605, 0x2606),  # black and white stars
        flat(  # mathematical symbols
            seq(0x2260, 0x2265),
            seq(0x2032, 0x2033, 0x203B),
            [0x00D7, 0x00F7, 0x222B, 0x2208, 0x2209, 0x2248, 0x221E],
        ),
    ]
    return set(map(chr, flat(base)))


def jamotbl(modmap=False):
    """modern-korean jamo characters (U+3131-U+3163):
    This is a subset of hangul-compatibility-jamo (U+3131-U+318E)
    specifically selected for practial use in modern korean.
    """
    elems = dict(
        H=[
            0x3131,  # ㄱ
            0x3132,  # ㄲ
            0x3134,  # ㄴ
            0x3137,  # ㄷ
            0x3138,  # ㄸ
            0x3139,  # ㄹ
            0x3141,  # ㅁ
            0x3142,  # ㅂ
            0x3143,  # ㅃ
            0x3145,  # ㅅ
            0x3146,  # ㅆ
            0x3147,  # ㅇ
            0x3148,  # ㅈ
            0x3149,  # ㅉ
            0x314A,  # ㅊ
            0x314B,  # ㅋ
            0x314C,  # ㅌ
            0x314D,  # ㅍ
            0x314E,  # ㅎ
        ],
        V=[
            0x314F,  # ㅏ
            0x3150,  # ㅐ
            0x3151,  # ㅑ
            0x3152,  # ㅒ
            0x3153,  # ㅓ
            0x3154,  # ㅔ
            0x3155,  # ㅕ
            0x3156,  # ㅖ
            0x3157,  # ㅗ
            0x3158,  # ㅘ
            0x3159,  # ㅙ
            0x315A,  # ㅚ
            0x315B,  # ㅛ
            0x315C,  # ㅜ
            0x315D,  # ㅝ
            0x315E,  # ㅞ
            0x315F,  # ㅟ
            0x3160,  # ㅠ
            0x3161,  # ㅡ
            0x3162,  # ㅢ
            0x3163,  # ㅣ
        ],
        T=[
            0x3131,  # ㄱ
            0x3132,  # ㄲ
            0x3133,  # ㄳ
            0x3134,  # ㄴ
            0x3135,  # ㄵ
            0x3136,  # ㄶ
            0x3137,  # ㄷ
            0x3139,  # ㄹ
            0x313A,  # ㄺ
            0x313B,  # ㄻ
            0x313C,  # ㄼ
            0x313D,  # ㄽ
            0x313E,  # ㄾ
            0x313F,  # ㄿ
            0x3140,  # ㅀ
            0x3141,  # ㅁ
            0x3142,  # ㅂ
            0x3144,  # ㅄ
            0x3145,  # ㅅ
            0x3146,  # ㅆ
            0x3147,  # ㅇ
            0x3148,  # ㅈ
            0x314A,  # ㅊ
            0x314B,  # ㅋ
            0x314C,  # ㅌ
            0x314D,  # ㅍ
            0x314E,  # ㅎ
        ],
    )
    return (
        {k: {chr(x): i for i, x in enumerate(v)} for k, v in elems.items()}
        if modmap
        else elems
    )


def to_jamos(xs):
    """Encode a given hangul-syllables (U+AC00-U+D7AF) string into the 'jamotbl',
    where jamotbl is modern korean jamo characters (U+3131-U+3163)
    """
    js = jamotbl()

    def go(x):
        return (chr(s) for s in split(x)) if 0xAC00 <= ord(x) <= 0xD7A3 else x

    def split(x):
        i = ord(x) - 0xAC00
        h, v, t = i // 28 // 21, i // 28 % 21, i % 28
        return (
            (js["H"][h], js["V"][v], js["T"][t - 1]) if t else (js["H"][h], js["V"][v])
        )

    return unchars(g for x in xs for g in go(x))


def from_jamos(xs):
    """decode the given 'jamotbl' string of an arbitrary H-V-T jamo sequence:
    it is automatically decoded according to the hangul combination rules
        - H: Cho-seong (head-consonant sounds)
        - V: Joong-seong (vowel sounds)
        - T: Jong-seong (tail-consonant sounds, optional)
    the resulting string is the hangul-syllables (U+AC00-U+D7AF)
    """
    js = jamotbl(1)
    res = []

    def o(i):
        try:
            return xs[i]
        except IndexError:
            return

    def update(i, r, n):
        r.append(
            join(o(i), o(i + 1))
            if n == 2
            else join(o(i), o(i + 1), o(i + 2)) if n == 3 else o(i)
        )
        return i + n

    def join(h, v, t=None):
        return chr(
            0xAC00
            + 28 * 21 * js["H"][h]
            + 28 * js["V"][v]
            + (0 if t is None else js["T"][t] + 1)
        )

    i = 0
    while True:
        if not o(i):  # reached end-of-input
            return unchars(res)
        elif o(i) not in js["H"] or o(i + 1) not in js["V"]:  # HV-guard
            i = update(i, res, 1)
        elif o(i + 2) in js["H"] and o(i + 3) in js["V"]:  # HVHV..
            i = update(i, res, 2)
        elif o(i + 2) in js["T"]:  # HVTX..
            i = update(i, res, 3)
        else:  # HVXX..
            i = update(i, res, 2)


# ----------
# tokenizer
# ----------
@fx
def refine(
    text,
    wspace=False,
    blanks=False,
    lower=False,
    digit=False,
    files=False,
    phone=False,
    date=False,
    email=False,
    url=False,
    tag=False,
    cparen=False,
    sparen=False,
    rparen=False,
    chinese=False,
    japanese=False,
    emoji=False,
    jamo=False,
    asciifb=False,
    custom=None,
):

    def rmcf(regex):
        return cfd_(*[rm(p, 1) for p in flatl(regex)[::-1]])(id)

    @fx
    def rm(regex, p, x, rep=""):
        return re.sub(regex, rep, x) if p else x

    lower_ = str.lower if lower else id
    digit_ = rm(r"\d+", digit)
    files_ = rm(r"\S+\.(jpe?g|png|gif|pdf|pptx?|docx?|xlsx?|txt)\b", files)
    phone_ = rm(r"[()+\d.\-]*[ ]?\d{2,4}[-. ]+\d{3,4}[-. ]+\d{3,4}", phone)
    date_ = rm(r"(\d{1,4}(?:/|-)\d{1,4}(?:/|-)\d{1,4})", date)
    email_ = rm(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b", email)
    url_ = rm(r"(https?|ftp|www)\S+", url)
    tag_ = rm(r"<[^>]*>", tag)
    cparen_ = rm(r"\{+[^\{\}]*\}+", cparen)
    sparen_ = rm(r"\[+[^\[\]]*\]+", sparen)
    rparen_ = rm(r"\(+[^\(\)]*\)+", rparen)
    jamo_ = to_jamos if jamo else id
    wspace_ = rm(r"\s+", wspace, rep=" ")
    blanks_ = rm(r" +", blanks, rep=" ")
    custom_ = rmcf(custom) if custom else id
    chinese_ = rm("[\u4e00-\u9fff]+", chinese)
    japanese_ = rm("[\u3040-\u30ff]+", japanese)
    emoji_ = rm(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "]+",
        emoji,
    )

    return cf_(
        unchars,
        custom_,
        lower_,
        digit_,
        files_,
        phone_,
        date_,
        email_,
        url_,
        tag_,
        cparen_,
        sparen_,
        rparen_,
        japanese_,
        chinese_,
        emoji_,
        jamo_,
        wspace_,
        blanks_,
    )(text)


def eop(x=""):
    """append the special token '<|endofprompt|>' at the end.
    the end of a prompt is the beginning of a new sequence.
    """
    return f"{x}<|endofprompt|>"


def eot(x=""):
    """append the special token '<|endoftext|>' at the end.
    the end of a sequence is the beginning of a new prompt.
    """
    return f"{x}<|endoftext|>"


def pad():
    """return the special token '<|pad|>': padding"""
    return "<|pad|>"


def unk():
    """return the special token '<|unk|>': out-of-vocab"""
    return "<|unk|>"


def read_tok(from_json=None, from_str=None):
    """read a byte|char BPE tokenizer."""
    if from_json:
        f = normpath(from_json)
        guard(exists(f), f"not found: {f}")
        return Tokenizer.from_file(f)
    elif from_str:
        return Tokenizer.from_str(from_str)
    else:
        error("Error, no given tokenizer to load")


def stream(src, from_str=False, jamo=False, size=2 << 12, **ref):
    chunks = flat(
        chunks_str(size, from_str)
        if from_str
        else (chunks_file(size, f) for f in flat(src))
    )
    refiner = refine(jamo=jamo, **ref)
    for chunk in chunks:
        yield refiner(chunk)


def train_tok(
    src,
    from_str=False,
    byte_level=True,
    size_vocab=5000,
    size_chunk=2 << 12,
    min_frequency=5,
    dropout=None,
    to_file=None,
    **ref,  # e.g., custom=rf"\s?<\|unk\|>|<\|pad\|>\s?"
):
    """train a byte|char BPE tokenizer"""
    special_tokens = [unk(), pad(), eop(), eot()]
    iterator = stream(
        src,
        from_str=from_str,
        jamo=not byte_level,
        size=size_chunk,
        **ref,  # kwargs for refiner or text-normalizer
    )

    def bytetok():
        t = ByteLevelBPETokenizer(
            unicode_normalizer="nfkc",
            trim_offsets=True,
            dropout=dropout,
        )
        t.train_from_iterator(
            iterator,
            vocab_size=size_vocab,
            min_frequency=min_frequency,
            special_tokens=special_tokens[1:],
        )
        return t

    def chartok():
        t = CharBPETokenizer(
            unk_token=unk(),
            bert_normalizer=False,
            split_on_whitespace_only=True,
            dropout=dropout,
        )
        t.train_from_iterator(
            iterator,
            vocab_size=size_vocab,
            initial_alphabet=list(charset()),
            limit_alphabet=len(charset()),
            min_frequency=min_frequency,
            special_tokens=special_tokens,
        )
        return t

    tok = bytetok() if byte_level else chartok()
    to_file and tok.save(normpath(to_file))
    return tok


def is_byte_tok(t):
    """Check if a given tokenizer is a byte-level BPE tokenizer."""
    vocab = t.get_vocab()
    return any(marker in vocab for marker in ["Ġ", "Ċ"])


def to_tokens(t):
    """token encoder builder using tokenizer 't'.
    valid only when 't' is a 'Tokenizer' instance
    """
    return cf_(
        _.tokens,
        t.encode,
        id if is_byte_tok(t) else to_jamos,
    )


def to_ids(t):
    """token-id encoder builder using tokenizer 't'.
    valid only when 't' is a 'Tokenizer' instance
    """
    return cf_(
        _.ids,
        t.encode,
        id if is_byte_tok(t) else to_jamos,
    )


def from_ids(t):
    """token-id decoder builder using tokenizer 't'.
    valid only when 't' is a 'Tokenizer' instance
    """
    return cf_(
        id if is_byte_tok(t) else from_jamos,
        lambda x: t.decode(x),
    )


@fx
def vocab_freq(t, d):
    """get vocab frequency from tokenizer and data-src"""
    return cf_(Counter, flatl, map(to_tokens(t)))(d)


# ----------
# nul-utils
# ----------


@fx
def dumper(x, **kwargs):
    print()
    pp(x | dmap(**kwargs), margin=16, width=120, sort=False)
    print()


@torch.no_grad()
def attention_mask(x, size_kv=0, ipad=0):
    """generate a padding-considered causual mask that controls which tokens
    the model should not pay attention to. (B, S) -> (B, 1, S, S)
    """
    L = x.size(-1) + size_kv
    causal_mask = torch.tril(
        torch.ones(1, 1, x.size(-1), L, dtype=torch.bool, device=x.device)
    )
    padding_mask = (x != ipad).unsqueeze(1).unsqueeze(1).repeat(1, 1, x.size(-1), 1)
    return causal_mask & padding_mask


@torch.no_grad()
def context_mask(x, ipad, ieop, ieot, device="cpu"):
    """generate a mask that ignores context tokens.
    'x' tensor must be the target part of a teacher-forcing pair.
    """
    B, _ = x.shape
    mask = torch.zeros_like(x, dtype=torch.bool, device=device)
    for i in range(B):
        p = (x[i] == ieop).nonzero(as_tuple=True)[0]
        s = (x[i] == ieot).nonzero(as_tuple=True)[0]
        P = p[-1] if len(p) > 0 else -1  # where the last <eop> is found
        if P >= 0:
            for start in p:
                end_candidates = s[s > start]
                if len(end_candidates) > 0:
                    end = end_candidates[0]
                    mask[i, start + 1 : end + 1] = 1
                else:
                    mask[i, start + 1 :] = 1
        else:
            mask[i, :] = 1  # open the entire tokens
        mask[i, x[i] == ipad] = 0  # masking pad tokens
    return mask


def bleu(refs, x, weights=(0.25, 0.25, 0.25, 0.25), epsilon=1e-12):
    """Calculate BLEU using references, ``refs`` and candidate, ``x``.

    BLEU = brevity-penalty * exp(sum_i(weights[i] * log(precisions[i])))
    """

    def count_ngram(tokens, n):
        o = {}
        for ngram in [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]:
            o[ngram] = o.get(ngram, 0) + 1
        return o

    def precision(refs, x, n):
        rs = [count_ngram(ref, n) for ref in refs]
        c = count_ngram(x, n)
        total = sum([min(c[ngram], max(r.get(ngram, 0) for r in rs)) for ngram in c])
        return total / (sum(c.values()) or 1)

    def penalty(refs, x):  # brevity penalty
        c = len(x)
        r = min(len(r) - c for r in refs)
        r = -1 if r < 0 else r + c
        return 0 if c == 0 else 1 if c > r else np.exp(1 - r / c)

    ps = [precision(refs, x, n) for n in range(1, len(weights) + 1)]
    if sum(ps):  # clipping precision to epsilon (if any matches)
        ps = [max(p, epsilon) for p in ps]
    sigma = np.sum([w * np.log(p) for w, p in zip(weights[: len(x)], ps)])
    return penalty(refs, x) * np.exp(sigma)


def cutoff(x, limit=0):
    """get the desired length of the last dimension for a given 'torch.Tensor'"""
    return x if x.size(-1) <= limit else x[..., -limit:]


def standardize(x, eps=1e-7):
    """transform 'torch.tensor' or 'numpy.array') with a mean of 0 and a stdev of 1"""
    return (x - x.mean()) / (x.std() + eps)


def normalize(x, dim=1, eps=1e-7):
    """normalize 'torch.tensor' so that its magnitude is 1"""
    return x / (x.norm(dim=dim, keepdim=True) + eps)


@fx
def sched_lr(it, lr=1e-3, lr_min=1e-5, steps=10000, warmup=1000):
    """Learning rate scheduler (cosine-annealing with warmup)"""
    it %= steps
    if it < warmup:
        return lr * it / warmup
    if it > steps:
        return lr_min
    decay_ratio = (it - warmup) / (steps - warmup)
    guard(0 <= decay_ratio <= 1, f"Error, invalid decay ratio: {decay_ratio}")
    return lr_min + 0.5 * (lr - lr_min) * (1.0 + math.cos(math.pi * decay_ratio))


def write_memmap(f, o):
    fp = np.memmap(f, dtype=np.uint16, mode="w+", shape=(len(o),))
    fp[:] = list(o)[:]
    fp.flush()
    return fp


def read_memmap(f):
    guard(exists(f, "f"), f"Error, not found memmap file: {f}")
    return np.memmap(f, dtype=np.uint16, mode="r")


def tloader(x, batch_size=1, shuffle=True, **kwargs):
    """Construct tensor-dataset loader for a given (tensor) data"""
    t = (
        TensorDataset(
            *mapl(torch.as_tensor, x),
        )
        if isinstance(x, (tuple, list))
        else x
    )
    return DataLoader(t, batch_size=batch_size, shuffle=shuffle, **kwargs)


def excerptor(src, length):
    fs = [open(f, mode="r", encoding="utf-8", errors="ignore") for f in flat(src)]
    sizes = [f.seek(0, 2) or f.tell() for f in fs]
    while True:
        i = randint(len(fs))
        f = fs[i]
        size = sizes[i]
        start = randint(0, size - length - 1)
        f.seek(start)
        while start > 0:
            f.seek(start)
            if f.read(1).isspace():
                break
            start -= 1
        end = start + length
        if end >= size:
            end = size - 1
        f.seek(end)
        while end < size:
            f.seek(end)
            if f.read(1).isspace():
                break
            end += 1
        f.seek(start)
        yield f.read(end - start)


def context_from_text(text):
    q = text.rfind(eop())
    if q == -1:
        return eop()
    return text[: q + len(eop())]


@torch.no_grad()
def batch_from_src(src, B, W, encoder, ipad=0, device="cpu"):
    """get a mini-batch (x[B, W], target[B, W]) from a given source"""

    def pad_(t, size, ipad=ipad):
        if t.size(-1) < size:
            return F.pad(t, (0, size - t.size(-1)), value=ipad)
        return t

    g = excerptor(src, W * 4)
    while True:
        yield mapl(
            torch.stack,
            zip(
                *[
                    cf_(
                        lambda x: (pad_(x[..., :W], W), pad_(x[..., 1 : W + 1], W)),
                        lambda x: encoder(next(x))[..., -W - 1 :].squeeze(),
                    )(g)
                    for _ in range(B)
                ]
            ),
        )


def ansicolor(code):
    return lambda x: f"\033[{code}m{x}\033[0m"


def red(x):
    return ansicolor(31)(x)


def green(x):
    return ansicolor(32)(x)


def yellow(x):
    return ansicolor(33)(x)


def blue(x):
    return ansicolor(34)(x)


def purple(x):
    return ansicolor(35)(x)


def cyan(x):
    return ansicolor(36)(x)
