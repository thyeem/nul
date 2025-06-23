from dataclasses import dataclass
from typing import Tuple

from ouch import *


@dataclass
class nulconf(autocast):
    size_vocab: int = 20480
    size_embed: int = 512
    size_block: int = 128
    num_layers: int = 1
    num_heads: int = 16
    ratio_ffn: int = 1
    bias: bool = True
    dropout: float = 0.05
    tok: str = "t/byte-tok.json"
    strict: bool = True
    seed: int = 42
    tset: str = "training-set"
    vset: str = "validation-set"
    size_batch: int = 32
    size_text: int = 128
    top_k: int = 50
    top_p: float = 0.9
    temperature: float = 1.0
    lr: float = 1e-3
    lr_min: float = 1e-4
    warmup: int = 1000
    optim: str = "adam"
    weight_decay: float = 0.01
    momentum: float = 0.9
    betas: Tuple[float, float] = (0.9, 0.999)
    EOT: float = 1.5
    reset: bool = False
    epochs: int = 1
    steps: int = 100000
    intv_lr: int = 20
    intv_log: int = 20
    intv_shot: int = 100
    intv_val: int = -1
    intv_ckpt: int = 5000
