from dataclasses import dataclass
from typing import Tuple

from ouch import *


@dataclass
class nulconf(autocast):
    size_vocab: int = -1
    size_embed: int = 256
    size_block: int = 128
    num_layers: int = 3
    num_heads: int = 2
    size_mux: int = 1
    ratio_ffn: int = 2
    bias: bool = True
    dropout: float = 0.1
    tok: str = "tok/byte-10240.json"
    strict: bool = True
    seed: int = 42
    tset: str = "training-set"
    vset: str = "validation-set"
    size_batch: int = 8
    size_text: int = 128
    top_k: int = 50
    top_p: float = 0.9
    temperature: float = 1.0
    lr: float = 1e-4
    lr_min: float = 1e-5
    warmup: int = 1000
    optim: str = "adam"
    weight_decay: float = 0.01
    momentum: float = 0.9
    betas: Tuple[float, float] = (0.9, 0.999)
    EOT: float = 1.5
    epochs: int = 2
    steps: int = 100000
    it: int = 0
    intv_lr: int = 10
    intv_log: int = 100
    size_val: int = 50
    intv_val: int = 500
    intv_shot: int = 500
    intv_ckpt: int = 5000
