import random

import numpy as np
import torch
import torch._dynamo
from foc import *
from ouch import *
from torch import nn

from .inference import *
from .model import Decoder, LayerNorm
from .utils import *


def nulconf(conf=None, **kwds):
    o = dmap(
        strict=True,
        size_vocab=4096,
        size_embed=64,
        size_block=64,
        num_layers=4,
        num_heads=4,
        ratio_ffn=2,
        bias=True,
        dropout=0.1,
        seed=42,
        tokenizer="t/bbpe.json",
    )

    def quar(x):
        x = x or {}
        for k in x:
            if k not in o:
                error(f"found invalid key: {k}")
        return dmap(x)

    return o | quar(conf) | quar(dict(**kwds))


class nul(nn.Module):
    # -------------------
    # Setup/Info
    # -------------------
    @classmethod
    def new(cls, conf=None, **kwds):
        """Create a new model"""
        o = nul()
        o.set_conf(conf, **kwds)
        o.set_seed()
        o.set_tokenizer(f=o.conf.tokenizer)
        o.set_model()
        o = o.into()
        return o.optimize()

    @classmethod
    def load(self, model):
        """Load the pre-trained"""
        guard(exists(model, "f"), f"Not found model: {model}")
        o = torch.load(model, map_location="cpu")
        self.set_conf(o["conf"])
        self.set_seed()
        self.set_tokenizer(s=o["tokenizer"])
        self.set_model()
        self.load_model(o["model"])
        return self.optimize()

    def set_conf(self, conf=None, **kwds):
        self.conf = nulconf(conf, **kwds)

    def set_seed(self, seed=None):
        seed = seed or self.conf.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def set_tokenizer(self, f=None, s=None):
        self.tokenizer = read_tok(from_json=f, from_str=s)

    def set_model(self):
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(self.conf.size_vocab, self.conf.size_embed),
                wpe=nn.Embedding(self.conf.size_block, self.conf.size_embed),
                decoder=Decoder(self.conf),
                ln=LayerNorm(self.conf.size_embed, bias=self.conf.bias),
            )
        )
        self.lm_head = nn.Linear(self.conf.size_embed, self.conf.size_vocab, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

    def load_model(self, model):
        self.load_state_dict(model, strict=self.conf.strict)

    def into(self, device=None, dtype=None):
        """set a default dtype and device based on availability"""
        device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        dtype = dtype or (
            torch.bfloat16 if torch.cuda.is_available() else torch.float16
        )
        return self.to(device=device, dtype=dtype)

    def optimize(self):
        import torch._dynamo

        torch._dynamo.config.suppress_errors = True
        return torch.compile(self, backend="eager")

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def numel(self):
        return (
            sum(p.numel() for p in self.parameters())
            - self.transformer.wpe.weight.numel()
        )

    def info(self):
        dumper(dict(self.transformer))
        dumper(dict(num_params=f"{self.numel:_}"))
        dumper(self.conf)

    # -------------------
    # Fundamental
    # -------------------
    @torch.no_grad()
    def to_ids(self, x):
        """from string to tensor"""
        return torch.as_tensor(
            to_ids(self.tokenizer)(x),
            device=self.device,
        ).unsqueeze(0)

    def from_ids(self, x):
        """from tensor to string"""
        return cf_(
            unwords,
            map(cf_(from_ids(self.tokenizer), list)),
        )(torch.unbind(x, dim=0))

    def forward(self, x):
        """forward
        input(B,S) -> logits(B,S,V)
        """
        x = cutoff(x, self.conf.size_block)
        B, S = x.size()  # batch size, sequence length
        mask = attention_mask(x, self.tokenizer.token_to_id(pad()))
        return cf_(
            self.lm_head,  # (B,S,E) -> (B,S,V)
            self.transformer.ln,
            f_(self.transformer.decoder, mask),  # (B,S,E)
            _ + self.pos(S),  # W_p[:,t]: (B,S,E) + (1,S,E)
            self.transformer.wte,  # W_e[:,x(t)]: (B,S) -> (B,S,E)
        )(x)

    def pos(self, S):
        """Incremental positional vector"""
        return self.transformer.wpe(  # W_p[:,t] -> (1,S,E)
            torch.arange(  # [[0,1,..,S]] -> (1,S)
                0, S, dtype=torch.long, device=self.device
            ).unsqueeze(0)
        )

    @torch.no_grad()
    def embed(self, x, mean=True, norm=True):
        """Get embedding"""
        return cf_(
            lambda x: x.mean(dim=1) if mean else x,
            f_(normalize, dim=2) if norm else id,
            self._forward,
        )(x)

    @torch.no_grad()
    def _forward(self, x):
        """"""
        self.eval()
        x = cutoff(x, self.conf.size_block)
        S = x.size(1)
        mask = attention_mask(x, self.tokenizer.token_to_id(pad()))
        return cf_(
            self.transformer.ln,
            f_(self.transformer.decoder, mask),
            _ + self.pos(S),
            self.transformer.wte,
        )(x)

    @torch.no_grad()
    def broca(self, x, stat=False):
        "Broca's area in the frontal lobe"
        self.eval()
        return cf_(
            lambda x: process(
                self,
                f_(
                    decode_sample,
                    self.conf.temperature,
                    self.conf.top_k,
                    self.conf.top_p,
                ),
                self.conf.size_block,
                3,
                None,
                x,
                stat=stat,
            ),
            self.lm_head,
        )(x)
