import logging
import random
from dataclasses import asdict, dataclass, field
from typing import List, Tuple

import numpy as np
import torch
import torch._dynamo
from foc import *
from ouch import *
from torch import nn

from .inference import *
from .model import Transformer
from .utils import *


@dataclass
class nulconf(autocast):
    strict: bool = True
    size_vocab: int = 20480
    size_embed: int = 256
    size_block: int = 128
    num_layers: int = 6
    num_heads: int = 8
    ratio_ffn: int = 4
    bias: bool = True
    dropout: float = 0.1
    tok: str = "t/byte-tok.json"
    seed: int = 42
    tset: List[str] = field(default_factory=list)
    vset: List[str] = field(default_factory=list)
    size_batch: int = 8
    size_chat: int = 128
    top_k: int = 100
    top_p: float = 0.9
    temperature: float = 1.0
    lr: float = 5e-4
    lr_min: float = 5e-5
    warmup: int = 100
    optim: str = "adamw"
    weight_decay: float = 0.01
    momentum: float = 0.9
    betas: Tuple[float, float] = (0.9, 0.999)
    EOT: float = 1.5
    reset: bool = False
    epoch: int = 1
    steps: int = 50000
    intv_lr: int = 20
    intv_log: int = 20
    intv_shot: int = 100
    intv_val: int = -1
    intv_ckpt: int = -1


class nul(nn.Module):
    # -----------
    # setup/info
    # -----------
    @classmethod
    def new(cls, name=None, conf=None, **kwargs):
        """Create a new model"""
        return (
            nul()
            .set_name(name)
            .set_conf(conf=conf, **kwargs)
            .set_seed()
            .set_tok()
            .set_model()
            .set_optim()
            .into()
            .finalize()
        )

    @classmethod
    def load(self, name, **kwargs):
        """Load the pre-trained"""
        path = f"o/{name}"
        guard(exists(path, "f"), f"Not found model: {name}")
        o = torch.load(path, map_location="cpu")
        return (
            nul()
            .set_name(o["name"])
            .set_conf(conf=o["conf"], **kwargs)
            .set_seed()
            .set_tok(o["tok"])
            .set_model()
            .load_model(o["model"])
            .set_optim(o["optim"])
            .finalize()
        )

    def save(self, name=None, ckpt=False):
        name = name or self.name
        path = f"o/{name}"
        d = dirname(path)
        mkdir(d)
        torch.save(
            dict(
                name=name,
                it=self.it,
                conf=asdict(self.conf) | dict(reset=False),
                optim=self.optim.state_dict() if self.optim else None,
                tok=self.tok.to_str(),
                model=self.state_dict(),
            ),
            normpath(name),
        )
        if ckpt:
            mkdir(f"{path}.snap")
            shell(f"cp -f {path} {path}.snap/{self.it:06d}")

    def set_name(self, name):
        self.name = name or base58e(randbytes(5))
        return self

    def set_conf(self, conf=None, **kwargs):
        conf = asdict(conf) if conf else {}
        self.conf = nulconf(**(conf | kwargs))
        return self

    def set_seed(self, seed=None):
        seed = seed or self.conf.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        return self

    def set_tok(self, from_str=None):
        if from_str is None:
            self.tok = read_tok(from_json=self.conf.tok)
        else:
            self.tok = read_tok(from_str=from_str)
        self.tid = self.tok.token_to_id
        return self

    def set_model(self):
        self.transformer = Transformer(self.conf)
        self.lm_head = nn.Linear(self.conf.size_embed, self.conf.size_vocab, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        return self

    def set_optim(self, optim=None):
        o = self.conf.optim
        if not o:
            self.optim = None
            return self
        self.optim = dict(
            sgd=torch.optim.SGD(
                self.transformer.parameters(),
                lr=self.conf.lr,
                weight_decay=self.conf.weight_decay,
                momentum=self.conf.momentum,
            ),
            adamw=torch.optim.AdamW(
                self.transformer.parameters(),
                lr=self.conf.lr,
                betas=self.conf.betas,
                weight_decay=self.conf.weight_decay,
            ),
            adam=torch.optim.Adam(
                self.transformer.parameters(),
                lr=self.conf.lr,
                betas=self.conf.betas,
            ),
        ).get(o) or error(f"No such optim supported: {o}")
        if not self.conf.reset and optim:
            self.optim.load_state_dict(optim)
        return self

    def load_model(self, model):
        self.load_state_dict(model, strict=self.conf.strict)
        return self

    def into(self, device=None, dtype=None):
        """set a default dtype and device based on availability"""
        device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        dtype = dtype or (
            torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )
        if hasattr(self.optim, "state"):
            for state in self.optim.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
        return self.to(device=device, dtype=dtype)

    def optimize(self):
        return torch.compile(self, backend="eager")

    def finalize(self):
        torch._dynamo.config.suppress_errors = True
        torch._logging.set_logs(dynamo=logging.ERROR)
        torch._dynamo.eval_frame.OptimizedModule.__repr__ = self.__repr__
        return self.into().optimize()

    def __repr__(self):
        return self.name if hasattr(self, "name") else ""

    @property
    def device(self):
        return next(self.transformer.parameters()).device

    @property
    def dtype(self):
        return next(self.transformer.parameters()).dtype

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

    # ------------
    # fundamental
    # ------------
    @torch.no_grad()
    def to_ids(self, x):
        """from string to tensor"""
        return torch.as_tensor(
            to_ids(self.tok)(x),
            device=self.device,
        ).unsqueeze(0)

    def from_ids(self, x):
        """from tensor to string"""
        return cf_(
            unwords,
            map(cf_(from_ids(self.tok), ob(_.tolist)())),
        )(torch.unbind(x, dim=0))

    def forward(self, x, cached=None):
        # TODO: keep context without cut-off
        x = cutoff(x, self.conf.size_block)
        return cf_(  # lm-head: (B, S, E) -> (B, S, V) logits
            self.lm_head if cached is None else bimap(self.lm_head, id),
            f_(
                self.transformer,
                cached=cached,
                ipad=self.tid(pad()),
            ),  # Transformer: (B, S) -> (B, S, E)
        )(x)

    @torch.no_grad()
    def embed(self, x, mean=True, norm=True):
        """Get embedding"""
        self.eval()
        return cf_(
            lambda x: x.mean(dim=1) if mean else x,
            f_(normalize, dim=2) if norm else id,
            f_(self.forward, embedding=True),
        )(x)

    def chat(
        self,
        prompt,
        size_chat=None,
        temperature=None,
        top_k=None,
        top_p=None,
        stopper=None,
    ):
        """generate a sequence"""
        processor = f_(
            process,  # text generator
            self,  # model
            size_chat or self.conf.size_chat,  # length of chat
            temperature or self.conf.temperature,
            top_k or self.conf.top_k,  # k in top-k filter
            top_p or self.conf.top_p,  # p in nucleus filter
            stopper or self.tid(eot()),  # token-id for early-stop
        )
        return cf_(
            self.from_ids,
            processor,
            self.to_ids,
        )(prompt)

    def when(self, x):
        return dict(
            lr=self.it % self.conf.intv_lr == 0,
            val=self.it % self.conf.intv_val == 0,
            log=self.it % self.conf.intv_log == 0,
            shot=self.it % self.conf.intv_shot == 0,
            ckpt=self.it % self.conf.intv_ckpt == 0,
        ).get(x, False)

    def update_lr(self, lr=None, lr_min=None, steps=None, warmup=None):
        """update the current learning rate (optimizer's step size)"""
        lr = lr or self.conf.lr
        lr_min = lr_min or self.conf.lr_min
        steps = steps or self.conf.steps
        warmup = warmup or self.conf.warmup
        self._lr = (
            lr
            if lr_min == -1
            else sched_lr(
                self.it,
                lr=lr,
                lr_min=lr_min,
                steps=steps,
                warmup=warmup,
            )
        )
        for param_group in self.optim.param_groups:
            param_group["lr"] = self._lr

    def token_weights(self):
        weights = torch.ones(self.tok.get_vocab_size(), device=self.device)
        weights[self.tid(eot())] = self.conf.EOT
        return weights

    def get_loss(self, x, target, mask=None, weight=None):
        logits = self(x)
        logits = logits.contiguous().view(-1, logits.size(-1))
        target = target.contiguous().view(-1)
        loss = F.cross_entropy(
            logits,
            target,
            reduction="mean" if mask is None else "none",
            weight=weight,
            ignore_index=self.tid(pad()),
        )
        if mask is None:
            return loss
        loss *= mask.view(-1)
        return loss.sum() / mask.sum()

    def human_feedback(self, prompt, response, lr=None, steps=None):
        # TODO: sequence-length
        t = self.to_ids(eop(prompt) + eot(response))
        x = t[:, :-1]
        target = t[:, 1:]
        mask = context_mask(
            target,
            self.tid(pad()),
            self.tid(eop()),
            self.tid(eot()),
            device=self.device,
        )
        self.train()
        self.update_lr(lr or self.conf.lr)
        print(purple(prompt), prompt)
        print()
        for i in tracker(range(steps or 1), "repeating"):
            self.optim.zero_grad(set_to_none=True)
            loss = self.get_loss(x, target, mask=mask)
            loss.backward()
            self.optim.step()
            print(cyan(i), self.chat(eop(prompt)))
        return self

    def self_supervised(self, src=None):
        src = src or self.conf.src
        dl = batch_from_src(
            src,
            self.conf.size_batch,
            self.conf.size_block,
            self.to_ids,
            ipad=self.tid(pad()),
            device=self.device,
        )
        g = excerptor(src, self.conf.size_block)
        self.it = 0
        for _ in tracker(range(self.conf.steps), "reading"):
            self.train()
            if self.when("lr"):
                self.update_lr()
            x, target = next(dl)
            # mask = context_mask(
            # target,
            # self.tid(pad()),
            # self.tid(eop()),
            # self.tid(eot()),
            # device=self.device,
            # )
            mask = None
            loss = self.get_loss(x, target, mask=mask)
            self.optim.zero_grad(set_to_none=True)
            loss.backward()
            self.optim.step()
            if self.when("log"):
                self.log(loss)
            if self.when("shot"):
                self.shot(context_from_text(next(g), pre=True))
            self.it += 1
        return self

    def log(self, loss):
        print()
        print(f"iter  |  {self.it}")
        print(f"  lr  |  {self._lr:.8f}")
        print(f"loss  |  {loss:.4f}")

    def shot(self, prompt):
        print()
        print(purple("prompt"), prompt)
        print()
        print(cyan("response"), self.chat(prompt))
