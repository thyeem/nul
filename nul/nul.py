import logging
import os
import random
from dataclasses import asdict

import numpy as np
import torch
import torch._dynamo
from foc import *
from ouch import *
from torch import nn

from .conf import nulconf
from .inference import *
from .model import *
from .utils import *


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
        )

    @classmethod
    def load(self, name, conf=None, **kwargs):
        """Load the pre-trained"""
        path = which_model(name)
        o = torch.load(path, map_location="cpu", weights_only=False)
        return (
            nul()
            .set_name(o.get("name"))
            .set_conf(conf=conf, **o["conf"], **kwargs)
            .set_seed()
            .set_tok(o["tok"])
            .set_model()
            .load_model(o["model"])
            .set_iter(o.get("it"))
            .set_stat(o.get("stat"))
            .set_optim(o.get("optim"))
            .finalize()
        )

    def save(self, path=None, checkpoint=False):
        path = path or path_model(self.name)
        mkdir(dirname(path))
        obj = dict(
            name=basename(path),
            conf=asdict(self.conf),
            tok=self.tok.to_str(),
            model=self.state_dict(),
        )
        if checkpoint:
            obj |= dict(
                name=stripext(basename(path)),
                it=self.it,
                optim=self.optim.state_dict(),
                stat=deepdict(self.stat),
            )
        torch.save(obj, normpath(path))
        return path

    def checkpoint(self, best=False, retain=12):
        if not is_checkpoint(self):
            die("Error, cannot create a checkpoint")
        if best:
            path = path_model(f"{self.name}.ckpt")
        else:
            dir = path_model(f"{self.name}.train")
            mkdir(dir)
            suffix = f"-{self.stat.valoss:.2f}-{self.it:06d}"
            path = f"{dir}/{self.name}{suffix}"
            for f in shell(f"find {dir} -type f | sort -V")[retain:]:
                os.remove(f)
        self.save(path=path, checkpoint=True)

    def set_name(self, name):
        self.name = name
        return self

    def set_conf(self, conf=None, **kwargs):
        conf = read_conf(conf) if conf is not None else {}
        self.conf = nulconf(**(kwargs | dict(conf)))
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
        self.conf.size_vocab = self.tok.get_vocab_size()
        self.tid = self.tok.token_to_id
        return self

    def set_model(self):
        self.transformer = Transformer(self.conf)
        self.lm_head = nn.Linear(self.conf.size_embed, self.conf.size_vocab, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        return self

    def set_iter(self, it):
        self.it = self.conf.it or it or 0
        return self

    def set_stat(self, stat=None):
        if stat is None:
            self.stat = dmap(
                loss=float("inf"),
                valoss=float("inf"),
                minloss=float("inf"),
                alpha=0.1,
            )
        else:
            self.stat = dmap(stat)
        self.dq = dmap(
            loss=dataq(self.conf.intv_log),
            valoss=dataq(self.conf.intv_val),
        )
        self.ema = ema(alpha=self.stat.alpha)
        return self

    def set_optim(self, optim=None):
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
        ).get(self.conf.optim or "sgd") or die(f"No such optim supported: {o}")
        if optim:
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
        if hasattr(self, "optim") and hasattr(self.optim, "state"):
            for state in self.optim.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
        return self.to(device=device, dtype=dtype)

    def optimize(self):
        return torch.compile(self, backend="eager")

    def finalize(self):
        torch._dynamo.config.suppress_errors = True
        torch._dynamo.eval_frame.OptimizedModule.__repr__ = self.__repr__
        torch._logging.set_logs(dynamo=logging.ERROR)
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

    @property
    def summary(self):
        path = path_model(self.name)
        extra = {"size": du_hs(path), "path": path} if exists(path) else {}
        return {
            "model": self.name,
            "parameters": f"{self.numel:_}",
            "vocab-size": self.conf.size_vocab,
            "embedding-size": self.conf.size_embed,
            "block-size": self.conf.size_block,
            "num-layers": self.conf.num_layers,
            "MUX-size": self.conf.size_mux,
            "num-heads": self.conf.num_heads,
            "FFN-ratio": self.conf.ratio_ffn,
        } | extra

    def info(self, arch=False):
        if arch:
            print(self.transformer)
        dumper(self.summary)

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

    def forward(self, x):
        if not isinstance(x, X):
            x = X(x)
        x.x = cutoff(x.x, limit=self.conf.size_block)
        return cf_(
            with_x(self.lm_head),  # (B, S, E) -> (B, S, V) logits
            f_(self.transformer, ipad=self.tid(pad())),  # (B, S) -> (B, S, E)
        )(x)

    @torch.no_grad()
    def embed(self, x, mean=True, norm=True):
        self.eval()
        return cf_(
            lambda x: x.mean(dim=1) if mean else x,
            f_(normalize, dim=2) if norm else id,
            _.x,
            f_(self.transformer, ipad=self.tid(pad())),
            X,
        )(x)

    def invoke(
        self,
        prompt,
        size_text=None,
        temperature=None,
        top_k=None,
        top_p=None,
        stopper=None,
        chat=False,
    ):
        """generate a sequence"""
        processor = f_(
            process,  # text generator
            self,  # model
            size_text or self.conf.size_text,  # length of text
            temperature or self.conf.temperature,
            top_k or self.conf.top_k,  # k in top-k filter
            top_p or self.conf.top_p,  # p in nucleus filter
            stopper or self.tid(eot()),  # token-id for early-stop
        )
        x = eop(prompt) if chat else prompt
        return cf_(
            self.from_ids,
            processor,
            self.to_ids,
        )(x)

    def when(self, x):
        return dict(
            lr=self.it % self.conf.intv_lr == 0,
            val=self.it % self.conf.intv_val == 0,
            log=self.it % self.conf.intv_log == 0,
            shot=self.it % self.conf.intv_shot == 0,
            checkpoint=self.it % self.conf.intv_ckpt == 0,
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

    def data_loader(self):
        return map(
            lambda x: batch_from_src(
                x,
                self.conf.size_batch,
                self.conf.size_block,
                self.to_ids,
                ipad=self.tid(pad()),
                device=self.device,
            ),
            (self.conf.tset, self.conf.vset),
        )

    def self_supervised(self):
        tl, vl = self.data_loader()  # train-set, validation-set
        g = excerptor(self.conf.tset, self.conf.size_block)
        steps = self.conf.steps * self.conf.epochs  # global steps
        for _ in tracker(range(steps), "supervised", start=self.it):
            self.train()
            if self.when("lr"):
                self.update_lr()
            x, target = next(tl)
            logits = self(x).x
            loss = F.cross_entropy(
                logits.contiguous().view(-1, logits.size(-1)),
                target.contiguous().view(-1),
                reduction="mean",
            )
            self.optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=2.0)
            self.optim.step()
            self.dq.loss.update(loss.item())
            if self.when("val"):
                self.validate(vl)
            if self.when("shot"):
                self.shot(context_from_text(next(g), pre=True))
            if self.when("log"):
                self.log()
            if self.when("checkpoint"):
                self.checkpoint()
            self.it += 1
        return self

    @torch.no_grad()
    def validate(self, vl):
        self.eval()
        for _ in tracker(range(self.conf.size_val), "validation"):
            x, target = next(vl)
            logits = self(x).x
            loss = F.cross_entropy(
                logits.contiguous().view(-1, logits.size(-1)),
                target.contiguous().view(-1),
                reduction="mean",
            )
            self.dq.valoss.update(loss.item())
        self.stat.valoss = self.ema(self.stat.valoss, self.dq.valoss.median)
        if self.stat.valoss < self.stat.minloss:
            self.stat.minloss = self.stat.valoss
            self.checkpoint(best=True)
            self.save()

    def log(self):
        self.stat.loss = self.ema(self.stat.loss, self.dq.loss.median)
        valmin = f"{self.stat.valoss:.4f} >= {self.stat.minloss:.4f}"
        print()
        print(f"      STEP  |  {self.it:06d}")
        print(f"        LR  |  {self._lr:.8f}")
        print(f" VAL (MIN)  |  {self.dq.valoss.median:.4f} ({valmin})")
        print(f"LOSS (EMA)  |  {self.dq.loss.median:.4f} ({self.stat.loss:.4f})")

    def shot(self, prompt):
        print()
        print(purple("prompt"), prompt)
        print()
        print(cyan("response"), self.invoke(prompt))
