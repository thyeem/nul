import random

import numpy as np
import torch
import torch._dynamo
from foc import *
from ouch import *
from rich.console import Console
from torch import nn

from .inference import *
from .model import Decoder, LayerNorm
from .utils import *

con = Console(width=100)


def nulconf(conf=None, **kwds):
    o = dmap(
        strict=True,
        size_vocab=4096,
        size_embed=128,
        size_block=128,
        num_layers=8,
        num_heads=8,
        ratio_ffn=2,
        bias=True,
        dropout=0.1,
        seed=42,
        texts=[],
        size_batch=8,
        size_chat=64,
        top_k=100,
        top_p=0.9,
        temperature=0.7,
        lr=1e-3,
        lr_min=1e-4,
        optim="sgd",
        weight_decay=1e-4,
        momentum=0.9,
        betas=(0.9, 0.999),
        tokenizer="t/cbpe.json",
    )

    def quar(x):
        x = x or {}
        for k in x:
            if k not in o:
                error(f"found invalid key: {k}")
        return dmap(x)

    return o | quar(conf) | quar(dict(**kwds))


class nul(nn.Module):
    # -----------
    # setup/info
    # -----------
    @classmethod
    def new(cls, conf=None, **kwds):
        """Create a new model"""
        return (
            nul()
            .set_conf(conf, **kwds)
            .set_seed()
            .set_tokenizer()
            .set_model()
            .set_optim()
            .into()
            .optimize()
        )

    @classmethod
    def load(self, model):
        """Load the pre-trained"""
        guard(exists(model, "f"), f"Not found model: {model}")
        o = torch.load(model, map_location="cpu")
        return (
            nul()
            .set_conf(o["conf"])
            .set_seed()
            .set_tokenizer(o["tokenizer"])
            .set_model()
            .load_model(o["model"])
            .set_optim()
            .optimize()
        )

    def set_conf(self, conf=None, **kwds):
        self.conf = nulconf(conf, **kwds)
        return self

    def set_seed(self, seed=None):
        seed = seed or self.conf.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        return self

    def set_tokenizer(self, from_str=None):
        if from_str is None:
            self.tokenizer = read_tok(from_json=self.conf.tokenizer)
        else:
            self.tokenizer = read_tok(from_str=from_str)
        return self

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
        return self.to(device=device, dtype=dtype)

    def optimize(self):
        import torch._dynamo

        torch._dynamo.config.suppress_errors = True
        return torch.compile(self, backend="eager")

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
            to_ids(self.tokenizer)(x),
            device=self.device,
        ).unsqueeze(0)

    def from_ids(self, x):
        """from tensor to string"""
        return cf_(
            unwords,
            map(cf_(from_ids(self.tokenizer), ob(_.tolist)())),
        )(torch.unbind(x, dim=0))

    def forward(self, x, embedding=False):
        """forward
        input(B,S) -> logits(B,S,V)
        """
        x = cutoff(x, self.conf.size_block)
        B, S = x.size()  # batch size, sequence length
        mask = attention_mask(x, self.tokenizer.token_to_id(pad()))
        return cf_(
            id if embedding else self.lm_head,  # (B,S,E) -> (B,S,V)
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
            stopper or self.tokenizer.token_to_id(eot()),  # token-id for early-stop
        )
        return cf_(
            self.from_ids,
            processor,
            self.to_ids,
        )(prompt)

    def get_loss(self, x, target, mask=None, weight=None):
        logits = self(x)
        logits = logits.contiguous().view(-1, logits.size(-1))
        target = target.contiguous().view(-1)
        loss = F.cross_entropy(
            logits,
            target,
            reduction="mean" if mask is None else "none",
            weight=weight,
            ignore_index=self.tokenizer.token_to_id(pad()),
        )
        if mask is None:
            return loss
        loss *= mask.view(-1)
        return loss.sum() / mask.sum()

    def update_lr(self, lr=None):
        """update the current learning rate (optimizer's step size)"""
        if lr is None:
            self._lr = sched_lr(
                self.it,
                lr=self.conf.lr,
                lr_min=self.conf.lr_min,
                steps=self.conf.steps,
                warmup=self.conf.warmup,
            )
        else:
            self._lr = lr
        for param_group in self.optim.param_groups:
            param_group["lr"] = self._lr

    def token_weights(self):
        weights = torch.ones(self.tokenizer.get_vocab_size(), device=self.device)
        weights[self.tokenizer.token_to_id(eot())] = 2.0
        return weights

    def train_by_chat(self, prompt, target, lr=None, steps=None):
        # TODO: sequence-length
        t = self.to_ids(eop(prompt) + eot(target))
        x = t[:, :-1]
        target = t[:, 1:]
        mask = context_mask(
            t[:, 1:],
            self.tokenizer.token_to_id(pad()),
            self.tokenizer.token_to_id(eop()),
            self.tokenizer.token_to_id(eot()),
            device=self.device,
        )
        weight = self.token_weights()
        self.train()
        self.update_lr(lr or self.conf.lr)
        print(f"\033[35mprompt\033[0m {prompt}")
        print()
        for i in tracker(range(steps or 1), "repeating"):
            loss = self.get_loss(x, target, mask=mask, weight=weight)
            self.optim.zero_grad(set_to_none=True)
            loss.backward()
            self.optim.step()
            print(f"\033[36m{i}\033[0m {self.chat(prompt)}")
        return self

    def train_by_reading(self, texts=None, lr=None, steps=None):
        self.train()
        self.update_lr(lr or self.conf.lr)
        g = excerpt_text(texts or self.conf.texts, 3 * self.conf.size_block)
        dl = batch_from_g(
            g,
            size_batch=self.conf.size_batch,
            size_block=self.conf.size_block,
            encoder=self.to_ids,
            ipad=self.tokenizer.token_to_id(pad()),
            device=self.device,
        )
        weight = self.token_weights()
        self.it = 0
        for _ in tracker(range(steps or self.conf.step), "reading"):
            x, target = next(dl)
            loss = self.get_loss(x, target, weight=weight)
            self.optim.zero_grad(set_to_none=True)
            loss.backward()
            self.optim.step()
            if self.it % 100 == 0:
                prompt = next(g)
                print()
                print(f"\033[35mprompt\033[0m {prompt}")
                print()
                print(f"\033[36mresponse\033[0m {self.chat(prompt)}")
            self.it += 1
        return self


def dataloader():
    return
