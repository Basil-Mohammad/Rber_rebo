"""Language-model planner (Qwen2.5-0.5B-Instruct) trained with group-baseline policy gradients.

Policy: for step k of a task, the model scores every candidate skill *name* by its length-normalized
log-likelihood after the prompt and the prefix "Step k:"; the policy is the softmax over the candidates
(SayCan-style scoring), independently per step. Rewards come from ``rber.rewards`` (identical code to the
tabular experiments). Evaluation: true success of the greedy plan on training and held-out tasks.
"""
from __future__ import annotations

import copy
import gc
import json
import os
import time

import numpy as np
import torch

from .domain import N_OPT, prompt_text
from .rewards import SkillPosterior, batch_rewards, group_advantage


class LMPolicy:
    def __init__(self, model_id: str, lr: float, device: str | None = None, dtype=torch.float32, micro: int = 24):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_id)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        try:
            m = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
        except TypeError:  # older transformers
            m = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
        self.model = m.to(dtype).to(self.dev)  # fp32 weights: T4 GPUs have no native bf16
        if self.dev == "cuda":
            self.model.gradient_checkpointing_enable()
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.0)
        self.micro = micro
        self._cache = {}

    def enc(self, x):
        return self.tok.encode(x, add_special_tokens=False)

    def close(self):
        """Free GPU memory (model, gradients, optimizer state) immediately."""
        self.opt.zero_grad(set_to_none=True)
        del self.opt, self.model
        self._cache.clear(); gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- scoring --------------------------------------------------------------------------------------
    def _task_seqs(self, w, t):
        key = (id(w), t.id)
        if key not in self._cache:
            chat = self.tok.apply_chat_template([{"role": "user", "content": prompt_text(w, t)}], tokenize=False,
                                                add_generation_prompt=True)
            p = self.enc(chat); out = []
            for k in range(w.K):
                pre = p + self.enc(f"Step {k + 1}:")
                for j in range(N_OPT):
                    c = self.enc(" " + w.lib.names[t.opts[k, j]])
                    out.append((pre + c, len(pre), c))
            self._cache[key] = out
        return self._cache[key]

    def _seq_logp(self, items):
        """Length-normalized log-likelihood of each candidate name (full-vocabulary softmax only at the
        candidate positions; the full logit tensor is never materialized)."""
        n = max(len(x[0]) for x in items); pad = self.tok.pad_token_id
        ids = torch.full((len(items), n), pad, dtype=torch.long); att = torch.zeros_like(ids); off = []
        for i, (x, _, _) in enumerate(items):
            ids[i, n - len(x):] = torch.tensor(x); att[i, n - len(x):] = 1; off.append(n - len(x))
        ids, att = ids.to(self.dev), att.to(self.dev)
        pos = (att.cumsum(1) - 1).clamp(min=0)
        h = self.model.base_model(input_ids=ids, attention_mask=att, position_ids=pos).last_hidden_state
        head = self.model.get_output_embeddings()
        rows, cols, tgt, owner = [], [], [], []
        for i, (x, st, c) in enumerate(items):
            for q, tokid in enumerate(c):
                rows.append(i); cols.append(off[i] + st + q - 1); tgt.append(tokid); owner.append(i)
        hs = h[torch.tensor(rows, device=self.dev), torch.tensor(cols, device=self.dev)]
        logits = (hs @ head.weight.T).float()
        if getattr(head, "bias", None) is not None:
            logits = logits + head.bias.float()
        lp = torch.log_softmax(logits, -1).gather(1, torch.tensor(tgt, device=self.dev)[:, None]).squeeze(1)
        tot = torch.zeros(len(items), device=self.dev).index_add(0, torch.tensor(owner, device=self.dev), lp)
        return tot / torch.tensor([len(c) for _, _, c in items], dtype=torch.float32, device=self.dev)

    def dist(self, w, tasks):
        """log pi[t, k, j] (differentiable)."""
        items = [it for t in tasks for it in self._task_seqs(w, t)]
        parts = [self._seq_logp(items[i:i + self.micro]) for i in range(0, len(items), self.micro)]
        return torch.log_softmax(torch.cat(parts).view(len(tasks), w.K, N_OPT), -1)

    def fast_dist(self, w, tasks):
        """No-grad policy for sampling/evaluation; fp16 autocast on GPU with an fp32 fallback."""
        self.model.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=self.dev == "cuda"):
            lp = self.dist(w, tasks).float()
        if not torch.isfinite(lp).all():
            with torch.no_grad():
                lp = self.dist(w, tasks)
        return lp

    def sample(self, w, tasks, n, rng, greedy=False):
        p = self.fast_dist(w, tasks).exp().cpu().numpy().astype(np.float64)
        ch = np.zeros((len(tasks), n, w.K), int)
        for ti in range(len(tasks)):
            for k in range(w.K):
                pk = p[ti, k] / p[ti, k].sum()
                ch[ti, :, k] = pk.argmax() if greedy else rng.choice(N_OPT, size=n, p=pk)
        return ch

    def update(self, w, tasks, choices, adv):
        """One on-policy step: loss = -(1/N) sum_g adv_g sum_k log pi(choice_gk). Backward task by task."""
        self.model.train(); self.opt.zero_grad(set_to_none=True)
        N = choices.shape[0] * choices.shape[1]
        for ti, t in enumerate(tasks):
            a = torch.tensor(adv[ti], dtype=torch.float32, device=self.dev)
            if a.abs().sum() == 0:
                continue
            lp = self.dist(w, [t])[0]
            ch = torch.tensor(choices[ti], device=self.dev)
            lpc = lp[torch.arange(w.K, device=self.dev)[None, :], ch].sum(1)
            (-(a * lpc).sum() / N).backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()


def evaluate(pol, w, tasks, rng):
    ch = pol.sample(w, tasks, 1, rng, greedy=True)[:, 0]
    return float(np.mean([w.success(t, c) for t, c in zip(tasks, ch)]))


def entropy(pol, w, tasks):
    p = pol.fast_dist(w, tasks).exp().cpu().numpy()
    return float((-(p * np.log(p + 1e-12)).sum(-1)).mean())


def train(w, method, seed, out_dir, model_id="Qwen/Qwen2.5-0.5B-Instruct", lr=3e-6, n_updates=250,
          tasks_per_update=4, G=8, eval_every=25, log=print):
    """Resumable at the run level: a finished run (done=True) is skipped."""
    path = os.path.join(out_dir, f"llm_{method}_s{seed}.json")
    if os.path.exists(path) and json.load(open(path)).get("done"):
        log(f"{method} s{seed}: already done"); return path
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed); exec_rng = np.random.default_rng(seed + 10_000)
    pol = LMPolicy(model_id, lr)
    post = SkillPosterior(len(w.lib.p)) if method.startswith("RBER") else None
    opt_tr = float(np.mean([w.optimum(t) for t in w.train])); opt_he = float(np.mean([w.optimum(t) for t in w.held]))
    hist, execs, t0 = [], 0, time.time()

    def do_eval(u):
        tr = evaluate(pol, w, w.train, rng); he = evaluate(pol, w, w.held, rng); ent = entropy(pol, w, w.train)
        hist.append(dict(update=u, execs=execs, train=tr, held=he, train_norm=tr / opt_tr, held_norm=he / opt_he,
                         entropy=ent, time=time.time() - t0))
        log(f"[{method} s{seed}] upd {u:3d} execs {execs:5d} | train {tr:.3f}/{opt_tr:.3f} held {he:.3f}/{opt_he:.3f}"
            f" entropy {ent:.2f} {time.time() - t0:.0f}s")
        json.dump(dict(method=method, seed=seed, lr=lr, hist=hist, opt_train=opt_tr, opt_held=opt_he, done=False),
                  open(path, "w"))

    do_eval(0)
    for u in range(1, n_updates + 1):
        idx = rng.choice(len(w.train), tasks_per_update, replace=False)
        tasks = [w.train[i] for i in idx]
        ch = pol.sample(w, tasks, G, rng)
        adv = np.zeros((tasks_per_update, G))
        snap = copy.deepcopy(post) if post is not None else None   # lag: posterior before the whole batch
        for ti, t in enumerate(tasks):
            r, n = batch_rewards(method, w, t, ch[ti], exec_rng, post, reward_post=snap)
            execs += n; adv[ti] = group_advantage(r)
        pol.update(w, tasks, ch, adv)
        if u % eval_every == 0 or u == n_updates:
            do_eval(u)
    json.dump(dict(method=method, seed=seed, lr=lr, hist=hist, opt_train=opt_tr, opt_held=opt_he, done=True),
              open(path, "w"))
    pol.close(); del pol
    return path
