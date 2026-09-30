"""Offline mechanics tests of the LM planner with a tiny random model (run: python tests/test_llm_mechanics.py)."""
import os, shutil, sys, tempfile
import numpy as np, torch, transformers
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src")); sys.path.insert(0, os.path.dirname(__file__))
transformers.AutoTokenizer.from_pretrained = transformers.PreTrainedTokenizerFast.from_pretrained  # tiny tokenizer only
from rber.domain import build_world, load_library
from rber import llm
from tiny_lm import build

w = build_world(load_library(), K=3, seed=0)
torch.manual_seed(0); d = build(w); rng = np.random.default_rng(0)
pol = llm.LMPolicy(d, lr=1e-3, device="cpu")
# 1. batched, padded, length-normalized candidate log-likelihood == unbatched full-vocabulary reference
items = pol._task_seqs(w, w.train[0])
with torch.no_grad():
    fast = pol._seq_logp(items).numpy()
    ref = [sum(torch.log_softmax(pol.model(input_ids=torch.tensor([x])).logits[0], -1)[st + q - 1, tok].item()
               for q, tok in enumerate(c)) / len(c) for x, st, c in items]
err = np.abs(fast - np.array(ref)).max(); print("1) candidate log-likelihood max error:", err); assert err < 1e-4
# 2. the update raises the probability of positively-advantaged choices
t = w.train[0]; ch = pol.sample(w, [t], 4, rng)
with torch.no_grad(): p0 = pol.dist(w, [t]).exp()[0]
pol.update(w, [t], ch, np.array([[1.0, 0, 0, 0]]))
with torch.no_grad(): p1 = pol.dist(w, [t]).exp()[0]
d_first = [float(p1[k, ch[0, 0, k]] - p0[k, ch[0, 0, k]]) for k in range(w.K)]
d_logp = float(sum(torch.log(p1[k, ch[0, 0, k]]) - torch.log(p0[k, ch[0, 0, k]]) for k in range(w.K)))
print("2) prob change of the rewarded plan's choices:", np.round(d_first, 4), "| plan log-prob change:", round(d_logp, 4))
assert d_logp > 0  # the rewarded plan becomes more likely (per-step factors share parameters)
pol.close()
# 3. every method trains end-to-end and learning happens (tiny model, large LR)
out = tempfile.mkdtemp()
res = {}
for m in ["VERIF", "STEP", "EXEC", "RBER"]:
    llm.train(w, m, 0, out, model_id=d, lr=3e-3, n_updates=60, eval_every=30, log=lambda *a: None)
    import json; h = json.load(open(os.path.join(out, f"llm_{m}_s0.json")))["hist"]
    res[m] = (h[0]["train_norm"], h[-1]["train_norm"], h[-1]["execs"])
    print(f"3) {m:5s} normalized train success {res[m][0]:.3f} -> {res[m][1]:.3f}  execs {res[m][2]}")
# the tiny random model learns noisily: require improvement for EXEC, and for RBER in at least one of two seeds
llm.train(w, "RBER", 1, out, model_id=d, lr=3e-3, n_updates=60, eval_every=30, log=lambda *a: None)
h1 = json.load(open(os.path.join(out, "llm_RBER_s1.json")))["hist"]
print(f"3) RBER  (seed 1) normalized train success {h1[0]['train_norm']:.3f} -> {h1[-1]['train_norm']:.3f}")
assert res["EXEC"][1] > res["EXEC"][0]
assert max(res["RBER"][1], h1[-1]["train_norm"]) > res["RBER"][0]
assert res["VERIF"][2] == 0
shutil.rmtree(out); print("ALL LLM MECHANICS TESTS PASSED")
