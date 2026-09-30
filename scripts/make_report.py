"""Generates every figure (paper/figures/*.pdf) and table (paper/tables/*.tex) from results/.

    python scripts/make_report.py

Language-model figures/tables are produced only if results/llm/*.json exist (Kaggle runs); otherwise the
paper shows a clearly marked 'pending' box instead of numbers.
"""
import glob, itertools, json, os, sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from rber.stats import iqm, boot_ci, prob_improvement, wilcoxon_paired, holm  # noqa: E402

RES, FIG, TAB = (os.path.join(ROOT, p) for p in ("results", "paper/figures", "paper/tables"))
os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)

# ---------------------------------------------------------------- style (IEEE: Times-like serif, 8 pt)
# TrueType Times-metric font (embeds cleanly as Type 42; OTF/CFF fonts do not)
for f in glob.glob("/usr/share/fonts/truetype/liberation/LiberationSerif-*.ttf"):
    font_manager.fontManager.addfont(f)
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
    "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df",
    "grid.linewidth": 0.5, "lines.linewidth": 1.4, "pdf.fonttype": 42, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "legend.frameon": False,
})
COL1, COL2 = 3.5, 7.16  # IEEE column / page width (inches)
# fixed identity per method (validated categorical order; secondary encoding by line style + marker)
STY = {
    "RBER":    dict(color="#2a78d6", ls="-",  marker="o", label="RBER (ours)"),
    "EXEC":    dict(color="#eb6834", ls="--", marker="s", label="EXEC"),
    "STEP":    dict(color="#1baf7a", ls="-.", marker="^", label="STEP"),
    "RBER-L":  dict(color="#4a3aa7", ls=(0, (5, 1.5)), marker="D", label="RBER-L (no sharing)"),
    "RBER-NL": dict(color="#e87ba4", ls=(0, (1, 1)), marker="v", label="RBER-NL (non-lagged)"),
    "RB*":     dict(color="#0b0b0b", ls=":",  marker=None, label=r"RB$^\star$ (oracle)"),
    "VERIF":   dict(color="#8a8985", ls=(0, (3, 2, 1, 2)), marker=None, label="VERIF (no execution)"),
}


def load_jsonl(name):
    p = os.path.join(RES, "cpu", f"{name}.jsonl")
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def sel(rows, **kw):
    out = [r for r in rows if all((r.get(k) == v) if not isinstance(v, float) else abs(r.get(k, -1) - v) < 1e-9
                                  for k, v in kw.items())]
    return sorted(out, key=lambda r: r["seed"])


def interp_curve(curve, grid):
    e = np.array([c[1] for c in curve], float); v = np.array([c[2] for c in curve], float)
    out = np.interp(grid, e, v); out[grid > e[-1]] = np.nan
    return out


def band(ax, grid, Y, sty, min_frac=0.8):
    Y = np.array(Y); ok = np.mean(~np.isnan(Y), 0) >= min_frac
    m = np.nanmean(Y, 0); n = np.sum(~np.isnan(Y), 0)
    se = np.nanstd(Y, 0, ddof=1) / np.sqrt(np.maximum(n, 1)); lo, hi = m - 1.96 * se, m + 1.96 * se
    ax.fill_between(grid[ok], lo[ok], hi[ok], color=sty["color"], alpha=0.15, lw=0)
    ax.plot(grid[ok], m[ok], color=sty["color"], ls=sty["ls"], label=sty["label"])


def fmt_ci(v, ci, d=3):
    return f"{v:.{d}f} [{ci[0]:.{d}f}, {ci[1]:.{d}f}]"


# ================================================================ Fig: main learning curves (E1)
def fig_main(rows):
    fig, axes = plt.subplots(1, 4, figsize=(COL2, 1.85), sharey=True)
    grid = np.linspace(0, 20000, 201)
    for ax, (K, inter) in zip(axes, [(3, 0.0), (3, 0.3), (4, 0.0), (4, 0.3)]):
        for m in ["STEP", "EXEC", "RBER-L", "RBER", "RB*"]:
            R = sel(rows, K=K, inter=inter, method=m)
            band(ax, grid, [interp_curve(r["curve"], grid) for r in R], STY[m])
        v = np.mean([r["final"] for r in sel(rows, K=K, inter=inter, method="VERIF")])
        ax.axhline(v, color=STY["VERIF"]["color"], ls=STY["VERIF"]["ls"], lw=1.1, label=STY["VERIF"]["label"])
        ax.axhline(0.9, color="#b9b8b3", lw=0.6, ls="-", zorder=0)
        ax.set_title(f"$K={K}$, " + ("independent" if inter == 0 else r"interaction $\delta=0.3$"))
        ax.set_xlabel("executed plans"); ax.set_xlim(0, 20000); ax.set_ylim(0, 1.02)
        ax.set_xticks([0, 10000, 20000]); ax.set_xticklabels(["0", "10k", "20k"])
    axes[0].set_ylabel("normalized success")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=6, bbox_to_anchor=(0.5, -0.17))
    fig.savefig(os.path.join(FIG, "main_curves.pdf")); plt.close(fig)


# ================================================================ Table: main results (E1)
def tab_main(rows):
    conds = [(3, 0.0), (3, 0.3), (4, 0.0), (4, 0.3)]
    methods = ["VERIF", "STEP", "EXEC", "RBER-L", "RBER", "RB*"]
    budget = {}
    lines = []
    for m in methods:
        cells = [STY[m]["label"].replace("RB$^\\star$ (oracle)", "RB$^\\star$ (oracle)")]
        for K, inter in conds:
            R = sel(rows, K=K, inter=inter, method=m)
            a = [r["auc"] for r in R]; ia = iqm(a); ci = boot_ci(a, iqm)
            if m == "VERIF":
                e_txt = "n/a"
            else:
                e = np.array([r["execs90"] for r in R]); hit = sum(r["reached"] for r in R)
                e_txt = (f"{np.median(e) / 1000:.1f}k" if hit >= len(R) / 2 else "n.r.") + f" ({hit}/{len(R)})"
            cells.append(f"{ia:.3f}"); cells.append(e_txt)
        lines.append(cells)
    # bold best AUC among non-oracle methods
    body = []
    for c in range(len(conds)):
        vals = [float(l[1 + 2 * c]) for l, m in zip(lines, methods) if m != "RB*"]
        best = max(vals)
        for l, m in zip(lines, methods):
            if m != "RB*" and abs(float(l[1 + 2 * c]) - best) < 1e-12:
                l[1 + 2 * c] = r"\textbf{" + l[1 + 2 * c] + "}"
    for l in lines:
        body.append(" & ".join(l) + r" \\")
    hdr = (r"\begin{tabular}{l" + "rr" * 4 + "}\n\\toprule\n"
           r" & \multicolumn{2}{c}{$K=3$, indep.} & \multicolumn{2}{c}{$K=3$, $\delta=0.3$} & \multicolumn{2}{c}{$K=4$, indep.} & \multicolumn{2}{c}{$K=4$, $\delta=0.3$} \\" "\n"
           r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}" "\n"
           r"Method & AUC & $E_{0.9}$ & AUC & $E_{0.9}$ & AUC & $E_{0.9}$ & AUC & $E_{0.9}$ \\" "\n\\midrule\n")
    body.insert(len(body) - 1, r"\midrule")
    open(os.path.join(TAB, "main.tex"), "w").write(hdr + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n")


def tab_tests(rows):
    conds = [(3, 0.0), (3, 0.3), (4, 0.0), (4, 0.3)]
    recs, ps = [], []
    for K, inter in conds:
        rb = sel(rows, K=K, inter=inter, method="RBER")
        for b in ["STEP", "EXEC", "RBER-L"]:
            bb = sel(rows, K=K, inter=inter, method=b)
            assert [r["seed"] for r in rb] == [r["seed"] for r in bb]
            pe = wilcoxon_paired([r["execs90"] for r in rb], [r["execs90"] for r in bb])
            pa = wilcoxon_paired([r["auc"] for r in rb], [r["auc"] for r in bb])
            pi, pci = prob_improvement([r["auc"] for r in rb], [r["auc"] for r in bb])
            d = np.array([r["auc"] for r in rb]) - np.array([r["auc"] for r in bb])
            recs.append([K, inter, b, d.mean(), boot_ci(d), pi, pci, pa, pe]); ps += [pa, pe]
    adj = holm(ps)
    lines = []
    for i, (K, inter, b, dm, dci, pi, pci, pa, pe) in enumerate(recs):
        cond = f"$K={K}$, " + ("indep." if inter == 0 else r"$\delta=0.3$")
        f = lambda p: (r"$<10^{-4}$" if p < 1e-4 else f"{p:.4f}")
        lines.append(f"{cond} & {b} & {dm:+.3f} [{dci[0]:+.3f}, {dci[1]:+.3f}] & {pi:.2f} [{pci[0]:.2f}, {pci[1]:.2f}] & "
                     f"{f(adj[2 * i])} & {f(adj[2 * i + 1])} \\\\")
    hdr = (r"\begin{tabular}{llccrr}" "\n\\toprule\n"
           r"Condition & vs. & $\Delta$AUC [95\% CI] & $P(\text{RBER}>\cdot)$ & $p_{\text{AUC}}$ & $p_{E_{0.9}}$ \\" "\n\\midrule\n")
    open(os.path.join(TAB, "tests.tex"), "w").write(hdr + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    return recs, adj


# ================================================================ Fig + table: horizon / variance (E2, E3)
def fig_horizon(hrows, var):
    fig, axes = plt.subplots(1, 2, figsize=(COL1 * 2, 1.9))
    Ks = [2, 3, 4, 5, 6]
    ax = axes[0]
    for tag, sty, lab in (("uniform", "o-", "uniform policy"), ("random", "s--", "random policy")):
        ratio = [[r["exec"] / r["rb"] for r in var if r["K"] == K and r["policy"] == tag] for K in Ks]
        med = [np.median(x) for x in ratio]; q1 = [np.quantile(x, .25) for x in ratio]; q3 = [np.quantile(x, .75) for x in ratio]
        ax.errorbar(Ks, med, yerr=[np.subtract(med, q1), np.subtract(q3, med)], fmt=sty, color="#2a78d6" if tag == "uniform" else "#4a3aa7",
                    ms=4, capsize=2, lw=1.2, label=f"variance ratio, {lab}")
    bound = [[r["m2_exec0"] / r["m2_rb0"] * r["Rmax"] for r in var if r["K"] == K and r["policy"] == "uniform"] for K in Ks]
    ax.set_yscale("log"); ax.set_xlabel("horizon $K$ (steps per plan)"); ax.set_xticks(Ks)
    ax.set_ylabel(r"$\mathrm{tr\,Cov}[g_{\mathrm{EXEC}}]\,/\,\mathrm{tr\,Cov}[g_{\mathrm{RB}^\star}]$")
    ax.set_title("(a) exact gradient-variance ratio (Thm. 1)")
    ax.legend(loc="upper left")
    ax = axes[1]
    for m in ["STEP", "EXEC", "RBER", "RB*"]:
        med, lo, hi, cens = [], [], [], []
        for K in Ks:
            R = sel(hrows, K=K, method=m); e = np.array([r["execs90"] for r in R])
            med.append(np.median(e)); lo.append(np.quantile(e, .25)); hi.append(np.quantile(e, .75))
            cens.append(np.mean([r["reached"] for r in R]) < 0.5)
        s = STY[m]
        ax.plot(Ks, med, color=s["color"], ls=s["ls"], marker=s["marker"] or "x", ms=4, label=s["label"])
        ax.fill_between(Ks, lo, hi, color=s["color"], alpha=0.12, lw=0)
        for K, y, c in zip(Ks, med, cens):
            if c:
                ax.plot(K, y, marker="o", mfc="white", mec=s["color"], ms=6, zorder=5)
    ax.axhline(24000, color="#b9b8b3", lw=0.6); ax.text(2.05, 26500, "budget (not reached)", fontsize=6.5, color="#52514e")
    ax.set_yscale("log"); ax.set_xticks(Ks); ax.set_xlabel("horizon $K$ (steps per plan)")
    ax.set_ylabel(r"executions to 90% of optimum"); ax.set_title("(b) sample efficiency vs. horizon")
    ax.legend(loc="lower right", ncol=2)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(os.path.join(FIG, "horizon.pdf")); plt.close(fig)
    # table
    lines = []
    for K in Ks:
        vr = [r["exec"] / r["rb"] for r in var if r["K"] == K and r["policy"] == "uniform"]
        rm = [r["Rmax"] for r in var if r["K"] == K and r["policy"] == "uniform"]
        cells = [str(K), f"{np.median(vr):.1f}", f"{np.median(rm):.3f}"]
        for m in ["STEP", "EXEC", "RBER", "RB*"]:
            R = sel(hrows, K=K, method=m); hit = sum(r["reached"] for r in R)
            e = np.median([r["execs90"] for r in R])
            cells.append((f"{e / 1000:.1f}k" if hit >= len(R) / 2 else "n.r.") + f" ({hit})")
        lines.append(" & ".join(cells) + r" \\")
    err = max(r["identity_err"] for r in var)
    hdr = (r"\begin{tabular}{rrrrrrr}" "\n\\toprule\n"
           r"$K$ & $\frac{\mathrm{tr\,Cov}[g_{\mathrm{EXEC}}]}{\mathrm{tr\,Cov}[g_{\mathrm{RB}^\star}]}$ & $R_{\max}$ & STEP & EXEC & RBER & RB$^\star$ \\" "\n\\midrule\n")
    open(os.path.join(TAB, "horizon.tex"), "w").write(hdr + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    return err


# ================================================================ Fig: interaction sweep (E4)
def fig_interaction(irows, gaps):
    ds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    fig, axes = plt.subplots(1, 2, figsize=(COL1 * 2, 1.85))
    for m in ["EXEC", "RBER", "RB*"]:
        s = STY[m]
        for ax, key in zip(axes, ["auc", "final"]):
            vals = [[r[key] for r in sel(irows, inter=d, method=m)] for d in ds]
            mu = [np.mean(v) for v in vals]; ci = [boot_ci(v) for v in vals]
            ax.plot(ds, mu, color=s["color"], ls=s["ls"], marker=s["marker"] or "x", ms=4, label=s["label"])
            ax.fill_between(ds, [c[0] for c in ci], [c[1] for c in ci], color=s["color"], alpha=0.15, lw=0)
    axes[0].set_ylabel("AUC (normalized success)"); axes[1].set_ylabel("final normalized success")
    axes[0].set_title("(a) learning speed"); axes[1].set_title("(b) final performance")
    for ax in axes:
        ax.set_xlabel(r"interaction strength $\delta$"); ax.set_xticks(ds)
    axes[1].legend(loc="lower left")
    fig.tight_layout(w_pad=2.0); fig.savefig(os.path.join(FIG, "interaction.pdf")); plt.close(fig)
    lines = []
    for d in ds:
        e = [r["auc"] for r in sel(irows, inter=d, method="EXEC")]; b = [r["auc"] for r in sel(irows, inter=d, method="RBER")]
        fe = [r["final"] for r in sel(irows, inter=d, method="EXEC")]; fb = [r["final"] for r in sel(irows, inter=d, method="RBER")]
        dd = np.array(b) - np.array(e); ci = boot_ci(dd)
        g = gaps[str(d)]
        lines.append(f"{d:.1f} & {np.median(g):.3f} & {np.mean(e):.3f} & {np.mean(b):.3f} & {dd.mean():+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] & "
                     f"{np.mean(fe):.3f} & {np.mean(fb):.3f} \\\\")
    hdr = (r"\begin{tabular}{rrrrcrr}" "\n\\toprule\n"
           r" & & \multicolumn{3}{c}{AUC} & \multicolumn{2}{c}{final} \\ \cmidrule(lr){3-5}\cmidrule(lr){6-7}" "\n"
           r"$\delta$ & $\varepsilon_{\mathrm{int}}$ & EXEC & RBER & $\Delta$ [95\% CI] & EXEC & RBER \\" "\n\\midrule\n")
    open(os.path.join(TAB, "interaction.tex"), "w").write(hdr + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")


# ================================================================ Table: ablations (E5)
ABL = {}
def auc_common(r, every=800):
    """AUC on a grid shared by all group sizes: one point every `every` sampled plans (31 points for 24k plans)."""
    G = r.get("G", 8)
    return float(np.mean([v for u, _, v in r["curve"] if (u * G) % every == 0]))


def tab_ablations(arows, mrows):
    lines = []
    for inter in (0.0, 0.3):
        cond = "indep." if inter == 0 else r"$\delta=0.3$"
        def row(name, R):
            a = [auc_common(r) for r in R]; f = [r["final"] for r in R]
            ABL[(inter, name)] = (np.mean(a), a)
            return f"{cond} & {name} & {fmt_ci(np.mean(a), boot_ci(a))} & {np.mean(f):.3f} \\\\"
        lines.append(row("RBER (Beta(1,1), shared, lagged, $G{=}8$)", sel(arows, inter=inter, method="RBER", G=8, prior=[1.0, 1.0])))
        lines.append(row("\\quad no sharing (RBER-L)", sel(mrows, K=3, inter=inter, method="RBER-L")))
        lines.append(row("\\quad non-lagged (RBER-NL)", sel(arows, inter=inter, method="RBER-NL")))
        for pr in ([0.5, 0.5], [2.0, 2.0], [5.0, 5.0]):
            lines.append(row(f"\\quad prior Beta({pr[0]:g},{pr[1]:g})", sel(arows, inter=inter, method="RBER", G=8, prior=pr)))
        for g in (4, 16):
            lines.append(row(f"\\quad group size $G={g}$", sel(arows, inter=inter, method="RBER", G=g)))
            lines.append(row(f"\\quad EXEC, $G={g}$", sel(arows, inter=inter, method="EXEC", G=g)))
        lines.append(row("\\quad EXEC, $G=8$", sel(mrows, K=3, inter=inter, method="EXEC")))
        if inter == 0.0:
            lines.append(r"\midrule")
    hdr = (r"\begin{tabular}{llcr}" "\n\\toprule\n"
           r"Condition & Variant & AUC [95\% CI] & final \\" "\n\\midrule\n")
    open(os.path.join(TAB, "ablations.tex"), "w").write(hdr + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")


# ================================================================ Fig + table: skill library
def fig_skills():
    d = json.load(open(os.path.join(ROOT, "data", "skills_metaworld.json")))
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from rber.domain import CATEGORIES
    order = [t for ts in CATEGORIES.values() for t in ts]
    rows = sorted(d["skills"], key=lambda s: (order.index(s["task"]), s["sigma"]))
    fig, ax = plt.subplots(figsize=(COL2, 1.7))
    x = np.arange(len(rows)); cols = {0.3: "#2a78d6", 0.8: "#eb6834", 1.5: "#1baf7a"}
    for s0, lab in ((0.3, r"$\sigma=0.3$"), (0.8, r"$\sigma=0.8$"), (1.5, r"$\sigma=1.5$")):
        idx = [i for i, r in enumerate(rows) if r["sigma"] == s0]
        p = np.array([rows[i]["p"] for i in idx]); lo = np.array([rows[i]["ci95"][0] for i in idx]); hi = np.array([rows[i]["ci95"][1] for i in idx])
        ax.errorbar(x[idx], p, yerr=[p - lo, hi - p], fmt="o", ms=2.8, color=cols[s0], elinewidth=0.8, capsize=0, label=lab)
    ticks = [np.mean([i for i, r in enumerate(rows) if r["task"] == t]) for t in order]
    ax.set_xticks(ticks); ax.set_xticklabels([t.replace("-v3", "") for t in order], rotation=40, ha="right", fontsize=6.5)
    b = 0
    short = {"move the object to the target": "move object", "open the container": "open",
             "press the control": "press", "close the container": "close", "use the tool on the part": "use tool",
             "move the gripper to the marked point": "reach"}
    tr = matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes)
    for c, ts in CATEGORIES.items():
        a = b; b += 3 * len(ts)
        ax.axvline(b - 0.5, color="#b9b8b3", lw=0.5)
        ax.text((a + b - 1) / 2, 1.04, short[c], transform=tr, ha="center", va="bottom", fontsize=6.5, color="#52514e")
    ax.set_xlim(-0.8, len(rows) - 0.2)
    ax.set_ylabel("success probability"); ax.set_ylim(-0.03, 1.03); ax.grid(axis="x", visible=False)
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.12))
    fig.savefig(os.path.join(FIG, "skills.pdf")); plt.close(fig)
    return d["meta"]


# ================================================================ LLM (Kaggle) results
def llm_report():
    files = sorted(glob.glob(os.path.join(RES, "llm", "**", "llm_*_s*.json"), recursive=True))
    runs = [json.load(open(f)) for f in files]
    runs = [r for r in runs if r.get("done")]
    macros = {}
    if not runs:
        open(os.path.join(TAB, "llm_status.tex"), "w").write("\\newcommand{\\LLMready}{0}\n")
        return None
    by = {}
    for r in runs:
        by.setdefault(r["method"], {})[r["seed"]] = r
    methods = [m for m in ["VERIF", "STEP", "EXEC", "RBER", "RBER-NL"] if m in by]
    nseeds = {m: len(by[m]) for m in methods}
    # figure: normalized train / held-out success vs executions (VERIF vs updates, shown flat at its final)
    fig, axes = plt.subplots(1, 2, figsize=(COL1 * 2, 1.9), sharey=True)
    grid = np.linspace(0, 8000, 161)
    for m in methods:
        for ax, key in zip(axes, ["train_norm", "held_norm"]):
            if m == "VERIF":
                v = np.mean([rr["hist"][-1][key] for rr in by[m].values()])
                ax.axhline(v, color=STY[m]["color"], ls=STY[m]["ls"], lw=1.1, label=STY[m]["label"]); continue
            Y = []
            for rr in by[m].values():
                c = [(h["update"], h["execs"], h[key]) for h in rr["hist"]]
                Y.append(interp_curve(c, grid))
            band(ax, grid, Y, STY[m])
    axes[0].set_ylabel("normalized success"); axes[0].set_title("(a) training tasks"); axes[1].set_title("(b) held-out tasks")
    for ax in axes:
        ax.set_xlabel("executed plans"); ax.set_ylim(0, 1.02)
    axes[1].legend(loc="lower right")
    fig.tight_layout(w_pad=1.5); fig.savefig(os.path.join(FIG, "llm_curves.pdf")); plt.close(fig)

    def aucf(rr, key="train_norm"):
        return float(np.mean([h[key] for h in rr["hist"]]))
    lines = []
    for m in methods:
        R = list(by[m].values())
        tr = [rr["hist"][-1]["train_norm"] for rr in R]; he = [rr["hist"][-1]["held_norm"] for rr in R]
        au = [aucf(rr) for rr in R]; ex = [rr["hist"][-1]["execs"] for rr in R]
        lines.append(f"{STY[m]['label']} & {len(R)} & {fmt_ci(np.mean(au), boot_ci(au))} & {fmt_ci(np.mean(tr), boot_ci(tr))} & "
                     f"{fmt_ci(np.mean(he), boot_ci(he))} & {np.mean(ex):.0f} \\\\")
    hdr = (r"\begin{tabular}{lrcccr}" "\n\\toprule\n"
           r"Method & seeds & AUC (train) & final train & final held-out & executions \\" "\n\\midrule\n")
    open(os.path.join(TAB, "llm_main.tex"), "w").write(hdr + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    # pre-registered primary test H6 + paired tests
    tests = {}
    if "EXEC" in by and "RBER" in by:
        common = sorted(set(by["EXEC"]) & set(by["RBER"]))
        Efull = np.mean([by["EXEC"][s]["hist"][-1]["execs"] for s in common])
        half = []
        for s in common:
            ok = [h for h in by["RBER"][s]["hist"] if h["execs"] <= Efull / 2]
            half.append(ok[-1]["train_norm"])
        exec_final = [by["EXEC"][s]["hist"][-1]["train_norm"] for s in common]
        tests["H6"] = dict(rber_half=float(np.mean(half)), exec_full=float(np.mean(exec_final)),
                           holds=bool(np.mean(half) >= np.mean(exec_final)))
        for b in [m for m in methods if m not in ("RBER",)]:
            cs = sorted(set(by[b]) & set(by["RBER"]))
            a1 = [aucf(by["RBER"][s]) for s in cs]; a0 = [aucf(by[b][s]) for s in cs]
            h1 = [by["RBER"][s]["hist"][-1]["held_norm"] for s in cs]; h0 = [by[b][s]["hist"][-1]["held_norm"] for s in cs]
            tests[b] = dict(n=len(cs), dauc=float(np.mean(np.subtract(a1, a0))), p_auc=wilcoxon_paired(a1, a0),
                            dheld=float(np.mean(np.subtract(h1, h0))), p_held=wilcoxon_paired(h1, h0))
        ks = [k for k in tests if k != "H6"]
        adj = holm([tests[k]["p_auc"] for k in ks] + [tests[k]["p_held"] for k in ks])
        for i, k in enumerate(ks):
            tests[k]["p_auc_holm"] = float(adj[i]); tests[k]["p_held_holm"] = float(adj[len(ks) + i])
    json.dump(dict(nseeds=nseeds, tests=tests), open(os.path.join(RES, "llm", "summary.json"), "w"), indent=1)
    with open(os.path.join(TAB, "llm_status.tex"), "w") as f:
        f.write("\\newcommand{\\LLMready}{1}\n")
        f.write(f"\\newcommand{{\\LLMseeds}}{{{min(nseeds.values())}}}\n")
        if "H6" in tests:
            f.write(f"\\newcommand{{\\LLMhalf}}{{{tests['H6']['rber_half']:.3f}}}\n\\newcommand{{\\LLMexecfull}}{{{tests['H6']['exec_full']:.3f}}}\n")
            f.write("\\newcommand{\\LLMverdict}{" + ("supported" if tests["H6"]["holds"] else "not supported") + "}\n")
            fp = lambda p: "<10^{-4}" if p < 1e-4 else f"={p:.3f}"
            for b, tag in (("EXEC", "Exec"), ("STEP", "Step"), ("VERIF", "Verif"), ("RBER-NL", "Nl")):
                if b in tests:
                    t = tests[b]
                    f.write(f"\\newcommand{{\\LLMdauc{tag}}}{{{t['dauc']:+.3f}}}\n\\newcommand{{\\LLMpauc{tag}}}{{{fp(t['p_auc_holm'])}}}\n"
                            f"\\newcommand{{\\LLMdheld{tag}}}{{{t['dheld']:+.3f}}}\n\\newcommand{{\\LLMpheld{tag}}}{{{fp(t['p_held_holm'])}}}\n")
    return tests


# ================================================================ numbers used in the text
def macros(rows, hrows, var, irows, gaps, recs, adj, meta, err):
    M = {}
    def med_e(m, K, inter):
        return np.median([r["execs90"] for r in sel(rows, K=K, inter=inter, method=m)])
    for K in (3, 4):
        for inter, tag in ((0.0, "I"), (0.3, "D")):
            KW = {3: "Three", 4: "Four"}[K]
            M[f"speedExec{KW}{tag}"] = f"{med_e('EXEC', K, inter) / med_e('RBER', K, inter):.1f}"
            M[f"eRber{KW}{tag}"] = f"{med_e('RBER', K, inter):.0f}"
            M[f"eExec{KW}{tag}"] = f"{med_e('EXEC', K, inter):.0f}"
    M["identityErr"] = f"{err:.1e}".replace("e-", "\\times10^{-").rstrip() + "}"
    rel = max(abs(r["exec"] - r["rb"] - r["gap"]) / r["exec"] for r in var if r["exec"] > 0)
    M["identityRelErr"] = f"{rel:.1e}".replace("e-", "\\times10^{-").rstrip() + "}"
    base = "RBER (Beta(1,1), shared, lagged, $G{=}8$)"
    for inter, tag in ((0.0, "I"), (0.3, "D")):
        M[f"ablRber{tag}"] = f"{ABL[(inter, base)][0]:.3f}"
        M[f"ablNoShare{tag}"] = f"{ABL[(inter, chr(92) + 'quad no sharing (RBER-L)')][0]:.3f}"
        a1 = np.array(ABL[(inter, base)][1]); a0 = np.array(ABL[(inter, chr(92) + 'quad non-lagged (RBER-NL)')][1])
        d = a1 - a0; ci = boot_ci(d)
        M[f"lagDiff{tag}"] = f"{d.mean():+.4f}"; M[f"lagCI{tag}"] = f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"
        M[f"lagP{tag}"] = f"{wilcoxon_paired(a1, a0):.3f}"
    vr = {K: np.median([r["exec"] / r["rb"] for r in var if r["K"] == K and r["policy"] == "uniform"]) for K in (2, 6)}
    M["varRatioTwo"] = f"{vr[2]:.1f}"; M["varRatioSix"] = f"{vr[6]:.0f}"
    M["boundMin"] = f"{min(r['m2_exec0'] / r['m2_rb0'] * r['Rmax'] for r in var):.2f}"
    M["nVarCases"] = str(len(var))
    for K, w in zip((2, 3, 4, 5, 6), ("Two", "Three", "Four", "Five", "Six")):
        J = np.array([r["J"] for r in var if r["K"] == K and r["policy"] == "uniform"])
        M[f"initJ{w}"] = f"{np.median(J):.3f}"
        M[f"zeroSig{w}"] = f"{np.median((1 - J) ** 8 + J ** 8):.2f}"
    M["nSkillRollouts"] = str(meta["rollouts"]); M["nSkills"] = "54"
    M["maxHolmP"] = f"{max(adj):.1e}"
    # H2: final RBER - EXEC (mean and bootstrap CI) per main condition
    lows = []
    for K in (3, 4):
        for inter, tag in ((0.0, "I"), (0.3, "D")):
            d = np.array([r["final"] for r in sel(rows, K=K, inter=inter, method="RBER")]) - \
                np.array([r["final"] for r in sel(rows, K=K, inter=inter, method="EXEC")])
            ci = boot_ci(d); lows.append(ci[0]); KW = {3: "Three", 4: "Four"}[K]
            M[f"finalDiff{KW}{tag}"] = f"{d.mean():+.3f}"
            M[f"finalRber{KW}{tag}"] = f"{np.mean([r['final'] for r in sel(rows, K=K, inter=inter, method='RBER')]):.3f}"
            M[f"finalExec{KW}{tag}"] = f"{np.mean([r['final'] for r in sel(rows, K=K, inter=inter, method='EXEC')]):.3f}"
    M["finalDiffMinLow"] = f"{min(lows):+.3f}"
    with open(os.path.join(TAB, "numbers.tex"), "w") as f:
        for k, v in M.items():
            f.write(f"\\newcommand{{\\num{k}}}{{{v}}}\n")
    return M


if __name__ == "__main__":
    rows, hrows, irows, arows = load_jsonl("main"), load_jsonl("horizon"), load_jsonl("interaction"), load_jsonl("ablations")
    var = json.load(open(os.path.join(RES, "cpu", "variance.json")))
    gaps = json.load(open(os.path.join(RES, "cpu", "interaction_gaps.json")))
    meta = fig_skills()
    fig_main(rows); tab_main(rows); recs, adj = tab_tests(rows)
    err = fig_horizon(hrows, var); fig_interaction(irows, gaps); tab_ablations(arows, rows)
    M = macros(rows, hrows, var, irows, gaps, recs, adj, meta, err)
    t = llm_report()
    # PNG previews of every figure (for the README / quick viewing); the paper uses the PDFs
    import shutil, subprocess
    if shutil.which("pdftoppm"):
        os.makedirs(os.path.join(FIG, "png"), exist_ok=True)
        for f in glob.glob(os.path.join(FIG, "*.pdf")):
            stem = os.path.join(FIG, "png", os.path.basename(f)[:-4])
            subprocess.run(["pdftoppm", "-r", "300", "-png", "-singlefile", f, stem], check=False, stderr=subprocess.DEVNULL)
    print(json.dumps(M, indent=1)); print("LLM:", "pending" if t is None else t)
