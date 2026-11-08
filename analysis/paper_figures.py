"""Paper figures from the pilot D files (moment rule as base case) with the pilot C loggain twin where specified.

Usage: python analysis/paper_figures.py [--out figures/paper]

Figures (Section 4 doc, exhibit specification):
  fig1  scope K and pooled capital share versus alpha, by family; loggain dashed; K* markers
  fig2  K/K* versus search probability p at alpha = 0.3 and 0.5          (appendix)
  fig3  owner-return gap: mean/median by family and alpha; by K; floor-hit rate per status
  fig4  Hill exponent versus sharing rate, main block lines + other blocks     (appendix)
  fig5  floor-hit clustering: index of dispersion with binomial reference  (appendix)
  fig6  consent by scope: acceptance by K (moment, loggain), marginal gains and cost, binding gain

Every number is read from the CSVs; nothing is typed in.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FAMILIES = ["normal", "laplace", "t3"]
LABEL = {"normal": "normal", "laplace": "Laplace", "t3": "t$_3$"}
COLOR = {"normal": "#0072B2", "laplace": "#D55E00", "t3": "#009E73"}   # Okabe-Ito, validated (CVD dE >= 11)
MARKER = {"normal": "o", "laplace": "s", "t3": "^"}
TEXT = "#333333"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#888888",
    "axes.labelcolor": TEXT, "xtick.color": TEXT, "ytick.color": TEXT, "text.color": TEXT,
    "grid.color": "#e3e3e3", "grid.linewidth": 0.6, "axes.grid": True, "axes.axisbelow": True,
    "lines.linewidth": 1.6, "lines.markersize": 5, "legend.frameon": False,
    "figure.dpi": 150, "savefig.bbox": "tight",
})


def med(df, by, cols):
    return df.groupby(by)[cols].median().reset_index()


def load(root):
    d = pd.read_csv(os.path.join(root, "pilot_d", "tidy.csv"))
    c = pd.read_csv(os.path.join(root, "pilot_c", "tidy.csv"))
    b = pd.read_csv(os.path.join(root, "analytics", "benchmarks.csv"))
    b = b[(b.cost_multiplier == 1.0) & (b.cost_type == "power_law")]
    return d, c, b


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"))
    plt.close(fig)


# ---------------------------------------------------------------- fig 1
def fig1(d, c, b, out):
    m = d[(d.block == "main") & (d.cost_type == "power_law") & (d.alpha > 0)]
    mc = c[(c.block == "main") & (c.cost_type == "power_law") & (c.alpha > 0) & (c.decision_rule == "loggain")]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.0), gridspec_kw={"wspace": 0.3})
    for f in FAMILIES:
        g = med(m[m.log_family == f], "alpha", ["K_mean", "cong_capital_share_median"])
        gl = med(mc[mc.log_family == f], "alpha", ["K_mean"])
        ax1.plot(g.alpha, g.K_mean, color=COLOR[f], marker=MARKER[f], label=LABEL[f])
        ax1.plot(gl.alpha, gl.K_mean, color=COLOR[f], ls="--", lw=1.0, marker=MARKER[f], mfc="white", ms=4)
        bk = b[b.family == f]
        ax1.scatter(bk.alpha, bk.K_star, color=COLOR[f], marker="_", s=120, lw=1.8, zorder=5)
        ax2.plot(g.alpha, g.cong_capital_share_median, color=COLOR[f], marker=MARKER[f], label=LABEL[f])
    ax1.set_xlabel(r"sharing rate $\alpha$"); ax1.set_ylabel("mean members per unit, $K$")
    ax1.set_yscale("log"); ax1.set_yticks([2, 3, 5, 10, 20, 50]); ax1.set_yticklabels(["2", "3", "5", "10", "20", "50"]); ax1.minorticks_off()
    ax1.set_title("(a) scope; $K^*$ as ticks", loc="left")
    ax2.set_xlabel(r"sharing rate $\alpha$"); ax2.set_ylabel("pooled capital share")
    ax2.set_ylim(0, 1); ax2.set_title("(b) pooled capital", loc="left")
    ax1.legend(loc="upper left")
    save(fig, out, "fig1_scope")


# ---------------------------------------------------------------- fig 2
def fig2(d, b, out):
    s = d[(d.cost_type == "power_law") & (d.block.isin(["main", "search"])) & (d.alpha.isin([0.3, 0.5]))]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, a in zip(axes, [0.3, 0.5]):
        for f in FAMILIES:
            g = med(s[(s.log_family == f) & (s.alpha == a)], "merge_thresh", ["K_mean"])
            ks = float(b[(b.family == f) & (b.alpha == a)].K_star.iloc[0])
            ax.plot(g.merge_thresh, g.K_mean / ks, color=COLOR[f], marker=MARKER[f], label=LABEL[f])
        ax.set_xscale("log"); ax.set_xticks([0.05, 0.2, 1]); ax.set_xticklabels(["0.05", "0.2", "1"])
        ax.set_xlabel("search probability $p$"); ax.set_title(rf"$\alpha$ = {a}", loc="left")
        ax.set_ylim(0, 1); ax.axhline(1, color="#888888", lw=0.8, ls=":")
    axes[0].set_ylabel("$K / K^*$"); axes[0].legend(loc="upper left")
    save(fig, out, "fig2_search")


# ---------------------------------------------------------------- fig 3
def fig3(root, out):
    p = pd.read_csv(os.path.join(root, "pilot_d", "status_panel_summary.csv"))
    g = p.groupby(["family", "alpha"]).mean(numeric_only=True).reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.0), gridspec_kw={"wspace": 0.42})
    ax = axes[0]
    for f in FAMILIES:
        h = g[g.family == f]
        ax.plot(h.alpha, 100 * h.diff_fe_mean, color=COLOR[f], marker=MARKER[f], label=LABEL[f])
        ax.plot(h.alpha, 100 * h.diff_fe_median, color=COLOR[f], marker=MARKER[f], mfc="white", ls="--", lw=1.0)
    ax.set_xlabel(r"$\alpha$"); ax.set_ylabel("member $-$ standalone owner return, pp per period")
    ax.set_title("(a) mean (solid) and median (dashed)", loc="left"); ax.legend(loc="upper left")
    ax = axes[1]
    w = 0.25; x = np.arange(3)
    for i, f in enumerate(FAMILIES):
        h = g[(g.family == f) & (g.alpha == 0.3)].iloc[0]
        vals = 100 * np.array([h.diff_K2_mean, h.diff_K3_4_mean, h.diff_K5p_mean])
        ax.bar(x + (i - 1) * w, vals, width=w * 0.9, color=COLOR[f], label=LABEL[f])
    ax.set_xticks(x); ax.set_xticklabels(["K = 2", "K = 3–4", "K ≥ 5"]); ax.set_ylabel("pp per period")
    ax.set_title(r"(b) by unit size, $\alpha$ = 0.3", loc="left")
    ax = axes[2]
    for f in FAMILIES:
        h = g[g.family == f]
        ax.plot(h.alpha, h.floor_rate_standalone, color=COLOR[f], marker=MARKER[f], label=f"{LABEL[f]}, standalone")
        ax.plot(h.alpha, h.floor_rate_member, color=COLOR[f], marker=MARKER[f], mfc="white", ls="--", lw=1.0)
    ax.set_yscale("log"); ax.set_xlabel(r"$\alpha$"); ax.set_ylabel("floor hits per firm-period of that status")
    ax.set_title("(c) hazard: standalone (solid), member (dashed)", loc="left")
    save(fig, out, "fig3_owner_return")


# ---------------------------------------------------------------- fig 4
def fig4(d, out):
    """Hill exponent against the sharing rate (8 Nov: x-axis changed from pooled share to alpha; the pooled
    share saturates by alpha = 0.05 and hid the proportionality). Main block as lines (medians over reps);
    the other blocks at the same floor and cost as markers; the endogenous block at its adopted rate."""
    x = d[d.K_mean.notna() & (d.decision_rule == "moment")]
    x = x[(x.cost_type == "power_law") & (x.floor_c.round(5) == 0.12717) & (x.sharing_rule == "proportional")]
    x = x[~x.block.isin(["nu", "size", "floor-level", "cost-level-high", "cost-level"])]
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    for f in FAMILIES:
        g = med(x[(x.block == "main") & (x.log_family == f)], "alpha", ["hill_exponent_median"])
        ax.plot(g.alpha, g.hill_exponent_median, color=COLOR[f], marker=MARKER[f], label=LABEL[f], zorder=4)
    shape = {"alpha-0.9": ("P", r"main, $\alpha$ = 0.9"), "search": ("s", "search"), "lookback": ("v", "lookback"),
             "correlation": ("x", "correlation"), "homogeneous": ("+", "homogeneous"),
             "endogenous-alpha": ("D", "endogenous (at adopted rate)")}
    for blk, (mk, lab) in shape.items():
        y = x[x.block == blk]
        xs = y.alpha_adopted_median if blk == "endogenous-alpha" else y.alpha
        for f in FAMILIES:
            z = y[y.log_family == f]; xz = xs[y.log_family == f]
            hollow = mk in ("s", "v", "D")
            ax.scatter(xz, z.hill_exponent_median, marker=mk, s=18, alpha=0.8, lw=0.8,
                       facecolors="none" if hollow else COLOR[f], edgecolors=COLOR[f],
                       label=lab if f == "normal" else None, zorder=3)
    ax.set_xlabel(r"sharing rate $\alpha$"); ax.set_ylabel("Hill exponent, within-market shares")
    ax.set_xlim(-0.02, 1.0)
    h, l = ax.get_legend_handles_labels()
    ax.legend(h, l, loc="upper left", ncol=2, fontsize=7)
    save(fig, out, "fig4_hill_alpha")


# ---------------------------------------------------------------- fig 5
def fig5(root, out):
    bs = pd.read_csv(os.path.join(root, "pilot_d", "burstiness_summary.csv"))
    g = bs.groupby(["family", "alpha"]).mean(numeric_only=True).reset_index()
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for f in FAMILIES:
        h = g[g.family == f]
        ax.plot(h.alpha, h.D_median, color=COLOR[f], marker=MARKER[f], label=LABEL[f])
        ax.fill_between(h.alpha, h.D_p25, h.D_p75, color=COLOR[f], alpha=0.15, lw=0)
        ax.plot(h.alpha, h.D_binomial, color=COLOR[f], ls=":", lw=1.0)
    ax.set_xlabel(r"$\alpha$"); ax.set_ylabel("index of dispersion of floor hits per market-period")
    ax.set_ylim(0, None); ax.legend(loc="lower left"); ax.set_title("solid: observed (IQR shaded); dotted: binomial", loc="left")
    save(fig, out, "fig5_burstiness")


# ---------------------------------------------------------------- fig 6
def fig6(root, out):
    am = pd.read_csv(os.path.join(root, "pilot_d", "acceptance_by_K.csv"))
    al = pd.read_csv(os.path.join(root, "pilot_c", "acceptance_by_K.csv"))
    kw = pd.read_csv(os.path.join(root, "pilot_d", "acceptance_by_K_w_pooled.csv"))
    sel = lambda a: a[(a.block == "main") & (a.alpha == 0.3) & (a.p == 0.05) & (a.K <= 12)]
    am, al = sel(am), sel(al)
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.1), gridspec_kw={"wspace": 0.38})
    for ax in axes:
        ax.set_xticks([2, 4, 6, 8, 10, 12])
    ax = axes[0]
    for f in FAMILIES:
        g = med(am[am.family == f], "K", ["acceptance"]); gl = med(al[al.family == f], "K", ["acceptance"])
        ax.plot(g.K, g.acceptance, color=COLOR[f], marker=MARKER[f], label=LABEL[f])
        ax.plot(gl.K, gl.acceptance, color=COLOR[f], marker=MARKER[f], mfc="white", ls="--", lw=1.0)
    ax.set_xlabel("size $K$ of the prospective unit"); ax.set_ylabel("share of feasible proposals accepted")
    ax.set_ylim(0, 1); ax.set_title("(a) acceptance: moment (solid),\nloggain (dashed)", loc="left", fontsize=8); ax.legend()
    ax = axes[1]
    for f in FAMILIES:
        g = med(am[am.family == f], "K", ["marginal_gain_into_K", "marginal_cost_into_K"])
        fl = kw[(kw.family == f) & (kw.alpha == 0.3) & (kw.group == "floor") & (kw.K <= 12) & (kw.K >= 3)]
        ax.plot(g.K, g.marginal_gain_into_K, color=COLOR[f], marker=MARKER[f], label=LABEL[f] if f == "normal" else None)
        ax.plot(fl.K, fl.gain_w_incumbent, color=COLOR[f], marker=MARKER[f], mfc="white", ls="--", lw=1.0)
    g = med(am, "K", ["marginal_cost_into_K"])
    ax.plot(g.K, g.marginal_cost_into_K, color="#555555", ls=":", lw=1.2, label="marginal cost")
    ax.set_yscale("log"); ax.set_xlabel("$K$"); ax.set_ylabel("gain or cost per period")
    ax.set_title("(b) marginal gain: equal weight (solid),\nfloor-sized recruit (dashed); cost (dotted)", loc="left", fontsize=8)
    h, l = ax.get_legend_handles_labels(); ax.legend([h[-1]], [l[-1]], loc="lower left")
    ax = axes[2]
    for f in FAMILIES:
        g = med(am[am.family == f], "K", ["binding_median", "binding_p25", "binding_p75"])
        ax.plot(g.K, 1e4 * g.binding_median, color=COLOR[f], marker=MARKER[f], label=LABEL[f])
        ax.fill_between(g.K, 1e4 * g.binding_p25, 1e4 * g.binding_p75, color=COLOR[f], alpha=0.12, lw=0)
    ax.axhline(0, color="#555555", lw=0.8)
    ax.set_xlabel("$K$"); ax.set_ylabel(r"binding member's estimated gain, $\times 10^{-4}$")
    ax.set_ylim(-8, 12); ax.set_title("(c) binding member's gain:\nmedian and IQR, moment rule", loc="left", fontsize=8)
    save(fig, out, "fig6_consent")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(a.root, "figures", "paper")
    os.makedirs(out, exist_ok=True)
    d, c, b = load(a.root)
    fig1(d, c, b, out); fig2(d, b, out); fig3(a.root, out); fig4(d, out); fig5(a.root, out); fig6(a.root, out)
    print("written to", out)


if __name__ == "__main__":
    main()
