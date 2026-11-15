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


CAPTIONS = {
    "fig1_scope": ("Scope and pooled capital by sharing rate. (a) Mean members per unit $K$ against $\\alpha$, one line per family, with the loggain rule as a dashed twin and the equal-members reference $K^*$ at $\\alpha \\in \\{0.1, 0.3, 0.5\\}$ as ticks; (b) pooled capital share against $\\alpha$. Main block, power-law cost; medians over five replications.", "fig:scope"),
    "fig2_search": ("Decision parameters. (a), (b) Scope relative to the equal-members reference, $K/K^*$, against the proposal probability $p \\in \\{0.05, 0.2, 1\\}$ at $\\alpha = 0.3$ and $0.5$, one line per family; (c) scope against the window length $l$ at $\\alpha = 0.3$, Laplace and $t_3$, moment rule solid and loggain rule dashed. Search, lookback and main blocks, power-law cost; medians over five replications.", "fig:search"),
    "fig3_owner_return": ("Member--standalone owner-return gap. (a) Mean (solid) and median (dashed) gap by family and $\\alpha$, percentage points per period; (b) the gap by unit size ($K = 2$, 3--4, $5+$) at $\\alpha = 0.3$; (c) floor hits per firm-period by status (standalone solid, member dashed). The 27 traced runs.", "fig:owner_return"),
        "fig4_hill_alpha": ("Tail exponent against the sharing rate. Hill exponent of within-market shares: main-block medians as lines by family; the search block ($p = 0.2$ and $1$ against the baseline $0.05$) as hollow squares, one per cell; the fixed $\\alpha = 0.9$ cell as crosses; the endogenous block as one diamond per family at its median adopted rate. No cells were run between $\\alpha = 0.5$ and $0.9$. The exponent rises with $\\alpha$ and does not respond to scope: the search block raises scope by a third to two thirds at the same rate and sits on the curve.", "fig:hill"),
    "fig5_burstiness": ("Clustering of floor hits. Index of dispersion of floor hits per market-period by family and $\\alpha$ (median and IQR over the traced replications) against the binomial reference.", "fig:burstiness"),
    "fig6_consent": ("Consent by scope. (a) Acceptance rate by size $K$ of the prospective unit at $\\alpha = 0.3$ under the moment and loggain rules ($p = 0.05$ and $p = 1$); (b) the equal-weight marginal gain of the $K$-th member, the incumbents' gain from a floor-sized recruit and the marginal cost, log axis; (c) the binding member's estimated gain by $K$ (median and IQR).", "fig:consent"),
}


def write_figure_tex(out, name):
    """A \\input-able figure environment beside the PDF: latex/figures/<name>.tex (path relative to latex/)."""
    cap, label = CAPTIONS[name]
    texdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(out))), "latex", "figures")
    os.makedirs(texdir, exist_ok=True)
    import shutil
    os.makedirs(os.path.join(texdir, "paper"), exist_ok=True)
    shutil.copyfile(os.path.join(out, f"{name}.pdf"), os.path.join(texdir, "paper", f"{name}.pdf"))
    rel = f"figures/paper/{name}.pdf"   # relative to latex/, where collaboration.tex compiles
    tex = ["\\begin{figure}[htbp]", "\\centering", f"\\includegraphics[width=\\linewidth]{{{rel}}}",
           f"\\caption{{{cap}}}", f"\\label{{{label}}}", "\\end{figure}", ""]
    open(os.path.join(texdir, f"{name}.tex"), "w").write("\n".join(tex))


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"))
    plt.close(fig)
    if name in CAPTIONS:
        write_figure_tex(out, name)


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
def fig2(d, c, b, out):
    """Figure 2 (14 Nov): decision parameters. (a), (b) scope relative to K* against the search probability p at
    alpha = 0.3 and 0.5; (c) scope against the window length l at alpha = 0.3 (Laplace and t3; moment solid, loggain
    dashed), which replaces the old Table 3b in the main text."""
    s = d[(d.cost_type == "power_law") & (d.block.isin(["main", "search"])) & (d.alpha.isin([0.3, 0.5]))]
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.0), gridspec_kw={"wspace": 0.32})
    for ax, a in zip(axes[:2], [0.3, 0.5]):
        for f in FAMILIES:
            g = med(s[(s.log_family == f) & (s.alpha == a)], "merge_thresh", ["K_mean"])
            ks = float(b[(b.family == f) & (b.alpha == a)].K_star.iloc[0])
            ax.plot(g.merge_thresh, g.K_mean / ks, color=COLOR[f], marker=MARKER[f], label=LABEL[f])
        ax.set_xscale("log"); ax.set_xticks([0.05, 0.2, 1]); ax.set_xticklabels(["0.05", "0.2", "1"])
        ax.set_xlabel("search probability $p$")
        ax.set_ylim(0, 1); ax.axhline(1, color="#888888", lw=0.8, ls=":")
    axes[0].set_title(r"(a) search, $\alpha = 0.3$", loc="left"); axes[1].set_title(r"(b) search, $\alpha = 0.5$", loc="left")
    axes[0].set_ylabel("$K / K^*$"); axes[1].set_ylabel("$K / K^*$"); axes[0].legend(loc="upper left")
    ax = axes[2]
    w = d[(d.block.isin(["lookback", "main"])) & (d.cost_type == "power_law") & (d.alpha == 0.3) & (d.log_family.isin(["laplace", "t3"]))]
    wc = c[(c.block.isin(["lookback", "main"])) & (c.cost_type == "power_law") & (c.alpha == 0.3) & (c.log_family.isin(["laplace", "t3"])) & (c.decision_rule == "loggain")]
    for f in ["laplace", "t3"]:
        g = med(w[w.log_family == f], "lookback", ["K_mean"]); gl = med(wc[wc.log_family == f], "lookback", ["K_mean"])
        ax.plot(g.lookback, g.K_mean, color=COLOR[f], marker=MARKER[f], label=LABEL[f])
        ax.plot(gl.lookback, gl.K_mean, color=COLOR[f], ls="--", lw=1.0, marker=MARKER[f], mfc="white", ms=4)
    ax.set_xscale("log"); ax.set_xticks([20, 50, 100, 200, 500]); ax.set_xticklabels(["20", "50", "100", "200", "500"]); ax.minorticks_off()
    ax.set_xlabel("window length $l$"); ax.set_ylabel("mean members per unit, $K$")
    ax.set_title(r"(c) window, $\alpha = 0.3$", loc="left"); ax.legend(loc="upper left")
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
    """Hill exponent against the sharing rate (15 Nov redraw). Main block as lines (medians over replications); the
    search block (p = 0.2 and 1) as hollow squares, one marker per cell = median over replications; the fixed
    alpha = 0.9 cell as crosses; the endogenous block as one diamond per family at its median adopted rate. The
    lookback, correlation and homogeneous markers of the earlier version are dropped (they sit on the curve and are
    in the robustness tables); the earlier version also drew every replication, hence five markers per cell."""
    x = d[d.K_mean.notna() & (d.decision_rule == "moment")]
    x = x[(x.cost_type == "power_law") & (x.floor_c.round(5) == 0.12717) & (x.sharing_rule == "proportional")]
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    for f in FAMILIES:
        g = med(x[(x.block == "main") & (x.log_family == f)], "alpha", ["hill_exponent_median"])
        ax.plot(g.alpha, g.hill_exponent_median, color=COLOR[f], marker=MARKER[f], ms=4, label=LABEL[f], zorder=4)
        sr = med(x[(x.block == "search") & (x.log_family == f)], ["alpha", "merge_thresh"], ["hill_exponent_median"])
        ax.scatter(sr.alpha, sr.hill_exponent_median, marker="s", s=28, facecolors="none", edgecolors=COLOR[f], lw=0.9,
                   label="search, $p = 0.2$ and $1$" if f == "normal" else None, zorder=3)
        a9 = med(x[(x.block == "alpha-0.9") & (x.log_family == f)], "alpha", ["hill_exponent_median"])
        ax.scatter(a9.alpha, a9.hill_exponent_median, marker="P", s=30, color=COLOR[f],
                   label=r"fixed $\alpha = 0.9$" if f == "normal" else None, zorder=4)
        e = x[(x.block == "endogenous-alpha") & (x.log_family == f)]
        ax.scatter([e.alpha_adopted_median.median()], [e.hill_exponent_median.median()], marker="D", s=30, facecolors="none",
                   edgecolors=COLOR[f], lw=1.0, label="endogenous, at adopted rate" if f == "normal" else None, zorder=4)
    ax.axvspan(0.5, 0.9, color="#000000", alpha=0.04, lw=0)
    ax.text(0.7, ax.get_ylim()[0] + 0.03, "no runs", ha="center", va="bottom", fontsize=7, color="#555555")
    ax.set_xlabel(r"sharing rate $\alpha$"); ax.set_ylabel("Hill exponent, within-market shares")
    ax.set_xlim(-0.02, 1.0)
    h, l = ax.get_legend_handles_labels()
    order = [l.index(LABEL[f]) for f in FAMILIES] + [i for i, lab in enumerate(l) if lab not in LABEL.values()]
    ax.legend([h[i] for i in order], [l[i] for i in order], loc="upper left", fontsize=7)
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
    fig1(d, c, b, out); fig2(d, c, b, out); fig3(a.root, out); fig4(d, out); fig5(a.root, out); fig6(a.root, out)
    print("written to", out)


if __name__ == "__main__":
    main()
