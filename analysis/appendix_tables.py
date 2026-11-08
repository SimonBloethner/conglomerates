"""Appendix tables from the pilot D / pilot C files, as Markdown (for the shared draft) and LaTeX (for the transfer).

Usage: python analysis/appendix_tables.py [--out latex/tables/appendix] [--md appendix_tables.md]
Every number is read from the files; nothing is typed in.
"""
import argparse
import os

import numpy as np
import pandas as pd

FAM = ["normal", "laplace", "t3"]
FAMTX = {"normal": "normal", "laplace": "Laplace", "t3": "t₃"}
FAMTEX = {"normal": "normal", "laplace": "Laplace", "t3": "$t_3$"}


def med(df, by, cols):
    return df.groupby(by)[cols].median()


class Out:
    def __init__(self, texdir):
        self.md = []
        self.texdir = texdir

    def section(self, title, text=None):
        self.md.append(f"\n## {title}\n")
        if text:
            self.md.append(text + "\n")

    def table(self, name, header, rows, caption, note=None):
        # markdown
        self.md.append(f"**{caption}**\n")
        self.md.append("| " + " | ".join(header) + " |")
        self.md.append("|" + "---|" * len(header))
        for r in rows:
            self.md.append("| " + " | ".join(str(c) for c in r) + " |")
        if note:
            self.md.append(f"\n{note}\n")
        self.md.append("")
        # latex
        tex = ["\\begin{table}[htbp]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{tab:{name}}}", "\\small",
               f"\\begin{{tabular}}{{l{'r' * (len(header) - 1)}}}", "\\toprule",
               " & ".join(h.replace("₃", "$_3$").replace("σ̂²/σ²", "$\\hat\\sigma^2/\\sigma^2$").replace("α", "$\\alpha$").replace("ν", "$\\nu$").replace("Δ̃", "$\\tilde\\Delta$").replace("Δ̂", "$\\hat\\Delta$") for h in header) + " \\\\", "\\midrule"]
        for r in rows:
            tex.append(" & ".join(str(c).replace("t₃", "$t_3$").replace("%", "\\%") for c in r) + " \\\\")
        tex += ["\\bottomrule", "\\end{tabular}"]
        if note:
            tex.append(f"\\par\\smallskip\\begin{{minipage}}{{0.95\\linewidth}}\\footnotesize {note.replace('%', chr(92) + '%')}\\end{{minipage}}")
        tex.append("\\end{table}\n")
        open(os.path.join(self.texdir, f"{name}.tex"), "w").write("\n".join(tex))


def f(x, nd=2):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--out", default=None)
    ap.add_argument("--md", default=None)
    ap.add_argument("--burn-in", choices=["c40", "c33"], default="c40",
                    help="A3 source: c40 (moment rule, pilot D; default) or c33 (the C33-ext loggain check, pilot C)")
    a = ap.parse_args()
    root = a.root
    texdir = a.out or os.path.join(root, "latex", "tables", "appendix")
    os.makedirs(texdir, exist_ok=True)
    o = Out(texdir)
    d = pd.read_csv(os.path.join(root, "pilot_d", "tidy.csv"))
    c = pd.read_csv(os.path.join(root, "pilot_c", "tidy.csv"))
    b = pd.read_csv(os.path.join(root, "analytics", "benchmarks.csv"))
    P = lambda name: os.path.join(root, "pilot_d", name)

    # A1 planner benchmark
    o.section("A1. Planner benchmark K*", "Equal members, i.i.d. shocks at the median market volatility q = 0.20, Monte Carlo (2·10⁶ draws), capped at M = 50; ×5–×20 for the power-law form only.")
    x = b[b.cost_multiplier == 1.0].pivot_table(index=["family", "cost_type"], columns="alpha", values="K_star")
    rows = [[FAMTX[fm], ct.replace("_", " ")] + [str(int(x.loc[(fm, ct)][al])) for al in [0.1, 0.3, 0.5]] for fm in FAM for ct in ["power_law", "linear", "quadratic", "exponential"]]
    o.table("a1_kstar", ["Family", "Cost form", "α = 0.1", "0.3", "0.5"], rows, "Table A1 — Planner's optimum K* by family, cost form and α")
    xh = b[(b.cost_multiplier > 1.0) & (b.cost_type == "power_law")].pivot_table(index=["family", "cost_multiplier"], columns="alpha", values="K_star")
    rows = [[FAMTX[fm], f"×{m:g}"] + [str(int(xh.loc[(fm, m)][al])) for al in [0.1, 0.3, 0.5]] for (fm, m) in xh.index]
    o.table("a1b_kstar_cost", ["Family", "Cost level", "α = 0.1", "0.3", "0.5"], rows, "Table A1b — K* at higher cost levels (power law)")

    # A2 floor calibration
    fc = pd.read_csv(os.path.join(root, "analytics", "floor_calibration.csv"))
    o.section("A2. Floor calibration", "Unaffiliated Laplace economy, α = 0, Hill exponent on the top decile of within-market shares; c* is the value at which the median Hill equals 1.06 (Axtell, 2001). The calibration is to the light-tailed families: at c* the normal and Laplace economies give 1.07 and 1.06, t₃ 1.00 and ν = 2.5 0.96.")
    rows = [[str(int(r.N)), f"{r.c:.4f}", f"{r.theoretical_hill:.3f}", f"{r.hill_median:.3f} [{r.hill_p25:.3f}, {r.hill_p75:.3f}]", f"{r.hhi_median:.3f}", f"{r.floor_frac_median:.3f}"] for _, r in fc.sort_values(["N", "c"]).iterrows()]
    o.table("a2_floor_calibration", ["N", "c", "1/(1 − c)", "Hill [IQR]", "HHI within", "share at floor"], rows, "Table A2 — Floor calibration grid")

    # A3 burn-in: C40 (moment rule, pilot D) by default; --burn-in c33 for the C33-ext loggain check (pilot C)
    if a.burn_in == "c40":
        bi = pd.read_csv(os.path.join(root, "diagnostics", "burn_in_c40.csv"))
        o.section("A3. Burn-in", "Twenty pilot D runs under the moment rule (Laplace α ∈ {0, 0.1, 0.3}, t₃ α = 0.3, main block, power law, five seeds) extended to T = 16,000 (C40): means over periods 11,000–16,000 against the pilot window 8,000–11,000, 95% CI across replications, and the OLS trend over 8,000–16,000. The 8,000-period burn-in stands if every headline series is within 2%. Each run reproduces its pilot D result up to period 11,000. The same check under the loggain rule (C33, pilot C) is in diagnostics/burn_in_c33_ext.md.")
    else:
        bi = pd.read_csv(os.path.join(root, "diagnostics", "burn_in_c33_ext.csv"))
        o.section("A3. Burn-in", "Twenty C33 runs (Laplace α ∈ {0, 0.1, 0.3}, t₃ α = 0.3, five seeds) extended to T = 16,000 under the loggain rule: means over periods 11,000–16,000 against the pilot window 8,000–11,000, 95% CI across replications, and the OLS trend over 8,000–16,000. The 8,000-period burn-in stands if every headline series is within 2%.")
    series = {"mean_k": "Mean K", "ccs": "Pooled capital share", "fhr_standalone": "Floor-hit rate, standalone (all-firm)", "fhr_member": "Floor-hit rate, member (all-firm)", "hill": "Hill exponent"}
    rows = []
    for (fm, al), g_ in bi.groupby(["log_family", "alpha"]):
        for key, lab in series.items():
            per_rep = []
            for rep, gr in g_.groupby("rep"):
                a_ = gr[(gr.step > 8000) & (gr.step <= 11000)][key].mean(); b_ = gr[(gr.step > 11000) & (gr.step <= 16000)][key].mean()
                if np.isfinite(a_) and np.isfinite(b_) and a_ != 0:
                    per_rep.append((a_, b_, 100 * (b_ - a_) / a_))
            if not per_rep:
                continue
            pr = np.array(per_rep)
            ci = 1.96 * pr[:, 2].std(ddof=1) / np.sqrt(len(pr)) if len(pr) > 1 else float("nan")
            rows.append([FAMTX.get(fm, fm), f"{al:g}", lab, f"{pr[:, 0].mean():.4f}", f"{pr[:, 1].mean():.4f}", f"{pr[:, 2].mean():+.2f}%", f"±{ci:.2f}"])
    o.table("a3_burnin", ["Family", "α", "Series", "Pilot window", "Periods 11,000–16,000", "Difference, % of pilot mean", "95% CI of the difference, ± % points"], rows, "Table A3 — Burn-in check: later window against the pilot window", note="Means over replications of each run's window means; difference in per cent of the pilot-window mean; CI across the five seeds.")

    # A4 loggain main block
    o.section("A4. The loggain rule as the bounded variant", "Main block, power-law cost, pilot C (same seeds as pilot D); medians over five replications. Compare Table 3.")
    mc = c[(c.block == "main") & (c.cost_type == "power_law") & (c.decision_rule == "loggain") & (c.alpha.isin([0.05, 0.1, 0.3, 0.5]))]
    g = med(mc, ["log_family", "alpha"], ["K_mean", "cong_capital_share_median", "hill_exponent_median", "acceptance_rate", "voluntary_exits_per_period", "floor_exits_per_period"])
    rows = [[FAMTX[fm], f"{al:g}", f(g.loc[(fm, al)].K_mean, 1), f(g.loc[(fm, al)].cong_capital_share_median), f(g.loc[(fm, al)].hill_exponent_median), f(g.loc[(fm, al)].acceptance_rate), f(g.loc[(fm, al)].voluntary_exits_per_period, 1), f(g.loc[(fm, al)].floor_exits_per_period, 1)] for fm in FAM for al in [0.05, 0.1, 0.3, 0.5]]
    o.table("a4_loggain_main", ["Family", "α", "K", "Pooled", "Hill", "Acceptance", "Voluntary exits / period", "Floor exits / period"], rows, "Table A4 — Main block under the loggain rule")

    # A5 acceptance by K both rules
    am = pd.read_csv(P("acceptance_by_K.csv")); al_ = pd.read_csv(os.path.join(root, "pilot_c", "acceptance_by_K.csv"))
    sel = lambda t: t[(t.block == "main") & (t.alpha == 0.3) & (t.p == 0.05) & (t.K <= 12)]
    gm = med(sel(am), ["family", "K"], ["acceptance", "binding_median", "marginal_gain_into_K", "marginal_cost_into_K"]); gl = med(sel(al_), ["family", "K"], ["acceptance", "binding_median"])
    o.section("A5. Consent by scope", "Acceptance of feasible proposals by size K of the prospective unit, α = 0.3, p = 0.05; binding gain = min over members of Δ̃ᵢ(Ĉ) − max{0, Δ̃ᵢ(Cᵢ)}; marginal gain = equal-weight Section 2 gain into K; medians over five replications. Figure 6.")
    rows = []
    for fm in FAM:
        for K in range(2, 13):
            r = gm.loc[(fm, K)]; rl = gl.loc[(fm, K)]
            rows.append([FAMTX[fm] if K == 2 else "", str(K), f(r.acceptance), f(rl.acceptance), f"{r.binding_median:.1e}", f"{rl.binding_median:.1e}", f"{r.marginal_gain_into_K:.1e}", f"{r.marginal_cost_into_K:.1e}"])
    o.table("a5_acceptance_by_K", ["Family", "K", "Acceptance, moment", "loggain", "Binding median, moment", "loggain", "Marginal gain", "Marginal cost"], rows, "Table A5 — Acceptance and binding gain by K, both rules")

    # A6 recruit size (C38)
    kw = pd.read_csv(P("acceptance_by_K_w_pooled.csv"))
    o.section("A6. Recruit size and the incumbents' gain (C38)", "Main block, power law, α = 0.3, moment rule, five replications pooled. Floor: standalone entrant with market share below 2c/N. Incumbent gain: exact second-order gain of an incumbent from an entrant of the observed mean weight; margin = that gain minus the marginal cost Φ(K) − Φ(K − 1).")
    rows = []
    for fm in FAM:
        for K in [3, 4, 5, 6, 7, 8, 10, 12]:
            fl = kw[(kw.family == fm) & (kw.alpha == 0.3) & (kw.K == K) & (kw.group == "floor")]
            big = kw[(kw.family == fm) & (kw.alpha == 0.3) & (kw.K == K) & (kw.group.str.contains(">=|≥"))]
            if len(fl) == 0:
                continue
            r = fl.iloc[0]
            bb = big.iloc[0] if len(big) else None
            rows.append([FAMTX[fm] if K == 3 else "", str(K), f(r.share_of_K), f(r.w_mean, 3), f(r.acceptance), f"{r.binding_mean:.1e}", f"{r.incumbent_margin:.1e}", f"{r.gain_equal:.1e}",
                         f(bb.acceptance) if bb is not None else "--", f"{bb.binding_mean:.1e}" if bb is not None else "--"])
    o.table("a6_recruit_size", ["Family", "K", "Share floor", "w̄ floor", "Acceptance, floor", "Binding mean, floor", "Incumbent margin, floor", "Equal-weight gain", "Acceptance, w ≥ 0.2", "Binding mean, w ≥ 0.2"], rows, "Table A6 — Proposals by recruit size")

    # A7 approximation error
    ae = pd.read_csv(P("approximation_error.csv")); ae = ae[ae.alpha == "all"]
    o.section("A7. The second-order approximation (C38)", "27 pilot D traces, 2,000 random decision points each (a standalone firm and K − 1 standalone partners from other markets), Δ̃ (moment) and Δ̂ (loggain replay) on the same 50-period window. Higher-order: Δ̃ − Δ̂ with the size weights frozen at the window's first period, i.e. the omitted terms proper; replay weight term: the replay's first-order term with in-window weights.")
    rows = [[FAMTX[r.family], str(int(r.K)), f"{r.ratio_median:.2f} [{r.ratio_p5:.2f}, {r.ratio_p95:.2f}]", f"{r.higher_order_ratio_fixed_median:.3f} [{r.higher_order_ratio_fixed_p5:.3f}, {r.higher_order_ratio_fixed_p95:.3f}]", f"{r.weight_term_hat_ratio_median:.2f}", f"{100*r.sign_disagree:.1f}%"] for _, r in ae.iterrows()]
    o.table("a7_approximation", ["Family", "K", "(Δ̃ − Δ̂)/Δ̂ median [5%, 95%]", "higher-order share, median [5%, 95%]", "replay weight term", "sign differs"], rows, "Table A7 — Size of the omitted terms and of the replay's first-order term")


    # A6b unit+unit by volatility bin (C39)
    uu = pd.read_csv(P("unit_unit_by_volbin_pooled.csv"))
    o.section("A6b. Unit\u2013unit proposals by relative volatility (C39)", "Same 15 runs as A6 with the proposal type recorded. Among proposals whose entrant side holds w \u2265 0.2, unit\u2013unit mergers are 42\u201346% at K = 4 and 88\u2013100% at K \u2265 8. Entrant calmer: the smaller unit has the lower mean member IQR. Calm binds: share of proposals in which the binding member (smallest margin) sits in the calmer unit.")
    rows = []
    for fm in FAM:
        for K in [4, 5, 6, 7, 8, 10, 12]:
            a_ = uu[(uu.family == fm) & (uu.alpha == 0.3) & (uu.K == K)]
            if len(a_) < 2:
                continue
            c_ = a_[a_.volbin == "entrant calmer"].iloc[0]; v_ = a_[a_.volbin == "entrant more volatile"].iloc[0]
            rows.append([FAMTX[fm] if K == 4 else "", str(K), f"{int(c_.proposals + v_.proposals):,}", f(c_.acceptance), f(v_.acceptance), f(c_.share_calm_binds), f(v_.share_calm_binds), f"{c_.binding_mean:.1e}", f"{v_.binding_mean:.1e}"])
    o.table("a6b_unit_unit", ["Family", "K", "Unit\u2013unit proposals", "Acceptance, entrant calmer", "entrant more volatile", "Calm unit binds, entrant calmer", "entrant more volatile", "Binding mean, entrant calmer", "entrant more volatile"], rows, "Table A6b \u2014 Unit\u2013unit proposals: who refuses", note="The volatile side refuses: when the entrant unit is calmer, the incumbent (volatile) unit holds the binding member in 67\u201387% of proposals and acceptance is about half that of the other bin.")

    # A7b cv rule (C39)
    cvr = pd.read_csv(P("approximation_error_cv.csv")); cvr = cvr[cvr.alpha == "all"]
    cvc = pd.read_csv(P("cv_comparison.csv"))
    o.section("A7b. The exact replay without its first-order term: the cv rule (C39)", "decision_rule = cv averages the exact replayed log gain minus its first-order term r\u03c4 \u03b1(\u03a3\u2c7c c\u2c7c\u03c4 u\u2c7c\u03c4 \u2212 u\u1d62\u03c4), which has expectation zero under every arrangement; it is exact in expectation and expands nothing. Left: the statistic on the A7 draws. Right: behaviour on the main power-law cells, 45 runs on the main cells' seeds, medians over five replications; paired differences on the seed.")
    rows = [[FAMTX[r.family], str(int(r.K)), f"{r.rel_cv_moment_median:.3f} [{r.rel_cv_moment_p5:.3f}, {r.rel_cv_moment_p95:.3f}]", f"{r.rel_hat_moment_median:.3f}", f"{100*r.sign_disagree_cv_moment:.2f}%", f"{100*r.sign_disagree_hat_moment:.2f}%"] for _, r in cvr.iterrows()]
    o.table("a7b_cv_statistic", ["Family", "K", "(\u0394\u0302\u1d9c\u1d5b \u2212 \u0394\u0303)/\u0394\u0303 median [5%, 95%]", "(\u0394\u0302 \u2212 \u0394\u0303)/\u0394\u0303 median", "sign differs, cv vs moment", "loggain vs moment"], rows, "Table A7b \u2014 The cv statistic against moment and loggain on the same windows")
    rows = [[FAMTX[r.family], f"{r.alpha:g}", f"{r.K_mean_cv:.2f} / {r.K_mean_moment:.2f} / {r.K_mean_loggain:.2f}", f"{r.K_mean_cv_minus_moment_median:+.2f} [{r.K_mean_cv_minus_moment_p25:+.2f}, {r.K_mean_cv_minus_moment_p75:+.2f}]", f"{r.K_mean_moment_minus_loggain_median:+.2f}", f"{r.cong_capital_share_median_cv:.3f} / {r.cong_capital_share_median_moment:.3f}", f"{r.hill_exponent_median_cv:.3f} / {r.hill_exponent_median_moment:.3f}"] for _, r in cvc.iterrows()]
    o.table("a7c_cv_behaviour", ["Family", "\u03b1", "K: cv / moment / loggain", "cv \u2212 moment, median [IQR]", "moment \u2212 loggain", "Pooled: cv / moment", "Hill: cv / moment"], rows, "Table A7c \u2014 Scope under the cv rule against moment and loggain", note="The cv rule reproduces the moment rule to within 0.11 members, 0.005 in pooled share and 0.03 in Hill; its difference to loggain is the first-order term alone.")

    # A8 homogeneous
    o.section("A8. Homogeneous markets", "Every market at q = 0.20, power law; both rules; medians over five replications. Compare Table 3c.")
    h = d[d.block == "homogeneous"]
    g = med(h, ["log_family", "decision_rule", "alpha"], ["K_mean", "cong_capital_share_median", "acceptance_rate", "hill_exponent_median"])
    rows = [[FAMTX[fm], rule, f"{al:g}", f(g.loc[(fm, rule, al)].K_mean, 1), f(g.loc[(fm, rule, al)].cong_capital_share_median), f(g.loc[(fm, rule, al)].acceptance_rate), f(g.loc[(fm, rule, al)].hill_exponent_median)] for fm in FAM for rule in ["moment", "loggain"] for al in [0.05, 0.1, 0.3, 0.5]]
    o.table("a8_homogeneous", ["Family", "Rule", "α", "K", "Pooled", "Acceptance", "Hill"], rows, "Table A8 — Homogeneous block")

    # A9 nu
    vr = pd.read_csv(P("variance_ratio.csv"))
    o.section("A9. Tail weight", "Student-t with ν ∈ {2.5, 3, 4} at equal IQR, power law, moment rule; σ̂²/σ² over 50-period i.i.d. windows at q = 0.2 (200,000 windows).")
    nu = d[(d.block == "nu")].copy(); t3 = d[(d.block == "main") & (d.cost_type == "power_law") & (d.log_family == "t3") & (d.alpha.isin([0.1, 0.3, 0.5]))].copy(); t3["nu"] = 3.0
    g = med(pd.concat([nu, t3]), ["nu", "alpha"], ["K_mean", "cong_capital_share_median", "hill_exponent_median", "acceptance_rate", "floor_exits_per_period", "floor_hit_rate_member_status", "floor_hit_rate_standalone_status"])
    rows = []
    for n_ in [2.5, 3.0, 4.0]:
        v = vr[(vr.family == "t") & (vr.nu == n_) & (vr.source.str.contains("iid"))].iloc[0]
        for al in [0.1, 0.3, 0.5]:
            r = g.loc[(n_, al)]
            rows.append([f"{n_:g}" if al == 0.1 else "", f"{al:g}", f(r.K_mean, 1), f(r.cong_capital_share_median), f(r.hill_exponent_median), f(r.acceptance_rate), f(r.floor_exits_per_period, 1), f(r.floor_hit_rate_member_status / r.floor_hit_rate_standalone_status), f"{v.q50:.2f} [{v.q05:.2f}, {v.q95:.2f}]" if al == 0.1 else "", f"{v.share_below_half:.2f}" if al == 0.1 else ""])
    o.table("a9_nu", ["ν", "α", "K", "Pooled", "Hill", "Acceptance", "Floor exits / period", "Hazard ratio", "σ̂²/σ² median [5%, 95%]", "share < ½"], rows, "Table A9 — ν block")

    # A10 cost forms and levels
    o.section("A10. Cost forms and levels", "Laplace (cost forms: all three families at α = 0.3 in the first table), power law unless stated; medians over five replications. Table 6b has the power-law level series.")
    m = d[(d.block == "main") & (d.alpha == 0.3)]
    g = med(m, ["log_family", "cost_type"], ["K_mean", "cong_capital_share_median", "hill_exponent_median", "acceptance_rate"])
    rows = [[FAMTX[fm], ct.replace("_", " "), f(g.loc[(fm, ct)].K_mean, 1), f(g.loc[(fm, ct)].cong_capital_share_median), f(g.loc[(fm, ct)].hill_exponent_median), f(g.loc[(fm, ct)].acceptance_rate)] for fm in FAM for ct in ["power_law", "linear", "quadratic", "exponential"]]
    o.table("a10_cost_forms", ["Family", "Cost form", "K", "Pooled", "Hill", "Acceptance"], rows, "Table A10 — Cost forms at equal Φ(40), α = 0.3")
    cl = d[(d.block.isin(["cost-level", "cost-level-full"])) & (d.alpha.isin([0.1, 0.3, 0.5]))]
    g = med(cl, ["block", "cost_type", "cost_multiplier", "alpha"], ["K_mean", "cong_capital_share_median", "acceptance_rate"])
    rows = [[blk.replace("cost-level-full", "all coefficients").replace("cost-level", "b₀ only"), ct.replace("_", " "), f"×{mlt:g}", f"{al:g}", f(g.loc[(blk, ct, mlt, al)].K_mean, 1), f(g.loc[(blk, ct, mlt, al)].cong_capital_share_median), f(g.loc[(blk, ct, mlt, al)].acceptance_rate)] for (blk, ct, mlt, al) in g.index]
    o.table("a10b_cost_levels", ["Scaled", "Cost form", "Level", "α", "K", "Pooled", "Acceptance"], rows, "Table A10b — Cost level ×0.5 and ×2, Laplace")

    # A11 floor, correlation, size
    o.section("A11. Floor level, correlation, economy size", "Laplace, power law; medians over five replications.")
    for blk, keys, name, cap in [("floor-level", ["floor_c"], "a11a_floor", "Table A11a — Floor level"), ("correlation", ["cross_corr"], "a11b_correlation", "Table A11b — Cross-market correlation 0.3"), ("size", ["M", "N"], "a11c_size", "Table A11c — Economy size with c*(N)")]:
        x = d[(d.block == blk) & (d.alpha.isin([0.1, 0.3, 0.5]))]
        g = med(x, keys + ["alpha"], ["K_mean", "cong_capital_share_median", "hill_exponent_median", "acceptance_rate", "floor_exits_per_period"])
        rows = [[" / ".join(f"{k:g}" for k in (idx[:-1] if isinstance(idx, tuple) else (idx,))), f"{idx[-1]:g}", f(r.K_mean, 1), f(r.cong_capital_share_median), f(r.hill_exponent_median), f(r.acceptance_rate), f(r.floor_exits_per_period, 1)] for idx, r in g.iterrows()]
        o.table(name, [" / ".join(keys), "α", "K", "Pooled", "Hill", "Acceptance", "Floor exits / period"], rows, cap)

    # A12 equal split
    o.section("A12. Equal sharing", "Normal shocks, four cost forms; medians over five replications; acceptance by proposal type (standalone–standalone, standalone–unit); pool ruin: members capped at limited liability per period.")
    eq = d[(d.block == "equal-split") & (d.alpha.isin([0.05, 0.1, 0.3, 0.5]))]
    g = med(eq, ["cost_type", "alpha"], ["K_mean", "cong_capital_share_median", "acceptance_rate_ss", "acceptance_rate_sc", "voluntary_exits_per_period", "pool_ruin_per_period", "pool_shortfall_per_period"])
    rows = [[ct.replace("_", " ") if al == 0.05 else "", f"{al:g}", f(r.K_mean, 1), f(r.cong_capital_share_median), f(r.acceptance_rate_ss), f(r.acceptance_rate_sc), f(r.voluntary_exits_per_period, 1), f(r.pool_ruin_per_period, 3), f"{r.pool_shortfall_per_period:.1e}"] for (ct, al), r in g.iterrows()]
    o.table("a12_equal_split", ["Cost form", "α", "K", "Pooled", "Acceptance s–s", "Acceptance s–u", "Voluntary exits / period", "Pool ruin / period", "Shortfall / period"], rows, "Table A12 — Equal-split block")

    # A13 endogenous alpha all cost forms + fixed 0.9
    o.section("A13. Endogenous sharing rate, all cost forms", "Start α = 0.1; medians over five replications; moment / loggain where both exist.")
    e = d[d.block == "endogenous-alpha"]; ec = c[(c.block == "endogenous-alpha")]
    g = med(e, ["log_family", "cost_type"], ["alpha_adopted_mean", "alpha_adopted_median", "alpha_adopted_share1", "K_mean", "cong_capital_share_median", "hill_exponent_median"]); gc = med(ec, ["log_family", "cost_type"], ["alpha_adopted_median", "K_mean"])
    rows = [[FAMTX[fm] if ct == "power_law" else "", ct.replace("_", " "), f(r.alpha_adopted_mean), f"{r.alpha_adopted_median:.2f} / {gc.loc[(fm, ct)].alpha_adopted_median:.2f}", f(r.alpha_adopted_share1), f"{r.K_mean:.1f} / {gc.loc[(fm, ct)].K_mean:.1f}", f(r.cong_capital_share_median), f(r.hill_exponent_median)] for fm in FAM for ct in ["power_law", "linear", "quadratic", "exponential"] for r in [g.loc[(fm, ct)]]]
    o.table("a13_endogenous", ["Family", "Cost form", "ᾱ", "median, moment / loggain", "share at 1", "K, moment / loggain", "Pooled", "Hill"], rows, "Table A13 — Endogenous α by cost form")

    # A14 event study
    es = pd.read_csv(P("event_study_matching.csv"))
    o.section("A14. Event study of entry (C38)", "27 traces, moment rule; joiner matched to a standalone in the same market; difference in differences of log-share growth over 50 periods before and after entry, pp per period; owner path net of recapitalization. Current: raw share at entry; A: owner-path share; B: both sides hit-free for k = 10 periods before entry; C: same number of pre-window floor hits. Post gap: joiner − control owner-path growth after entry.")
    want = ["current", "A_owner", "B_k10", "C_recap"]
    rows = []
    for fm in FAM:
        for al in [0.05, 0.1, 0.3]:
            x = es[(es.family == fm) & (es.alpha == al)].set_index("variant")
            rows.append([FAMTX[fm] if al == 0.05 else "", f"{al:g}", f"{100*x.loc['current'].did_itt_median:.2f}"] + [f"{100*x.loc[v].did_owner_median:.2f}" for v in want] + [f"{100*x.loc['current'].post_gap_owner_mean:.2f}", f"{int(x.loc['B_k10'].n_matched):,}"])
    o.table("a14_event_study", ["Family", "α", "ITT DiD", "Owner DiD: raw-share matching", "owner-path matching", "hit-free k = 10", "same hit count", "Post-entry gap", "matched events (B)"], rows, "Table A14 — Event study under four matchings")

    # A15 burstiness
    bs = pd.read_csv(P("burstiness_summary.csv")).groupby(["family", "alpha"]).mean(numeric_only=True)
    o.section("A15. Clustering of floor hits", "Index of dispersion of floor hits per market-period against the binomial reference; share of market-periods with three or more hits; worst market-period in a run. Means over three traced replications.")
    rows = [[FAMTX[fm] if al == 0.05 else "", f"{al:g}", f(r.hits_per_market_period), f"{r.D_median:.2f} [{r.D_p25:.2f}, {r.D_p75:.2f}]", f(r.D_binomial), f(r.share_periods_3plus), f(r.share_periods_3plus_binom), f"{r.max_hits_one_period:.0f}"] for (fm, al), r in bs.iterrows()]
    o.table("a15_burstiness", ["Family", "α", "Hits / market-period", "D median [IQR]", "D binomial", "share ≥ 3 hits", "binomial", "max hits in one period"], rows, "Table A15 — Burstiness")

    md = "\n".join(o.md)
    mdpath = a.md or os.path.join(root, "diagnostics", "appendix_tables.md")
    open(mdpath, "w").write("# Appendix tables (generated by analysis/appendix_tables.py)\n" + md)
    print("written", texdir, mdpath)


if __name__ == "__main__":
    main()
