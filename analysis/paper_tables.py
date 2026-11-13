"""Paper tables as LaTeX (booktabs) from the pilot D files, with the pilot C loggain twin where the text quotes it.

Usage: python analysis/paper_tables.py [--out latex/tables] [--table3c-accept]

Writes table1_parameters.tex, table2_design.tex, table3_scope.tex, table3b_window.tex, table3c_decomposition.tex,
table4_owner_return.tex, table5_endogenous_alpha.tex, table6_robustness.tex, table6b_cost_level.tex.
C42: --table3c-accept also writes table3c_decomposition_accept.tex, Table 3c re-cut with the accept rule's K
(pilot_d/consent_gap_decomposition_accept.csv): environment / consent / transfer conflict / verifiability. The current
table3c_decomposition.tex is written as before.
Every number is computed from the CSVs / scenarios.json; nothing is typed in. Captions are placeholders for the
Section 4 doc's captions and can be edited in the tex that \\input{}s these files.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

FAM = ["normal", "laplace", "t3"]
FAMTEX = {"normal": "normal", "laplace": "Laplace", "t3": "$t_3$"}


def f(x, nd=1):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    return f"{x:.{nd}f}"


def table(rows, header, caption, label, colspec=None, note=None, small=True):
    n = len(header)
    colspec = colspec or ("l" + "r" * (n - 1))
    lines = ["\\begin{table}[htbp]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}"]
    if small:
        lines.append("\\small")
    lines.append(f"\\begin{{tabular}}{{{colspec}}}")
    lines.append("\\toprule")
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\midrule")
    for r in rows:
        if r == "midrule":
            lines.append("\\midrule")
        else:
            lines.append(" & ".join(str(c) for c in r) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    if note:
        lines.append(f"\\par\\smallskip\\begin{{minipage}}{{0.95\\linewidth}}\\footnotesize {note}\\end{{minipage}}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def med(df, by, cols):
    return df.groupby(by)[cols].median()


def load(root):
    d = pd.read_csv(os.path.join(root, "pilot_d", "tidy.csv"))
    c = pd.read_csv(os.path.join(root, "pilot_c", "tidy.csv"))
    b = pd.read_csv(os.path.join(root, "analytics", "benchmarks.csv"))
    sc = json.load(open(os.path.join(root, "pilot_d", "scenarios.json")))
    return d, c, b, sc


def kstar(b, fam, alpha, cost="power_law", mult=1.0):
    x = b[(b.family == fam) & (b.alpha == alpha) & (b.cost_type == cost) & (b.cost_multiplier == mult)]
    return int(x.K_star.iloc[0]) if len(x) else None


# ---------------------------------------------------------------- Table 1
def table1(sc, out):
    main = [s for s in sc if s["block"] == "main"]
    s0 = main[0]
    alphas = sorted({s["alpha"] for s in main})
    vals = lambda key, blk: sorted({s[key] for s in sc if s["block"] == blk})
    rows = [
        ["$M$, $N$", "markets, firms per market", f"{s0['M']}, {s0['N']}", "size: $\\{25, 100\\}^2$"],
        ["$T$, burn-in", "periods, discarded periods", f"{s0['T']:,}, {s0['burn_in']:,}", "--"],
        ["$F$", "shock family, equal IQR", "normal, Laplace, $t_3$", f"$\\nu$: Student-$t$, $\\nu \\in \\{{{', '.join(str(v) for v in vals('nu', 'nu'))}\\}}$"],
        ["$q_m$", "market IQR of $x_{it}$", f"rescaled onto $[{s0['sigma_range'][0]}, {s0['sigma_range'][1]}]$", "homogeneous: every market at 0.20"],
        ["$g$", "time-average growth rate", f"{s0['g']}", "--"],
        ["--", "cross-market correlation", f"{s0['cross_corr']}", f"correlation: {vals('cross_corr', 'correlation')[-1]}"],
        ["$c$", "floor, fraction of market mean", f"{s0['floor_c']}", f"floor: {', '.join(str(v) for v in vals('floor_c', 'floor-level'))}; size: $c^*(N)$"],
        ["$\\alpha$", "sharing rate", ", ".join(str(a) for a in alphas), "endogenous: chosen by each unit; fixed 0.9"],
        ["--", "sharing rule", "proportional to size", "equal split"],
        ["$\\Phi(K)$", "coordination cost, $\\Phi(40) = 0.0017$", "power law (main); linear, quadratic, exponential",
         "cost level: $\\times$" + ", $\\times$".join(str(v) for v in sorted({s.get('cost_multiplier', 1.0) for s in sc if s['block'] in ('cost-level', 'cost-level-high')} - {1.0}))],
        ["$p$", "search probability per firm-period", f"{s0['merge_thresh']}", f"search: {', '.join(str(v) for v in vals('merge_thresh', 'search'))}"],
        ["$l$", "decision window and lock-in", f"{s0['lookback']}", f"lookback: {', '.join(str(v) for v in vals('lookback', 'lookback') if v != s0['lookback'])}"],
        ["--", "decision rule", "second moments of the demeaned window (moment)", "sample mean of the replayed log gain (loggain), every cell"],
        ["--", "replications per cell", str(len({s["rep"] for s in main})), "--"],
    ]
    note = ("Cost parameters: power law $b_0 = 2.032\\cdot10^{-5}$, $b_1 = 1.2$; linear $b_0 = 10^{-5}$, $b_1 = 4.225\\cdot10^{-5}$; "
            "quadratic $b_0 = 10^{-4}$, $b_1 = 10^{-6}$, $b_2 = 9.7\\cdot10^{-7}$; exponential $b_0 = 10^{-3}$, $b_1 = 0.013266$.")
    open(os.path.join(out, "table1_parameters.tex"), "w").write(
        table(rows, ["Symbol", "Meaning", "Main block", "Varied in block"], "Parameters (main block, and the block in which each is varied)",
              "tab:parameters", colspec="llll", note=note))


# ---------------------------------------------------------------- Table 2
def table2(sc, out):
    df = pd.DataFrame(sc)
    order = ["main", "equal-split", "cost-level", "cost-level-full", "cost-level-high", "lookback", "correlation",
             "endogenous-alpha", "floor-level", "search", "size", "homogeneous", "nu", "alpha-0.9"]
    what = {"main": "--", "equal-split": "sharing rule: equal", "cost-level": "$b_0 \\times 0.5, \\times 2$ (power law, exponential: whole schedule)",
            "cost-level-full": "$b_0, b_1, b_2 \\times 0.5, \\times 2$", "cost-level-high": "$b_0 \\times 5, \\times 10, \\times 20$",
            "lookback": "$l \\in \\{20, 50, 100, 200, 500\\}$", "correlation": "cross-market correlation 0.3",
            "endogenous-alpha": "$\\alpha_C$ chosen by unit, start 0.1", "floor-level": "$c \\in \\{0.10, 0.35\\}$",
            "search": "$p \\in \\{0.2, 1\\}$", "size": "$(M, N) \\in \\{25, 100\\}^2$, $c^*(N)$",
            "homogeneous": "every market at $q = 0.20$; both decision rules", "nu": "Student-$t$, $\\nu \\in \\{2.5, 4\\}$",
            "alpha-0.9": "fixed $\\alpha = 0.9$ (C38)"}
    grid = sorted(df[df.block == "main"].alpha.unique())
    rows = []
    for blk in order:
        x = df[df.block == blk]
        if len(x) == 0:
            continue
        fams = sorted(x.log_family.unique(), key=lambda s: (FAM + ["student_t"]).index(s) if s in FAM + ["student_t"] else 9)
        famtxt = "all three" if set(fams) >= set(FAM) else ", ".join(FAMTEX.get(s, "$t$") for s in fams)
        costs = sorted(x.cost_type.unique())
        costtxt = "all four" if len(costs) == 4 else ", ".join(s.replace("_", " ") for s in costs)
        al = sorted(x.alpha.dropna().unique())
        altxt = "grid" if list(al) == list(grid) else ("--" if blk == "endogenous-alpha" else ", ".join(f"{a:g}" for a in al))
        rows.append([blk.replace("-", " ").replace("alpha", "$\\alpha$").replace("nu", "$\\nu$") if blk not in ("alpha-0.9",) else "fixed $\\alpha = 0.9$",
                     what[blk], famtxt, costtxt, altxt, f"{len(x):,}"])
    rows.append("midrule")
    rows.append(["total", "", "", "", "", f"{len(df):,}"])
    note = ("Five replications per cell; ``grid'' is the nine-value $\\alpha$ grid. Every cell except the $\\nu$ block is also run under the loggain rule "
            "with the same seeds (pilot C, 2,100 runs).")
    open(os.path.join(out, "table2_design.tex"), "w").write(
        table(rows, ["Block", "What varies", "Families", "Cost forms", "$\\alpha$", "Runs"], "Simulation design", "tab:design",
              colspec="llllll", note=note))


# ---------------------------------------------------------------- Table 3
def table3(d, c, b, out):
    A = [0.05, 0.1, 0.3, 0.5]
    m = d[(d.block == "main") & (d.cost_type == "power_law") & (d.alpha.isin(A))]
    mc = c[(c.block == "main") & (c.cost_type == "power_law") & (c.alpha.isin(A)) & (c.decision_rule == "loggain")]
    cols = ["K_mean", "cong_capital_share_median", "floor_exits_per_period", "voluntary_exits_per_period", "acceptance_rate",
            "floor_hit_rate_member_status", "floor_hit_rate_standalone_status"]
    g = med(m, ["log_family", "alpha"], cols)
    gl = med(mc, ["log_family", "alpha"], ["K_mean"])
    rows = []
    for fam in FAM:
        for a in A:
            r = g.loc[(fam, a)]
            ks = kstar(b, fam, a)
            rows.append([FAMTEX[fam] if a == A[0] else "", f"{a:g}", f(r.K_mean), f(gl.loc[(fam, a)].K_mean), str(ks) if ks else "--",
                         f(r.K_mean / ks, 2) if ks else "--", f(r.cong_capital_share_median, 2), f(r.floor_exits_per_period),
                         f(r.voluntary_exits_per_period, 2), f(r.acceptance_rate, 2), f(r.floor_hit_rate_member_status, 3),
                         f(r.floor_hit_rate_standalone_status, 3)])
        if fam != FAM[-1]:
            rows.append("midrule")
    note = ("$K$: mean members per active unit (moment rule; loggain twin from pilot C); $K^*$: planner's optimum at the median market volatility, capped at $M = 50$; "
            "pooled: share of economy capital held by members; exits per period among 2,500 firms (mergers per period equal floor exits to within 0.2); "
            "acceptance: share of feasible proposals accepted; floor hit: hits per firm-period of that status.")
    open(os.path.join(out, "table3_scope.tex"), "w").write(
        table(rows, ["Family", "$\\alpha$", "$K$", "$K$, loggain", "$K^*$", "$K/K^*$", "Pooled", "Floor exits", "Vol.\\ exits", "Accept.",
                     "Floor hit, member", "Floor hit, standalone"],
              "Scope, turnover and hazard; main block, power-law cost; medians over five replications", "tab:scope", note=note))


# ---------------------------------------------------------------- Table 3b
def table3b(d, c, out):
    L = [20, 50, 100, 200, 500]
    x = d[(d.block.isin(["lookback", "main"])) & (d.cost_type == "power_law") & (d.alpha.isin([0.1, 0.3])) & (d.log_family.isin(["laplace", "t3"]))]
    xc = c[(c.block.isin(["lookback", "main"])) & (c.cost_type == "power_law") & (c.alpha.isin([0.1, 0.3])) & (c.log_family.isin(["laplace", "t3"])) & (c.decision_rule == "loggain")]
    g = med(x, ["log_family", "alpha", "lookback"], ["K_mean"]); gc = med(xc, ["log_family", "alpha", "lookback"], ["K_mean"])
    rows = []
    for fam in ["laplace", "t3"]:
        for a in [0.1, 0.3]:
            rows.append([FAMTEX[fam] + ", $K$", f"{a:g}"] + [f"{g.loc[(fam, a, l)].K_mean:.1f} / {gc.loc[(fam, a, l)].K_mean:.1f}" for l in L])
    note = ("Cells: moment / loggain. Floor exits per period do not depend on $l$ (35--38 under Laplace, 41--47 under $t_3$ at $\\alpha = 0.3$); "
            "voluntary exits are nil at every $l$ under the moment rule and fall from about five per period at $l = 20$ to none beyond $l = 200$ under loggain.")
    open(os.path.join(out, "table3b_window.tex"), "w").write(
        table(rows, ["Family", "$\\alpha$"] + [f"$l = {L[0]}$"] + [str(l) for l in L[1:]], "Window length, power-law cost; medians over five replications",
              "tab:window", note=note))


# ---------------------------------------------------------------- Table 3c
def table3c(root, out):
    """Table 3c, five-scope form (10 Nov): the equal-members reference K*, the accept-rule control K_accept (what the
    environment permits), K_hom, K (base) and K_loggain, with two differences: consent (K_accept - K) and
    verifiability (K - K_loggain). The transfer conflict (K_hom - K) goes to the note; the pre-C42 three-part
    table is kept in a3c_legacy if ever needed (--table3c-legacy)."""
    x = pd.read_csv(os.path.join(root, "pilot_d", "consent_gap_decomposition_accept.csv"))
    rows = []
    for fam in FAM:
        y = x[x.family == fam].sort_values("alpha")
        for i, r in y.iterrows():
            consent = r.K_accept - r.K_mom_het
            verif = r.K_mom_het - r.K_log_het
            rows.append([FAMTEX[fam] if i == y.index[0] else "", f"{r.alpha:g}", str(int(r.K_star)), f(r.K_accept),
                         f(r.K_mom_hom), f(r.K_mom_het), f(r.K_log_het),
                         f"{consent:.1f} ({100 * consent / r.K_accept:.0f}\\%)", f"{verif:.1f} ({100 * verif / r.K_accept:.0f}\\%)"])
        if fam != FAM[-1]:
            rows.append("midrule")
    note = ("$K^*$: equal-members reference (Section 3.4). $K_{\\text{accept}}$: every feasible proposal accepted and no "
            "voluntary exit, i.e. the scope the floor and the one-firm-per-market constraint alone permit (a control without "
            "knowledge). $K_{\\text{hom}}$: homogeneous economy (every market at $q = 0.20$), moment rule. $K$: base case. "
            "$K_{\\text{loggain}}$: the same gain estimated by its sample mean. Consent and verifiability are shares of "
            "$K_{\\text{accept}}$. The transfer conflict between members of unequal volatility, $K_{\\text{hom}} - K$, is "
            "0.0--0.4 members in every cell (at most 4\\% of $K^* - K_{\\text{loggain}}$) and is not shown.")
    open(os.path.join(out, "table3c_decomposition.tex"), "w").write(
        table(rows, ["Family", "$\\alpha$", "$K^*$", "$K_{\\text{accept}}$", "$K_{\\text{hom}}$", "$K$", "$K_{\\text{loggain}}$",
                     "Consent: $K_{\\text{accept}} - K$", "Verifiability: $K - K_{\\text{loggain}}$"],
              "Scope under five rules and the two differences that bound it, power-law cost; medians over five replications",
              "tab:decomposition", note=note))


def table3c_legacy(root, out):
    """The pre-C42 three-part decomposition (survival+feasibility / transfer / verifiability), kept for the record."""
    x = pd.read_csv(os.path.join(root, "pilot_d", "consent_gap_decomposition.csv"))
    rows = []
    for fam in FAM:
        y = x[x.family == fam].sort_values("alpha")
        for i, r in y.iterrows():
            tot = r.total
            rows.append([FAMTEX[fam] if i == y.index[0] else "", f"{r.alpha:g}", str(int(r.K_star)), f(r.K_mom_hom), f(r.K_mom_het), f(r.K_log_het),
                         f"{r.a_attrition:.1f} ({100*r.a_attrition/tot:.0f}\\%)", f"{r.b_transfer:.1f} ({100*r.b_transfer/tot:.0f}\\%)",
                         f"{r.c_verif_het:.1f} ({100*r.c_verif_het/tot:.0f}\\%)"])
        if fam != FAM[-1]:
            rows.append("midrule")
    note = ("Shares in parentheses are of $K^* - K_{\\text{loggain}}$ (the three parts sum to it). $K_{\\text{hom}}$: homogeneous economy (every market at $q = 0.20$), moment rule. "
            "The first part holds attrition, feasibility, the size of the recruits on offer and residual estimation noise; the third the loggain estimate's sampling noise and finite-window bias.")
    open(os.path.join(out, "table3c_decomposition_legacy.tex"), "w").write(
        table(rows, ["Family", "$\\alpha$", "$K^*$", "$K_{\\text{hom}}$", "$K$", "$K_{\\text{loggain}}$", "Survival, feasibility: $K^* - K_{\\text{hom}}$",
                     "Transfer conflict: $K_{\\text{hom}} - K$", "Verifiability: $K - K_{\\text{loggain}}$"],
              "Decomposition of the distance to the planner's scope, power-law cost; medians over five replications", "tab:decomposition-legacy", note=note))


def table3c_accept(root, out):
    """C42: Table 3c re-cut with K_accept (the accept rule: every feasible proposal accepted, no voluntary exit)."""
    x = pd.read_csv(os.path.join(root, "pilot_d", "consent_gap_decomposition_accept.csv"))
    rows = []
    for fam in FAM:
        y = x[x.family == fam].sort_values("alpha")
        for i, r in y.iterrows():
            tot = r.total
            rows.append([FAMTEX[fam] if i == y.index[0] else "", f"{r.alpha:g}", str(int(r.K_star)), f(r.K_accept),
                         f(r.K_mom_hom), f(r.K_mom_het), f(r.K_log_het)] +
                        [f"{v:.1f} ({100 * v / tot:.0f}\\%)" for v in (r.environment, r.consent_size_noise,
                                                                      r.transfer, r.verifiability)])
        if fam != FAM[-1]:
            rows.append("midrule")
    note = ("Shares in parentheses are of $K^* - K_{\\text{loggain}}$ (the four parts sum to it; the first two sum to the "
            "first column of Table~\\ref{tab:decomposition}). $K_{\\text{accept}}$: heterogeneous economy, every feasible "
            "proposal accepted and no voluntary exit (a control without knowledge). A negative environment part means the "
            "environment alone permits units larger than $K^*$. The consent part compares the heterogeneous accept economy "
            "with the homogeneous moment economy, so it also holds the homogeneous-heterogeneous difference of the accept rule.")
    open(os.path.join(out, "table3c_decomposition_accept.tex"), "w").write(
        table(rows, ["Family", "$\\alpha$", "$K^*$", "$K_{\\text{accept}}$", "$K_{\\text{hom}}$", "$K$", "$K_{\\text{loggain}}$",
                     "Environment: $K^* - K_{\\text{accept}}$", "Consent: $K_{\\text{accept}} - K_{\\text{hom}}$",
                     "Transfer conflict: $K_{\\text{hom}} - K$", "Verifiability: $K - K_{\\text{loggain}}$"],
              "Decomposition of the distance to the planner's scope with the accept rule, power-law cost; medians over "
              "five replications", "tab:decomposition-accept", note=note))


# ---------------------------------------------------------------- Table 4
def table4(root, out):
    p = pd.read_csv(os.path.join(root, "pilot_d", "status_panel_summary.csv")).groupby(["family", "alpha"]).mean(numeric_only=True)
    dec = pd.read_csv(os.path.join(root, "pilot_d", "status_panel_decomposition.csv"))
    dec = dec.groupby(["family", "alpha"]).mean(numeric_only=True)
    rows = []
    for fam in FAM:
        for a in [0.05, 0.1, 0.3]:
            r = p.loc[(fam, a)]; s = dec.loc[(fam, a)]
            rows.append([FAMTEX[fam] if a == 0.05 else "", f"{a:g}", f(100 * r.diff_fe_mean, 2), f(100 * r.diff_fe_median, 2),
                         f(100 * r.diff_K2_mean, 2), f(100 * r.diff_K3_4_mean, 2), f(100 * r.diff_K5p_mean, 2),
                         f(s.share_ordinary, 2), f(r.floor_rate_member, 3), f(r.floor_rate_standalone, 3), f(r.floor_rate_member / r.floor_rate_standalone, 3)])
        if fam != FAM[-1]:
            rows.append("midrule")
    note = ("Owner return: next-period log change of the firm's market share net of floor recapitalization; gap = member $-$ standalone within market $\\times$ size decile, "
            "in percentage points per period; 27 traced runs, means over three replications. Share ordinary: part of the gap accruing in periods without a floor hit. "
            "Floor rates per firm-period of that status; hazard ratio = member / standalone.")
    open(os.path.join(out, "table4_owner_return.tex"), "w").write(
        table(rows, ["Family", "$\\alpha$", "Gap, mean", "Gap, median", "$K = 2$", "$K = 3$--4", "$K \\geq 5$", "Share ordinary", "Floor, member", "Floor, standalone", "Hazard ratio"],
              "Owners' time-average growth and ruin hazard by affiliation status", "tab:owner", note=note))


# ---------------------------------------------------------------- Table 5
def table5(d, root, out):
    e = d[(d.block == "endogenous-alpha") & (d.cost_type == "power_law")]
    g = med(e, "log_family", ["alpha_adopted_mean", "alpha_adopted_median", "alpha_adopted_share1", "K_mean", "cong_capital_share_median", "hill_exponent_median"])
    sc = pd.read_csv(os.path.join(root, "pilot_d", "alpha_scatter.csv"))
    f9 = d[d.block == "alpha-0.9"]
    g9 = med(f9, "log_family", ["K_mean", "cong_capital_share_median", "hill_exponent_median"]) if len(f9) else None
    rows = []
    from scipy.stats import spearmanr
    for fam in FAM:
        r = g.loc[fam]; s = sc[sc.log_family == fam]
        rho = spearmanr(s.alpha_tavg, s.mean_iqr).correlation
        rows.append([FAMTEX[fam], f(r.alpha_adopted_mean, 2), f(r.alpha_adopted_median, 2), f(r.alpha_adopted_share1, 2), f(r.K_mean),
                     f(r.cong_capital_share_median, 2), f(r.hill_exponent_median, 2), f(rho, 2), f"{len(s):,}",
                     f(g9.loc[fam].K_mean) if g9 is not None else "--", f(g9.loc[fam].hill_exponent_median, 2) if g9 is not None else "--"])
    note = ("Endogenous-$\\alpha$ block, power-law cost, start $\\alpha_C = 0.1$; adopted $\\alpha$ over unit-periods; share at 1: unit-periods at $\\alpha_C = 1$. "
            "$\\rho$: Spearman correlation of a unit's time-averaged rate with the mean IQR of its members' markets, units alive at $T$. "
            "Last two columns: the fixed $\\alpha = 0.9$ cell for comparison.")
    open(os.path.join(out, "table5_endogenous_alpha.tex"), "w").write(
        table(rows, ["Family", "$\\bar\\alpha$", "median", "share at 1", "$K$", "Pooled", "Hill", "$\\rho(\\alpha, \\bar q)$", "units", "$K$ at $\\alpha = 0.9$", "Hill at $\\alpha = 0.9$"],
              "The sharing rate units choose", "tab:endogenous", note=note))


def nu_row(d, cols):
    nu = d[(d.block == "nu") & (d.alpha == 0.3)].copy()
    t3 = d[(d.block == "main") & (d.cost_type == "power_law") & (d.log_family == "t3") & (d.alpha == 0.3)].copy()
    t3["nu"] = 3.0
    g = med(pd.concat([nu, t3]), "nu", cols)
    fmt = ("1", "2", "2", "0")
    return [" / ".join(f"{g.loc[k][col]:.{fmt[i]}f}" for k in [2.5, 3.0, 4.0]) for i, col in enumerate(cols)]


# ---------------------------------------------------------------- Table 6 and 6b
def table6(d, c, b, out):
    cols = ["K_mean", "cong_capital_share_median", "hill_exponent_median", "floor_exits_per_period"]
    def row(sel, by, keys, fmt=("1", "2", "2", "0")):
        g = med(sel, by, cols)
        vals = []
        for i, col in enumerate(cols):
            vals.append(" / ".join(f"{g.loc[k][col]:.{fmt[i]}f}" for k in keys))
        return vals
    L = d[(d.log_family == "laplace")]
    rows = [
        ["cost form", "linear / quadratic / exponential"] + row(L[(L.block == "main") & (L.alpha == 0.3)], "cost_type", ["linear", "quadratic", "exponential"]) + ["shape moves scope by under one member"],
        ["cost level", "$\\times 0.5$ / 2 / 5 / 10 / 20"] + row(L[(L.block.isin(["cost-level", "cost-level-high"])) & (L.cost_type == "power_law") & (L.alpha == 0.3)], "cost_multiplier", [0.5, 2.0, 5.0, 10.0, 20.0]) + ["about 0.6 members per doubling; $K^*$ never reached"],
        ["window $l$", "20 / 100 / 200 / 500"] + row(L[(L.block == "lookback") & (L.alpha == 0.3)], "lookback", [20, 100, 200, 500]) + ["0.3--0.5 members per grid step, no saturation"],
        ["search $p$", "0.2 / 1"] + row(L[(L.block == "search") & (L.alpha == 0.3)], "merge_thresh", [0.2, 1.0]) + ["50\\% of $K^*$ at $p = 1$"],
        ["correlation", "cross-market 0.3"] + row(L[(L.block == "correlation") & (L.alpha == 0.3)], "cross_corr", [0.3]) + ["within 0.2 of baseline"],
        ["floor $c$", "0.10 / 0.35"] + row(L[(L.block == "floor-level") & (L.alpha == 0.3)], "floor_c", [0.10, 0.35]) + ["$c$ sets the tail; scope $+3\\%$ / $-19\\%$"],
        ["economy size", "$(M,N)$ = (25,25) / (100,25) / (25,100) / (100,100)"] + row(L[(L.block == "size") & (L.alpha == 0.3)], ["M", "N"], [(25, 25), (100, 25), (25, 100), (100, 100)]) + ["firms per market matter, markets barely"],
        ["sharing rule", "equal split (normal, four cost forms)"] + row(d[(d.block == "equal-split") & (d.alpha == 0.3)], "cost_type", ["linear", "quadratic", "exponential", "power_law"]) + ["the contract decides existence"],
        ["sharing rate", "endogenous $\\alpha$ (three families)"] + row(d[(d.block == "endogenous-alpha") & (d.cost_type == "power_law")], "log_family", FAM) + ["adopted $\\alpha \\approx 0.9$"],
        ["", "fixed $\\alpha = 0.9$ (three families)"] + row(d[d.block == "alpha-0.9"], "log_family", FAM) + ["reproduces the endogenous block"],
        ["tail weight $\\nu$", "Student-$t$, $\\nu$ = 2.5 / 3 / 4"] + nu_row(d, cols) + ["attrition, not consent, takes the scope"],
        ["market heterogeneity", "every market at $q = 0.20$ (three families)"] + row(d[(d.block == "homogeneous") & (d.alpha == 0.3) & (d.decision_rule == "moment")], "log_family", FAM) + ["transfer conflict 0--4\\% of the gap"],
        ["decision rule", "loggain (three families)"] + row(c[(c.block == "main") & (c.cost_type == "power_law") & (c.alpha == 0.3) & (c.decision_rule == "loggain")], "log_family", FAM) + ["verifiability 3--20\\% of the gap"],
    ]
    base = med(L[(L.block == "main") & (L.cost_type == "power_law") & (L.alpha == 0.3)], "log_family", cols).loc["laplace"]
    note = (f"At $\\alpha = 0.3$, Laplace unless stated, power-law cost unless stated, moment rule; medians over five replications. Baseline: $K = {base.K_mean:.1f}$, "
            f"pooled {base.cong_capital_share_median:.2f}, Hill {base.hill_exponent_median:.2f}, {base.floor_exits_per_period:.0f} floor exits per period. "
            "Three-family rows are normal / Laplace / $t_3$; equal-split cells are linear / quadratic / exponential / power law.")
    open(os.path.join(out, "table6_robustness.tex"), "w").write(
        table(rows, ["Block", "What varies", "$K$", "Pooled", "Hill", "Floor exits", "Finding"], "Robustness summary", "tab:robustness",
              colspec="llrrrrl", note=note))
    # 6b
    rows = []
    for fam, mults in [("laplace", [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]), ("t3", [1.0, 5.0, 10.0, 20.0])]:
        x = d[(d.log_family == fam) & (d.cost_type == "power_law") & (d.alpha == 0.3) & (d.block.isin(["main", "cost-level", "cost-level-high"]))]
        xc = c[(c.log_family == fam) & (c.cost_type == "power_law") & (c.alpha == 0.3) & (c.block.isin(["main", "cost-level", "cost-level-high"])) & (c.decision_rule == "loggain")]
        g = med(x, "cost_multiplier", ["K_mean", "cong_capital_share_median", "acceptance_rate", "hill_exponent_median"]); gc = med(xc, "cost_multiplier", ["K_mean"])
        for i, mlt in enumerate(mults):
            if mlt not in g.index:
                continue
            ks = kstar(b, fam, 0.3, mult=mlt)
            r = g.loc[mlt]
            rows.append([FAMTEX[fam] if i == 0 else "", f"{mlt:g}", f(r.K_mean), f(gc.loc[mlt].K_mean), str(ks) if ks else "--", f(r.K_mean / ks, 2) if ks else "--",
                         f(r.cong_capital_share_median, 2), f(r.acceptance_rate, 2), f(r.hill_exponent_median, 2)])
        if fam == "laplace":
            rows.append("midrule")
    open(os.path.join(out, "table6b_cost_level.tex"), "w").write(
        table(rows, ["Family", "$\\times$ cost level", "$K$", "$K$, loggain", "$K^*$", "$K/K^*$", "Pooled", "Accept.", "Hill"],
              "Cost level, power-law form, $\\alpha = 0.3$; medians over five replications", "tab:costlevel"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--out", default=None)
    ap.add_argument("--table3c-accept", action="store_true", help="C42: also write table3c_decomposition_accept.tex")
    ap.add_argument("--table3c-legacy", action="store_true", help="also write the pre-C42 three-part Table 3c")
    a = ap.parse_args()
    out = a.out or os.path.join(a.root, "latex", "tables")
    os.makedirs(out, exist_ok=True)
    d, c, b, sc = load(a.root)
    table1(sc, out); table2(sc, out); table3(d, c, b, out); table3b(d, c, out); table3c(a.root, out)
    table4(a.root, out); table5(d, a.root, out); table6(d, c, b, out)
    if a.table3c_accept:
        table3c_accept(a.root, out)
    if a.table3c_legacy:
        table3c_legacy(a.root, out)
    print("written to", out)


if __name__ == "__main__":
    main()
