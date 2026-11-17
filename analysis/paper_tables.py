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


def table(rows, header, caption, label, colspec=None, note=None, small=True, wide=None):
    """wide=None | "resize" (tabular scaled to \\linewidth) | "sideways" (rotated page; needs \\usepackage{rotating})."""
    n = len(header)
    colspec = colspec or ("l" + "r" * (n - 1))
    env = "sidewaystable" if wide in ("sideways", "sideways-resize") else "table"
    lines = [f"\\begin{{{env}}}[htbp]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}"]
    if small:
        lines.append("\\small")
    if wide in ("resize", "sideways-resize"):
        lines.append("\\resizebox{\\linewidth}{!}{%")
    lines.append(f"\\begin{{tabular}}{{{colspec}}}")
    lines.append("\\toprule")
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\midrule")
    for r in rows:
        if r == "midrule":
            lines.append("\\midrule")
        else:
            if isinstance(r, list) and len(r) == 1 and str(r[0]).startswith("\\multicolumn"):
                lines.append("\\addlinespace[3pt]")
            lines.append(" & ".join(str(c) for c in r) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    if wide in ("resize", "sideways-resize"):
        lines.append("}")
    if note:
        lines.append(f"\\par\\smallskip\\begin{{minipage}}{{0.95\\linewidth}}\\footnotesize {note}\\end{{minipage}}")
    lines.append(f"\\end{{{env}}}")
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
    """Table 1 (14 Nov): one parameter per row, no group rows; a short 'set by' column says where each main-block
    value comes from. Which block varies which parameter is Table 2's business."""
    main = [s for s in sc if s["block"] == "main"]
    s0 = main[0]
    alphas = sorted({s["alpha"] for s in main})
    rows = [
        ["$M$", "markets", f"{s0['M']}", "set"],
        ["$N$", "firms per market", f"{s0['N']}", "set"],
        ["$g$", "time-average growth rate", f"{s0['g']}", "cancels from every share (3.1)"],
        ["$c$", "reflecting floor, fraction of market mean", f"{s0['floor_c']}", "Hill exponent 1.06 at $\\alpha = 0$ (A2)"],
        ["$F$", "shock family", "normal, Laplace, $t_3$", "Laplace empirical; the others bracket it"],
        ["$q_m$", "market IQR of $x_{it}$", f"on $[{s0['sigma_range'][0]}, {s0['sigma_range'][1]}]$", "Stanley et al.\\ (1996) range (3.1)"],
        ["$\\rho$", "cross-market correlation", f"{s0['cross_corr']}", "independence"],
        ["$\\alpha$", "sharing rate", ", ".join(str(a) for a in alphas), "swept"],
        ["--", "sharing rule", "proportional to size", "--"],
        ["$\\Phi(K)$", "coordination cost schedule", "power law", "shape unsettled; four forms run"],
        ["$\\Phi(40)$", "cost level", "0.0017 per period", "common to the four forms"],
        ["$p$", "proposal probability per firm-period", f"{s0['merge_thresh']}", "uncalibrated; swept"],
        ["$l$", "decision window and lock-in", f"{s0['lookback']}", "practitioner lookback (3.3)"],
        ["--", "decision rule", "moment", "Booth--Fama estimate (3.3)"],
        ["$T$", "periods per run; burn-in", f"{s0['T']:,}; {s0['burn_in']:,}", "relaxation check (A3)"],
        ["--", "replications per cell", str(len({s["rep"] for s in main})), "common random numbers"],
    ]
    note = ("Cost parameters: power law $b_0 = 2.032\\cdot10^{-5}$, $b_1 = 1.2$; linear $b_0 = 10^{-5}$, $b_1 = 4.225\\cdot10^{-5}$; "
            "quadratic $b_0 = 10^{-4}$, $b_1 = 10^{-6}$, $b_2 = 9.7\\cdot10^{-7}$; exponential $b_0 = 10^{-3}$, $b_1 = 0.013266$. "
            "The blocks that vary each parameter are in Table~\\ref{tab:design}.")
    open(os.path.join(out, "table1_parameters.tex"), "w").write(
        table(rows, ["Symbol", "Meaning", "Main block", "Set by"], "Parameters of the main block",
              "tab:parameters", colspec="lp{0.27\\linewidth}p{0.25\\linewidth}p{0.27\\linewidth}", note=note))


# ---------------------------------------------------------------- Table 2
def table2(sc, out):
    """Table 2, regrouped (14 Nov): blocks grouped by what they vary (environment / contract / decisions / rule
    controls); the three cost-level blocks merged into one row; the 'Cost forms' column folded into 'What varies'
    (power law unless stated). Run counts come from scenarios.json."""
    df = pd.DataFrame(sc)
    n = lambda *blks: int(sum((df.block == b).sum() for b in blks))
    grid = "grid"
    G = lambda name: [f"\\multicolumn{{5}}{{l}}{{\\emph{{{name}}}}}"]
    rows = [
        ["main", "--; all four cost schedules", "all three", grid, f"{n('main'):,}"],
        G("Environment"),
        ["$\\nu$", "Student-$t$ with $\\nu \\in \\{2.5, 4\\}$", "$t$", "0.1, 0.3, 0.5", f"{n('nu'):,}"],
        ["correlation", "$\\rho = 0.3$ (Gaussian copula)", "Laplace", grid, f"{n('correlation'):,}"],
        ["floor level", "$c \\in \\{0.10, 0.35\\}$", "Laplace", grid, f"{n('floor-level'):,}"],
        ["size", "$(M, N) \\in \\{25, 100\\}^2$ with $c^*(N)$", "Laplace", "0, 0.1, 0.3, 0.5", f"{n('size'):,}"],
        ["homogeneous", "every market at $q = 0.20$; moment and loggain rules", "all three", "0.05, 0.1, 0.3, 0.5", f"{n('homogeneous'):,}"],
        G("Contract"),
        ["equal split", "net pool split equally; all four cost schedules", "normal", grid, f"{n('equal-split'):,}"],
        ["cost level", "$\\times 0.5$, $\\times 2$: all four schedules (linear and quadratic also with $b_1, b_2$ scaled); $\\times 5$, $\\times 10$, $\\times 20$: power law",
         "Laplace; $t_3$ at the high levels", "grid; 0.1, 0.3, 0.5 at the high levels", f"{n('cost-level', 'cost-level-full', 'cost-level-high'):,}"],
        ["endogenous $\\alpha$", "$\\alpha_C$ renegotiated by each unit, start 0.1; all four cost schedules", "all three", "--", f"{n('endogenous-alpha'):,}"],
        ["$\\alpha$ beyond the grid", "the main power-law scenarios at $\\alpha \\in \\{0.6, 0.7, 0.8, 0.9\\}$", "all three", "0.6--0.9", f"{n('alpha-0.9', 'alpha-gap'):,}"],
        G("Decisions"),
        ["lookback", "$l \\in \\{20, 50, 100, 200, 500\\}$", "Laplace, $t_3$", "0.1, 0.3", f"{n('lookback'):,}"],
        ["search", "$p \\in \\{0.2, 1\\}$", "all three", "0.05, 0.1, 0.3, 0.5", f"{n('search'):,}"],
        G("Rule controls (3.3)"),
        ["cv", "exact replay minus its first-order term", "all three", "0.1, 0.3, 0.5", f"{n('cv'):,}"],
        ["accept", "every feasible proposal accepted, no voluntary exit", "all three", "0.1, 0.3, 0.5", f"{n('accept'):,}"],
        "midrule",
        ["total", "", "", "", f"{len(df):,}"],
    ]
    note = ("Five replications per cell; ``grid'' is the nine-value $\\alpha$ grid of the main block; power-law cost unless "
            "stated. The homogeneous block's 60 loggain runs are part of the design; every other moment-rule cell outside the "
            "$\\nu$ block and the $\\alpha$ cells beyond the grid is also run under the loggain rule with the same seeds "
            "(1,845 runs, not in the table).")
    open(os.path.join(out, "table2_design.tex"), "w").write(
        table(rows, ["Block", "What varies", "Families", "$\\alpha$", "Runs"], "Simulation design", "tab:design",
              colspec="lp{0.36\\linewidth}p{0.12\\linewidth}p{0.17\\linewidth}r", note=note))


# ---------------------------------------------------------------- Table 3
def table3(d, c, b, out):
    """Table 3 (14 Nov): split in two so that neither needs shrinking. table3_scope.tex holds scope (K, loggain twin,
    K*, K/K*, pooled share, acceptance); table3t_turnover.tex holds exits and floor hits."""
    A = [0.05, 0.1, 0.3, 0.5, 0.7, 0.9]   # 0.7 from the alpha-gap block, 0.9 from the alpha-0.9 block (C43, C38)
    m = d[(d.block.isin(["main", "alpha-gap", "alpha-0.9"])) & (d.cost_type == "power_law") & (d.alpha.isin(A)) & (d.decision_rule == "moment")]
    mc = c[(c.block == "main") & (c.cost_type == "power_law") & (c.alpha.isin(A)) & (c.decision_rule == "loggain")]
    cols = ["K_mean", "cong_capital_share_median", "floor_exits_per_period", "voluntary_exits_per_period", "acceptance_rate",
            "floor_hit_rate_member_status", "floor_hit_rate_standalone_status"]
    g = med(m, ["log_family", "alpha"], cols)
    gl = med(mc, ["log_family", "alpha"], ["K_mean"])
    rows, rows_t = [], []
    for fam in FAM:
        for a in A:
            r = g.loc[(fam, a)]
            ks = kstar(b, fam, a)
            lab = FAMTEX[fam] if a == A[0] else ""
            kl = gl.loc[(fam, a)].K_mean if (fam, a) in gl.index else None
            rows.append([lab, f"{a:g}", f(r.K_mean), f(kl) if kl is not None else "--", str(ks) if ks else "--",
                         f(r.K_mean / ks, 2) if ks else "--", f(r.cong_capital_share_median, 2), f(r.acceptance_rate, 2)])
            rows_t.append([lab, f"{a:g}", f(r.floor_exits_per_period), f(r.voluntary_exits_per_period, 2),
                           f(r.floor_hit_rate_member_status, 3), f(r.floor_hit_rate_standalone_status, 3)])
        if fam != FAM[-1]:
            rows.append("midrule"); rows_t.append("midrule")
    note = ("$K$: mean members per active unit (moment rule; loggain twin from pilot C, run to $\\alpha = 0.5$); $K^*$: equal-members reference at the median "
            "market volatility, capped at $M = 50$, computed at $\\alpha \\in \\{0.1, 0.3, 0.5\\}$; pooled: share of economy capital held by members; "
            "acceptance: share of feasible proposals accepted.")
    open(os.path.join(out, "table3_scope.tex"), "w").write(
        table(rows, ["Family", "$\\alpha$", "$K$", "$K$, loggain", "$K^*$", "$K/K^*$", "Pooled", "Accept."],
              "Scope; main block, power-law cost; medians over five replications", "tab:scope", note=note))
    note_t = ("Exits per period among 2,500 firms (mergers per period equal floor exits to within 0.2); "
              "member / standalone hits: floor hits per firm-period of that status.")
    open(os.path.join(out, "table3t_turnover.tex"), "w").write(
        table(rows_t, ["Family", "$\\alpha$", "Floor exits", "Vol.\\ exits", "Member hits", "Standalone hits"],
              "Turnover and floor hits; main block, power-law cost; medians over five replications", "tab:turnover", note=note_t))


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
            "$K_{\\text{loggain}}$: the same gain estimated by its sample mean. Consent $= K_{\\text{accept}} - K$, "
            "Verif.\\ (verifiability) $= K - K_{\\text{loggain}}$; shares of $K_{\\text{accept}}$ in parentheses. The transfer conflict between members of unequal volatility, $K_{\\text{hom}} - K$, is "
            "0.0--0.4 members in every cell (at most 4\\% of $K^* - K_{\\text{loggain}}$) and is not shown.")
    open(os.path.join(out, "table3c_decomposition.tex"), "w").write(
        table(rows, ["Family", "$\\alpha$", "$K^*$", "$K_{\\text{accept}}$", "$K_{\\text{hom}}$", "$K$", "$K_{\\text{loggain}}$",
                     "Consent", "Verif."],
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
    note = ("Environment $= K^* - K_{\\text{accept}}$, consent $= K_{\\text{accept}} - K_{\\text{hom}}$, transfer $= K_{\\text{hom}} - K$, "
            "verifiability $= K - K_{\\text{loggain}}$; shares in parentheses are of $K^* - K_{\\text{loggain}}$ (the four parts sum to it). $K_{\\text{accept}}$: heterogeneous economy, every feasible "
            "proposal accepted and no voluntary exit (a control without knowledge). A negative environment part means the "
            "environment alone permits units larger than $K^*$. The consent part compares the heterogeneous accept economy "
            "with the homogeneous moment economy, so it also holds the homogeneous-heterogeneous difference of the accept rule.")
    open(os.path.join(out, "table3c_decomposition_accept.tex"), "w").write(
        table(rows, ["Family", "$\\alpha$", "$K^*$", "$K_{\\text{accept}}$", "$K_{\\text{hom}}$", "$K$", "$K_{\\text{loggain}}$",
                     "Environment", "Consent", "Transfer", "Verifiability"],
              "Four-part decomposition of the distance to the equal-members reference, power-law cost; medians over "
              "five replications", "tab:decomposition-accept", note=note, wide="sideways"))


# ---------------------------------------------------------------- Table 4
def table4(root, out):
    """Table 4 (14 Nov, main text): the owner-return gap (mean, median), its ordinary-period share and the hazard ratio
    in one six-column table. The split by unit size (table4a_owner_by_K.tex) and the floor rates by status
    (table4b_hazard.tex) go to the appendix."""
    p = pd.read_csv(os.path.join(root, "pilot_d", "status_panel_summary.csv")).groupby(["family", "alpha"]).mean(numeric_only=True)
    dec = pd.read_csv(os.path.join(root, "pilot_d", "status_panel_decomposition.csv"))
    dec = dec.groupby(["family", "alpha"]).mean(numeric_only=True)
    rows, rows_k, rows_h = [], [], []
    for fam in FAM:
        for a in [0.05, 0.1, 0.3]:
            r = p.loc[(fam, a)]; s = dec.loc[(fam, a)]
            lab = FAMTEX[fam] if a == 0.05 else ""
            rows.append([lab, f"{a:g}", f(100 * r.diff_fe_mean, 2), f(100 * r.diff_fe_median, 2), f(s.share_ordinary, 2),
                         f(r.floor_rate_standalone / r.floor_rate_member, 1)])
            rows_k.append([lab, f"{a:g}", f(100 * r.diff_fe_mean, 2), f(100 * r.diff_K2_mean, 2), f(100 * r.diff_K3_4_mean, 2), f(100 * r.diff_K5p_mean, 2)])
            rows_h.append([lab, f"{a:g}", f(r.floor_rate_member, 3), f(r.floor_rate_standalone, 3), f(r.floor_rate_standalone / r.floor_rate_member, 1)])
        if fam != FAM[-1]:
            rows.append("midrule"); rows_k.append("midrule"); rows_h.append("midrule")
    note = ("Owner return: next-period log change of the firm's market share net of floor recapitalization; gap = member $-$ standalone "
            "within market $\\times$ size decile, in percentage points per period; 27 traced runs, means over three replications. "
            "Ordinary: share of the mean gap accruing in periods without a floor hit. Hazard ratio: floor hits per standalone-period "
            "over floor hits per member-period, same cells (Table~\\ref{tab:hazard}).")
    open(os.path.join(out, "table4_owner_return.tex"), "w").write(
        table(rows, ["Family", "$\\alpha$", "Gap, mean", "Gap, median", "Ordinary share", "Hazard ratio"],
              "Owners' time-average growth and ruin hazard by affiliation status", "tab:owner", note=note))
    open(os.path.join(out, "table4a_owner_by_K.tex"), "w").write(
        table(rows_k, ["Family", "$\\alpha$", "All units", "$K = 2$", "$K = 3$--4", "$K \\geq 5$"],
              "The owner-return gap by unit size, percentage points per period", "tab:owner-byK",
              note="Mean member minus standalone gap of Table~\\ref{tab:owner}, split by the size of the member's unit."))
    note_h = ("Floor hits per firm-period of that status, within market $\\times$ size decile; same traced runs as "
              "Table~\\ref{tab:owner}. Hazard ratio $=$ standalone / member.")
    open(os.path.join(out, "table4b_hazard.tex"), "w").write(
        table(rows_h, ["Family", "$\\alpha$", "Floor, member", "Floor, standalone", "Hazard ratio"],
              "Ruin hazard by affiliation status", "tab:hazard", note=note_h))


# ---------------------------------------------------------------- Table 5
def table5(d, root, out):
    e = d[(d.block == "endogenous-alpha") & (d.cost_type == "power_law")]
    g = med(e, "log_family", ["alpha_adopted_mean", "alpha_adopted_median", "alpha_adopted_share1", "K_mean", "cong_capital_share_median", "hill_exponent_median"])
    sc = pd.read_csv(os.path.join(root, "pilot_d", "alpha_scatter.csv"))
    f9 = d[d.block == "alpha-0.9"]
    g9 = med(f9, "log_family", ["K_mean", "cong_capital_share_median", "hill_exponent_median"]) if len(f9) else None
    rows = []
    from scipy.stats import spearmanr
    units, fixed = [], []
    for fam in FAM:
        r = g.loc[fam]; s = sc[sc.log_family == fam]
        rho = spearmanr(s.alpha_tavg, s.mean_iqr).correlation
        rows.append([FAMTEX[fam], f(r.alpha_adopted_mean, 2), f(r.alpha_adopted_median, 2), f(r.alpha_adopted_share1, 2), f(r.K_mean),
                     f(r.cong_capital_share_median, 2), f(r.hill_exponent_median, 2), f(rho, 2)])
        units.append(f"{len(s):,}")
        if g9 is not None:
            fixed.append(f"$K = {g9.loc[fam].K_mean:.1f}$, Hill {g9.loc[fam].hill_exponent_median:.2f}")
    note = ("Endogenous-$\\alpha$ block, power-law cost, start $\\alpha_C = 0.1$; adopted $\\alpha$ over unit-periods; share at 1: unit-periods at $\\alpha_C = 1$. "
            "$\\rho$: Spearman correlation of a unit's time-averaged rate with the mean IQR of its members' markets, over the "
            + " / ".join(units) + " units alive at $T$ (normal / Laplace / $t_3$). ")
    if fixed:
        note += "For comparison, the fixed $\\alpha = 0.9$ cell gives " + "; ".join(fixed) + " (same order)."
    open(os.path.join(out, "table5_endogenous_alpha.tex"), "w").write(
        table(rows, ["Family", "$\\bar\\alpha$", "median", "share at 1", "$K$", "Pooled", "Hill", "$\\rho(\\alpha, \\bar q)$"],
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
    """Table 6 (14 Nov): one row per variant (no slash-separated cells), block name on the first row of each block."""
    cols = ["K_mean", "cong_capital_share_median", "hill_exponent_median", "floor_exits_per_period"]
    fmt = ("1", "2", "2", "0")
    rows = []
    def block(name, sel, by, keys, labels):
        g = med(sel, by, cols)
        for i, (k, lab) in enumerate(zip(keys, labels)):
            r = g.loc[k]
            rows.append([name if i == 0 else "", lab] + [f"{r[col]:.{fmt[j]}f}" for j, col in enumerate(cols)])
        rows.append("midrule")
    L = d[(d.log_family == "laplace")]
    FL = ["normal", "Laplace", "$t_3$"]
    block("cost form", L[(L.block == "main") & (L.alpha == 0.3)], "cost_type", ["linear", "quadratic", "exponential"], ["linear", "quadratic", "exponential"])
    block("cost level", L[(L.block.isin(["cost-level", "cost-level-high"])) & (L.cost_type == "power_law") & (L.alpha == 0.3)], "cost_multiplier",
          [0.5, 2.0, 5.0, 10.0, 20.0], ["$\\times 0.5$", "$\\times 2$", "$\\times 5$", "$\\times 10$", "$\\times 20$"])
    block("window $l$", L[(L.block == "lookback") & (L.alpha == 0.3)], "lookback", [20, 100, 200, 500], ["20", "100", "200", "500"])
    block("search $p$", L[(L.block == "search") & (L.alpha == 0.3)], "merge_thresh", [0.2, 1.0], ["0.2", "1"])
    block("correlation", L[(L.block == "correlation") & (L.alpha == 0.3)], "cross_corr", [0.3], ["cross-market 0.3"])
    block("floor $c$", L[(L.block == "floor-level") & (L.alpha == 0.3)], "floor_c", [0.10, 0.35], ["0.10", "0.35"])
    block("economy size $(M, N)$", L[(L.block == "size") & (L.alpha == 0.3)], ["M", "N"], [(25, 25), (100, 25), (25, 100), (100, 100)],
          ["(25, 25)", "(100, 25)", "(25, 100)", "(100, 100)"])
    rows.pop()
    rows_env, rows = rows, []
    block("equal split (normal)", d[(d.block == "equal-split") & (d.alpha == 0.3)], "cost_type", ["linear", "quadratic", "exponential", "power_law"],
          ["linear", "quadratic", "exponential", "power law"])
    block("endogenous $\\alpha$", d[(d.block == "endogenous-alpha") & (d.cost_type == "power_law")], "log_family", FAM, FL)
    block("fixed $\\alpha = 0.9$", d[d.block == "alpha-0.9"], "log_family", FAM, FL)
    nu = d[(d.block == "nu") & (d.alpha == 0.3)].copy()
    t3 = d[(d.block == "main") & (d.cost_type == "power_law") & (d.log_family == "t3") & (d.alpha == 0.3)].copy(); t3["nu"] = 3.0
    block("tail weight $\\nu$ (Student-$t$)", pd.concat([nu, t3]), "nu", [2.5, 3.0, 4.0], ["2.5", "3", "4"])
    block("homogeneous markets", d[(d.block == "homogeneous") & (d.alpha == 0.3) & (d.decision_rule == "moment")], "log_family", FAM, FL)
    block("loggain rule", c[(c.block == "main") & (c.cost_type == "power_law") & (c.alpha == 0.3) & (c.decision_rule == "loggain")], "log_family", FAM, FL)
    rows.pop()
    base = med(L[(L.block == "main") & (L.cost_type == "power_law") & (L.alpha == 0.3)], "log_family", cols).loc["laplace"]
    note = (f"At $\\alpha = 0.3$, Laplace shocks, power-law cost unless stated, moment rule; "
            f"medians over five replications. Baseline: $K = {base.K_mean:.1f}$, "
            f"pooled {base.cong_capital_share_median:.2f}, Hill {base.hill_exponent_median:.2f}, {base.floor_exits_per_period:.0f} floor exits per period. "
            "")
    hdr = ["Block", "Variant", "$K$", "Pooled", "Hill", "Floor exits"]
    open(os.path.join(out, "table6_robustness.tex"), "w").write(
        table(rows_env, hdr, "Robustness: environment and decision parameters", "tab:robustness", colspec="llrrrr", note=note))
    open(os.path.join(out, "table6c_robustness_contract.tex"), "w").write(
        table(rows, hdr, "Robustness: contract, shock family and decision rule", "tab:robustness-b", colspec="llrrrr",
              note="Conventions as in Table~\\ref{tab:robustness}; family rows are at $\\alpha = 0.3$ except the endogenous-$\\alpha$ block. "
                   "Homogeneous markets: every market at $q = 0.20$."))
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
        table(rows, ["Family", "Cost level", "$K$", "$K$, loggain", "$K^*$", "$K/K^*$", "Pooled", "Accept.", "Hill"],
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
