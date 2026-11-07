#!/usr/bin/env python3
"""
C36: outputs of the moment block (decision_rule='moment') against its loggain counterparts.

    python analytics/moment_c36.py --acceptance          # pilot_c/acceptance_by_K_moment.csv (+ _pooled)
    python analytics/moment_c36.py --variance <trace_dir> # sigma-hat^2 / sigma^2 over 50-period windows
    python analytics/moment_c36.py --tables               # markdown tables -> pilot_c/moment_tables_c36.md

--acceptance reads the moment pickles in pilot_c/results (alpha = 0.3, fixed alpha; p 0.05 and 1, and the
cost-level runs at x5 and x20) and writes the C35 columns (analytics/acceptance_c35.rows_for).
--variance writes pilot_c/variance_ratio_c36.csv: for each family, the distribution of the window variance
over its population value, (a) for 200,000 windows of 50 i.i.d. draws of the model's log shock at the median
IQR 0.2 (exact sigma^2: normal s^2, laplace 2 b^2, t3 3 s^2), and (b) from one trace per family (alpha = 0.3,
rep 0): non-overlapping 50-period stretches in which a firm is standalone and never at the floor, its
period log-share change (its relative return, up to the small common change of its market's total from
pooling and floor injections) against sigma^2 of its market (market IQR from the C34 result pickle).
--tables writes the tables of diagnostics/c36_note.md from tidy.csv, benchmarks.csv, the acceptance files
and the alpha-scatter files.
"""
import gzip
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'analytics'))
from acceptance_c35 import rows_for, hist_quantiles  # noqa: E402

FAMS = ['normal', 'laplace', 't3']


def load(i, d='pilot_c/results'):
    with gzip.open(os.path.join(ROOT, d, f'scenario_{i:04d}.pkl.gz')) as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------------------------------------
def acceptance():
    sc = json.load(open(os.path.join(ROOT, 'pilot_c', 'scenarios.json')))
    rows, hists, edges = [], {}, None
    for s in sc:
        if s['block'] != 'moment' or s['alpha'] != 0.3 or s.get('alpha_endogenous'):
            continue
        res = load(s['scenario_id'])
        r, H, edges = rows_for(res)
        rows += r
        key = (s['log_family'], s['merge_thresh'], s.get('cost_multiplier', 1.0))
        hists[key] = hists.get(key, 0) + H
    df = pd.DataFrame(rows).sort_values(['family', 'p', 'cost_multiplier', 'rep', 'K'])
    df.to_csv(os.path.join(ROOT, 'pilot_c', 'acceptance_by_K_moment.csv'), index=False)
    keys = ['block', 'family', 'alpha', 'p', 'cost_multiplier', 'K']
    pooled = df.groupby(keys).agg(proposals=('proposals', 'sum'), mergers=('mergers', 'sum'), n_runs=('rep', 'size'),
                                  sigma_median=('sigma_median', 'mean'),
                                  marginal_gain_next=('marginal_gain_next', 'mean'),
                                  marginal_cost_next=('marginal_cost_next', 'first'),
                                  marginal_gain_into_K=('marginal_gain_into_K', 'mean'),
                                  marginal_cost_into_K=('marginal_cost_into_K', 'first')).reset_index()
    pooled['acceptance'] = pooled.mergers / pooled.proposals.where(pooled.proposals > 0)
    q = []
    for _, r in pooled.iterrows():
        H = hists[(r.family, r.p, r.cost_multiplier)][int(r.K)]
        q25, q50, q75 = hist_quantiles(H, edges)
        n = H.sum()
        q.append(dict(binding_median=q50, binding_p25=q25, binding_p75=q75,
                      binding_share_positive=H[int(np.argmin(np.abs(edges))):].sum() / n if n else np.nan,
                      binding_share_clipped=(H[0] + H[-1]) / n if n else np.nan))
    pooled = pd.concat([pooled, pd.DataFrame(q)], axis=1)
    pooled.to_csv(os.path.join(ROOT, 'pilot_c', 'acceptance_by_K_moment_pooled.csv'), index=False)
    print(f'{len(df)} rows ({df.scenario_id.nunique()} runs), pooled {len(pooled)} rows')


# ---------------------------------------------------------------------------------------------------------
def pop_var(family, iqr):
    if family == 'normal':
        return (iqr / (2 * stats.norm.ppf(0.75))) ** 2
    if family == 'laplace':
        return 2 * (iqr / (2 * np.log(2))) ** 2
    return 3 * (iqr / (2 * stats.t.ppf(0.75, df=3))) ** 2


def draw(family, iqr, size, rng):
    if family == 'normal':
        return rng.normal(0, iqr / (2 * stats.norm.ppf(0.75)), size)
    if family == 'laplace':
        return rng.laplace(0, iqr / (2 * np.log(2)), size)
    return iqr / (2 * stats.t.ppf(0.75, df=3)) * rng.standard_t(3, size)


QS = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]


def variance(trace_dir, l=50):
    rng = np.random.default_rng(36)
    sc = json.load(open(os.path.join(ROOT, 'pilot_c', 'scenarios.json')))
    rows = []
    for fam in FAMS:
        x = draw(fam, 0.2, (200_000, l), rng)
        ratio = x.var(axis=1, ddof=1) / pop_var(fam, 0.2)
        rows.append(dict(family=fam, source='iid draws, IQR 0.2', n_windows=len(ratio), mean=ratio.mean(),
                         **{f'q{int(q * 100):02d}': np.quantile(ratio, q) for q in QS},
                         share_below_half=(ratio < 0.5).mean(), share_above_2=(ratio > 2).mean()))
        s = [c for c in sc if c['block'] == 'main' and c['cost_type'] == 'power_law' and c['log_family'] == fam
             and c['alpha'] == 0.3 and c['rep'] == 0][0]
        z = np.load(os.path.join(trace_dir, f'{fam}_a0.3_r0.npz'))
        res = load(s['scenario_id'])
        var_m = pop_var(fam, res['market_iqr'])
        ls, unit, floor, home, jump = z['logshare'].astype(np.float64), z['cong'], z['floor'], z['home'], z['jump']
        b = s['burn_in']
        inc = (ls[1:] - ls[:-1]) - jump[1:]                 # (T, F)
        ok = (unit[:-1] < 0) & (~floor.T)                   # standalone at the start, no floor hit in the period
        out = []
        T, F = inc.shape
        for f in range(F):
            t = b
            while t + l <= T:
                if ok[t:t + l, f].all():
                    out.append(inc[t:t + l, f].var(ddof=1) / var_m[home[f]])
                    t += l
                else:
                    t += int(np.nonzero(~ok[t:t + l, f])[0][-1]) + 1   # restart after the last bad period
        ratio = np.array(out)
        rows.append(dict(family=fam, source=f'trace {fam}_a0.3_r0, standalone no-floor windows', n_windows=len(ratio),
                         mean=ratio.mean(), **{f'q{int(q * 100):02d}': np.quantile(ratio, q) for q in QS},
                         share_below_half=(ratio < 0.5).mean(), share_above_2=(ratio > 2).mean()))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ROOT, 'pilot_c', 'variance_ratio_c36.csv'), index=False)
    print(df.to_string())


# ---------------------------------------------------------------------------------------------------------
def kstar_lookup():
    bm = pd.read_csv(os.path.join(ROOT, 'analytics', 'benchmarks.csv'))
    bm = bm[(bm.cost_type == 'power_law') & (bm.M == 50)]
    return {(r.family, float(r.cost_multiplier), float(r.alpha)): int(r.K_star) for r in bm.itertuples()}


COLS = ['K_mean', 'cong_capital_share_median', 'hill_exponent_median', 'mergers_per_period',
        'floor_exits_per_period', 'voluntary_exits_per_period', 'acceptance_rate']


def f(x, fmt='{:.3f}'):
    return '—' if pd.isna(x) else fmt.format(x)


def triplet_table(tidy, ks, sel_mom, title, by=('log_family', 'alpha')):
    """Moment runs selected by sel_mom against their counterparts (counterpart_id), medians over reps."""
    sc = json.load(open(os.path.join(ROOT, 'pilot_c', 'scenarios.json')))
    cp = {s['scenario_id']: s['counterpart_id'] for s in sc if s['block'] == 'moment'}
    mom = tidy[(tidy.block == 'moment') & sel_mom(tidy)].copy()
    log = tidy.set_index('scenario_id').loc[[cp[i] for i in mom.scenario_id]].reset_index()
    by = list(by)
    gm = mom.groupby(by)[COLS].median()
    gl = log.groupby(by)[COLS].median()
    lines = [f'**{title}** (medians over 5 reps; moment / loggain)', '',
             '| ' + ' | '.join(by) + ' | K* | K mom | K log | K/K* mom | K/K* log | pooled share mom / log | Hill mom / log '
             '| mergers/period mom / log | floor exits mom / log | voluntary exits mom / log | acceptance mom / log |',
             '|' + '---|' * (len(by) + 11)]
    for key in gm.index:
        key = key if isinstance(key, tuple) else (key,)
        d = dict(zip(by, key))
        mult = float(d.get('cost_multiplier', 1.0))
        kst = ks.get((d['log_family'], mult, float(d['alpha'])), np.nan)
        m, lg = gm.loc[key if len(key) > 1 else key[0]], gl.loc[key if len(key) > 1 else key[0]]
        lines.append('| ' + ' | '.join(f'{v:g}' if isinstance(v, float) else str(v) for v in key) + ' | '
                     + ' | '.join([f(kst, '{:.0f}'), f(m.K_mean), f(lg.K_mean), f(m.K_mean / kst), f(lg.K_mean / kst),
                                   f'{f(m.cong_capital_share_median)} / {f(lg.cong_capital_share_median)}',
                                   f'{f(m.hill_exponent_median)} / {f(lg.hill_exponent_median)}',
                                   f'{f(m.mergers_per_period, "{:.2f}")} / {f(lg.mergers_per_period, "{:.2f}")}',
                                   f'{f(m.floor_exits_per_period, "{:.2f}")} / {f(lg.floor_exits_per_period, "{:.2f}")}',
                                   f'{f(m.voluntary_exits_per_period, "{:.2f}")} / {f(lg.voluntary_exits_per_period, "{:.2f}")}',
                                   f'{f(m.acceptance_rate, "{:.3f}")} / {f(lg.acceptance_rate, "{:.3f}")}']) + ' |')
    return lines


def acceptance_profile():
    log = pd.read_csv(os.path.join(ROOT, 'pilot_c', 'acceptance_by_K_pooled.csv'))
    mom = pd.read_csv(os.path.join(ROOT, 'pilot_c', 'acceptance_by_K_moment_pooled.csv'))
    Ks = list(range(2, 13)) + [15, 20, 25, 30]
    lines = []
    for p, blk in ((0.05, 'main'), (1.0, 'search')):
        lines += [f'**α = 0.3, p = {p:g}** (pooled over 5 reps; acceptance, and binding-gain median, moment / loggain)', '',
                  '| K | ' + ' | '.join(f'{fam} acc. mom / log | {fam} proposals mom / log | {fam} binding median mom / log'
                                       for fam in FAMS) + ' | ΔG into K (normal) | ΔΦ into K |',
                  '|' + '---|' * (2 + 3 * len(FAMS) + 1)]
        for K in Ks:
            cells, dg, dphi = [], np.nan, np.nan
            any_ = False
            for fam in FAMS:
                a = mom[(mom.family == fam) & (mom.p == p) & (mom.cost_multiplier == 1.0) & (mom.K == K)]
                b = log[(log.block == blk) & (log.family == fam) & (log.alpha == 0.3) & (log.p == p) & (log.K == K)]
                am = a.iloc[0] if len(a) else None
                bl = b.iloc[0] if len(b) else None
                any_ |= (am is not None and am.proposals > 0) or (bl is not None and bl.proposals > 0)
                acc = lambda r: f(r.acceptance) if r is not None and r.proposals > 0 else '—'  # noqa: E731
                pr = lambda r: str(int(r.proposals)) if r is not None else '0'  # noqa: E731
                bm = lambda r: f(r.binding_median, '{:.1e}') if r is not None and r.proposals > 0 else '—'  # noqa: E731
                cells += [f'{acc(am)} / {acc(bl)}', f'{pr(am)} / {pr(bl)}', f'{bm(am)} / {bm(bl)}']
                if fam == 'normal' and bl is not None:
                    dg, dphi = bl.marginal_gain_into_K, bl.marginal_cost_into_K
            if any_:
                lines.append(f'| {K} | ' + ' | '.join(cells) + f' | {f(dg, "{:.1e}")} | {f(dphi, "{:.1e}")} |')
        lines.append('')
    return lines


def alpha_table(tidy):
    lines = ['**Adopted α, endogenous-α runs, power law** (moment block vs the loggain endogenous-alpha cells; '
             'tidy: medians over 5 runs of the run-level adopted α; alpha_scatter: units alive at T)', '',
             '| family | α median (run medians) mom / log | α mean mom / log | units at T mom / log | α_tavg mean mom / log '
             '| share α_tavg ≥ 0.95 mom / log | Spearman ρ(α_tavg, mean IQR) mom / log | K mom / log |',
             '|---|---|---|---|---|---|---|---|']
    am = pd.read_csv(os.path.join(ROOT, 'pilot_c', 'alpha_scatter_moment.csv'))
    al = pd.read_csv(os.path.join(ROOT, 'pilot_c', 'alpha_scatter.csv'))
    al = al[al.cost_type == 'power_law']
    for fam in FAMS:
        tm = tidy[(tidy.block == 'moment') & (tidy.alpha_endogenous) & (tidy.log_family == fam)]
        tl = tidy[(tidy.block == 'endogenous-alpha') & (tidy.cost_type == 'power_law') & (tidy.log_family == fam)]
        sm, sl = am[am.log_family == fam].dropna(subset=['alpha_tavg']), al[al.log_family == fam].dropna(subset=['alpha_tavg'])
        rho = lambda d: stats.spearmanr(d.alpha_tavg, d.mean_iqr)[0] if len(d) > 2 else np.nan  # noqa: E731
        lines.append(f'| {fam} | {f(tm.alpha_adopted_median.median())} / {f(tl.alpha_adopted_median.median())} | '
                     f'{f(tm.alpha_adopted_mean.median())} / {f(tl.alpha_adopted_mean.median())} | {len(sm)} / {len(sl)} | '
                     f'{f(sm.alpha_tavg.mean())} / {f(sl.alpha_tavg.mean())} | '
                     f'{f((sm.alpha_tavg >= 0.95).mean())} / {f((sl.alpha_tavg >= 0.95).mean())} | '
                     f'{f(rho(sm))} / {f(rho(sl))} | {f(tm.K_mean.median())} / {f(tl.K_mean.median())} |')
    return lines


def tables():
    tidy = pd.read_csv(os.path.join(ROOT, 'pilot_c', 'tidy.csv'))
    ks = kstar_lookup()
    fixed = lambda t: (~t.alpha_endogenous.astype(bool))  # noqa: E731
    lines = ['## Main block (power law, p = 0.05)', '']
    lines += triplet_table(tidy, ks, lambda t: fixed(t) & (t.merge_thresh == 0.05) & (t.cost_multiplier == 1.0),
                           'Main, power law')
    lines += ['', '## Cost level (power law, c0 × 5 and × 20, p = 0.05)', '']
    lines += triplet_table(tidy, ks, lambda t: fixed(t) & (t.cost_multiplier > 1.0), 'Cost level',
                           by=('log_family', 'cost_multiplier', 'alpha'))
    lines += ['', '## Search p = 1', '']
    lines += triplet_table(tidy, ks, lambda t: fixed(t) & (t.merge_thresh == 1.0), 'Search p = 1')
    lines += ['', '## Acceptance by K, moment against loggain', '']
    lines += acceptance_profile()
    lines += ['## Adopted α', '']
    lines += alpha_table(tidy)
    out = os.path.join(ROOT, 'pilot_c', 'moment_tables_c36.md')
    with open(out, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    if '--acceptance' in sys.argv:
        acceptance()
    elif '--variance' in sys.argv:
        variance(sys.argv[sys.argv.index('--variance') + 1])
    elif '--tables' in sys.argv:
        tables()
