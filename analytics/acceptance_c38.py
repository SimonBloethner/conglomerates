#!/usr/bin/env python3
"""
C38 Task 2: consent profile by recruit size -- acceptance and binding gain by K x entrant capital share w.

    python analytics/acceptance_c38.py [<dir>]      # default pilot_d/results_c38
    python analytics/acceptance_c38.py --report     # markdown tables from pilot_d/acceptance_by_K_w.csv

Reads the C38 result pickles (with `proposals_by_K_w`, see merger_kernel_numba) and writes
- pilot_d/acceptance_by_K_w.csv: one row per run, K = 2..M and w column with proposals > 0:
    family, alpha, rep, scenario_id, K, w_bin ('floor' or 'lo-hi'), w_lo, w_hi, proposals, mergers, acceptance,
    binding_n (proposals with a finite binding gain), binding_mean, binding_sd (over those), w_mean,
    sigma_median (median over markets of the s.d. of the log shock, as in C35),
    gain_equal = 1/2 a (2 - a) sigma^2 [1/(K-1) - 1/K]            (Section 2, equal weights, K-1 -> K)
    gain_w     = 1/2 a (2 - a) sigma^2 [(1 - sum w_new^2) - (1 - sum w_old^2)]
               = 1/2 a (2 - a) sigma^2 [1/(K-1) - (1-w)^2/(K-1) - w^2]
                 (the brief's w-adjusted formula: incumbents of total weight 1 - w at equal weights, entrant w,
                 evaluated at w_mean of the cell)
    gain_w_incumbent = 1/2 sigma^2 [2 a (1-a) w/(K-1) + a^2 (1/(K-1) - (1-w)^2/(K-1) - w^2)]
                 (an incumbent's exact second-order gain under proportional sharing, equal variances: the member
                 gain 1/2 sigma^2 [1 - (1-a)^2 - 2a(1-a) c_i - a^2 sum c^2] at c_i = (1-w)/(K-1) minus at 1/(K-1))
    gain_entrant = 1/2 sigma^2 [1 - (1-a)^2 - 2a(1-a) w - a^2 ((1-w)^2/(K-1) + w^2)]   (from standalone)
    marginal_cost = Phi(K) - Phi(K-1), phi_K = Phi(K)
    incumbent_margin = gain_w_incumbent - marginal_cost  (the incumbent's analytical binding margin: its proposed
                 minus current gain, net of the cost increase)
  gain_w is the capital-weighted average of the members' gains, 1/2 a (2-a) sigma^2 (1 - sum c^2), not an
  incumbent's: it falls below zero once w > about 2/K, while an incumbent's gain stays positive.
  The formulas treat the entrant as a standalone joining K-1 incumbents; unit+unit proposals (entrant = the
  smaller unit) are in the same w columns.
- pilot_d/acceptance_by_K_w_pooled.csv: pooled over reps by family, K, group (floor; w < 0.2; w >= 0.2):
    counts summed, binding mean / sd from the summed moments, w_mean, the gain columns at the pooled w_mean;
- diagnostics/pilot_d_acceptance_by_K_w.png: per family, against K: gain_equal, gain_w (floor and w >= 0.2),
  incumbent_margin (floor),
  gain_w_incumbent (floor) and the binding mean (floor and w >= 0.2), with marginal cost.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from collaborative_growth import coordination_cost  # noqa: E402
from analytics.acceptance_c35 import log_shock_sd  # noqa: E402
from analytics.identity_c35 import load  # noqa: E402

OUT = os.path.join(ROOT, 'pilot_d', 'acceptance_by_K_w.csv')
POOLED = os.path.join(ROOT, 'pilot_d', 'acceptance_by_K_w_pooled.csv')
FIG = os.path.join(ROOT, 'diagnostics', 'pilot_d_acceptance_by_K_w.png')
GROUPS = ('floor', 'w<0.2', 'w>=0.2')
FAMILIES = ('normal', 'laplace', 't3')


def gain_equal(a, s2, K):
    return 0.5 * a * (2 - a) * s2 * (1.0 / (K - 1) - 1.0 / K)


def gain_w(a, s2, K, w):
    return 0.5 * a * (2 - a) * s2 * (1.0 / (K - 1) - (1 - w) ** 2 / (K - 1) - w ** 2)


def gain_w_incumbent(a, s2, K, w):
    return 0.5 * s2 * (2 * a * (1 - a) * w / (K - 1) + a * a * (1.0 / (K - 1) - (1 - w) ** 2 / (K - 1) - w ** 2))


def gain_entrant(a, s2, K, w):
    return 0.5 * s2 * (1 - (1 - a) ** 2 - 2 * a * (1 - a) * w - a * a * ((1 - w) ** 2 / (K - 1) + w ** 2))


def add_gains(df):
    s2 = df.sigma_median ** 2
    df['gain_equal'] = gain_equal(df.alpha, s2, df.K)
    df['gain_w'] = gain_w(df.alpha, s2, df.K, df.w_mean)
    df['gain_w_incumbent'] = gain_w_incumbent(df.alpha, s2, df.K, df.w_mean)
    df['gain_entrant'] = gain_entrant(df.alpha, s2, df.K, df.w_mean)
    df['incumbent_margin'] = df.gain_w_incumbent - df.marginal_cost
    return df


def rows_for(res):
    s = res['scenario']
    P, Mg = res['proposals_by_K_w'], res['mergers_by_K_w']
    n, s1, sq, ws, e = (res['binding_n_by_K_w'], res['binding_sum_by_K_w'], res['binding_sq_by_K_w'],
                        res['w_sum_by_K_w'], res['K_w_edges'])
    sigma = float(np.median(log_shock_sd(s['log_family'], res['market_iqr'])))
    phi = lambda K: coordination_cost(K, s['cost_type'], s['c0'], s['c1'], s['c2'])  # noqa: E731
    out = []
    for K in range(2, s['M'] + 1):
        for j in range(P.shape[1]):
            if P[K, j] == 0:
                continue
            mean = s1[K, j] / n[K, j] if n[K, j] else np.nan
            var = (sq[K, j] - n[K, j] * mean ** 2) / (n[K, j] - 1) if n[K, j] > 1 else np.nan
            out.append(dict(family=s['log_family'], alpha=s['alpha'], rep=s['rep'], scenario_id=s['scenario_id'],
                            K=K, w_bin='floor' if j == 0 else f'{e[j - 1]:.2f}-{e[j]:.2f}',
                            w_lo=np.nan if j == 0 else e[j - 1], w_hi=np.nan if j == 0 else e[j],
                            proposals=int(P[K, j]), mergers=int(Mg[K, j]), acceptance=Mg[K, j] / P[K, j],
                            binding_n=int(n[K, j]), binding_mean=mean, binding_sd=np.sqrt(max(var, 0.0)),
                            w_mean=ws[K, j] / P[K, j], sigma_median=sigma,
                            marginal_cost=phi(K) - phi(K - 1), phi_K=phi(K),
                            _s1=s1[K, j], _sq=sq[K, j], _ws=ws[K, j]))
    return out


def group_of(df):
    return np.where(df.w_bin == 'floor', 'floor', np.where(df.w_lo < 0.2 - 1e-12, 'w<0.2', 'w>=0.2'))


def pooled(df):
    df = df.assign(group=group_of(df))
    g = df.groupby(['family', 'alpha', 'K', 'group'], sort=True)
    p = g.agg(proposals=('proposals', 'sum'), mergers=('mergers', 'sum'), binding_n=('binding_n', 'sum'),
              s1=('_s1', 'sum'), sq=('_sq', 'sum'), ws=('_ws', 'sum'), sigma_median=('sigma_median', 'mean'),
              marginal_cost=('marginal_cost', 'first'), n_reps=('rep', 'nunique')).reset_index()
    p['acceptance'] = p.mergers / p.proposals
    p['binding_mean'] = p.s1 / p.binding_n.where(p.binding_n > 0)
    var = (p.sq - p.binding_n * p.binding_mean ** 2) / (p.binding_n - 1).where(p.binding_n > 1)
    p['binding_sd'] = np.sqrt(var.clip(lower=0))
    p['w_mean'] = p.ws / p.proposals
    tot = p.groupby(['family', 'alpha', 'K']).proposals.transform('sum')
    p['share_of_K'] = p.proposals / tot
    return add_gains(p.drop(columns=['s1', 'sq', 'ws']))


def figure(p):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fams = FAMILIES
    fig, axes = plt.subplots(1, len(fams), figsize=(5 * len(fams), 4), sharey=False)
    for ax, fam in zip(np.atleast_1d(axes), fams):
        q = p[(p.family == fam) & (p.K <= 16)]
        fl, hi = q[q.group == 'floor'], q[q.group == 'w>=0.2']
        eq = q.groupby('K').first().reset_index()
        ax.plot(eq.K, eq.gain_equal, 'k-', label='equal-weight gain (K-1→K)')
        ax.plot(fl.K, fl.gain_w, 'C0--', label='w-adjusted gain, floor entrants')
        ax.plot(fl.K, fl.gain_w_incumbent, 'C0:', label='incumbent gain (exact 2nd order), floor')
        ax.plot(fl.K, fl.incumbent_margin, 'C1-.', label='incumbent gain − marginal cost, floor')
        ax.plot(hi.K, hi.gain_w, 'C2--', label='w-adjusted gain, w ≥ 0.2')
        ax.plot(fl.K, fl.binding_mean, 'C0o-', ms=3, label='binding mean, floor entrants')
        ax.plot(hi.K, hi.binding_mean, 'C2s-', ms=3, label='binding mean, w ≥ 0.2')
        ax.plot(eq.K, eq.marginal_cost, 'C3-', lw=0.8, label='marginal cost Φ(K)-Φ(K-1)')
        ax.axhline(0, color='grey', lw=0.5)
        ax.set_yscale('symlog', linthresh=1e-5)
        ax.set_title(f'{fam}, α = {q.alpha.iloc[0]:g}')
        ax.set_xlabel('K (size of the merged set)')
    np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG, dpi=130)


def e(x):
    return '–' if pd.isna(x) else f'{x:.2e}'


def report():
    p = pd.read_csv(POOLED)
    for fam in FAMILIES:
        q = p[p.family == fam]
        print(f'\n**{fam}** (α = {q.alpha.iloc[0]:g}, 5 reps pooled)\n')
        print('| K | share floor | w̄ floor | acc. floor | binding mean floor | incumbent margin floor | '
              'share w ≥ 0.2 | w̄ w ≥ 0.2 | acc. w ≥ 0.2 | binding mean w ≥ 0.2 | equal-weight gain | '
              'w-adj. gain floor | incumbent gain floor | marginal cost |')
        print('|' + '---|' * 14)
        for K, r in q.groupby('K'):
            if K > 12:
                break
            f = r[r.group == 'floor']
            h = r[r.group == 'w>=0.2']
            fv = lambda c, d=f: d[c].iloc[0] if len(d) else np.nan  # noqa: E731
            hv = lambda c, d=h: d[c].iloc[0] if len(d) else np.nan  # noqa: E731
            print(f'| {K} | {fv("share_of_K"):.2f} | {fv("w_mean"):.3f} | {fv("acceptance"):.3f} | '
                  f'{e(fv("binding_mean"))} | {e(fv("incumbent_margin"))} | {hv("share_of_K"):.2f} | '
                  f'{hv("w_mean"):.3f} | {hv("acceptance"):.3f} | {e(hv("binding_mean"))} | '
                  f'{e(r.gain_equal.iloc[0])} | {e(fv("gain_w"))} | {e(fv("gain_w_incumbent"))} | '
                  f'{e(r.marginal_cost.iloc[0])} |')
        k4 = q[q.K >= 4]
        print(f'\nShare of proposals at K ≥ 4 whose entrant is at the floor: '
              f'{k4[k4.group == "floor"].proposals.sum() / k4.proposals.sum():.3f}')


def main():
    if '--report' in sys.argv:
        report()
        return
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'pilot_d', 'results_c38')
    ids = sorted(int(f[9:13]) for f in os.listdir(d) if f.startswith('scenario_'))
    rows = []
    for i in ids:
        res = load(d, i)
        assert res.get('proposals_by_K_w') is not None, i
        rows += rows_for(res)
    df = add_gains(pd.DataFrame(rows))
    p = pooled(df)
    df.drop(columns=['_s1', '_sq', '_ws']).to_csv(OUT, index=False)
    p.to_csv(POOLED, index=False)
    figure(p)
    print(f'{len(ids)} runs, {len(df)} rows -> {OUT}; pooled {len(p)} rows -> {POOLED}; figure {FIG}')


if __name__ == '__main__':
    main()
