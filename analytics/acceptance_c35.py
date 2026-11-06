#!/usr/bin/env python3
"""
C35 Task 2: acceptance by K = size of the merged set, against the analytical marginal gain and marginal cost.

    python analytics/acceptance_c35.py <dir> [<dir> ...]
    python analytics/acceptance_c35.py --report     # markdown tables from acceptance_by_K_pooled.csv

Reads every C35 result pickle (one that has `proposals_by_K`) in the given directories and writes
- pilot_c/acceptance_by_K.csv: one row per run and K = 2..M with proposals > 0 or K <= 12:
    block, family, alpha, p (search probability), cost_multiplier, rep, scenario_id, K,
    proposals, mergers, acceptance (post-burn-in sums; NaN without proposals),
    binding_median, binding_p25, binding_p75, binding_share_positive, binding_share_clipped
        (quantiles of the binding gain min_i [Delta_hat_i(C_hat) - max(0, Delta_hat_i(C_i))] from the
        post-burn-in histogram, linear inside a bin; positive: share in bins above 0 (accepted);
        clipped: share in the first or last bin, i.e. below -0.02 (including -inf) or at or above 0.02),
    sigma_median (median over markets of the s.d. of the log shock, from the run's market IQRs),
    marginal_gain_next = 1/2 alpha (2 - alpha) sigma^2 [(1 - 1/(K+1)) - (1 - 1/K)]   (the (K+1)-th member)
    marginal_cost_next = Phi(K+1) - Phi(K)
    marginal_gain_into_K, marginal_cost_into_K: the same for the K-th member (K-1 -> K), the step a
        proposal of size K takes when a standalone joins a unit of K-1;
- pilot_c/acceptance_by_K_pooled.csv: the same pooled over reps (sums of counts, binding quantiles from the
  summed histogram) by block, family, alpha, p, cost_multiplier, K;
- diagnostics/pilot_c_acceptance_by_K.png: acceptance, marginal gain and marginal cost against K, alpha = 0.3,
  p in {0.05, 1}, one panel per family.
"""
import gzip
import os
import pickle
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from collaborative_growth import coordination_cost  # noqa: E402

K_MIN_ROWS = 12


def log_shock_sd(family, iqr):
    """S.d. of the log shock for a market with IQR `iqr` (normal, laplace, t3 with nu = 3)."""
    iqr = np.asarray(iqr, dtype=float)
    if family == 'normal':
        return iqr / (2 * stats.norm.ppf(0.75))
    if family == 'laplace':
        return np.sqrt(2.0) * iqr / (2 * np.log(2))
    if family == 't3':
        return np.sqrt(3.0) * iqr / (2 * stats.t.ppf(0.75, df=3))
    raise ValueError(family)


def hist_quantiles(h, edges, qs=(0.25, 0.5, 0.75)):
    n = h.sum()
    if n == 0:
        return [np.nan] * len(qs)
    c = np.cumsum(h)
    out = []
    for q in qs:
        target = q * n
        b = int(np.searchsorted(c, target))
        prev = c[b - 1] if b > 0 else 0
        frac = (target - prev) / h[b] if h[b] > 0 else 0.0
        out.append(edges[b] + frac * (edges[b + 1] - edges[b]))
    return out


def gain_increment(alpha, sigma, K):
    """1/2 alpha (2 - alpha) sigma^2 [(1 - 1/(K+1)) - (1 - 1/K)]."""
    return 0.5 * alpha * (2 - alpha) * sigma ** 2 * (1.0 / K - 1.0 / (K + 1))


def rows_for(res):
    s = res['scenario']
    P, Mg, H, edges = res['proposals_by_K'], res['mergers_by_K'], res['binding_gain_hist_by_K'], res['binding_gain_edges']
    sigma = float(np.median(log_shock_sd(s['log_family'], res['market_iqr'])))
    phi = lambda K: coordination_cost(K, s['cost_type'], s['c0'], s['c1'], s['c2'])  # noqa: E731
    out = []
    for K in range(2, s['M'] + 1):
        if P[K] == 0 and K > K_MIN_ROWS:
            continue
        q25, q50, q75 = hist_quantiles(H[K], edges)
        n = H[K].sum()
        out.append(dict(
            block=s['block'], family=s['log_family'], alpha=s['alpha'], p=s['merge_thresh'],
            cost_multiplier=s.get('cost_multiplier', 1.0), rep=s['rep'], scenario_id=s['scenario_id'], K=K,
            proposals=int(P[K]), mergers=int(Mg[K]), acceptance=Mg[K] / P[K] if P[K] else np.nan,
            binding_median=q50, binding_p25=q25, binding_p75=q75,
            binding_share_clipped=(H[K, 0] + H[K, -1]) / n if n else np.nan,
            binding_share_positive=H[K, int(np.argmin(np.abs(edges))):].sum() / n if n else np.nan,
            sigma_median=sigma,
            marginal_gain_next=gain_increment(s['alpha'], sigma, K), marginal_cost_next=phi(K + 1) - phi(K),
            marginal_gain_into_K=gain_increment(s['alpha'], sigma, K - 1), marginal_cost_into_K=phi(K) - phi(K - 1)))
    return out, H, edges


def main():
    rows, hists = [], {}
    for d in sys.argv[1:]:
        for f in sorted(os.listdir(d)):
            if not f.startswith('scenario_'):
                continue
            with gzip.open(os.path.join(d, f)) as fh:
                res = pickle.load(fh)
            if 'proposals_by_K' not in res:
                continue
            s = res['scenario']
            if s['block'] not in ('main', 'search', 'cost-level-high') or s['cost_type'] != 'power_law' \
                    or s['alpha'] not in (0.1, 0.3, 0.5):
                continue
            r, H, edges = rows_for(res)
            rows += r
            key = (s['block'], s['log_family'], s['alpha'], s['merge_thresh'], s.get('cost_multiplier', 1.0))
            hists[key] = hists.get(key, 0) + H
    df = pd.DataFrame(rows).sort_values(['block', 'family', 'alpha', 'p', 'cost_multiplier', 'rep', 'K'])
    df.to_csv(os.path.join(ROOT, 'pilot_c', 'acceptance_by_K.csv'), index=False)

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
        H = hists[(r.block, r.family, r.alpha, r.p, r.cost_multiplier)][int(r.K)]
        q25, q50, q75 = hist_quantiles(H, edges)
        n = H.sum()
        q.append(dict(binding_median=q50, binding_p25=q25, binding_p75=q75,
                      binding_share_positive=H[int(np.argmin(np.abs(edges))):].sum() / n if n else np.nan,
                      binding_share_clipped=(H[0] + H[-1]) / n if n else np.nan))
    pooled = pd.concat([pooled, pd.DataFrame(q)], axis=1)
    pooled.to_csv(os.path.join(ROOT, 'pilot_c', 'acceptance_by_K_pooled.csv'), index=False)
    print(f'{len(df)} rows ({df.scenario_id.nunique()} runs), pooled {len(pooled)} rows')
    figure(pooled)


def figure(pooled):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fams = ['normal', 'laplace', 't3']
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, fam in zip(axes, fams):
        ax2 = ax.twinx()
        for p, ls in ((0.05, '-'), (1.0, '--')):
            blk = 'main' if p == 0.05 else 'search'
            sub = pooled[(pooled.block == blk) & (pooled.family == fam) & (pooled.alpha == 0.3)
                         & (pooled.p == p) & (pooled.proposals >= 20)]
            if sub.empty:
                continue
            ax.plot(sub.K, sub.marginal_gain_into_K, 'C0' + ls, label=f'marginal gain (K-1→K), p={p:g}')
            ax.plot(sub.K, sub.marginal_cost_into_K, 'C3' + ls, label=f'marginal cost (K-1→K), p={p:g}')
            ax2.plot(sub.K, sub.acceptance, 'k' + ls, marker='o', ms=3, label=f'acceptance, p={p:g}')
        ax.set_yscale('log')
        ax.set_xlabel('K (size of the merged set)')
        ax.set_title(f'{fam}, α = 0.3')
        ax2.set_ylim(0, None)
        if fam == 't3':
            ax2.set_ylabel('acceptance rate')
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, fontsize=7, loc='upper right')
    axes[0].set_ylabel('per period')
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, 'diagnostics', 'pilot_c_acceptance_by_K.png'), dpi=150)


REPORT_K = list(range(2, 13)) + [15, 20, 25]


def e(x):
    return 'n/a' if pd.isna(x) else f'{x:.1e}'


def report():
    """Markdown tables for diagnostics/c35_acceptance.md (alpha = 0.3, p in {0.05, 1}) from the pooled CSV."""
    pooled = pd.read_csv(os.path.join(ROOT, 'pilot_c', 'acceptance_by_K_pooled.csv'))
    for fam in ['normal', 'laplace', 't3']:
        for p, blk in ((0.05, 'main'), (1.0, 'search')):
            sub = pooled[(pooled.block == blk) & (pooled.family == fam) & (pooled.alpha == 0.3) & (pooled.p == p)
                         & pooled.K.isin(REPORT_K) & (pooled.proposals > 0)]
            if sub.empty:
                continue
            print(f'\n**{fam}, α = 0.3, p = {p:g}** (5 runs, σ_median = {sub.sigma_median.iloc[0]:.3f})\n')
            print('| K | proposals | acceptance | binding gain median [p25, p75] | share > 0 | ΔG into K | ΔΦ into K '
                  '| ΔG next | ΔΦ next |')
            print('|---|---|---|---|---|---|---|---|---|')
            for _, r in sub.iterrows():
                print(f'| {int(r.K)} | {int(r.proposals)} | {r.acceptance:.3f} | {r.binding_median:.1e} '
                      f'[{r.binding_p25:.1e}, {r.binding_p75:.1e}] | {r.binding_share_positive:.3f} | '
                      f'{e(r.marginal_gain_into_K)} | {e(r.marginal_cost_into_K)} | {e(r.marginal_gain_next)} | '
                      f'{e(r.marginal_cost_next)} |')


if __name__ == '__main__':
    if '--report' in sys.argv:
        report()
    else:
        main()
