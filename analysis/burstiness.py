"""
Clustering of floor hits within markets, from the 27 traced runs.

Usage (from the repository root):
    python analysis/burstiness.py <trace_dir>      # writes pilot_c/burstiness_summary.csv

For each trace: n[m, t] = number of firms in market m that hit the floor in period t (t >= burn_in).
Under a constant hazard with independent firms, n[m, t] is binomial and its index of dispersion
D_m = Var_t(n[m, t]) / Mean_t(n[m, t]) is about 1 - p (p the per-firm hazard), i.e. just below 1.
D_m > 1 means hits cluster in time: a market's firms fail together.

Output, one row per trace:
  family, alpha, rep, scenario_id, burn_in, T
  hits_per_market_period     mean of n over markets and periods
  D_median, D_p25, D_p75     index of dispersion across the M markets
  D_binomial                 1 - p, the independent-firm reference for the same mean hazard
  share_periods_3plus        share of market-periods with >= 3 hits
  share_periods_3plus_binom  the same under the binomial reference with the same per-market hazard
  max_hits_one_period        largest n[m, t] observed
"""
import sys, os, json
import numpy as np
import pandas as pd
from scipy.stats import binom

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'analysis'))
from run_traces import select  # noqa: E402


def burstiness(floor, home, burn_in):
    """floor: (F, T) bool; home: (F,) market id."""
    F, T = floor.shape
    M = int(home.max()) + 1
    N = F // M
    hits = floor[:, burn_in:].astype(np.int32)                       # (F, T-b)
    n = np.zeros((M, hits.shape[1]), dtype=np.int32)
    np.add.at(n, home, hits)                                          # (M, T-b)
    mean = n.mean(axis=1)
    var = n.var(axis=1, ddof=1)
    D = var / np.where(mean > 0, mean, np.nan)
    p = mean / N
    D_binom = 1.0 - p
    share3 = (n >= 3).mean()
    share3_binom = float(np.mean([1.0 - binom.cdf(2, N, pi) for pi in p]))
    return dict(hits_per_market_period=float(mean.mean()),
                D_median=float(np.nanmedian(D)), D_p25=float(np.nanpercentile(D, 25)),
                D_p75=float(np.nanpercentile(D, 75)), D_binomial=float(np.mean(D_binom)),
                share_periods_3plus=float(share3), share_periods_3plus_binom=share3_binom,
                max_hits_one_period=int(n.max()))


def main():
    trace_dir = sys.argv[1]
    scenarios = json.load(open(os.path.join(ROOT, 'pilot_c', 'scenarios.json')))
    rows = []
    for s in select(scenarios):
        tag = f"{s['log_family']}_a{s['alpha']}_r{s['rep']}"
        z = np.load(os.path.join(trace_dir, tag + '.npz'))
        row = dict(family=s['log_family'], alpha=s['alpha'], rep=s['rep'], scenario_id=s['scenario_id'],
                   burn_in=s['burn_in'], T=z['floor'].shape[1])
        row.update(burstiness(z['floor'], z['home'], s['burn_in']))
        rows.append(row)
        print(tag, {k: round(v, 4) for k, v in row.items() if k.startswith(('D_', 'share', 'hits'))}, flush=True)
    out = os.path.join(ROOT, 'pilot_c', 'burstiness_summary.csv')
    pd.DataFrame(rows).to_csv(out, index=False)
    print('wrote', out, len(rows))


if __name__ == '__main__':
    main()
