#!/usr/bin/env python3
"""C33 stop-rule follow-up: drift or noise? Reads diagnostics/burn_in_c33.csv only."""
import numpy as np
import pandas as pd
from scipy import stats

from run_burnin_c14 import compute_convergence_step
from run_burnin_c33 import rolling_max_deviation

df = pd.read_csv('diagnostics/burn_in_c33.csv')
HEAD = ['ccs', 'fhr_standalone', 'fhr_member', 'mean_k']
CELLS = [('laplace', 0.1), ('laplace', 0.3), ('t3', 0.3)]
ME, BURN = 100, 8000
rng = np.random.default_rng(33)


def panel(family, alpha, key):
    d = df[(df.log_family == family) & (df.alpha == alpha)].pivot(index='rep', columns='step', values=key)
    return d.to_numpy(), d.columns.to_numpy()


def surrogate(x_rep, n_obs, L=10):
    """Stationary surrogate of one rep: circular blocks of length L resampled from its post-burn-in window."""
    n = len(x_rep)
    idx = []
    while len(idx) < n_obs:
        s = rng.integers(n)
        idx.extend((s + j) % n for j in range(L))
    return x_rep[np.array(idx[:n_obs])]


print('=== Trend after 8000 (reps as units, df=4) and burn-in sensitivity ===')
print(f"{'cell':16s}{'series':16s}{'mean>8000':>10s}{'slope %/1000':>13s}{'t':>7s}{'p':>7s}"
      f"{'split %':>9s}{'p':>7s}{'5-8k vs 8-11k %':>17s}{'p':>7s}{'>9000 %':>9s}{'>10000 %':>10s}")
for family, alpha in CELLS:
    for key in HEAD:
        x, steps = panel(family, alpha, key)
        post = steps > BURN
        base = x[:, post].mean()
        # OLS slope per rep over the post-burn-in window, as % of the cell mean per 1000 steps
        slopes = np.array([np.polyfit(steps[post], r[post], 1)[0] for r in x]) * 1000 / base * 100
        t_slope = stats.ttest_1samp(slopes, 0.0)
        early = x[:, (steps > 8000) & (steps <= 9500)].mean(axis=1)
        late = x[:, steps > 9500].mean(axis=1)
        split = stats.ttest_rel(late, early)
        before = x[:, (steps > 5000) & (steps <= 8000)].mean(axis=1)
        after = x[:, post].mean(axis=1)
        prior = stats.ttest_rel(after, before)
        m9 = (x[:, steps > 9000].mean() / base - 1) * 100
        m10 = (x[:, steps > 10000].mean() / base - 1) * 100
        print(f"{family + ' a' + str(alpha):16s}{key:16s}{base:10.4f}{slopes.mean():13.3f}{t_slope.statistic:7.2f}"
              f"{t_slope.pvalue:7.3f}{(late.mean() / early.mean() - 1) * 100:9.2f}{split.pvalue:7.3f}"
              f"{(after.mean() / before.mean() - 1) * 100:17.2f}{prior.pvalue:7.3f}{m9:9.2f}{m10:10.2f}")

print()
print('=== Placebo: criterion on stationary surrogates built from the post-8000 window (300 draws, blocks of 1000 steps) ===')
print(f"{'cell':16s}{'series':16s}{'obs max dev':>12s}{'surr median':>12s}{'surr p95':>10s}"
      f"{'P(surr dev>=obs)':>18s}{'surr pass by 8000':>19s}")
for family, alpha in CELLS:
    for key in ['ccs', 'fhr_standalone', 'fhr_member', 'mean_k', 'hill', 'hhi_within']:
        x, steps = panel(family, alpha, key)
        post = steps > BURN
        obs = rolling_max_deviation(x.mean(axis=0), BURN, ME)
        devs, passed = [], []
        for _ in range(300):
            s = np.stack([surrogate(r[post], x.shape[1]) for r in x]).mean(axis=0)
            devs.append(rolling_max_deviation(s, BURN, ME))
            conv = compute_convergence_step(s, window=500, tolerance=0.02, final_window=2000, metric_every=ME)
            passed.append(conv <= BURN)
        devs = np.array(devs)
        print(f"{family + ' a' + str(alpha):16s}{key:16s}{obs:12.2%}{np.median(devs):12.2%}{np.percentile(devs, 95):10.2%}"
              f"{(devs >= obs).mean():18.2f}{np.mean(passed):19.2f}")

print()
print('=== 500-step rolling cell mean of the failing series (every 500 steps from 6000), % of final-2000 mean ===')
for family, alpha, key in [('laplace', 0.3, 'ccs'), ('t3', 0.3, 'ccs'), ('t3', 0.3, 'fhr_standalone'), ('t3', 0.3, 'fhr_member')]:
    x, steps = panel(family, alpha, key)
    m = x.mean(axis=0)
    final = m[-20:].mean()
    roll = pd.Series(m).rolling(5).mean().to_numpy()
    pts = {int(steps[i]): (roll[i] / final - 1) * 100 for i in range(len(m)) if steps[i] >= 6000 and steps[i] % 500 == 0}
    print(f"{family} a{alpha} {key:15s} final {final:.4f}: " + ' '.join(f"{k}:{v:+.1f}" for k, v in pts.items()))
