"""
C41: growth-rate distributions of members and standalones, from the 27 pilot D traces (no model run).

Usage (from the repository root):
    python analysis/growth_distributions.py <trace_dir> [--pilot d]       # CSVs and the figure
    python analysis/growth_distributions.py --tables [--pilot d]          # markdown tables from the CSVs (stdout)

Trace layout (collaborative_growth.model, trace_path): logshare and cong are (T + 1, F), row t + 1 is the state
after step t (row 0 the initial state); floor is (F, T), floor[i, t] is a floor lift in step t; home is (F,).
Within a step the order is: mergers, pooling (members) / own shock (standalones), floor, exits. So the growth
over step t (row t -> row t + 1) is

- a member's if the firm was pooled in step t: cong[t] != -1 (members are not removed before pooling) or
  cong[t + 1] != -1 (it joined in step t, before pooling); its unit is cong[t + 1], or cong[t] if it left
  after pooling, and K is the number of firms pooled in that unit in step t;
- a standalone's otherwise.

Firm-periods with a floor lift in step t (the lift is in the growth) or in step t - 1 (row t is the lifted
level, a recapitalization) are excluded; their counts go to <pilot>/growth_distributions_exclusions.csv.

Three measures of the per-period growth g_it, steps t >= burn_in (growth into rows burn_in + 1 .. T):
- raw:          d log s_it, the change in the firm's log market share. Pilot D fixes each market's mean size
                (market_size_fixed), so this is the firm's log growth minus the log growth of its market's mean
                size, a common term that moves with the large firms' own shocks.
- demeaned:     raw minus its mean over the market's non-excluded firms in the same period. The common term
                cancels exactly: what is left is the firm's own log growth minus the market-period mean.
- standardized: demeaned / q_m, the market's shock IQR (result pickle's market_iqr). Shape statistics pooled
                across markets need this: a pool of markets with different q_m is a scale mixture, fatter-tailed
                than the family even for standalones.
- prelift:      every firm-period, lifts included, with the growth before the lift (raw minus the trace's
                floor jump), demeaned over all the market's firms, / q_m. The exclusion rule above drops exactly
                the periods in which a firm near the floor draws a large negative shock, so it truncates the lower
                tail of small firms (most of them standalones); this measure does not, and is the one to compare
                tails and size slopes with.

ref_sd: the root of the mean over the group of (1 - alpha + alpha w_i)^2 + alpha^2 (sum_j w_j^2 - w_i^2), the
member's growth sd relative to its own shock's with the unit's actual capital weights w (equal shock variances,
linear in the log shocks); 1 for standalones. With equal weights it is sqrt(1 - alpha (2 - alpha)(1 - 1/K)).

Statistics per group: n, mean, sd, IQR, skewness, excess kurtosis, 1/5/95/99th percentiles, MAD/SD (mean
absolute deviation from the mean over sd: 0.707 Laplace, 0.798 normal, lower for t3), and the per-observation
log-likelihoods of the ML Laplace (median, mean |x - median|) and normal fits.

Tables (rows of <pilot>/growth_distributions.csv, column `table`):
- status:  family x alpha x status (member / standalone) x measure, reps pooled;
- K:       members by K bin (2, 3-4, 5+, as Table 4);
- decile:  by decile of the firm's log share at the start of the step (row t), within family x alpha x status
           (by rank, ties broken at random: every firm lifted in the previous step sits at log(c / N));
  `mean_lag_logshare` gives the decile's mean.
"""
import gzip
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'analysis'))
from pilots import pop_pilot  # noqa: E402

MEASURES = ('raw', 'demeaned', 'standardized')
K_BINS = (('2', 2, 2), ('3-4', 3, 4), ('5+', 5, 10 ** 6))
STAT_COLS = ['n', 'mean', 'median', 'sd', 'iqr', 'skew', 'exkurt', 'p1', 'p5', 'p95', 'p99', 'mad_sd', 'laplace_b',
             'll_laplace', 'll_normal']
FIG_FAMILIES = ('normal', 'laplace', 't3')
FIG_ALPHA = 0.3


def market_period_mean(x, mask, home, M):
    """(P, F) -> (P, F): mean of x over the firms of the same market in the same period where mask holds."""
    P = x.shape[0]
    sums = np.zeros((P, M))
    nums = np.zeros((P, M))
    for m in range(M):
        cols = home == m
        sums[:, m] = np.where(mask[:, cols], x[:, cols], 0.0).sum(axis=1)
        nums[:, m] = mask[:, cols].sum(axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        return (sums / nums)[:, home]


def growth_panel(logshare, cong, floor, home, burn_in, q_m=None, jump=None, alpha=None):
    """
    Per-period growth of every firm over steps burn_in .. T - 1, flattened over (step, firm).

    Returns a dict of 1-d arrays over the kept firm-periods (raw, demeaned, standardized if q_m is given,
    member, K, lag, market, ref_var) and the exclusion counts. logshare may hold log levels instead of log shares:
    raw is then the firm's log growth.

    ref_var (alpha given): the member's growth variance relative to its own shock's, if every member had the same
    shock variance and the log shocks entered linearly: (1 - alpha + alpha w_i)^2 + alpha^2 (sum_j w_j^2 - w_i^2),
    w the capital weights in the unit at the start of the step (shares; pilot D fixes every market's mean size, so
    shares are proportional to capital across markets); 1 for standalones. With equal weights it is
    1 - alpha (2 - alpha)(1 - 1/K).

    jump (the trace's (T + 1, F) log jump at the floor) adds out['all']: every firm-period, lifts included, with
    the growth before the lift (raw - jump), demeaned over all the market's firms and divided by q_m
    ('prelift'). The exclusion rule truncates the lower tail of firms near the floor (a large negative shock is
    lifted, so the period is dropped); this sample does not.
    """
    ls = np.asarray(logshare, dtype=np.float64)
    T = ls.shape[0] - 1
    F = ls.shape[1]
    b = burn_in
    P = T - b
    d = ls[b + 1:] - ls[b:-1]                                   # (P, F), growth over step t = b .. T - 1
    c0, c1 = cong[b:T].astype(np.int64), cong[b + 1:T + 1].astype(np.int64)
    fl = np.asarray(floor, dtype=bool)
    fl_prev = np.concatenate([np.zeros((F, 1), dtype=bool), fl[:, :T - 1]], axis=1)  # fl_prev[:, t] = fl[:, t - 1]
    lift_end = fl[:, b:T].T                                     # lift in step t
    lift_start = fl_prev[:, b:T].T                              # lift in step t - 1
    excluded = lift_end | lift_start

    member = (c0 != -1) | (c1 != -1)
    unit = np.where(c1 != -1, c1, c0)                           # unit pooled in step t (members only)
    n_ids = int(max(unit.max(), 0)) + 1
    flat = np.where(member, unit + np.arange(P)[:, None] * n_ids, -1)
    fm = flat[member]
    K = np.zeros((P, F), dtype=np.int64)
    K[member] = np.bincount(fm, minlength=P * n_ids)[fm]
    ref_var = np.ones((P, F))
    if alpha is not None and member.any():
        s = np.exp(ls[b:T] - ls[b:T].max())                     # shares at the start of the step (common scale)
        w = np.zeros((P, F))
        w[member] = s[member] / np.bincount(fm, weights=s[member], minlength=P * n_ids)[fm]
        H = np.bincount(fm, weights=w[member] ** 2, minlength=P * n_ids)[fm]
        wm = w[member]
        ref_var[member] = (1 - alpha + alpha * wm) ** 2 + alpha ** 2 * (H - wm ** 2)

    home = np.asarray(home, dtype=np.int64)
    M = int(home.max()) + 1
    keep = ~excluded
    demeaned = d - market_period_mean(d, keep, home, M)         # over non-excluded firms
    q = None if q_m is None else np.asarray(q_m, dtype=np.float64)[home]

    out = {
        'raw': d[keep], 'demeaned': demeaned[keep], 'member': member[keep], 'K': K[keep],
        'lag': ls[b:T][keep], 'market': np.broadcast_to(home, (P, F))[keep], 'ref_var': ref_var[keep],
        'n_total': P * F, 'n_lift_end': int(lift_end.sum()), 'n_lift_start_only': int((lift_start & ~lift_end).sum()),
        'n_excluded': int(excluded.sum()),
        'n_excluded_member': int((excluded & member).sum()), 'n_excluded_standalone': int((excluded & ~member).sum()),
    }
    if q is not None:
        out['standardized'] = (demeaned / q)[keep]
    if jump is not None:
        pre = d - np.asarray(jump, dtype=np.float64)[b + 1:T + 1]
        pre = pre - market_period_mean(pre, np.ones_like(keep), home, M)
        out['all'] = {'prelift': (pre if q is None else pre / q).ravel(), 'member': member.ravel(),
                      'K': K.ravel(), 'lag': ls[b:T].ravel(), 'ref_var': ref_var.ravel()}
    return out


def stats(x):
    """The statistics of STAT_COLS for a 1-d sample."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < 10:
        return dict(n=n, **{c: np.nan for c in STAT_COLS[1:]})
    mu = x.mean()
    dev = x - mu
    var = np.mean(dev ** 2)
    sd = np.sqrt(var)
    p1, p5, p25, p50, p75, p95, p99 = np.percentile(x, [1, 5, 25, 50, 75, 95, 99])
    b = np.mean(np.abs(x - p50))                                # ML Laplace scale (location = median)
    return dict(n=n, mean=mu, median=p50, sd=sd, iqr=p75 - p25,
                skew=np.mean(dev ** 3) / sd ** 3, exkurt=np.mean(dev ** 4) / var ** 2 - 3.0,
                p1=p1, p5=p5, p95=p95, p99=p99, mad_sd=np.mean(np.abs(dev)) / sd, laplace_b=b,
                ll_laplace=-np.log(2.0 * b) - 1.0, ll_normal=-0.5 * np.log(2.0 * np.pi * var) - 0.5)


def k_bin(K):
    lab = np.full(K.shape, '', dtype=object)
    for name, lo, hi in K_BINS:
        lab[(K >= lo) & (K <= hi)] = name
    return lab


def deciles(x, seed=41):
    """Deciles 1..10 by rank, ties broken at random (firms lifted in the previous step all sit at the same log
    share, log(c / N), so quantile edges would leave the bottom deciles empty)."""
    n = len(x)
    order = np.lexsort((np.random.default_rng(seed).random(n), x))
    dec = np.empty(n, dtype=np.int64)
    dec[order] = np.arange(n) * 10 // n + 1
    return dec


def cell_rows(family, alpha, panel, measures=MEASURES):
    """All CSV rows of one family x alpha cell for the given measures of one sample (reps pooled in panel)."""
    rows = []
    base = dict(family=family, alpha=alpha)

    def extra(sel_K, sel_lag, sel_ref, is_member):
        return dict(mean_K=float(sel_K.mean()) if is_member and len(sel_K) else np.nan,
                    mean_lag_logshare=float(sel_lag.mean()) if len(sel_lag) else np.nan,
                    ref_sd=float(np.sqrt(sel_ref.mean())) if len(sel_ref) else np.nan)

    for status, mask in (('member', panel['member']), ('standalone', ~panel['member'])):
        if not mask.any():
            continue
        lag, Km, ref = panel['lag'][mask], panel['K'][mask], panel['ref_var'][mask]
        is_m = status == 'member'
        dec = deciles(lag)
        kb = k_bin(Km) if is_m else None
        for meas in measures:
            x = panel[meas][mask]
            r = dict(base, table='status', measure=meas, status=status, group='all', **extra(Km, lag, ref, is_m))
            r.update(stats(x))
            rows.append(r)
            if is_m:
                for name, _, _ in K_BINS:
                    sel = kb == name
                    r = dict(base, table='K', measure=meas, status=status, group=name,
                             **extra(Km[sel], lag[sel], ref[sel], True))
                    r.update(stats(x[sel]))
                    rows.append(r)
            for k in range(1, 11):
                sel = dec == k
                r = dict(base, table='decile', measure=meas, status=status, group=str(k),
                         **extra(Km[sel], lag[sel], ref[sel], is_m))
                r.update(stats(x[sel]))
                rows.append(r)
    return rows


def select_traces(scenarios):
    from run_traces import select
    return select(scenarios)


def load_q(pdir, scenario_id):
    with gzip.open(os.path.join(ROOT, pdir, 'results', f'scenario_{scenario_id:04d}.pkl.gz'), 'rb') as f:
        res = pickle.load(f)
    return np.asarray(res['market_iqr'], dtype=np.float64), res['floor_hits_by_status']


def histogram(x, edges):
    """Density over all of x (mass outside the edges counts in the normalization)."""
    h, _ = np.histogram(x, bins=edges)
    return h / (len(x) * np.diff(edges))


def main_compute(trace_dir, pdir):
    scenarios = json.load(open(os.path.join(ROOT, pdir, 'scenarios.json')))
    sel = select_traces(scenarios)
    rows, excl = [], []
    edges = np.linspace(-8, 8, 161)
    fig_data = {}
    cells = {}
    for s in sel:
        cells.setdefault((s['log_family'], s['alpha']), []).append(s)
    for (family, alpha), group in cells.items():
        parts = []
        for s in group:
            tag = f"{s['log_family']}_a{s['alpha']}_r{s['rep']}"
            z = np.load(os.path.join(trace_dir, tag + '.npz'))
            q_m, fhs = load_q(pdir, s['scenario_id'])
            # the trace and the result pickle are the same run: floor hits per step agree
            assert np.array_equal(z['floor'].sum(axis=0), np.asarray(fhs).sum(axis=1)), tag
            p = growth_panel(z['logshare'], z['cong'], z['floor'], z['home'], s['burn_in'], q_m,
                             jump=z['jump'], alpha=s['alpha'])
            excl.append(dict(family=family, alpha=alpha, rep=s['rep'], scenario_id=s['scenario_id'],
                             **{k: v for k, v in p.items() if k.startswith('n_')},
                             n_kept_member=int(p['member'].sum()), n_kept_standalone=int((~p['member']).sum())))
            parts.append(p)
            print(tag, 'kept', len(p['raw']), 'excluded', p['n_excluded'], flush=True)
        panel = {k: np.concatenate([p[k] for p in parts]) for k in
                 ('raw', 'demeaned', 'standardized', 'member', 'K', 'lag', 'market', 'ref_var')}
        rows += cell_rows(family, alpha, panel)
        del panel
        allp = {k: np.concatenate([p['all'][k] for p in parts]) for k in parts[0]['all']}
        del parts
        rows += cell_rows(family, alpha, allp, measures=('prelift',))
        if alpha == FIG_ALPHA:
            fig_data[family] = {st: (histogram(allp['prelift'][m], edges), stats(allp['prelift'][m]))
                                for st, m in (('standalone', ~allp['member']), ('member', allp['member']))}
        del allp
    df = pd.DataFrame(rows)
    cols = ['table', 'family', 'alpha', 'measure', 'status', 'group', 'mean_K', 'mean_lag_logshare', 'ref_sd'] + STAT_COLS
    df = df[cols]
    df.to_csv(os.path.join(ROOT, pdir, 'growth_distributions.csv'), index=False)
    pd.DataFrame(excl).to_csv(os.path.join(ROOT, pdir, 'growth_distributions_exclusions.csv'), index=False)
    figure(fig_data, edges)
    print('wrote', os.path.join(pdir, 'growth_distributions.csv'), len(df), 'rows')


def figure(fig_data, edges):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    mid = 0.5 * (edges[1:] + edges[:-1])
    fams = [f for f in FIG_FAMILIES if f in fig_data]
    fig, axes = plt.subplots(1, len(fams), figsize=(4.2 * len(fams), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    names = {'normal': 'normal', 'laplace': 'Laplace', 't3': '$t_3$'}
    colors = {'standalone': '#4C72B0', 'member': '#C44E52'}
    for ax, fam in zip(axes, fams):
        for st in ('standalone', 'member'):
            h, s = fig_data[fam][st]
            ok = h > 0
            ax.plot(mid[ok], h[ok], 'o', ms=2.5, color=colors[st], label=f'{st}s (sd {s["sd"]:.2f})')
            # ML Laplace fit: location the sample median, scale mean |x - median|
            xx = np.linspace(edges[0], edges[-1], 400)
            ax.plot(xx, np.exp(-np.abs(xx - s['median']) / s['laplace_b']) / (2 * s['laplace_b']),
                    '-', lw=1, color=colors[st], alpha=0.8)
        ax.set_yscale('log')
        ax.set_ylim(1e-6, 2)
        ax.set_xlim(-6, 6)
        ax.set_title(f'{names[fam]}, α = {FIG_ALPHA:g}')
        ax.set_xlabel('log growth before any floor lift, market-demeaned, / $q_m$', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc='lower center')
    axes[0].set_ylabel('density (log scale)')
    fig.suptitle('Growth-rate densities, pilot D traces, all firm-periods (lines: ML Laplace fits)', fontsize=10)
    fig.tight_layout()
    out = os.path.join(ROOT, 'figures', 'appendix')
    os.makedirs(out, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(out, f'growth_density_alpha{FIG_ALPHA:g}.{ext}'), dpi=150)
    plt.close(fig)
    print('wrote figures/appendix/growth_density_alpha0.3.{pdf,png}')


def main():
    pdir = 'pilot_' + pop_pilot()
    args = [a for a in sys.argv[1:]]
    if args == ['--tables']:
        tables(pdir)
    elif len(args) == 1:
        main_compute(args[0], pdir)
    else:
        print(__doc__)
        sys.exit(1)


def tables(pdir):
    """Markdown tables for the note, from the CSVs."""
    df = pd.read_csv(os.path.join(ROOT, pdir, 'growth_distributions.csv'))
    ex = pd.read_csv(os.path.join(ROOT, pdir, 'growth_distributions_exclusions.csv'))
    g = ex.groupby(['family', 'alpha'])[['n_total', 'n_lift_end', 'n_lift_start_only', 'n_excluded',
                                         'n_excluded_member', 'n_excluded_standalone',
                                         'n_kept_member', 'n_kept_standalone']].sum()
    print('### Exclusions\n')
    print('| Family | α | firm-periods | lift in step | lift in previous step only | excluded % | of which members | kept members | kept standalones |')
    print('|---|---|---|---|---|---|---|---|---|')
    for (f, a), r in g.iterrows():
        print(f'| {f} | {a:g} | {r.n_total:,} | {r.n_lift_end:,} | {r.n_lift_start_only:,} | '
              f'{100 * r.n_excluded / r.n_total:.2f} | {r.n_excluded_member:,} | {r.n_kept_member:,} | {r.n_kept_standalone:,} |')
    s = df[df.table == 'status']
    for meas in MEASURES + ('prelift',):
        print(f'\n### By status, {meas}\n')
        print('| Family | α | status | n | mean | sd | IQR | skew | ex. kurt | p1 | p5 | p95 | p99 | MAD/SD | LL Laplace | LL normal |')
        print('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
        for _, r in s[s.measure == meas].iterrows():
            print(f'| {r.family} | {r.alpha:g} | {r.status} | {int(r.n):,} | {r["mean"]:+.4f} | {r.sd:.4f} | {r.iqr:.4f} | '
                  f'{r["skew"]:+.2f} | {r.exkurt:.2f} | {r.p1:+.3f} | {r.p5:+.3f} | {r.p95:+.3f} | {r.p99:+.3f} | '
                  f'{r.mad_sd:.3f} | {r.ll_laplace:.3f} | {r.ll_normal:.3f} |')
    for meas in ('standardized', 'prelift'):
        sd_alone = s[(s.measure == meas) & (s.status == 'standalone')].set_index(['family', 'alpha'])
        k = pd.concat([s[(s.measure == meas) & (s.status == 'member')],
                       df[(df.table == 'K') & (df.measure == meas)]])
        k = k.sort_values(['family', 'alpha'], kind='stable')
        print(f'\n### Members by K, {meas} (ratio: member / standalone of the same cell and measure)\n')
        print('| Family | α | K | mean K | n | sd | sd ratio | equal weights √(1 − α(2 − α)(1 − 1/K̄)) | '
              'actual weights √mean[(1 − α + αw)² + α²(Σw² − w²)] | p1 ratio | p99 ratio | p1 ratio / sd ratio | '
              'ex. kurt | MAD/SD | LL Laplace − LL normal |')
        print('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
        for _, r in k.iterrows():
            a0 = sd_alone.loc[(r.family, r.alpha)]
            ref = np.sqrt(1 - r.alpha * (2 - r.alpha) * (1 - 1 / r.mean_K)) if r.n > 0 else np.nan
            ratio = r.sd / a0.sd
            print(f'| {r.family} | {r.alpha:g} | {r.group} | {r.mean_K:.2f} | {int(r.n):,} | {r.sd:.3f} | {ratio:.3f} | '
                  f'{ref:.3f} | {r.ref_sd:.3f} | {r.p1 / a0.p1:.3f} | {r.p99 / a0.p99:.3f} | '
                  f'{r.p1 / a0.p1 / ratio:.3f} | {r.exkurt:.2f} | {r.mad_sd:.3f} | {r.ll_laplace - r.ll_normal:+.3f} |')
    d = df[df.table == 'decile']
    for meas in MEASURES + ('prelift',):
        print(f'\n### SD by decile of log share at the start of the step, {meas}\n')
        print('| Family | α | status | ' + ' | '.join(f'D{i}' for i in range(1, 11)) +
              ' | D10 / D1 | ref_sd D10 / D1 (actual weights) |')
        print('|---|---|---|' + '---|' * 12)
        for (f, a, st), gr in d[d.measure == meas].groupby(['family', 'alpha', 'status'], sort=False):
            gr = gr.sort_values('group', key=lambda c: c.astype(int))
            sd, ref = gr.sd.to_numpy(), gr.ref_sd.to_numpy()
            print(f'| {f} | {a:g} | {st} | ' + ' | '.join(f'{v:.4f}' for v in sd) +
                  f' | {sd[-1] / sd[0]:.3f} | {ref[-1] / ref[0]:.3f} |')


if __name__ == '__main__':
    main()
