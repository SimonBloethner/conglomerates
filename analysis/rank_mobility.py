"""
C42 Task 1: rank mobility and persistence within markets, from the pilot D traces (no model run).

Usage (from the repository root):
    python analysis/rank_mobility.py <trace_dir> [--pilot d]                # -> <pilot>/rank_mobility.csv
    python analysis/rank_mobility.py --tables [--pilot d]                   # markdown tables from the CSV (stdout)
    python analysis/rank_mobility.py <trace_dir> --range-sd [--pilot d]     # -> <pilot>/rank_range_sd.csv
    python analysis/rank_mobility.py --range-sd-tables [--pilot d]          # markdown tables (stdout)

--range-sd: the May 2025 article's mobility measures (Section 5, Figure fig:mobility_ridges), the same quantities
the model accumulates online (collaborative_growth.model: rank_range, rank_std): for each firm, over a window of
periods, the range (max - min) and the standard deviation of its within-market rank, ranks 1..N with 1 the smallest,
from the double argsort of the shares as in the model. Windows: `all` = rows 1..T (every step, as the model's online
statistics), `post` = rows burn_in + 1..T. Per trace and window, the distribution over the M x N firms: mean, p10,
p25, p50, p75, p90, and for the range the share of firms at the maximum N - 1.

<trace_dir> holds <family>_a<alpha>_r<rep>.npz for the 27 traced main power-law cells (alpha 0.05, 0.1, 0.3) and,
if present, the nine alpha = 0 cells (analysis/run_traces.py --alpha0, the standalone baseline). Each trace is
checked against its result pickle (<pilot>/results): floor hits per step agree.

Trace layout: logshare and cong are (T + 1, F), row t is the state after step t - 1 (row 0 the initial state);
floor is (F, T), floor[i, s] a floor lift in step s (it shows in row s + 1). Firm f is in market f // N.

Origins t: rows burn_in + 100, burn_in + 200, ... with t + h <= T, for each lag h in LAGS. At each origin and lag:

Within markets (ranks of the log market share among the market's firms):
- spearman:        Spearman correlation of the firms' ranks at t and t + h (average ranks for ties), averaged over
                   markets and origins;
- top_persist:     share of the market's top decile at t (the top N // 10 = 5 of 50) still in it at t + h;
- bottom_persist:  the same for the bottom decile (ties at the floor broken at random, seeded);
- leader_turnover: share of (market, t) in which the largest firm at t is not the largest at t + h;
- mean_abs_drank:  mean |rank at t + h - rank at t| (ranks 1..n);
- hit_share:       share of firms with a floor lift in (t, t + h], i.e. in steps t .. t + h - 1.
Samples:
- all:      every firm;
- hitfree:  only the firms without a floor lift in (t, t + h]: ranks, deciles (round(n / 10) firms) and the leader
            are recomputed among them, so that a lifted firm moves neither its own rank nor anyone else's. The
            difference all - hitfree is the floor's share of mobility (recapitalization at the floor against
            mobility among going concerns). Markets with fewer than 10 such firms are skipped.
By status at t (member: cong[t] != -1): top_persist, bottom_persist and mean_abs_drank over the firms of each status
(the deciles are the market's, the shares are over the decile's firms of that status), in both samples.

Across the economy (sample `owners`): owning units, a unit counted as one owner and a standalone as its own owner,
with capital = the sum of its firms' market shares (pilot D fixes every market's mean size, so shares are
comparable across markets). An owner at t is followed to t + h through its anchor, its largest firm at t: its owner
at t + h is the anchor's unit at t + h (the anchor itself if standalone then).
- spearman:        Spearman correlation over the owners at t of their capital at t and their successor's capital
                   at t + h;
- top_persist:     share of the top decile of owners at t (n_owners // 10, as top10pct_aggregate) whose successor
                   is in the top decile at t + h;
- leader_turnover: whether the largest owner's successor is not the largest owner at t + h (one per origin);
- top_share:       the top decile's share of capital at t (the ownership-concentration measure of Section 4.6,
                   top10pct_aggregate, on the trace's origins).

Output rows (one per family, alpha, rep, h, sample, status, measure): value (the mean over origins and markets),
n (the number of (origin, market) cells, or of firms for the status rows).
"""
import gzip
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'analysis'))
from pilots import pop_pilot  # noqa: E402

LAGS = [10, 50, 100, 500]
STEP = 100
MIN_FREE = 10
FAMILIES = ['normal', 'laplace', 't3']
ALPHAS = [0.0, 0.05, 0.1, 0.3]
REPS = [0, 1, 2]
SEED = 42


def origins(T, burn_in, h):
    return list(range(burn_in + STEP, T - h + 1, STEP))


def ordinal(x, rnd):
    """Ordinal ranks 0..n-1 along the last axis, ties broken by rnd."""
    return np.argsort(np.lexsort((rnd, x), axis=-1), axis=-1)


def pearson_rows(a, b):
    a = a - a.mean(-1, keepdims=True)
    b = b - b.mean(-1, keepdims=True)
    den = np.sqrt((a * a).sum(-1) * (b * b).sum(-1))
    with np.errstate(invalid='ignore', divide='ignore'):
        return (a * b).sum(-1) / den


class Acc:
    """Sums and counts per (sample, status, measure)."""

    def __init__(self):
        self.s, self.n = {}, {}

    def add(self, key, total, count):
        self.s[key] = self.s.get(key, 0.0) + float(total)
        self.n[key] = self.n.get(key, 0) + int(count)

    def rows(self, **meta):
        return [dict(meta, sample=k[0], status=k[1], measure=k[2], value=self.s[k] / self.n[k] if self.n[k] else np.nan,
                     n=self.n[k]) for k in self.s]


def market_block(acc, sample, x0, x1, member, rnd0, rnd1, k):
    """x0, x1, member, rnd*: (n_markets, n) arrays of the same market size n; k: decile size."""
    n = x0.shape[1]
    r0, r1 = rankdata(x0, axis=1), rankdata(x1, axis=1)
    rho = pearson_rows(r0, r1)
    ok = np.isfinite(rho)
    acc.add((sample, 'all', 'spearman'), rho[ok].sum(), ok.sum())
    o0, o1 = ordinal(x0, rnd0), ordinal(x1, rnd1)
    top0, top1 = o0 >= n - k, o1 >= n - k
    bot0, bot1 = o0 < k, o1 < k
    lead0, lead1 = o0 == n - 1, o1 == n - 1
    m = x0.shape[0]
    acc.add((sample, 'all', 'top_persist'), (top0 & top1).sum(), k * m)
    acc.add((sample, 'all', 'bottom_persist'), (bot0 & bot1).sum(), k * m)
    acc.add((sample, 'all', 'leader_turnover'), (lead0 & ~lead1).sum(), m)
    d = np.abs(o1 - o0)
    acc.add((sample, 'all', 'mean_abs_drank'), d.sum(), d.size)
    for st, mask in (('member', member), ('standalone', ~member)):
        acc.add((sample, st, 'top_persist'), (top0 & top1 & mask).sum(), (top0 & mask).sum())
        acc.add((sample, st, 'bottom_persist'), (bot0 & bot1 & mask).sum(), (bot0 & mask).sum())
        acc.add((sample, st, 'mean_abs_drank'), d[mask].sum(), mask.sum())
        acc.add((sample, st, 'top_share_of_decile'), (top0 & mask).sum(), top0.sum())


def owners_at(logshare_row, cong_row):
    """Owner id per firm (unit id, or F + firm for a standalone), owner capital (sum of shares) per firm's owner,
    and the anchor firm (largest member) of each owner."""
    F = len(cong_row)
    own = np.where(cong_row >= 0, cong_row.astype(np.int64), F + np.arange(F))
    share = np.exp(logshare_row.astype(np.float64))
    ids, inv = np.unique(own, return_inverse=True)
    cap = np.bincount(inv, weights=share)
    # anchor: the firm with the largest share within each owner
    order = np.lexsort((share, inv))           # by owner, then share ascending
    last = np.r_[np.flatnonzero(np.diff(inv[order])), len(order) - 1]
    anchor = order[last]
    return ids, inv, cap, anchor


def owner_block(acc, ls, cg, t, h):
    ids0, inv0, cap0, anchor0 = owners_at(ls[t], cg[t])
    ids1, inv1, cap1, _ = owners_at(ls[t + h], cg[t + h])
    succ = inv1[anchor0]                     # successor owner index at t + h of each owner at t
    c_succ = cap1[succ]
    r = pearson_rows(rankdata(cap0)[None], rankdata(c_succ)[None])[0]
    acc.add(('owners', 'all', 'spearman'), r, 1)
    n0, n1 = len(cap0), len(cap1)
    k0, k1 = max(1, n0 // 10), max(1, n1 // 10)
    top0 = np.argsort(cap0)[::-1][:k0]
    top1 = np.zeros(n1, bool)
    top1[np.argsort(cap1)[::-1][:k1]] = True
    acc.add(('owners', 'all', 'top_persist'), top1[succ[top0]].sum(), k0)
    acc.add(('owners', 'all', 'leader_turnover'), succ[np.argmax(cap0)] != np.argmax(cap1), 1)
    acc.add(('owners', 'all', 'top_share'), np.sort(cap0)[::-1][:k0].sum() / cap0.sum(), 1)
    acc.add(('owners', 'all', 'n_owners'), n0, 1)


def trace_rows(z, M, N, burn_in, meta, seed=SEED):
    ls, cg, fl = z['logshare'], z['cong'], z['floor']
    T = ls.shape[0] - 1
    rng = np.random.default_rng(seed)
    rows = []
    for h in LAGS:
        acc = Acc()
        for t in origins(T, burn_in, h):
            x0 = ls[t].astype(np.float64).reshape(M, N)
            x1 = ls[t + h].astype(np.float64).reshape(M, N)
            member = (cg[t] != -1).reshape(M, N)
            hit = fl[:, t:t + h].any(axis=1).reshape(M, N)
            rnd0, rnd1 = rng.random((M, N)), rng.random((M, N))
            acc.add(('all', 'all', 'hit_share'), hit.sum(), hit.size)
            for st, mask in (('member', member), ('standalone', ~member)):
                acc.add(('all', st, 'hit_share'), (hit & mask).sum(), mask.sum())
            market_block(acc, 'all', x0, x1, member, rnd0, rnd1, N // 10)
            for mk in range(M):
                free = ~hit[mk]
                n = int(free.sum())
                if n < MIN_FREE:
                    continue
                market_block(acc, 'hitfree', x0[mk, free][None], x1[mk, free][None], member[mk, free][None],
                             rnd0[mk, free][None], rnd1[mk, free][None], max(1, int(round(n / 10))))
                acc.add(('hitfree', 'all', 'n_free'), n, 1)
            owner_block(acc, ls, cg, t, h)
        rows += acc.rows(**meta, h=h)
    return rows


def firm_range_sd(ls, M, N, a, b, chunk=1000):
    """Per firm (F,), over rows a..b-1 of the (T + 1, F) log-share trace: max - min and sd of the within-market rank
    (1..N, 1 the smallest; double argsort as in the model)."""
    F = M * N
    mn = np.full(F, N + 1, np.int64)
    mx = np.zeros(F, np.int64)
    s = np.zeros(F)
    s2 = np.zeros(F)
    for c0 in range(a, b, chunk):
        x = np.asarray(ls[c0:min(c0 + chunk, b)]).reshape(-1, M, N)
        r = (np.argsort(np.argsort(x, axis=2), axis=2) + 1).reshape(-1, F)
        mn = np.minimum(mn, r.min(0))
        mx = np.maximum(mx, r.max(0))
        rf = r.astype(np.float64)
        s += rf.sum(0)
        s2 += (rf * rf).sum(0)
    n = b - a
    mean = s / n
    return (mx - mn).astype(float), np.sqrt(np.maximum(s2 / n - mean * mean, 0.0))


def range_sd_rows(ls, M, N, burn_in, meta):
    T = ls.shape[0] - 1
    rows = []
    for window, a in (('all', 1), ('post', burn_in + 1)):
        rr, sd = firm_range_sd(ls, M, N, a, T + 1)
        for measure, x in (('rank_range', rr), ('rank_sd', sd)):
            st = dict(mean=x.mean(), p10=np.percentile(x, 10), p25=np.percentile(x, 25), p50=np.median(x),
                      p75=np.percentile(x, 75), p90=np.percentile(x, 90))
            if measure == 'rank_range':
                st['share_at_max'] = float((x == N - 1).mean())
            rows += [dict(meta, window=window, measure=measure, stat=k, value=v) for k, v in st.items()]
    return rows


def main_range_sd(trace_dir, pdir):
    scenarios = json.load(open(os.path.join(ROOT, pdir, 'scenarios.json')))
    rows = []
    for s in select(scenarios, ALPHAS):
        tag = f"{s['log_family']}_a{s['alpha']}_r{s['rep']}"
        path = os.path.join(trace_dir, tag + '.npz')
        if not os.path.exists(path):
            assert s['alpha'] == 0.0, f'missing trace {path}'
            continue
        ls = np.load(path)['logshare']
        meta = dict(family=s['log_family'], alpha=s['alpha'], rep=s['rep'], scenario_id=s['scenario_id'])
        part = range_sd_rows(ls, s['M'], s['N'], s['burn_in'], meta)
        rows += part
        v = {(r['window'], r['measure'], r['stat']): r['value'] for r in part}
        print(tag, f"post: range p50 {v[('post', 'rank_range', 'p50')]:.0f} sd p50 {v[('post', 'rank_sd', 'p50')]:.2f}; "
                   f"all: range p50 {v[('all', 'rank_range', 'p50')]:.0f} sd p50 {v[('all', 'rank_sd', 'p50')]:.2f}",
              flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ROOT, pdir, 'rank_range_sd.csv'), index=False)
    print('wrote', os.path.join(pdir, 'rank_range_sd.csv'), len(df), 'rows')


def range_sd_tables(pdir):
    df = pd.read_csv(os.path.join(ROOT, pdir, 'rank_range_sd.csv'))
    m = df.groupby(['family', 'alpha', 'window', 'measure', 'stat']).value.mean()
    alphas = sorted(df.alpha.unique())
    for window, label in (('post', 'post-burn-in (3,000 periods)'), ('all', 'whole run (11,000 periods, as the model)')):
        print(f'\n**Per-firm rank range and rank SD, {label}** (distribution over the 2,500 firms; mean over the 3 reps)\n')
        print('| family | α | range p10 / p25 / median / p75 | share at max (49) | SD p10 / p25 / median / p75 / p90 | SD mean |')
        print('|---|---|---|---|---|---|')
        for fam in FAMILIES:
            for a in alphas:
                g = lambda meas, st: m[(fam, a, window, meas, st)]  # noqa: E731
                print(f'| {FAMTXT[fam]} | {a:g} | ' +
                      ' / '.join(f"{g('rank_range', st):.1f}" for st in ('p10', 'p25', 'p50', 'p75')) +
                      f" | {g('rank_range', 'share_at_max'):.3f} | " +
                      ' / '.join(f"{g('rank_sd', st):.2f}" for st in ('p10', 'p25', 'p50', 'p75', 'p90')) +
                      f" | {g('rank_sd', 'mean'):.2f} |")


def load_pickle(pdir, sid):
    with gzip.open(os.path.join(ROOT, pdir, 'results', f'scenario_{sid:04d}.pkl.gz'), 'rb') as f:
        return pickle.load(f)


def select(scenarios, alphas):
    out = []
    for fam in FAMILIES:
        for a in alphas:
            for r in REPS:
                m = [s for s in scenarios if s['block'] == 'main' and s['cost_type'] == 'power_law'
                     and s['log_family'] == fam and abs(s['alpha'] - a) < 1e-9 and s['rep'] == r]
                assert len(m) == 1, (fam, a, r, len(m))
                out.append(m[0])
    return out


def main_compute(trace_dir, pdir):
    scenarios = json.load(open(os.path.join(ROOT, pdir, 'scenarios.json')))
    rows = []
    for s in select(scenarios, ALPHAS):
        tag = f"{s['log_family']}_a{s['alpha']}_r{s['rep']}"
        path = os.path.join(trace_dir, tag + '.npz')
        if not os.path.exists(path):
            assert s['alpha'] == 0.0, f'missing trace {path}'
            continue
        z = np.load(path)
        res = load_pickle(pdir, s['scenario_id'])
        assert np.array_equal(z['floor'].sum(axis=0), np.asarray(res['floor_hits_by_status']).sum(axis=1)), tag
        meta = dict(family=s['log_family'], alpha=s['alpha'], rep=s['rep'], scenario_id=s['scenario_id'],
                    top10pct_aggregate_median=res['summary']['top10pct_aggregate_median'])
        part = trace_rows(z, s['M'], s['N'], s['burn_in'], meta)
        rows += part
        v = {(r['h'], r['sample'], r['status'], r['measure']): r['value'] for r in part}
        print(tag, ' '.join(f"h={h}: rho {v.get((h, 'all', 'all', 'spearman'), np.nan):.3f}/"
                            f"{v.get((h, 'hitfree', 'all', 'spearman'), np.nan):.3f}" for h in LAGS), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ROOT, pdir, 'rank_mobility.csv'), index=False)
    print('wrote', os.path.join(pdir, 'rank_mobility.csv'), len(df), 'rows')


FAMTXT = {'normal': 'normal', 'laplace': 'Laplace', 't3': 't₃'}


def tables(pdir, h=100):
    df = pd.read_csv(os.path.join(ROOT, pdir, 'rank_mobility.csv'))
    g = df.groupby(['family', 'alpha', 'h', 'sample', 'status', 'measure']).value
    mean, lo, hi = g.mean(), g.min(), g.max()

    def v(fam, a, sample, status, measure, hh=h, nd=3):
        k = (fam, a, hh, sample, status, measure)
        if k not in mean.index or np.isnan(mean[k]):
            return '–'
        return f'{mean[k]:.{nd}f}'

    alphas = sorted(df.alpha.unique())
    print(f'**Within markets, h = {h}** (mean over the 3 reps; all firms / hit-free)\n')
    print('| family | α | Spearman | top-decile persistence | bottom-decile persistence | leader turnover | '
          'mean abs rank change | share with a lift in (t, t+h] | firms per hit-free market |')
    print('|---|---|---|---|---|---|---|---|---|')
    for fam in FAMILIES:
        for a in alphas:
            cells = [f"{v(fam, a, 'all', 'all', m)} / {v(fam, a, 'hitfree', 'all', m)}"
                     for m in ('spearman', 'top_persist', 'bottom_persist', 'leader_turnover')]
            cells.append(f"{v(fam, a, 'all', 'all', 'mean_abs_drank', nd=2)} / "
                         f"{v(fam, a, 'hitfree', 'all', 'mean_abs_drank', nd=2)}")
            cells.append(v(fam, a, 'all', 'all', 'hit_share'))
            cells.append(v(fam, a, 'hitfree', 'all', 'n_free', nd=1))
            print(f'| {FAMTXT[fam]} | {a:g} | ' + ' | '.join(cells) + ' |')
    print(f'\n**By status at t, h = {h}** (members / standalones; all firms, then hit-free)\n')
    print('| family | α | top-decile persistence | bottom-decile persistence | mean abs rank change | '
          'members\' share of the top decile | lift in (t, t+h] |')
    print('|---|---|---|---|---|---|---|')
    for fam in FAMILIES:
        for a in alphas:
            if a == 0:
                continue
            cells = []
            for m, nd in (('top_persist', 3), ('bottom_persist', 3), ('mean_abs_drank', 2)):
                cells.append(f"{v(fam, a, 'all', 'member', m, nd=nd)} / {v(fam, a, 'all', 'standalone', m, nd=nd)}; "
                             f"{v(fam, a, 'hitfree', 'member', m, nd=nd)} / {v(fam, a, 'hitfree', 'standalone', m, nd=nd)}")
            cells.append(v(fam, a, 'all', 'member', 'top_share_of_decile'))
            cells.append(f"{v(fam, a, 'all', 'member', 'hit_share')} / {v(fam, a, 'all', 'standalone', 'hit_share')}")
            print(f'| {FAMTXT[fam]} | {a:g} | ' + ' | '.join(cells) + ' |')
    print(f'\n**Owners across the economy** (a unit counted once; Spearman, top-decile persistence, leader turnover at '
          f'each lag; top-decile capital share)\n')
    print('| family | α | ' + ' | '.join(f'h = {hh}: ρ / top / leader' for hh in LAGS) +
          ' | top-decile share (trace) | top10pct_aggregate (pickle) | owners |')
    print('|---|---|' + '---|' * (len(LAGS) + 3))
    t10 = df.groupby(['family', 'alpha']).top10pct_aggregate_median.mean()
    for fam in FAMILIES:
        for a in alphas:
            cells = [f"{v(fam, a, 'owners', 'all', 'spearman', hh)} / {v(fam, a, 'owners', 'all', 'top_persist', hh)} / "
                     f"{v(fam, a, 'owners', 'all', 'leader_turnover', hh, nd=2)}" for hh in LAGS]
            cells += [v(fam, a, 'owners', 'all', 'top_share'), f'{t10[(fam, a)]:.3f}',
                      v(fam, a, 'owners', 'all', 'n_owners', nd=0)]
            print(f'| {FAMTXT[fam]} | {a:g} | ' + ' | '.join(cells) + ' |')
    print('\n**Spearman by lag** (all firms / hit-free)\n')
    print('| family | α | ' + ' | '.join(f'h = {hh}' for hh in LAGS) + ' |')
    print('|---|---|' + '---|' * len(LAGS))
    for fam in FAMILIES:
        for a in alphas:
            print(f'| {FAMTXT[fam]} | {a:g} | ' + ' | '.join(
                f"{v(fam, a, 'all', 'all', 'spearman', hh)} / {v(fam, a, 'hitfree', 'all', 'spearman', hh)}"
                for hh in LAGS) + ' |')


def main():
    pdir = 'pilot_' + pop_pilot()
    if '--range-sd-tables' in sys.argv:
        range_sd_tables(pdir)
        return
    if '--range-sd' in sys.argv:
        main_range_sd(sys.argv[1], pdir)
        return
    if '--tables' in sys.argv:
        hh = int(sys.argv[sys.argv.index('--h') + 1]) if '--h' in sys.argv else 100
        tables(pdir, hh)
        return
    main_compute(sys.argv[1], pdir)


if __name__ == '__main__':
    main()
