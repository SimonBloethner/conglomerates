"""
Owner-return growth by affiliation status, from a per-firm trace.

Why: shocks are i.i.d. across periods, so a firm's next-period owner return does not depend on how it
got to its current state, only on its market, its current size (through the floor) and whether it is
pooled. Comparing next-period owner returns of members and standalones within the same market and
size bin therefore identifies the effect of pooling on owners' growth without selecting on an entry
event or on survival. The event study (event_study.py) conditions on an entry at t and on a control
sitting at the same size, which selects both groups on a low endpoint; this panel does not.

Input: the .npz written by the model with trace_path (keys logshare (T+1,F), cong (T+1,F), floor (F,T),
home (F,), jump (T+1,F)).

Definitions, for every firm i and step t >= burn_in with t+1 <= T:
  inc[t, i]   = logshare[t+1, i] - logshare[t, i] - jump[t+1, i]      owner return (floor recapitalisation removed)
  member[t,i] = cong[t, i] >= 0                                       status at the START of the period
  K[t, i]     = size of the conglomerate firm i belongs to at t (0 for standalones)
  size bin    = decile of logshare[t, i] among all firm-periods of the same market (t >= burn_in)

Output of summarize(): one row of aggregates
  n_member, n_standalone                      firm-periods
  inc_member_mean, inc_standalone_mean        unconditional means
  diff_fe_mean    sum over (market, decile) cells of (mean_member - mean_standalone) x cell weight, where the
                  weight is the number of member firm-periods in the cell (cells with both statuses only)
  diff_fe_median  same with medians inside each cell
  diff_dec{1..10}_mean, n_dec{1..10}          the cell difference by size decile (market-weighted as above)
  diff_K2_mean, diff_K3_4_mean, diff_K5p_mean the same with members restricted to K = 2, 3-4, >= 5
  floor_rate_member, floor_rate_standalone    share of firm-periods ending with a floor hit, by status
"""
import sys
import numpy as np
import pandas as pd


def panel_from_trace(logshare, cong, floor, home, jump, burn_in=0):
    T1, F = logshare.shape
    ls = logshare.astype(np.float64)
    inc = (ls[1:] - ls[:-1]) - jump[1:].astype(np.float64)        # (T, F): period t -> t+1
    member = cong[:-1] >= 0                                        # status at start of period
    hit = floor.T                                                  # (T, F): floor hit during period t
    # K at start of period: count members per (t, cong id)
    K = np.zeros_like(inc, dtype=np.int16)
    for t in range(T1 - 1):
        row = cong[t]
        m = row >= 0
        if m.any():
            ids, counts = np.unique(row[m], return_counts=True)
            K[t, m] = counts[np.searchsorted(ids, row[m])]
    t0 = burn_in
    df = pd.DataFrame({
        'market': np.repeat(home[np.newaxis, :], T1 - 1 - t0, axis=0).ravel(),
        'size': ls[t0:-1].ravel(),
        'inc': inc[t0:].ravel(),
        'member': member[t0:].ravel(),
        'K': K[t0:].ravel(),
        'hit': hit[t0:].ravel(),
    })
    df['decile'] = df.groupby('market')['size'].transform(
        lambda s: pd.qcut(s.rank(method='first'), 10, labels=False) + 1).astype(np.int8)
    return df


def _fe_diff(df, stat='mean'):
    """Member - standalone difference of `stat`(inc) within (market, decile) cells, weighted by member count."""
    g = df.groupby(['market', 'decile', 'member'])['inc']
    agg = (g.mean() if stat == 'mean' else g.median()).unstack('member').reindex(columns=[False, True])
    n_mem = df[df.member].groupby(['market', 'decile']).size()
    ok = agg.notna().all(axis=1)
    if not ok.any():
        return np.nan, 0
    d = (agg.loc[ok, True] - agg.loc[ok, False])
    w = n_mem.reindex(d.index).fillna(0).values
    return float(np.average(d.values, weights=w)) if w.sum() > 0 else np.nan, int(w.sum())


def summarize(df):
    out = dict(n_member=int(df.member.sum()), n_standalone=int((~df.member).sum()),
               inc_member_mean=float(df.inc[df.member].mean()),
               inc_standalone_mean=float(df.inc[~df.member].mean()),
               inc_member_median=float(df.inc[df.member].median()),
               inc_standalone_median=float(df.inc[~df.member].median()),
               floor_rate_member=float(df.hit[df.member].mean()),
               floor_rate_standalone=float(df.hit[~df.member].mean()))
    out['diff_fe_mean'], _ = _fe_diff(df, 'mean')
    out['diff_fe_median'], _ = _fe_diff(df, 'median')
    for d in range(1, 11):
        sub = df[df.decile == d]
        out[f'diff_dec{d}_mean'], out[f'n_dec{d}'] = _fe_diff(sub, 'mean')
    for name, sel in (('K2', df.K == 2), ('K3_4', (df.K >= 3) & (df.K <= 4)), ('K5p', df.K >= 5)):
        sub = df[(~df.member) | sel]
        out[f'diff_{name}_mean'], out[f'n_{name}'] = _fe_diff(sub, 'mean')
        out[f'diff_{name}_median'], _ = _fe_diff(sub, 'median')
    return out


if __name__ == '__main__':
    z = np.load(sys.argv[1])
    burn_in = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    df = panel_from_trace(z['logshare'], z['cong'], z['floor'], z['home'], z['jump'], burn_in=burn_in)
    for k, v in summarize(df).items():
        print(f'{k:26s} {v:.6f}' if isinstance(v, float) else f'{k:26s} {v}')
