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

decompose() (C35): the same gap on the firm-periods that do not end at the floor (hit == 0, i.e. no floor
recapitalisation between t and t+1: floor[i, t] in the trace's (F, T) array, which is the hit that sets
jump[t+1, i]) and on those that do, plus an exact split of the total. Inside every (market, decile) cell
with members and standalones in the no-floor sample, with p = floor rate and m = mean owner return by status
and sample (nf = no floor, f = floor):
  total = mean_m - mean_s
        = (1 - p_s)(m_m,nf - m_s,nf)          A  ordinary periods (drag reduction)
        + [p_m m_m,f - p_s m_s,f]             B  floor periods (their frequency times their return)
        + (p_s - p_m) m_m,nf                  C  composition: members' extra ordinary periods
B + C = p_m (m_m,f - m_m,nf) - p_s (m_s,f - m_m,nf) is the part due to members reaching the floor less
often (ruin protection). Cell values are aggregated with the member-count weights of _fe_diff, so A + B + C
equals the weighted total gap on these cells exactly; share_ordinary = A / total, share_ruin = (B + C) / total.
"""
import sys
import numpy as np
import pandas as pd


def panel_from_trace(logshare, unit, floor, home, jump, burn_in=0):
    T1, F = logshare.shape
    ls = logshare.astype(np.float64)
    inc = (ls[1:] - ls[:-1]) - jump[1:].astype(np.float64)        # (T, F): period t -> t+1
    member = unit[:-1] >= 0                                        # status at start of period
    hit = floor.T                                                  # (T, F): floor hit during period t
    # K at start of period: count members per (t, cong id)
    K = np.zeros_like(inc, dtype=np.int16)
    for t in range(T1 - 1):
        row = unit[t]
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


def decompose(df):
    """C35: member - standalone gap on the no-floor and floor samples and the split total = A + B + C
    (module docstring). Returns one row of aggregates."""
    df = df.assign(hit=df.hit.astype(bool), member=df.member.astype(bool))
    nf, fl = df[~df.hit], df[df.hit]
    out = dict(n_nofloor_member=int(nf.member.sum()), n_nofloor_standalone=int((~nf.member).sum()),
               n_floor_member=int(fl.member.sum()), n_floor_standalone=int((~fl.member).sum()),
               inc_member_floor_mean=float(fl.inc[fl.member].mean()),
               inc_standalone_floor_mean=float(fl.inc[~fl.member].mean()),
               inc_member_nofloor_mean=float(nf.inc[nf.member].mean()),
               inc_standalone_nofloor_mean=float(nf.inc[~nf.member].mean()))
    out['gap_total_mean_fe'], _ = _fe_diff(df, 'mean')
    out['gap_total_median'], _ = _fe_diff(df, 'median')
    out['gap_nofloor_mean'], _ = _fe_diff(nf, 'mean')
    out['gap_nofloor_median'], _ = _fe_diff(nf, 'median')
    out['gap_floor_mean'], out['n_floor_cells_member'] = _fe_diff(fl, 'mean')
    out['gap_floor_median'], _ = _fe_diff(fl, 'median')

    cols = pd.MultiIndex.from_product([['sum', 'count'], [False, True], [False, True]],
                                      names=[None, 'member', 'hit'])
    a = (df.groupby(['market', 'decile', 'member', 'hit'])['inc'].agg(['sum', 'count'])
         .unstack(['member', 'hit']).reindex(columns=cols).fillna(0.0))
    S, C = a['sum'], a['count']
    n_m, n_s = C[True].sum(axis=1), C[False].sum(axis=1)
    ok = (C[(True, False)] > 0) & (C[(False, False)] > 0)
    S, C, n_m, n_s = S[ok], C[ok], n_m[ok], n_s[ok]
    p_m, p_s = C[(True, True)] / n_m, C[(False, True)] / n_s
    m_m_nf, m_s_nf = S[(True, False)] / C[(True, False)], S[(False, False)] / C[(False, False)]
    A = (1 - p_s) * (m_m_nf - m_s_nf)
    B = S[(True, True)] / n_m - S[(False, True)] / n_s
    Cc = (p_s - p_m) * m_m_nf
    total = (S[(True, False)] + S[(True, True)]) / n_m - (S[(False, False)] + S[(False, True)]) / n_s
    w = n_m.values
    avg = lambda x: float(np.average(x.values, weights=w))  # noqa: E731
    out.update(gap_total_mean=avg(total), decomp_ordinary=avg(A), decomp_floor_periods=avg(B),
               decomp_composition=avg(Cc), p_floor_member_cells=avg(p_m), p_floor_standalone_cells=avg(p_s),
               n_cells=int(ok.sum()), n_member_in_cells=int(w.sum()))
    out['decomp_residual'] = out['gap_total_mean'] - (out['decomp_ordinary'] + out['decomp_floor_periods']
                                                      + out['decomp_composition'])
    out['share_ordinary'] = out['decomp_ordinary'] / out['gap_total_mean']
    out['share_ruin'] = (out['decomp_floor_periods'] + out['decomp_composition']) / out['gap_total_mean']
    return out


if __name__ == '__main__':
    z = np.load(sys.argv[1])
    burn_in = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    df = panel_from_trace(z['logshare'], z['cong'], z['floor'], z['home'], z['jump'], burn_in=burn_in)
    for k, v in summarize(df).items():
        print(f'{k:26s} {v:.6f}' if isinstance(v, float) else f'{k:26s} {v}')
