"""
Event study of conglomerate entry on a per-firm share trace.

Input: an .npz with
  logshare : float array (T, F)  log market share of each firm at each step
  cong     : int   array (T, F)  conglomerate id of each firm (-1 = standalone)
  floor    : bool  array (F, T)  True where the firm hit the floor at that step
  home     : int   array (F,)    market of each firm

Definitions (l = window length):
  event      = step t at which cong[t-1, i] == -1 and cong[t, i] >= 0, with t-l >= 0,
               t+l <= T-1, and the firm affiliated throughout [t, t+l).
  before_i   = (logshare[t, i]   - logshare[t-l, i]) / l
  after_i    = (logshare[t+l, i] - logshare[t, i])   / l
  control    = firm c in the same market, standalone throughout [t-l, t+l],
               minimising |logshare[t, c] - logshare[t, i]|, with that gap <= tol.
  did        = (after_i - before_i) - (after_c - before_c)
  floor_before = floor[i, t-l : t].any()

Output: one row per event, plus a summary function.
"""
import sys
import numpy as np
import pandas as pd


def events_from_trace(logshare, cong, floor, home, l=50, tol=0.25):
    T, F = logshare.shape
    entered = (cong[:-1] == -1) & (cong[1:] >= 0)          # (T-1, F): entry at step t+1
    steps, firms = np.nonzero(entered)
    steps = steps + 1
    rows = []
    for t, i in zip(steps, firms):
        if t - l < 0 or t + l > T - 1:
            continue
        if not (cong[t:t + l, i] >= 0).all():
            continue
        m = home[i]
        in_market = np.nonzero(home == m)[0]
        in_market = in_market[in_market != i]
        standalone = (cong[t - l:t + l + 1][:, in_market] == -1).all(axis=0)
        cands = in_market[standalone]
        before_i = (logshare[t, i] - logshare[t - l, i]) / l
        after_i = (logshare[t + l, i] - logshare[t, i]) / l
        row = dict(step=t, firm=i, market=m, K_at_entry=int((cong[t] == cong[t, i]).sum()),
                   logshare_entry=float(logshare[t, i]), before=before_i, after=after_i,
                   floor_before=bool(floor[i, t - l:t].any()), control=-1,
                   control_before=np.nan, control_after=np.nan, did=np.nan)
        if len(cands):
            gaps = np.abs(logshare[t, cands] - logshare[t, i])
            j = int(np.argmin(gaps))
            if gaps[j] <= tol:
                c = cands[j]
                cb = (logshare[t, c] - logshare[t - l, c]) / l
                ca = (logshare[t + l, c] - logshare[t, c]) / l
                row.update(control=int(c), control_before=cb, control_after=ca,
                           did=(after_i - before_i) - (ca - cb))
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(ev):
    m = ev[ev.control >= 0]
    out = dict(n_events=len(ev), n_matched=len(m),
               n_nofloor=int((~m.floor_before).sum()),
               joiner_before=ev.before.median(), joiner_after=ev.after.median(),
               control_before=m.control_before.median(), control_after=m.control_after.median(),
               did_median=m.did.median(),
               did_p25=m.did.quantile(.25), did_p75=m.did.quantile(.75),
               did_nofloor_median=m[~m.floor_before].did.median())
    # by size tercile of the joiner's market share at entry
    if len(m) >= 9:
        m = m.assign(tercile=pd.qcut(m.logshare_entry, 3, labels=['small', 'mid', 'large']))
        for k, g in m.groupby('tercile', observed=True):
            out[f'did_{k}'] = g.did.median()
            out[f'n_{k}'] = len(g)
    return out


if __name__ == '__main__':
    z = np.load(sys.argv[1])
    l = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    ev = events_from_trace(z['logshare'], z['cong'], z['floor'], z['home'], l=l)
    ev.to_csv(sys.argv[1].replace('.npz', '_events.csv'), index=False)
    for k, v in summarize(ev).items():
        print(f'{k:22s} {v:.5f}' if isinstance(v, float) else f'{k:22s} {v}')
