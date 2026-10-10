"""
Event study of conglomerate entry on a per-firm share trace.

Input: an .npz with
  logshare : float array (T, F)  log market share of each firm at each step
  cong     : int   array (T, F)  conglomerate id of each firm (-1 = standalone)
  floor    : bool  array (F, T)  True where the firm hit the floor at that step
  home     : int   array (F,)    market of each firm
  jump     : float array (T, F)  log-size jump applied by the floor at that step (0 elsewhere); optional

Definitions (l = window length):
  event        = step t at which cong[t-1, i] == -1 and cong[t, i] >= 0, with t-l >= 0 and
                 t+l <= T-1. Every such entry is an event (intention to treat); nothing is
                 conditioned on what happens after t.
  before_i     = (logshare[t, i]   - logshare[t-l, i]) / l
  after_i      = (logshare[t+l, i] - logshare[t, i])   / l   (the firm id's realised path,
                 whether or not it stays affiliated; a floor hit and recapitalisation is part
                 of the path, for joiner and control alike)
  stayed       = cong[t:t+l, i] >= 0 throughout
  floor_before = floor[i, t-l:t].any();  floor_after = floor[i, t:t+l].any()
  control      = firm c in the same market, standalone throughout [t-l, t] (not conditioned
                 on [t, t+l]), minimising |logshare[t, c] - logshare[t, i]|, gap <= tol.
  control_stayed_standalone = cong[t:t+l+1, c] == -1 throughout
  did          = (after_i - before_i) - (after_c - before_c)
  owner path   = logshare - cumsum(jump): the firm's log share net of recapitalisation at the floor,
                 i.e. what the sequence of owners earned. before_owner/after_owner/did_owner are the
                 same quantities on that path (matching still uses raw log share). Needs `jump`.

Summary (over matched events):
  did_itt       all matched events (no selection on the after-window)
  did_stayer    joiner stayed l periods AND control stayed standalone (the pre-C28 definition;
                 selected on survival, reported for comparison)
  did_nofloor   ITT, joiner and control hit no floor in [t-l, t)
  did_small/mid/large   ITT by tercile of the joiner's log share at entry
  did_owner_*   the same three (all, nofloor, terciles) on the owner path, when `jump` is given,
                plus means (did_itt_mean, did_owner_mean, did_owner_trim_mean) and the owner DiD by
                conglomerate size at entry (K2, K3_4, K5p), median and mean

Output: one row per event, plus a summary function.
"""
import sys
import numpy as np
import pandas as pd


def events_from_trace(logshare, cong, floor, home, l=50, tol=0.25, jump=None):
    T, F = logshare.shape
    owner = logshare - np.cumsum(jump, axis=0) if jump is not None else None
    entered = (cong[:-1] == -1) & (cong[1:] >= 0)          # (T-1, F): entry at step t+1
    steps, firms = np.nonzero(entered)
    steps = steps + 1
    rows = []
    for t, i in zip(steps, firms):
        if t - l < 0 or t + l > T - 1:
            continue
        m = home[i]
        in_market = np.nonzero(home == m)[0]
        in_market = in_market[in_market != i]
        standalone_before = (cong[t - l:t + 1][:, in_market] == -1).all(axis=0)
        cands = in_market[standalone_before]
        before_i = (logshare[t, i] - logshare[t - l, i]) / l
        after_i = (logshare[t + l, i] - logshare[t, i]) / l
        if owner is not None:
            before_o = (owner[t, i] - owner[t - l, i]) / l
            after_o = (owner[t + l, i] - owner[t, i]) / l
        else:
            before_o = after_o = np.nan
        row = dict(step=t, firm=i, market=m, K_at_entry=int((cong[t] == cong[t, i]).sum()),
                   logshare_entry=float(logshare[t, i]), before=before_i, after=after_i,
                   stayed=bool((cong[t:t + l, i] >= 0).all()),
                   floor_before=bool(floor[i, t - l:t].any()),
                   floor_after=bool(floor[i, t:t + l].any()),
                   control=-1, control_before=np.nan, control_after=np.nan,
                   control_stayed_standalone=False, control_floor_before=False, did=np.nan,
                   before_owner=before_o, after_owner=after_o, did_owner=np.nan)
        if len(cands):
            gaps = np.abs(logshare[t, cands] - logshare[t, i])
            j = int(np.argmin(gaps))
            if gaps[j] <= tol:
                c = cands[j]
                cb = (logshare[t, c] - logshare[t - l, c]) / l
                ca = (logshare[t + l, c] - logshare[t, c]) / l
                row.update(control=int(c), control_before=cb, control_after=ca,
                           control_stayed_standalone=bool((cong[t:t + l + 1, c] == -1).all()),
                           control_floor_before=bool(floor[c, t - l:t].any()),
                           did=(after_i - before_i) - (ca - cb))
                if owner is not None:
                    cbo = (owner[t, c] - owner[t - l, c]) / l
                    cao = (owner[t + l, c] - owner[t, c]) / l
                    row['did_owner'] = (after_o - before_o) - (cao - cbo)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(ev):
    m = ev[ev.control >= 0]
    stay = m[m.stayed & m.control_stayed_standalone]
    nofloor = m[~m.floor_before & ~m.control_floor_before]
    out = dict(n_events=len(ev), n_matched=len(m), n_stayed=len(stay), n_nofloor=len(nofloor),
               share_joiner_stayed=float(ev.stayed.mean()) if len(ev) else np.nan,
               share_floor_before=float(ev.floor_before.mean()) if len(ev) else np.nan,
               joiner_before=ev.before.median(), joiner_after=ev.after.median(),
               control_before=m.control_before.median(), control_after=m.control_after.median(),
               did_itt_median=m.did.median(),
               did_itt_p25=m.did.quantile(.25), did_itt_p75=m.did.quantile(.75),
               did_stayer_median=stay.did.median(),
               did_nofloor_median=nofloor.did.median())
    has_owner = 'did_owner' in m.columns and m.did_owner.notna().any()
    if has_owner:
        out['did_owner_median'] = m.did_owner.median()
        out['did_owner_p25'] = m.did_owner.quantile(.25)
        out['did_owner_p75'] = m.did_owner.quantile(.75)
        out['did_owner_nofloor_median'] = nofloor.did_owner.median()
        # means: with heavy tails the median of a 50-period window hides the rare large loss that
        # pooling averts, so the mean (and a 1%-trimmed mean) is reported next to the median
        out['did_itt_mean'] = m.did.mean()
        out['did_owner_mean'] = m.did_owner.mean()
        lo, hi = m.did_owner.quantile([.01, .99])
        out['did_owner_trim_mean'] = m.did_owner[(m.did_owner >= lo) & (m.did_owner <= hi)].mean()
        out['joiner_before_owner'] = ev.before_owner.median()
        out['joiner_after_owner'] = ev.after_owner.median()
        out['joiner_before_owner_mean'] = ev.before_owner.mean()
        out['joiner_after_owner_mean'] = ev.after_owner.mean()
        # by conglomerate size at entry (K = 2 is a two-firm pool; the diversification gain grows with K)
        for name, sel in (('K2', m.K_at_entry == 2), ('K3_4', m.K_at_entry.between(3, 4)),
                          ('K5p', m.K_at_entry >= 5)):
            g = m[sel]
            out[f'n_{name}'] = len(g)
            out[f'did_owner_{name}_median'] = g.did_owner.median()
            out[f'did_owner_{name}_mean'] = g.did_owner.mean()
    # by size tercile of the joiner's market share at entry (ITT)
    if len(m) >= 9:
        m = m.assign(tercile=pd.qcut(m.logshare_entry, 3, labels=['small', 'mid', 'large']))
        for k, g in m.groupby('tercile', observed=True):
            out[f'did_{k}'] = g.did.median()
            out[f'n_{k}'] = len(g)
            if has_owner:
                out[f'did_owner_{k}'] = g.did_owner.median()
    return out


if __name__ == '__main__':
    z = np.load(sys.argv[1])
    l = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    ev = events_from_trace(z['logshare'], z['cong'], z['floor'], z['home'], l=l,
                           jump=z['jump'] if 'jump' in z.files else None)
    ev.to_csv(sys.argv[1].replace('.npz', '_events.csv'), index=False)
    for k, v in summarize(ev).items():
        print(f'{k:28s} {v:.5f}' if isinstance(v, float) else f'{k:28s} {v}')
