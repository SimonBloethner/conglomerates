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
                 same quantities on that path (matching uses raw log share unless match='owner'). Needs `jump`.

Matching (C38; the default reproduces the definitions above exactly):
  match='raw'    the control minimises |logshare[t, c] - logshare[t, i]| (the default)
  match='owner'  the control minimises the gap in the owner-path log share at entry over the window, i.e. the
                 raw share net of the recapitalisation in the window: logshare[t] - sum(jump[t-l+1 .. t]); same
                 market, same standalone condition, same calliper tol. Needs `jump`.
  match='recap'  raw-share matching among candidates with exactly the joiner's number of floor hits in
                 [t-l, t) (n_hits_before), i.e. the same recapitalisation history count.
  hitfree_k=k    (any match) the joiner must have no floor hit in [t-k, t) -- events that fail are dropped, so
                 n_events counts the survivors -- and candidates are restricted to firms with none either.
Per event (C38): n_hits_before, control_n_hits_before, control_before_owner, control_after_owner.

Summary (over matched events):
  did_itt       all matched events (no selection on the after-window)
  did_stayer    joiner stayed l periods AND control stayed standalone (the pre-C28 definition;
                 selected on survival, reported for comparison)
  did_nofloor   ITT, joiner and control hit no floor in [t-l, t)
  did_small/mid/large   ITT by tercile of the joiner's log share at entry
  did_owner_*   the same three (all, nofloor, terciles) on the owner path, when `jump` is given,
                plus means (did_itt_mean, did_owner_mean, did_owner_trim_mean) and the owner DiD by
                conglomerate size at entry (K2, K3_4, K5p), median and mean
  C38, decomposition of the DiD into its pre- and post-entry parts, over matched events (means are additive:
                did_mean = post_gap_mean - pre_gap_mean, and the same on the owner path):
                joiner_/control_ before/after(_owner)_m_mean, pre_gap(_owner)_mean = mean(before_i - before_c),
                post_gap(_owner)_mean = mean(after_i - after_c), and the medians of the four owner-path means;
                control_before_owner, control_after_owner (medians over matched events, the counterparts of
                joiner_before_owner / joiner_after_owner, which are over all events)

Output: one row per event, plus a summary function.
"""
import sys
import numpy as np
import pandas as pd

MATCHES = ('raw', 'owner', 'recap')


def events_from_trace(logshare, unit, floor, home, l=50, tol=0.25, jump=None, match='raw', hitfree_k=None):
    assert match in MATCHES, match
    assert match != 'owner' or jump is not None, "match='owner' needs jump"
    T, F = logshare.shape
    cumjump = np.cumsum(jump, axis=0) if jump is not None else None
    owner = logshare - cumjump if jump is not None else None
    entered = (unit[:-1] == -1) & (unit[1:] >= 0)          # (T-1, F): entry at step t+1
    steps, firms = np.nonzero(entered)
    steps = steps + 1
    rows = []
    for t, i in zip(steps, firms):
        if t - l < 0 or t + l > T - 1:
            continue
        if hitfree_k is not None and floor[i, t - hitfree_k:t].any():
            continue
        m = home[i]
        in_market = np.nonzero(home == m)[0]
        in_market = in_market[in_market != i]
        standalone_before = (unit[t - l:t + 1][:, in_market] == -1).all(axis=0)
        cands = in_market[standalone_before]
        n_hits_i = int(floor[i, t - l:t].sum())
        if hitfree_k is not None and len(cands):
            cands = cands[~floor[cands, t - hitfree_k:t].any(axis=1)]
        if match == 'recap' and len(cands):
            cands = cands[floor[cands, t - l:t].sum(axis=1) == n_hits_i]
        before_i = (logshare[t, i] - logshare[t - l, i]) / l
        after_i = (logshare[t + l, i] - logshare[t, i]) / l
        if owner is not None:
            before_o = (owner[t, i] - owner[t - l, i]) / l
            after_o = (owner[t + l, i] - owner[t, i]) / l
        else:
            before_o = after_o = np.nan
        row = dict(step=t, firm=i, market=m, K_at_entry=int((unit[t] == unit[t, i]).sum()),
                   logshare_entry=float(logshare[t, i]), before=before_i, after=after_i,
                   stayed=bool((unit[t:t + l, i] >= 0).all()),
                   floor_before=bool(floor[i, t - l:t].any()),
                   floor_after=bool(floor[i, t:t + l].any()),
                   control=-1, control_before=np.nan, control_after=np.nan,
                   control_stayed_standalone=False, control_floor_before=False, did=np.nan,
                   before_owner=before_o, after_owner=after_o, did_owner=np.nan,
                   n_hits_before=n_hits_i, control_n_hits_before=-1,
                   control_before_owner=np.nan, control_after_owner=np.nan)
        if len(cands):
            if match == 'owner':   # owner-path share at entry, net of the window's recapitalisation
                key = logshare[t] - (cumjump[t] - cumjump[t - l])
            else:
                key = logshare[t]
            gaps = np.abs(key[cands] - key[i])
            j = int(np.argmin(gaps))
            if gaps[j] <= tol:
                c = cands[j]
                cb = (logshare[t, c] - logshare[t - l, c]) / l
                ca = (logshare[t + l, c] - logshare[t, c]) / l
                row.update(control=int(c), control_before=cb, control_after=ca,
                           control_stayed_standalone=bool((unit[t:t + l + 1, c] == -1).all()),
                           control_floor_before=bool(floor[c, t - l:t].any()),
                           control_n_hits_before=int(floor[c, t - l:t].sum()),
                           did=(after_i - before_i) - (ca - cb))
                if owner is not None:
                    cbo = (owner[t, c] - owner[t - l, c]) / l
                    cao = (owner[t + l, c] - owner[t, c]) / l
                    row.update(did_owner=(after_o - before_o) - (cao - cbo),
                               control_before_owner=cbo, control_after_owner=cao)
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
        # C38: the control counterparts and the pre/post decomposition over matched events
        out['control_before_owner'] = m.control_before_owner.median()
        out['control_after_owner'] = m.control_after_owner.median()
        for side, prefix in (('joiner', ''), ('control', 'control_')):
            for col in ('before', 'after', 'before_owner', 'after_owner'):
                out[f'{side}_{col}_m_mean'] = m[prefix + col].mean()
        out['pre_gap_mean'] = (m.before - m.control_before).mean()
        out['post_gap_mean'] = (m.after - m.control_after).mean()
        out['pre_gap_owner_mean'] = (m.before_owner - m.control_before_owner).mean()
        out['post_gap_owner_mean'] = (m.after_owner - m.control_after_owner).mean()
        out['pre_gap_owner_median'] = (m.before_owner - m.control_before_owner).median()
        out['post_gap_owner_median'] = (m.after_owner - m.control_after_owner).median()
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
    # python analysis/event_study.py <trace.npz> [l] [--match raw|owner|recap] [--hitfree k]   (C38 flags)
    args = sys.argv[1:]
    match, hitfree_k = 'raw', None
    if '--match' in args:
        k = args.index('--match')
        match = args[k + 1]
        del args[k:k + 2]
    if '--hitfree' in args:
        k = args.index('--hitfree')
        hitfree_k = int(args[k + 1])
        del args[k:k + 2]
    z = np.load(args[0])
    l = int(args[1]) if len(args) > 1 else 50
    ev = events_from_trace(z['logshare'], z['cong'], z['floor'], z['home'], l=l,
                           jump=z['jump'] if 'jump' in z.files else None, match=match, hitfree_k=hitfree_k)
    suffix = '' if match == 'raw' else f'_{match}'
    suffix += '' if hitfree_k is None else f'_k{hitfree_k}'
    ev.to_csv(args[0].replace('.npz', suffix + '_events.csv'), index=False)
    for k, v in summarize(ev).items():
        print(f'{k:28s} {v:.5f}' if isinstance(v, float) else f'{k:28s} {v}')
