#!/usr/bin/env python3
"""
C39 Task 1: acceptance and binding gain by K x entrant share x proposal type, and unit+unit proposals by volatility
bin.

    python analytics/acceptance_c39.py [<dir>]     # default pilot_d/results_c39 (the 15 Task 1 runs in it)
    python analytics/acceptance_c39.py --report    # markdown tables from the pooled files

Reads the C39 result pickles of the main power-law alpha = 0.3 moment runs (`proposals_by_K_w_type`, see
merger_kernel_numba) and writes
- pilot_d/acceptance_by_K_w_type.csv: one row per run, K, w column and type with proposals > 0:
    family, alpha, rep, scenario_id, K, w_bin ('floor' or 'lo-hi'), w_lo, type (solo+solo, solo+unit, unit+unit),
    proposals, mergers, acceptance, binding_n, binding_mean, binding_sd (over the finite binding gains);
- pilot_d/unit_unit_by_volbin.csv: one row per run, K and volatility bin of the unit+unit proposals:
    volbin ('entrant calmer' if the entrant unit -- the smaller by capital -- has the lower mean member IQR,
    'entrant more volatile' otherwise), proposals, mergers, acceptance, binding_n, binding_mean, binding_sd,
    binding_entrant (proposals whose binding member is in the entrant unit), share_calm_binds (share of proposals
    whose binding member is in the calmer unit: binding_entrant / proposals when the entrant is calmer, else
    1 - binding_entrant / proposals);
- the same two pooled over reps (counts and moments summed): *_pooled.csv.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from analytics.identity_c35 import load  # noqa: E402

D = os.path.join(ROOT, 'pilot_d')
TYPES = ('solo+solo', 'solo+unit', 'unit+unit')
VOLBINS = ('entrant calmer', 'entrant more volatile')
FAMILIES = ('normal', 'laplace', 't3')


def moments(s1, sq, n):
    mean = s1 / n if n else np.nan
    sd = np.sqrt(max((sq - n * mean ** 2) / (n - 1), 0.0)) if n > 1 else np.nan
    return mean, sd


def rows_for(res):
    s = res['scenario']
    P, Mg = res['proposals_by_K_w_type'], res['mergers_by_K_w_type']
    s1, sq, n = res['binding_sum_by_K_w_type'], res['binding_sq_by_K_w_type'], res['binding_n_by_K_w_type']
    e = res['K_w_edges']
    base = dict(family=s['log_family'], alpha=s['alpha'], rep=s['rep'], scenario_id=s['scenario_id'])
    out = []
    for K, j, t in zip(*np.nonzero(P)):
        mean, sd = moments(s1[K, j, t], sq[K, j, t], n[K, j, t])
        out.append(dict(base, K=int(K), w_bin='floor' if j == 0 else f'{e[j - 1]:.2f}-{e[j]:.2f}',
                        w_lo=np.nan if j == 0 else e[j - 1], type=TYPES[t], proposals=int(P[K, j, t]),
                        mergers=int(Mg[K, j, t]), acceptance=Mg[K, j, t] / P[K, j, t], binding_n=int(n[K, j, t]),
                        binding_mean=mean, binding_sd=sd, _s1=s1[K, j, t], _sq=sq[K, j, t]))
    vb = []
    VP, VM = res['proposals_by_K_type_volbin'], res['mergers_by_K_type_volbin']
    v1, vq, vn, ve = (res['binding_sum_by_K_type_volbin'], res['binding_sq_by_K_type_volbin'],
                      res['binding_n_by_K_type_volbin'], res['binding_entrant_by_K_type_volbin'])
    for K, b in zip(*np.nonzero(VP)):
        mean, sd = moments(v1[K, b], vq[K, b], vn[K, b])
        sh_ent = ve[K, b] / VP[K, b]
        vb.append(dict(base, K=int(K), volbin=VOLBINS[b], proposals=int(VP[K, b]), mergers=int(VM[K, b]),
                       acceptance=VM[K, b] / VP[K, b], binding_n=int(vn[K, b]), binding_mean=mean, binding_sd=sd,
                       binding_entrant=int(ve[K, b]), share_calm_binds=sh_ent if b == 0 else 1 - sh_ent,
                       _s1=v1[K, b], _sq=vq[K, b]))
    return out, vb


def pool(df, keys):
    g = df.groupby(keys, sort=True, dropna=False)   # w_lo is NaN in the floor column
    agg = dict(proposals=('proposals', 'sum'), mergers=('mergers', 'sum'), binding_n=('binding_n', 'sum'),
               s1=('_s1', 'sum'), sq=('_sq', 'sum'), n_reps=('rep', 'nunique'))
    if 'binding_entrant' in df:
        agg['binding_entrant'] = ('binding_entrant', 'sum')
    p = g.agg(**agg).reset_index()
    p['acceptance'] = p.mergers / p.proposals
    p['binding_mean'] = p.s1 / p.binding_n.where(p.binding_n > 0)
    var = (p.sq - p.binding_n * p.binding_mean ** 2) / (p.binding_n - 1).where(p.binding_n > 1)
    p['binding_sd'] = np.sqrt(var.clip(lower=0))
    if 'binding_entrant' in p:
        sh = p.binding_entrant / p.proposals
        p['share_calm_binds'] = np.where(p.volbin == VOLBINS[0], sh, 1 - sh)
    return p.drop(columns=['s1', 'sq'])


def e(x):
    return '–' if pd.isna(x) else f'{x:.1e}'


def report():
    t = pd.read_csv(os.path.join(D, 'acceptance_by_K_w_type_pooled.csv'))
    v = pd.read_csv(os.path.join(D, 'unit_unit_by_volbin_pooled.csv'))
    for fam in FAMILIES:
        q = t[(t.family == fam) & (t.w_lo >= 0.2 - 1e-12) & t.K.between(4, 12)]
        print(f'\n**{fam}**, entrant w ≥ 0.2 (α = 0.3, 5 reps pooled)\n')
        print('| K | proposals | share solo+unit | share unit+unit | acc. solo+unit | acc. unit+unit | '
              'binding mean solo+unit | binding mean unit+unit |')
        print('|---|---|---|---|---|---|---|---|')
        for K, r in q.groupby('K'):
            by = r.groupby('type').agg(p=('proposals', 'sum'), m=('mergers', 'sum'), n=('binding_n', 'sum'))
            tot = by.p.sum()
            sums = {ty: (r[r.type == ty].binding_mean * r[r.type == ty].binding_n).sum() for ty in TYPES}
            f = lambda ty, c: by.loc[ty, c] if ty in by.index else 0  # noqa: E731
            acc = lambda ty: f(ty, 'm') / f(ty, 'p') if f(ty, 'p') else np.nan  # noqa: E731
            bm = lambda ty: sums[ty] / f(ty, 'n') if f(ty, 'n') else np.nan  # noqa: E731
            print(f'| {K} | {tot} | {f("solo+unit", "p") / tot:.2f} | {f("unit+unit", "p") / tot:.2f} | '
                  f'{acc("solo+unit"):.3f} | {acc("unit+unit"):.3f} | {e(bm("solo+unit"))} | {e(bm("unit+unit"))} |')
        u = v[(v.family == fam) & v.K.between(4, 12)]
        print(f'\n**{fam}**, unit+unit proposals by volatility bin\n')
        print('| K | volbin | proposals | acceptance | binding mean | share where the calmer unit binds |')
        print('|---|---|---|---|---|---|')
        for _, r in u.iterrows():
            print(f'| {r.K} | {r.volbin} | {r.proposals} | {r.acceptance:.3f} | {e(r.binding_mean)} | '
                  f'{r.share_calm_binds:.2f} |')


def main():
    if '--report' in sys.argv:
        report()
        return
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(D, 'results_c39')
    rows, vb = [], []
    for f in sorted(os.listdir(d)):
        i = int(f[9:13])
        res = load(d, i)
        s = res['scenario']
        if not (s['block'] == 'main' and s['cost_type'] == 'power_law' and s['alpha'] == 0.3):
            continue
        a, b = rows_for(res)
        rows += a
        vb += b
    df, vdf = pd.DataFrame(rows), pd.DataFrame(vb)
    assert df.scenario_id.nunique() == 15, df.scenario_id.nunique()
    tp = pool(df, ['family', 'alpha', 'K', 'w_bin', 'w_lo', 'type'])
    vp = pool(vdf, ['family', 'alpha', 'K', 'volbin'])
    df.drop(columns=['_s1', '_sq']).to_csv(os.path.join(D, 'acceptance_by_K_w_type.csv'), index=False)
    vdf.drop(columns=['_s1', '_sq']).to_csv(os.path.join(D, 'unit_unit_by_volbin.csv'), index=False)
    tp.to_csv(os.path.join(D, 'acceptance_by_K_w_type_pooled.csv'), index=False)
    vp.to_csv(os.path.join(D, 'unit_unit_by_volbin_pooled.csv'), index=False)
    print(f'{df.scenario_id.nunique()} runs: {len(df)} type rows, {len(vdf)} volatility-bin rows')


if __name__ == '__main__':
    main()
