"""
C35 Task 1: floor / no-floor decomposition of the member - standalone owner-return gap on the 27 traces.

Usage (from the repository root):
    python analysis/run_decomposition.py <trace_dir> --index K   # trace K (0..26), for a SLURM array
    python analysis/run_decomposition.py <trace_dir> --merge     # merge, write the CSV and the markdown table
Reads <trace_dir>/<family>_a<alpha>_r<rep>.npz (the traces of analysis/run_traces.py). Per trace it writes
pilot_c/panel/<tag>_decomp.csv; --merge writes pilot_c/status_panel_decomposition.csv and
pilot_c/panel/decomposition_table.md (the family x alpha table, means over the three reps, for
diagnostics/c35_decomposition.md).

Checks per trace (columns check_*):
- check_floor_jump_aligned: the floor hit in floor[:, t] is exactly the one that sets jump[t+1] != 0;
- check_floor_matches_pickle: per-step floor-hit totals of the trace equal those of the C34 result pickle
  (pilot_c/results/scenario_XXXX.pkl.gz, floor_hits_by_status), i.e. the trace is the C34 run.
"""
import gzip
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'analysis'))
from run_traces import select                                  # noqa: E402
from status_panel import panel_from_trace, summarize, decompose  # noqa: E402
from pilots import pop_pilot                                     # noqa: E402

PDIR = 'pilot_c'  # C37: set from --pilot c|d (pilot D: the traces and result pickles of pilot_d/)

OUT_COLS = ['family', 'alpha', 'rep', 'scenario_id', 'gap_total_mean', 'gap_total_mean_fe', 'gap_total_median',
            'gap_nofloor_mean', 'gap_nofloor_median', 'gap_floor_mean', 'gap_floor_median',
            'decomp_ordinary', 'decomp_floor_periods', 'decomp_composition', 'decomp_residual',
            'share_ordinary', 'share_ruin',
            'inc_member_floor_mean', 'inc_standalone_floor_mean',
            'inc_member_nofloor_mean', 'inc_standalone_nofloor_mean',
            'floor_rate_member', 'floor_rate_standalone', 'p_floor_member_cells', 'p_floor_standalone_cells',
            'n_nofloor_member', 'n_nofloor_standalone', 'n_floor_member', 'n_floor_standalone',
            'n_cells', 'n_member_in_cells', 'check_floor_jump_aligned', 'check_floor_matches_pickle', 'seconds']


def one(trace_dir, k):
    scenarios = json.load(open(os.path.join(ROOT, PDIR, 'scenarios.json')))
    s = select(scenarios)[k]
    tag = f"{s['log_family']}_a{s['alpha']}_r{s['rep']}"
    z = np.load(os.path.join(trace_dir, tag + '.npz'))
    t0 = time.time()
    floor, jump = z['floor'], z['jump']
    aligned = bool(np.array_equal(floor, (jump[1:] != 0).T))
    with gzip.open(os.path.join(ROOT, PDIR, 'results', f"scenario_{s['scenario_id']:04d}.pkl.gz")) as f:
        res = pickle.load(f)
    matches = bool(np.array_equal(floor.sum(axis=0), res['floor_hits_by_status'].sum(axis=1)))
    df = panel_from_trace(z['logshare'], z['cong'], floor, z['home'], jump, burn_in=s['burn_in'])
    row = dict(family=s['log_family'], alpha=s['alpha'], rep=s['rep'], scenario_id=s['scenario_id'])
    sm = summarize(df)
    row.update(floor_rate_member=sm['floor_rate_member'], floor_rate_standalone=sm['floor_rate_standalone'],
               diff_fe_mean_panel=sm['diff_fe_mean'])
    row.update(decompose(df))
    row.update(check_floor_jump_aligned=aligned, check_floor_matches_pickle=matches,
               seconds=round(time.time() - t0, 1))
    out = os.path.join(ROOT, PDIR, 'panel', tag + '_decomp.csv')
    pd.DataFrame([row]).to_csv(out, index=False)
    print(tag, {c: row[c] for c in ('gap_total_mean', 'gap_total_mean_fe', 'gap_nofloor_mean', 'share_ordinary',
                                    'check_floor_jump_aligned', 'check_floor_matches_pickle', 'seconds')},
          flush=True)


def fmt(x, f='{:.5f}'):
    return 'n/a' if pd.isna(x) else f.format(x)


def merge():
    d = os.path.join(ROOT, PDIR, 'panel')
    parts = sorted(f for f in os.listdir(d) if f.endswith('_decomp.csv'))
    assert len(parts) == 27, f'expected 27 per-trace decompositions, found {len(parts)}'
    df = pd.concat([pd.read_csv(os.path.join(d, f)) for f in parts]).sort_values(['family', 'alpha', 'rep'])
    df[OUT_COLS + [c for c in df.columns if c not in OUT_COLS]].to_csv(
        os.path.join(ROOT, PDIR, 'status_panel_decomposition.csv'), index=False)
    g = df.groupby(['family', 'alpha'])
    m = g[['gap_total_mean', 'gap_total_median', 'gap_nofloor_mean', 'gap_nofloor_median', 'gap_floor_mean',
           'decomp_ordinary', 'decomp_floor_periods', 'decomp_composition', 'share_ordinary', 'share_ruin',
           'floor_rate_member', 'floor_rate_standalone']].mean()
    lo, hi = g['share_ordinary'].min(), g['share_ordinary'].max()
    lines = ['| family | α | total gap (mean) | total gap (median) | no-floor gap (mean) | no-floor gap (median) '
             '| floor gap (mean) | A ordinary | B floor periods | C composition | share ordinary [min, max over reps] '
             '| share ruin | floor rate m / s |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for (fam, a), r in m.iterrows():
        lines.append(f"| {fam} | {a} | {fmt(r.gap_total_mean)} | {fmt(r.gap_total_median)} | "
                     f"{fmt(r.gap_nofloor_mean)} | {fmt(r.gap_nofloor_median)} | {fmt(r.gap_floor_mean)} | "
                     f"{fmt(r.decomp_ordinary)} | {fmt(r.decomp_floor_periods)} | {fmt(r.decomp_composition)} | "
                     f"{fmt(r.share_ordinary, '{:.2f}')} [{fmt(lo[(fam, a)], '{:.2f}')}, {fmt(hi[(fam, a)], '{:.2f}')}] | "
                     f"{fmt(r.share_ruin, '{:.2f}')} | "
                     f"{fmt(r.floor_rate_member, '{:.3f}')} / {fmt(r.floor_rate_standalone, '{:.3f}')} |")
    checks = (f"\nChecks: floor hits aligned with the jump in {int(df.check_floor_jump_aligned.sum())}/27 traces; "
              f"trace floor hits equal the C34 pickle in {int(df.check_floor_matches_pickle.sum())}/27; "
              f"max |A + B + C - total| = {df.decomp_residual.abs().max():.2e}; "
              f"max |total (decomposition cells) - total (panel cells)| = "
              f"{(df.gap_total_mean - df.gap_total_mean_fe).abs().max():.2e}.\n")
    with open(os.path.join(ROOT, PDIR, 'panel', 'decomposition_table.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n' + checks)
    print('\n'.join(lines) + checks)


if __name__ == '__main__':
    PDIR = 'pilot_' + pop_pilot()  # C37: --pilot c|d
    if '--merge' in sys.argv:
        merge()
    else:
        one(sys.argv[1], int(sys.argv[sys.argv.index('--index') + 1]))
