#!/usr/bin/env python3
"""
C37: pilot D (the moment rule as the base case) - reuse of the C36 runs, variance ratios, pilot C vs D comparison.

    python analytics/pilot_d.py --copy-c36                 # pilot_d/results/ (255 reused runs), pilot_d/provenance.csv
    python analytics/pilot_d.py --check-copies <rerun_dir> # copied results vs fresh pilot D reruns, field by field
    python analytics/pilot_d.py --variance                 # pilot_d/variance_ratio.csv
    python analytics/pilot_d.py --comparison               # pilot_d/comparison_c_vs_d.csv
    python analytics/pilot_d.py --tables                   # pilot_d/tables_c37.md (the tables of diagnostics/c37_note.md)

--copy-c36: 255 pilot D scenarios (source_id set in pilot_design_d.py) are the same model runs as pilot C's C36
  moment block. Their pickles are copied from pilot_c/results/scenario_<source_id>.pkl.gz to
  pilot_d/results/scenario_<pilot D id>.pkl.gz with the `scenario` field replaced by the pilot D scenario (all other
  fields byte-identical). Before copying, every scenario key that reaches the model is checked equal: the two
  scenarios may differ only in block, scenario_id, counterpart_id (C36) and pilot, twin_id, source_id, nu (pilot D;
  nu is None or 3.0, the model's default). provenance.csv has one row per pilot D scenario: origin (copied / run),
  the pilot C loggain twin, the C36 source and the md5 of the source file.
--check-copies: for each rerun in <rerun_dir> (pilot D ids), every field except `scenario` and the runtime is
  compared as pickle bytes with the copy in pilot_d/results; writes diagnostics/identity_c37_copies.csv.
--variance: sigma-hat^2 / sigma^2 over 50-period windows of i.i.d. draws at IQR 0.2 (200,000 windows each) for
  normal, laplace and Student-t with nu in {2.5, 3, 4} (sigma^2 = nu / (nu - 2) s^2), plus the standalone
  no-floor windows of the pilot D t3 trace at alpha = 0.3, rep 0 (as in C36).
--comparison: for every pilot D scenario with a pilot C twin, the paired difference moment - loggain of the tidy
  metrics, then median and IQR over the reps of each cell.
"""
import gzip
import hashlib
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'analytics'))
from pilots import pilot_dir  # noqa: E402

RUNTIME = ('elapsed_seconds', 'ms_per_step')
C36_ONLY = {'block', 'scenario_id', 'counterpart_id'}
D_ONLY = {'pilot', 'twin_id', 'source_id', 'nu'}


def P(*parts):
    return os.path.join(ROOT, *parts)


def load_pkl(path):
    with gzip.open(path, 'rb') as f:
        return pickle.load(f)


def scen(pilot):
    return json.load(open(pilot_dir(pilot, 'scenarios.json', absolute=True)))


# ---------------------------------------------------------------------------------------------------------
def model_keys_match(c36, d):
    """The C36 and pilot D scenarios describe the same model run."""
    kc, kd = set(c36) - C36_ONLY, set(d) - D_ONLY - {'block', 'scenario_id'}
    if kc != kd:
        return False
    if d.get('nu') not in (None, 3.0):
        return False
    return all(c36[k] == d[k] for k in kc) and c36['counterpart_id'] == d['twin_id']


def copy_c36():
    c = {s['scenario_id']: s for s in scen('c')}
    out_dir = pilot_dir('d', 'results', absolute=True)
    os.makedirs(out_dir, exist_ok=True)
    rows, n = [], 0
    for s in scen('d'):
        row = dict(scenario_id=s['scenario_id'], block=s['block'], log_family=s['log_family'], alpha=s['alpha'],
                   rep=s['rep'], decision_rule=s['decision_rule'], twin_id=s['twin_id'], source_id=s['source_id'],
                   origin='run (C37)', source_file='', source_md5='')
        if s['source_id'] is not None:
            src = P('pilot_c', 'results', f"scenario_{s['source_id']:04d}.pkl.gz")
            c36 = c[s['source_id']]
            assert c36['block'] == 'moment' and model_keys_match(c36, s), s['scenario_id']
            raw = open(src, 'rb').read()
            res = pickle.loads(gzip.decompress(raw))
            assert res['scenario'] == c36, s['scenario_id']
            res['scenario'] = s
            with gzip.open(os.path.join(out_dir, f"scenario_{s['scenario_id']:04d}.pkl.gz"), 'wb') as f:
                pickle.dump(res, f)
            row.update(origin='copied from pilot C (C36 moment block)',
                       source_file=f"pilot_c/results/scenario_{s['source_id']:04d}.pkl.gz",
                       source_md5=hashlib.md5(raw).hexdigest())
            n += 1
        rows.append(row)
    df = pd.DataFrame(rows).astype({'twin_id': 'Int64', 'source_id': 'Int64'})
    df.to_csv(pilot_dir('d', 'provenance.csv', absolute=True), index=False)
    print(f'copied {n} results; provenance for {len(rows)} scenarios')


def check_copies(rerun_dir):
    rows = []
    for f in sorted(os.listdir(rerun_dir)):
        if not f.startswith('scenario_'):
            continue
        new = load_pkl(os.path.join(rerun_dir, f))
        cp = load_pkl(pilot_dir('d', 'results', f, absolute=True))
        keys = sorted((set(new) | set(cp)) - set(RUNTIME) - {'scenario'})
        diffs = [k for k in keys if k not in new or k not in cp
                 or pickle.dumps(new[k], protocol=4) != pickle.dumps(cp[k], protocol=4)]
        s = cp['scenario']
        rows.append(dict(scenario_id=s['scenario_id'], source_id=s['source_id'], log_family=s['log_family'],
                         alpha=s['alpha'], merge_thresh=s['merge_thresh'],
                         cost_multiplier=s.get('cost_multiplier', 1.0), alpha_endogenous=s['alpha_endogenous'],
                         same_scenario=new['scenario'] == s, identical=not diffs, differing_fields='|'.join(diffs)))
    df = pd.DataFrame(rows)
    df.to_csv(P('diagnostics', 'identity_c37_copies.csv'), index=False)
    print(f"{len(df)} reruns compared with the copied C36 results: {int(df.identical.sum())} identical, "
          f"scenario equal in {int(df.same_scenario.sum())}")


# ---------------------------------------------------------------------------------------------------------
def t_scale(iqr, nu):
    return iqr / (2 * stats.t.ppf(0.75, df=nu))


def pop_var(family, iqr, nu=3.0):
    if family == 'normal':
        return (iqr / (2 * stats.norm.ppf(0.75))) ** 2
    if family == 'laplace':
        return 2 * (iqr / (2 * np.log(2))) ** 2
    return nu / (nu - 2) * t_scale(iqr, nu) ** 2


def draw(family, iqr, size, rng, nu=3.0):
    if family == 'normal':
        return rng.normal(0, iqr / (2 * stats.norm.ppf(0.75)), size)
    if family == 'laplace':
        return rng.laplace(0, iqr / (2 * np.log(2)), size)
    return t_scale(iqr, nu) * rng.standard_t(nu, size)


QS = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]


def ratio_row(family, nu, source, ratio):
    return dict(family=family, nu=nu, source=source, n_windows=len(ratio), mean=ratio.mean(),
                **{f'q{int(q * 100):02d}': np.quantile(ratio, q) for q in QS},
                share_below_half=(ratio < 0.5).mean(), share_above_2=(ratio > 2).mean())


def variance(l=50):
    rng = np.random.default_rng(37)
    rows = []
    for family, nu in (('normal', None), ('laplace', None), ('t', 2.5), ('t', 3.0), ('t', 4.0)):
        x = draw(family, 0.2, (200_000, l), rng, nu)
        ratio = x.var(axis=1, ddof=1) / pop_var(family, 0.2, nu)
        rows.append(ratio_row(family, nu, 'iid draws, IQR 0.2', ratio))
    # the pilot D t3 trace, alpha = 0.3, rep 0: standalone windows without a floor hit (C36's selection)
    s = [c for c in scen('d') if c['block'] == 'main' and c['cost_type'] == 'power_law' and c['log_family'] == 't3'
         and c['alpha'] == 0.3 and c['rep'] == 0][0]
    z = np.load(pilot_dir('d', 'traces', 't3_a0.3_r0.npz', absolute=True))
    res = load_pkl(pilot_dir('d', 'results', f"scenario_{s['scenario_id']:04d}.pkl.gz", absolute=True))
    var_m = pop_var('t', res['market_iqr'], 3.0)
    ls, unit, floor, home, jump = z['logshare'].astype(np.float64), z['cong'], z['floor'], z['home'], z['jump']
    inc = (ls[1:] - ls[:-1]) - jump[1:]
    ok = (unit[:-1] < 0) & (~floor.T)
    out, (T, F) = [], inc.shape
    for f in range(F):
        t = s['burn_in']
        while t + l <= T:
            if ok[t:t + l, f].all():
                out.append(inc[t:t + l, f].var(ddof=1) / var_m[home[f]])
                t += l
            else:
                t += int(np.nonzero(~ok[t:t + l, f])[0][-1]) + 1
    rows.append(ratio_row('t', 3.0, 'pilot D trace t3_a0.3_r0, standalone no-floor windows', np.array(out)))
    df = pd.DataFrame(rows)
    df.to_csv(pilot_dir('d', 'variance_ratio.csv', absolute=True), index=False)
    print(df.to_string())


# ---------------------------------------------------------------------------------------------------------
CELL = ['block', 'log_family', 'cost_type', 'alpha', 'sharing_rule', 'lookback', 'cross_corr', 'alpha_endogenous',
        'cost_multiplier', 'merge_thresh', 'M', 'N', 'floor_c']
METRICS = ['K_mean', 'cong_capital_share_median', 'hill_exponent_median', 'hhi_within_median',
           'top10pct_aggregate_median', 'acceptance_rate', 'mergers_per_period', 'floor_exits_per_period',
           'voluntary_exits_per_period', 'floor_hit_rate_member', 'floor_hit_rate_standalone',
           'member_firm_share', 'floor_hit_rate_member_status', 'floor_hit_rate_standalone_status',
           'alpha_adopted_median']
STATUS_COLS = ['member_firm_share', 'floor_hit_rate_member_status', 'floor_hit_rate_standalone_status']


def pilot_c_status_rates(ids):
    """Pilot C's tidy.csv has no status occupancy; compute it from the pilot C pickles (summarize_pilot_c's
    pilot_d_metrics, which reads only the result) for the given scenario ids."""
    from summarize_pilot_c import pilot_d_metrics
    out = {}
    for i in ids:
        m = pilot_d_metrics(load_pkl(P('pilot_c', 'results', f'scenario_{i:04d}.pkl.gz')))
        out[i] = [m[k] for k in STATUS_COLS]
    return pd.DataFrame.from_dict(out, orient='index', columns=STATUS_COLS)


def comparison():
    """Paired differences moment - loggain. The pilot C status rates (STATUS_COLS) are computed from the pilot C
    pickles. Metrics: METRICS; cells: CELL; per cell the medians of each rule and the median, p25, p75 of the
    paired differences over the reps."""
    d = pd.read_csv(pilot_dir('d', 'tidy.csv', absolute=True))
    c = pd.read_csv(pilot_dir('c', 'tidy.csv', absolute=True)).set_index('scenario_id')
    d = d[d.twin_id.notna()].copy()
    d['twin_id'] = d.twin_id.astype(int)
    tw = c.loc[d.twin_id].copy()
    tw[STATUS_COLS] = pilot_c_status_rates(d.twin_id).loc[d.twin_id].values
    assert (tw.decision_rule.values == 'loggain').all() and (tw.seed.values == d.seed.values).all()
    for col in CELL + ['rep']:
        if col != 'block':
            assert (tw[col].values == d[col].values).all(), col
    pairs = d[CELL + ['rep', 'scenario_id', 'twin_id']].copy()
    for m in METRICS:
        pairs[m + '_moment'] = d[m].values
        pairs[m + '_loggain'] = tw[m].values
        pairs[m + '_diff'] = d[m].values - tw[m].values
    g = pairs.groupby(CELL, dropna=False)
    out = g.size().rename('n_pairs').to_frame()
    for m in METRICS:
        out[m + '_moment_median'] = g[m + '_moment'].median()
        out[m + '_loggain_median'] = g[m + '_loggain'].median()
        out[m + '_diff_median'] = g[m + '_diff'].median()
        out[m + '_diff_p25'] = g[m + '_diff'].quantile(0.25)
        out[m + '_diff_p75'] = g[m + '_diff'].quantile(0.75)
    out = out.reset_index()
    out.to_csv(pilot_dir('d', 'comparison_c_vs_d.csv', absolute=True), index=False)
    print(f'{len(pairs)} pairs in {len(out)} cells')


if __name__ == '__main__':
    if '--copy-c36' in sys.argv:
        copy_c36()
    elif '--check-copies' in sys.argv:
        check_copies(sys.argv[sys.argv.index('--check-copies') + 1])
    elif '--variance' in sys.argv:
        variance()
    elif '--comparison' in sys.argv:
        comparison()
    elif '--tables' in sys.argv:
        from tables_c37 import main as tables  # noqa: E402
        tables()
    else:
        raise SystemExit(__doc__)
