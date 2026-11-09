"""
Run the 27 traced scenarios and the event study on each.

Usage (from the repository root):
    python analysis/run_traces.py <trace_dir> [--quick]            # all 27 in sequence
    python analysis/run_traces.py <trace_dir> --index K [--quick]  # only scenario K (0..26), for a SLURM array
    python analysis/run_traces.py <trace_dir> --merge              # merge the 27 per-scenario summaries
    --reuse: if <trace_dir>/<tag>.npz already exists, skip the model run and only redo the event study.

Selects, from pilot_c/scenarios.json, the main-block power_law scenarios for
family in {normal, laplace, t3}, alpha in {0.05, 0.1, 0.3}, rep in {0, 1, 2};
runs each with exactly the parameters run_pilot_c.run_scenario uses, writing the
trace to <trace_dir>/<family>_a<alpha>_r<rep>.npz; then runs
analysis.event_study on the trace and writes
    pilot_c/events/<family>_a<alpha>_r<rep>_events.csv
    pilot_c/event_study_summary.csv   (one row per scenario, appended as it goes)
--quick overrides M=N=10, T=300 for a smoke test (outputs go to the same places).
--pilot d (C37): select from pilot_d/scenarios.json (moment rule) and write to pilot_d/events and
pilot_d/event_study_summary.csv; run_one passes the scenario's nu.

C38, alternative matchings on the existing traces (no model run):
    python analysis/run_traces.py <trace_dir> --matching [--index K] --pilot d
        for each trace and each variant in MATCHINGS, the event study (event_study.events_from_trace with that
        match / hitfree_k) and its summary -> <pilot>/events/<tag>_matching.csv (one row per variant)
    python analysis/run_traces.py <trace_dir> --merge-matching --pilot d
        -> <pilot>/event_study_matching_by_rep.csv (27 x variants) and <pilot>/event_study_matching.csv (means
        over the 3 reps by family x alpha x variant); checks that the 'current' variant reproduces
        <pilot>/event_study_summary.csv (n_matched, DiD medians) and stops otherwise.

C42, the alpha = 0 traces (the standalone baseline for analysis/rank_mobility.py):
    python analysis/run_traces.py <trace_dir> --alpha0 --index K --pilot d     # K in 0..8
        reruns the main power-law alpha = 0 scenario of family K // 3 and rep K % 3 with trace_path set ->
        <trace_dir>/<family>_a0.0_r<rep>.npz. Model run only: with no units there are no events.
"""
import sys, os, json, time
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'analysis'))
from collaborative_growth import model              # noqa: E402
from event_study import events_from_trace, summarize  # noqa: E402
from pilots import pop_pilot                          # noqa: E402

PDIR = 'pilot_c'  # C37: set from --pilot c|d in main(); pilot D reads pilot_d/scenarios.json (moment rule)
FAMILIES = ['normal', 'laplace', 't3']
ALPHAS = [0.05, 0.1, 0.3]
REPS = [0, 1, 2]
# C38: (variant, match, hitfree_k); 'current' is the C29-C37 matching (raw log share at entry)
MATCHINGS = [('current', 'raw', None), ('A_owner', 'owner', None),
             ('B_k5', 'raw', 5), ('B_k10', 'raw', 10), ('B_k20', 'raw', 20), ('C_recap', 'recap', None)]
MATCHING_COLS = ['n_events', 'n_matched', 'share_floor_before', 'did_itt_median', 'did_itt_mean',
                 'did_owner_median', 'did_owner_mean', 'did_owner_trim_mean',
                 'joiner_before_m_mean', 'joiner_after_m_mean', 'control_before_m_mean', 'control_after_m_mean',
                 'joiner_before_owner_m_mean', 'joiner_after_owner_m_mean',
                 'control_before_owner_m_mean', 'control_after_owner_m_mean',
                 'pre_gap_mean', 'post_gap_mean', 'pre_gap_owner_mean', 'post_gap_owner_mean',
                 'pre_gap_owner_median', 'post_gap_owner_median',
                 'joiner_before_owner', 'joiner_after_owner', 'control_before_owner', 'control_after_owner']


def matching_rows(s, tp):
    """C38: one summary row per matching variant for the trace of scenario s."""
    z = np.load(tp)
    ls, cg, fl, home = z['logshare'].astype(float), z['cong'], z['floor'], z['home']
    jump = z['jump'].astype(float)
    rows = []
    for name, match, k in MATCHINGS:
        t0 = time.time()
        ev = events_from_trace(ls, cg, fl, home, l=s['lookback'], jump=jump, match=match, hitfree_k=k)
        summ = summarize(ev)
        row = dict(family=s['log_family'], alpha=s['alpha'], rep=s['rep'], scenario_id=s['scenario_id'],
                   variant=name, match=match, hitfree_k=k if k is not None else np.nan)
        row.update({c: summ.get(c, np.nan) for c in MATCHING_COLS})
        rows.append(row)
        print(s['log_family'], s['alpha'], s['rep'], name, row['n_events'], row['n_matched'],
              f"itt={row['did_itt_median']:.5f} owner={row['did_owner_median']:.5f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    return rows


def merge_matching(events_dir):
    parts = sorted(f for f in os.listdir(events_dir) if f.endswith('_matching.csv'))
    assert len(parts) == 27, f'expected 27 per-trace matching files, found {len(parts)}'
    by_rep = pd.concat([pd.read_csv(os.path.join(events_dir, f)) for f in parts], ignore_index=True)
    order = {name: i for i, (name, _, _) in enumerate(MATCHINGS)}
    by_rep = by_rep.sort_values(['family', 'alpha', 'variant', 'rep'],
                                key=lambda c: c.map(order) if c.name == 'variant' else c).reset_index(drop=True)
    # the current matching must reproduce the C37 event-study summary
    ref = pd.read_csv(os.path.join(ROOT, PDIR, 'event_study_summary.csv'))
    cur = by_rep[by_rep.variant == 'current'].merge(ref, on=['family', 'alpha', 'rep'], suffixes=('', '_ref'))
    assert len(cur) == 27
    for c in ('n_matched', 'did_itt_median', 'did_owner_median'):
        assert np.allclose(cur[c], cur[c + '_ref'], rtol=1e-9, atol=1e-12), c
    by_rep.to_csv(os.path.join(ROOT, PDIR, 'event_study_matching_by_rep.csv'), index=False)
    means = (by_rep.groupby(['family', 'alpha', 'variant', 'match', 'hitfree_k'], dropna=False, sort=False)
             [MATCHING_COLS].mean().reset_index())
    means.insert(5, 'n_reps', by_rep.groupby(['family', 'alpha', 'variant'], sort=False).size().values)
    means.to_csv(os.path.join(ROOT, PDIR, 'event_study_matching.csv'), index=False)
    print('merged', len(parts), 'traces;', len(means), 'rows; current matching reproduces the C37 summary')


def select(scenarios, alphas=ALPHAS):
    out = []
    for fam in FAMILIES:
        for a in alphas:
            for r in REPS:
                m = [s for s in scenarios if s['block'] == 'main' and s['cost_type'] == 'power_law'
                     and s['log_family'] == fam and abs(s['alpha'] - a) < 1e-9 and s['rep'] == r]
                assert len(m) == 1, (fam, a, r, len(m))
                out.append(m[0])
    return out


def run_one(s, trace_path, quick):
    M, N, T = (10, 10, 300) if quick else (s['M'], s['N'], s['T'])
    assert not s['proportional'], "cost as a fixed amount was removed in C34"
    params = dict(M=M, N=N, T=T, alpha=s['alpha'], search_prob=s['merge_thresh'], window=s['lookback'],
                  cost_form=s['cost_type'], b0=s['c0'], b1=s['c1'], b2=s['c2'])
    model(**params, seed=s['seed'], growth_process=s['growth_process'], family=s['log_family'],
          nu=s.get('nu') or 3.0,
          mu_range=tuple(s['mu_range']) if s.get('mu_range') else (0.01, 0.1),
          iqr_range=tuple(s['sigma_range']), floor_c=s['floor_c'], cross_corr=s['cross_corr'],
          metric_every=s['metric_every'], sharing_rule=s['sharing_rule'], burn_in=s['burn_in'],
          alpha_endogenous=s.get('alpha_endogenous', False), g=s.get('g'),
          market_size_fixed=s.get('market_size_fixed', False),
          decision_rule=s.get('decision_rule', 'loggain'),
          exit_review_every=s.get('exit_review_every', 1),
          trace_path=trace_path)


def main():
    global PDIR
    PDIR = 'pilot_' + pop_pilot()
    trace_dir = sys.argv[1]
    quick = '--quick' in sys.argv
    events_dir = os.path.join(ROOT, PDIR, 'events')
    summary_path = os.path.join(ROOT, PDIR, 'event_study_summary.csv')
    os.makedirs(trace_dir, exist_ok=True)
    os.makedirs(events_dir, exist_ok=True)
    if '--merge' in sys.argv:
        parts = sorted(f for f in os.listdir(events_dir) if f.endswith('_summary.csv'))
        assert len(parts) == 27, f'expected 27 per-scenario summaries, found {len(parts)}'
        pd.concat([pd.read_csv(os.path.join(events_dir, f)) for f in parts]).to_csv(summary_path, index=False)
        print('merged', len(parts))
        return
    if '--merge-matching' in sys.argv:
        merge_matching(events_dir)
        return
    scenarios = json.load(open(os.path.join(ROOT, PDIR, 'scenarios.json')))
    if '--alpha0' in sys.argv:  # C42
        chosen = select(scenarios, alphas=[0.0])
        k = int(sys.argv[sys.argv.index('--index') + 1])
        s = chosen[k]
        tp = os.path.join(trace_dir, f"{s['log_family']}_a{s['alpha']}_r{s['rep']}.npz")
        t0 = time.time()
        run_one(s, tp, quick)
        z = np.load(tp)
        assert z['logshare'].shape[0] == (300 if quick else s['T']) + 1 and (z['cong'] == -1).all()
        print(tp, f'scenario {s["scenario_id"]}', f'{time.time() - t0:.0f}s', flush=True)
        return
    chosen = select(scenarios)
    if '--index' in sys.argv:
        k = int(sys.argv[sys.argv.index('--index') + 1])
        assert 0 <= k < 27, k
        chosen = [chosen[k]]
    if '--matching' in sys.argv:
        for s in chosen:
            tag = f"{s['log_family']}_a{s['alpha']}_r{s['rep']}"
            rows = matching_rows(s, os.path.join(trace_dir, tag + '.npz'))
            pd.DataFrame(rows).to_csv(os.path.join(events_dir, tag + '_matching.csv'), index=False)
        return
    rows = []
    for s in chosen:
        tag = f"{s['log_family']}_a{s['alpha']}_r{s['rep']}"
        tp = os.path.join(trace_dir, tag + '.npz')
        t0 = time.time()
        if not ('--reuse' in sys.argv and os.path.exists(tp)):
            run_one(s, tp, quick)
        z = np.load(tp)
        ls, cg, fl, home = z['logshare'].astype(float), z['cong'], z['floor'], z['home']
        assert ls.shape[0] == (300 if quick else s['T']) + 1, ls.shape
        sd = np.diff(ls, axis=0).std()
        ev = events_from_trace(ls, cg, fl, home, l=s['lookback'],
                               jump=z['jump'].astype(float) if 'jump' in z.files else None)
        ev.to_csv(os.path.join(ROOT, PDIR, 'events', tag + '_events.csv'), index=False)
        row = dict(family=s['log_family'], alpha=s['alpha'], rep=s['rep'], scenario_id=s['scenario_id'],
                   T=ls.shape[0] - 1, sigma_lo=s['sigma_range'][0], sigma_hi=s['sigma_range'][1],
                   dlogshare_sd=sd, seconds=round(time.time() - t0, 1))
        row.update(summarize(ev))
        rows.append(row)
        pd.DataFrame([row]).to_csv(os.path.join(events_dir, tag + '_summary.csv'), index=False)
        if '--index' not in sys.argv:
            pd.DataFrame(rows).to_csv(summary_path, index=False)
        print(tag, f"sd={sd:.4f}", {k: (round(v, 5) if isinstance(v, float) else v)
                                     for k, v in row.items() if k.startswith(('n_', 'did'))}, flush=True)


if __name__ == '__main__':
    main()
