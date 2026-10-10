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
"""
import sys, os, json, time
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'analysis'))
from collaborative_growth import model              # noqa: E402
from event_study import events_from_trace, summarize  # noqa: E402

FAMILIES = ['normal', 'laplace', 't3']
ALPHAS = [0.05, 0.1, 0.3]
REPS = [0, 1, 2]


def select(scenarios):
    out = []
    for fam in FAMILIES:
        for a in ALPHAS:
            for r in REPS:
                m = [s for s in scenarios if s['block'] == 'main' and s['cost_type'] == 'power_law'
                     and s['log_family'] == fam and abs(s['alpha'] - a) < 1e-9 and s['rep'] == r]
                assert len(m) == 1, (fam, a, r, len(m))
                out.append(m[0])
    return out


def run_one(s, trace_path, quick):
    M, N, T = (10, 10, 300) if quick else (s['M'], s['N'], s['T'])
    params = [M, N, T, s['alpha'], M * N, s['merge_thresh'], 4, 0.0, s['proportional'],
              s['lookback'], s['cost_type'], s['c0'], s['c1'], s['c2']]
    model(params, seed=s['seed'], growth_process=s['growth_process'], log_family=s['log_family'],
          mu_range=tuple(s['mu_range']) if s.get('mu_range') else (0.01, 0.1),
          sigma_range=tuple(s['sigma_range']), floor_c=s['floor_c'], cross_corr=s['cross_corr'],
          metric_every=s['metric_every'], sharing_rule=s['sharing_rule'], burn_in=s['burn_in'],
          alpha_endogenous=s.get('alpha_endogenous', False), g=s.get('g'),
          market_size_fixed=s.get('market_size_fixed', False),
          decision_rule=s.get('decision_rule', 'loggain'),
          exit_review_every=s.get('exit_review_every', 1),
          trace_path=trace_path)


def main():
    trace_dir = sys.argv[1]
    quick = '--quick' in sys.argv
    events_dir = os.path.join(ROOT, 'pilot_c', 'events')
    summary_path = os.path.join(ROOT, 'pilot_c', 'event_study_summary.csv')
    os.makedirs(trace_dir, exist_ok=True)
    os.makedirs(events_dir, exist_ok=True)
    if '--merge' in sys.argv:
        parts = sorted(f for f in os.listdir(events_dir) if f.endswith('_summary.csv'))
        assert len(parts) == 27, f'expected 27 per-scenario summaries, found {len(parts)}'
        pd.concat([pd.read_csv(os.path.join(events_dir, f)) for f in parts]).to_csv(summary_path, index=False)
        print('merged', len(parts))
        return
    scenarios = json.load(open(os.path.join(ROOT, 'pilot_c', 'scenarios.json')))
    chosen = select(scenarios)
    if '--index' in sys.argv:
        k = int(sys.argv[sys.argv.index('--index') + 1])
        assert 0 <= k < 27, k
        chosen = [chosen[k]]
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
        ev = events_from_trace(ls, cg, fl, home, l=s['lookback'])
        ev.to_csv(os.path.join(ROOT, 'pilot_c', 'events', tag + '_events.csv'), index=False)
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
