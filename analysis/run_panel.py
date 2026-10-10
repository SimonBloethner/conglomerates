"""
Status panel (analysis/status_panel.py) on the 27 traced scenarios.

Usage (from the repository root):
    python analysis/run_panel.py <trace_dir> --index K    # scenario K (0..26), for a SLURM array
    python analysis/run_panel.py <trace_dir> --merge      # merge the 27 per-scenario summaries
Reads <trace_dir>/<family>_a<alpha>_r<rep>.npz (written by run_traces.py; must contain `jump`),
writes pilot_c/panel/<tag>_summary.csv and, on --merge, pilot_c/status_panel_summary.csv.
"""
import sys, os, json, time
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'analysis'))
from run_traces import select                      # noqa: E402
from status_panel import panel_from_trace, summarize  # noqa: E402


def main():
    trace_dir = sys.argv[1]
    out_dir = os.path.join(ROOT, 'pilot_c', 'panel')
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(ROOT, 'pilot_c', 'status_panel_summary.csv')
    if '--merge' in sys.argv:
        parts = sorted(f for f in os.listdir(out_dir) if f.endswith('_summary.csv'))
        assert len(parts) == 27, f'expected 27 per-scenario summaries, found {len(parts)}'
        pd.concat([pd.read_csv(os.path.join(out_dir, f)) for f in parts]).to_csv(summary_path, index=False)
        print('merged', len(parts))
        return
    scenarios = json.load(open(os.path.join(ROOT, 'pilot_c', 'scenarios.json')))
    chosen = select(scenarios)
    k = int(sys.argv[sys.argv.index('--index') + 1])
    assert 0 <= k < 27, k
    s = chosen[k]
    tag = f"{s['log_family']}_a{s['alpha']}_r{s['rep']}"
    z = np.load(os.path.join(trace_dir, tag + '.npz'))
    assert 'jump' in z.files, 'trace has no jump key; regenerate with the C29 model'
    t0 = time.time()
    df = panel_from_trace(z['logshare'], z['cong'], z['floor'], z['home'], z['jump'], burn_in=s['burn_in'])
    row = dict(family=s['log_family'], alpha=s['alpha'], rep=s['rep'], scenario_id=s['scenario_id'],
               T=z['logshare'].shape[0] - 1, burn_in=s['burn_in'], n_rows=len(df))
    row.update(summarize(df))
    row['seconds'] = round(time.time() - t0, 1)
    pd.DataFrame([row]).to_csv(os.path.join(out_dir, tag + '_summary.csv'), index=False)
    print(tag, {k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()
                if k in ('n_rows', 'diff_fe_mean', 'diff_fe_median', 'diff_K2_mean', 'diff_K5p_mean', 'seconds')},
          flush=True)


if __name__ == '__main__':
    main()
