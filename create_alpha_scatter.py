#!/usr/bin/env python3
"""
Create pilot_c/alpha_scatter.csv from the endogenous-alpha scenario pickles.

Columns: scenario_id, rep, log_family, cost_type, cong_id, alpha_final, alpha_tavg, K, sd_iqr, mean_iqr
One row per active unit at T. alpha_tavg is the unit's mean alpha over post-burn-in snapshots (C31).

    python create_alpha_scatter.py            # endogenous-alpha block -> pilot_c/alpha_scatter.csv
    python create_alpha_scatter.py --moment   # C36: endogenous-alpha runs of the moment block
                                              #      -> pilot_c/alpha_scatter_moment.csv
"""
import sys
import csv
import gzip
import pickle
from pathlib import Path

RESULT_DIR = Path('pilot_c/results')
OUT_FILE = 'pilot_c/alpha_scatter.csv'
FIELDS = ['scenario_id', 'rep', 'log_family', 'cost_type', 'cong_id', 'alpha_final', 'alpha_tavg',
          'K', 'sd_iqr', 'mean_iqr']


def main():
    moment = '--moment' in sys.argv
    out_file = 'pilot_c/alpha_scatter_moment.csv' if moment else OUT_FILE
    rows = []
    n_scen = 0
    for fname in sorted(RESULT_DIR.glob('scenario_*.pkl.gz')):
        with gzip.open(fname, 'rb') as f:
            result = pickle.load(f)
        scenario = result.get('scenario', {})
        if moment:
            if scenario.get('block') != 'moment' or not scenario.get('alpha_endogenous', False):
                continue
        elif scenario.get('block') != 'endogenous-alpha':
            continue
        n_scen += 1
        for unit in result.get('alpha_scatter_final', []):
            rows.append({
                'scenario_id': scenario['scenario_id'],
                'rep': scenario['rep'],
                'log_family': scenario['log_family'],
                'cost_type': scenario['cost_type'],
                'cong_id': unit['cong_id'],
                'alpha_final': unit['alpha_final'],
                'alpha_tavg': unit['alpha_tavg'],
                'K': unit['K'],
                'sd_iqr': unit['sd_iqr'],
                'mean_iqr': unit['mean_iqr'],
            })

    with open(out_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {len(rows)} rows from {n_scen} endogenous-alpha scenarios to {out_file}')


if __name__ == '__main__':
    main()
