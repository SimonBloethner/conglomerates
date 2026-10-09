#!/usr/bin/env python3
"""
C23: Create alpha_scatter.csv from endogenous-α scenario pickles.
Columns: scenario_id, rep, cong_id, alpha_final, K, sd_iqr, mean_iqr
One row per active conglomerate at end of run.
"""
import gzip
import pickle
import csv
import os

RESULT_DIR = 'pilot_c/results'
OUT_FILE = '/tmp/alpha_scatter.csv'

ENDO_ALPHA_START = 1125
ENDO_ALPHA_END = 1184

rows = []

for idx in range(ENDO_ALPHA_START, ENDO_ALPHA_END + 1):
    fname = os.path.join(RESULT_DIR, f'scenario_{idx:04d}.pkl.gz')
    if not os.path.exists(fname):
        print(f'Missing: {fname}')
        continue
    
    with gzip.open(fname, 'rb') as f:
        result = pickle.load(f)
    
    scenario = result.get('scenario', {})
    alpha_scatter_final = result.get('alpha_scatter_final', [])
    
    if not alpha_scatter_final:
        print(f'No alpha_scatter_final in scenario {idx}')
        continue
    
    scenario_id = idx
    rep = scenario.get('rep', 0)
    
    for cong_data in alpha_scatter_final:
        rows.append({
            'scenario_id': scenario_id,
            'rep': rep,
            'cong_id': cong_data['cong_id'],
            'alpha_final': cong_data['alpha_final'],
            'K': cong_data['K'],
            'sd_iqr': cong_data['sd_iqr'],
            'mean_iqr': cong_data['mean_iqr'],
        })

print(f'Writing {len(rows)} rows to {OUT_FILE}')

if rows:
    fieldnames = ['scenario_id', 'rep', 'cong_id', 'alpha_final', 'K', 'sd_iqr', 'mean_iqr']
    with open(OUT_FILE, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f'Done. Unique scenarios: {len(set(r["scenario_id"] for r in rows))}')
else:
    print('No rows to write!')
