#!/usr/bin/env python3
"""
C35: scenario-id lists for the reruns (written to pilot_c/lists_c35/).

- task2.txt     main block, power law, 3 families, alpha in {0.1, 0.3, 0.5}, 5 reps (45) and the search block
                at p = 1 for the same cells (45): 90 runs, to populate the acceptance-by-K counters;
- identity.txt  one scenario of every other cell of the C34 design (cells 0..61 not covered by task2.txt):
                the alpha closest to 0.3 (the start alpha in the endogenous-alpha block), rep = cell_id % 5;
- cost_high.txt the cost-level-high block, ids 1800-1889.
task2.txt and identity.txt run to pilot_c/results_c35 and are compared with the C34 pickles
(analytics/identity_c35.py); cost_high.txt runs to pilot_c/results.
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    sc = json.load(open(os.path.join(ROOT, 'pilot_c', 'scenarios.json')))
    task2 = [s['scenario_id'] for s in sc
             if s['cost_type'] == 'power_law' and s['alpha'] in (0.1, 0.3, 0.5)
             and ((s['block'] == 'main') or (s['block'] == 'search' and s['merge_thresh'] == 1.0))]
    covered = {s['cell_id'] for s in sc if s['scenario_id'] in set(task2)}
    ident = []
    for cid in sorted({s['cell_id'] for s in sc if s['scenario_id'] < 1800} - covered):
        cell = [s for s in sc if s['cell_id'] == cid]
        a = min({s['alpha'] for s in cell}, key=lambda x: (abs(x - 0.3), x))
        pick = [s for s in cell if s['alpha'] == a and s['rep'] == cid % 5]
        ident.append(pick[0]['scenario_id'])
    high = [s['scenario_id'] for s in sc if s['block'] == 'cost-level-high']
    out = os.path.join(ROOT, 'pilot_c', 'lists_c35')
    os.makedirs(out, exist_ok=True)
    for name, ids in (('task2', task2), ('identity', ident), ('cost_high', high)):
        with open(os.path.join(out, name + '.txt'), 'w') as f:
            f.write('\n'.join(map(str, ids)) + '\n')
        blocks = sorted({sc[i]['block'] for i in ids})
        print(f'{name}: {len(ids)} runs, blocks {blocks}')
    print('covered cells', sorted(covered), 'identity cells', len(ident))


if __name__ == '__main__':
    main()
