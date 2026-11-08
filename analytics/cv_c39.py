#!/usr/bin/env python3
"""
C39 Task 2: the cv block (decision_rule='cv', pilot D ids 2010-2054, results in pilot_d/results_c39/).

    python analytics/cv_c39.py --append-tidy   # append the 45 tidy rows to pilot_d/tidy.csv
    python analytics/cv_c39.py --table         # pilot_d/cv_comparison.csv (+ markdown)
    python analytics/cv_c39.py --acceptance    # pilot_d/cv_acceptance_by_K.csv (+ markdown)

--append-tidy: tidy rows of the 45 cv runs (summarize_pilot_c.create_tidy_df), written under the existing
  pilot_d/tidy.csv only if its 2,010 rows stay byte-identical.
--table: by family x alpha, medians over the 5 reps of K_mean, pooled capital share, Hill, acceptance, voluntary
  and floor exits per period, for cv (this block), moment (the pilot D main scenarios on the same seeds) and loggain
  (their pilot C twins); and the paired differences cv - moment and moment - loggain over the reps (median, p25,
  p75; paired on family, alpha, rep, i.e. on the seed).
--acceptance: acceptance and binding-gain median by K at alpha = 0.3 (main, power law, search 0.05), cv pooled over
  its 5 reps from the pickles (as analytics/acceptance_c35.py), beside the pooled moment and loggain rows of
  pilot_d/acceptance_by_K_pooled.csv and pilot_c/acceptance_by_K_pooled.csv.
"""
import io
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import pilot_design_d as pdd  # noqa: E402
from analytics.acceptance_c35 import hist_quantiles  # noqa: E402
from analytics.identity_c35 import load  # noqa: E402
from summarize_pilot_c import create_tidy_df  # noqa: E402

D = os.path.join(ROOT, 'pilot_d')
TIDY = os.path.join(D, 'tidy.csv')
RESULTS = os.path.join(D, 'results_c39')
IDS = range(pdd.TOTAL_C38, pdd.TOTAL)
METRICS = ['K_mean', 'cong_capital_share_median', 'hill_exponent_median', 'acceptance_rate',
           'voluntary_exits_per_period', 'floor_exits_per_period']
FAMILIES = ('normal', 'laplace', 't3')


def append_tidy():
    old_text = open(TIDY).read()
    old = pd.read_csv(TIDY, float_precision='round_trip')
    assert len(old) == pdd.TOTAL_C38 and old.scenario_id.max() == pdd.TOTAL_C38 - 1, len(old)
    new = create_tidy_df([load(RESULTS, i) for i in IDS])
    assert list(new.columns) == list(old.columns), set(new.columns) ^ set(old.columns)
    assert (new.decision_rule == 'cv').all() and (new.block == 'cv').all()
    buf = io.StringIO()
    pd.concat([old, new], ignore_index=True).to_csv(buf, index=False)
    text = buf.getvalue()
    assert text.startswith(old_text), 'existing rows would change'
    with open(TIDY, 'w') as f:
        f.write(text)
    print(f'appended {len(new)} rows; the {pdd.TOTAL_C38} existing rows are byte-identical')


def triples():
    """cv, moment and loggain tidy rows on the same seeds, keyed by (family, alpha, rep)."""
    t = pd.read_csv(TIDY)
    cv = t[t.block == 'cv']
    mom = t[(t.block == 'main') & (t.cost_type == 'power_law') & t.alpha.isin(pdd.CV_ALPHAS)]
    c = pd.read_csv(os.path.join(ROOT, 'pilot_c', 'tidy.csv'))
    log = c[c.scenario_id.isin(cv.twin_id.astype(int))]
    key = ['log_family', 'alpha', 'rep']
    out = {}
    for name, d in (('cv', cv), ('moment', mom), ('loggain', log)):
        assert len(d) == 45, (name, len(d))
        out[name] = d.set_index(key).sort_index()
    assert (out['moment'].seed == out['cv'].seed).all() and (out['loggain'].seed == out['cv'].seed).all()
    return out


def table():
    tr = triples()
    rows = []
    for (fam, a), _ in tr['cv'].groupby(level=['log_family', 'alpha']):
        r = dict(family=fam, alpha=a)
        for m in METRICS:
            for name in ('cv', 'moment', 'loggain'):
                r[f'{m}_{name}'] = tr[name].loc[(fam, a)][m].median()
            for lab, (x, y) in (('cv_minus_moment', ('cv', 'moment')), ('moment_minus_loggain', ('moment', 'loggain'))):
                diff = tr[x].loc[(fam, a)][m] - tr[y].loc[(fam, a)][m]
                r[f'{m}_{lab}_median'] = diff.median()
                r[f'{m}_{lab}_p25'] = diff.quantile(.25)
                r[f'{m}_{lab}_p75'] = diff.quantile(.75)
        rows.append(r)
    out = pd.DataFrame(rows)
    out['family'] = pd.Categorical(out.family, FAMILIES, ordered=True)
    out = out.sort_values(['family', 'alpha'])
    out.to_csv(os.path.join(D, 'cv_comparison.csv'), index=False)
    lab = dict(K_mean='K', cong_capital_share_median='pooled share', hill_exponent_median='Hill',
               acceptance_rate='acceptance', voluntary_exits_per_period='voluntary exits / period',
               floor_exits_per_period='floor exits / period')
    fmt = dict(K_mean='.2f', cong_capital_share_median='.3f', hill_exponent_median='.3f', acceptance_rate='.3f',
               voluntary_exits_per_period='.2f', floor_exits_per_period='.1f')
    print('| family | α | ' + ' | '.join(f'{lab[m]} cv / moment / loggain' for m in METRICS) + ' |')
    print('|---|---|' + '---|' * len(METRICS))
    for _, r in out.iterrows():
        cells = [' / '.join(f'{r[f"{m}_{n}"]:{fmt[m]}}' for n in ('cv', 'moment', 'loggain')) for m in METRICS]
        print(f'| {r.family} | {r.alpha:g} | ' + ' | '.join(cells) + ' |')
    print('\nPaired differences (median [p25, p75] over the 5 reps)\n')
    print('| family | α | ' + ' | '.join(f'{lab[m]}: cv − moment | moment − loggain' for m in METRICS[:4]) + ' |')
    print('|---|---|' + '---|---|' * 4)
    for _, r in out.iterrows():
        cells = []
        for m in METRICS[:4]:
            for d in ('cv_minus_moment', 'moment_minus_loggain'):
                cells.append(f'{r[f"{m}_{d}_median"]:+{fmt[m]}} [{r[f"{m}_{d}_p25"]:+{fmt[m]}}, '
                             f'{r[f"{m}_{d}_p75"]:+{fmt[m]}}]')
        print(f'| {r.family} | {r.alpha:g} | ' + ' | '.join(cells) + ' |')


def acceptance():
    rows = []
    hists = {}
    for i in IDS:
        res = load(RESULTS, i)
        s = res['scenario']
        if s['alpha'] != 0.3:
            continue
        fam = s['log_family']
        P, Mg, H = res['proposals_by_K'], res['mergers_by_K'], res['binding_gain_hist_by_K']
        if fam not in hists:
            hists[fam] = [np.zeros_like(P), np.zeros_like(Mg), np.zeros_like(H), res['binding_gain_edges']]
        hists[fam][0] += P
        hists[fam][1] += Mg
        hists[fam][2] += H
    for fam, (P, Mg, H, edges) in hists.items():
        for K in range(2, 13):
            q25, q50, q75 = hist_quantiles(H[K], edges)
            rows.append(dict(family=fam, K=K, rule='cv', proposals=int(P[K]), mergers=int(Mg[K]),
                             acceptance=Mg[K] / P[K] if P[K] else np.nan, binding_median=q50))
    for rule, path in (('moment', os.path.join(D, 'acceptance_by_K_pooled.csv')),
                       ('loggain', os.path.join(ROOT, 'pilot_c', 'acceptance_by_K_pooled.csv'))):
        p = pd.read_csv(path)
        sel = (p.block == 'main') & (p.alpha == 0.3) & (p.p == 0.05) & p.K.between(2, 12)
        if 'decision_rule' in p:
            sel &= p.decision_rule == rule
        for _, r in p[sel].iterrows():
            rows.append(dict(family=r.family, K=int(r.K), rule=rule, proposals=int(r.proposals),
                             mergers=int(r.mergers), acceptance=r.acceptance, binding_median=r.binding_median))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(D, 'cv_acceptance_by_K.csv'), index=False)
    for fam in FAMILIES:
        q = out[out.family == fam].pivot_table(index='K', columns='rule', values=['acceptance', 'binding_median'])
        print(f'\n**{fam}** (α = 0.3)\n')
        print('| K | acceptance cv / moment / loggain | binding median cv / moment / loggain |')
        print('|---|---|---|')
        for K, r in q.iterrows():
            acc = ' / '.join(f'{r[("acceptance", n)]:.3f}' for n in ('cv', 'moment', 'loggain'))
            bm = ' / '.join(f'{r[("binding_median", n)]:.1e}' for n in ('cv', 'moment', 'loggain'))
            print(f'| {K} | {acc} | {bm} |')


if __name__ == '__main__':
    if '--append-tidy' in sys.argv:
        append_tidy()
    elif '--table' in sys.argv:
        table()
    elif '--acceptance' in sys.argv:
        acceptance()
    else:
        sys.exit(__doc__)
