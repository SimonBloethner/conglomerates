#!/usr/bin/env python3
"""
C42 Task 2: the accept block (decision_rule='accept', pilot D ids 2055-2099, results in pilot_d/results_c42/), the
survival-feasible ceiling. A control: every feasible proposal is accepted and no firm leaves voluntarily.

    python analytics/accept_c42.py --append-tidy                # append the 45 tidy rows to pilot_d/tidy.csv
    python analytics/accept_c42.py --table                      # pilot_d/accept_comparison.csv (+ markdown)
    python analytics/accept_c42.py --acceptance                 # pilot_d/accept_acceptance_by_K.csv (+ markdown)
    python analytics/accept_c42.py --composition [MOMENT_DIR]   # pilot_d/accept_composition.csv (+ markdown)
    python analytics/accept_c42.py --decomposition              # pilot_d/consent_gap_decomposition_accept.csv

--append-tidy: tidy rows of the 45 accept runs (summarize_pilot_c.create_tidy_df), written under the existing
  pilot_d/tidy.csv only if its 2,055 rows stay byte-identical.
--table: by family x alpha, medians over the 5 reps of K, pooled share, Hill, within-market HHI, floor exits, mergers
  and voluntary exits per period, for accept (this block), moment (the pilot D main scenarios on the same seeds) and
  loggain (their pilot C twins); paired differences accept - moment (median, p25, p75 over the reps).
--acceptance: proposals, mergers and acceptance by K (2..M) pooled over the reps, accept from the pickles, moment
  and loggain from pilot_d / pilot_c acceptance_by_K_pooled.csv (main, power law, search 0.05).
--composition: the accepted proposals (post-burn-in mergers, *_by_K_w_type) of accept and moment, pooled over the
  reps, per period: by type (solo+solo, solo+unit, unit+unit) and by entrant class (a floor-sized standalone; else
  the entrant's capital share w of the merged unit: <= 0.10, 0.10-0.25, 0.25-0.50, > 0.50); the excess accept -
  moment; and the moment rule's refusals (proposals - mergers) by the same classes. MOMENT_DIR (default
  pilot_d/results_c42_identity) holds the moment pickles with the C39 keys.
--decomposition: Table 3c with K_accept: K* - K_accept (environment: attrition at the floor, one firm per market),
  K_accept - K_hom (consent on recruit size and noise), K_hom - K (transfer conflict), K - K_loggain (verifiability).
"""
import io
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import pilot_design_d as pdd  # noqa: E402
from analytics.identity_c35 import load  # noqa: E402

D = os.path.join(ROOT, 'pilot_d')
TIDY = os.path.join(D, 'tidy.csv')
RESULTS = os.path.join(D, 'results_c42')
IDS = range(pdd.TOTAL_C39, pdd.TOTAL)
METRICS = ['K_mean', 'cong_capital_share_median', 'hill_exponent_median', 'hhi_within_median',
           'floor_exits_per_period', 'mergers_per_period', 'voluntary_exits_per_period', 'acceptance_rate',
           'floor_hit_rate_member_status', 'member_firm_share']
LAB = dict(K_mean='K', cong_capital_share_median='pooled share', hill_exponent_median='Hill',
           hhi_within_median='HHI within', floor_exits_per_period='floor exits / period',
           mergers_per_period='mergers / period', voluntary_exits_per_period='voluntary exits / period',
           acceptance_rate='acceptance', floor_hit_rate_member_status='member floor-hit rate',
           member_firm_share='share of firms in units')
FMT = dict(K_mean='.2f', cong_capital_share_median='.3f', hill_exponent_median='.3f', hhi_within_median='.4f',
           floor_exits_per_period='.1f', mergers_per_period='.1f', voluntary_exits_per_period='.2f',
           acceptance_rate='.3f', floor_hit_rate_member_status='.4f', member_firm_share='.3f')
FAMILIES = ('normal', 'laplace', 't3')
RULES = ('accept', 'moment', 'loggain')
TYPES = ('solo+solo', 'solo+unit', 'unit+unit')
# entrant classes over the 21 columns of *_by_K_w (0 = floor-sized standalone, 1 + bin of w in steps of 0.05)
CLASSES = [('floor', [0]), ('w<=0.10', [1, 2]), ('0.10-0.25', [3, 4, 5]), ('0.25-0.50', list(range(6, 11))),
           ('w>0.50', list(range(11, 21)))]


def append_tidy():
    from summarize_pilot_c import create_tidy_df
    old_text = open(TIDY).read()
    old = pd.read_csv(TIDY, float_precision='round_trip')
    assert len(old) == pdd.TOTAL_C39 and old.scenario_id.max() == pdd.TOTAL_C39 - 1, len(old)
    new = create_tidy_df([load(RESULTS, i) for i in IDS])
    assert list(new.columns) == list(old.columns), set(new.columns) ^ set(old.columns)
    assert (new.decision_rule == 'accept').all() and (new.block == 'accept').all()
    buf = io.StringIO()
    pd.concat([old, new], ignore_index=True).to_csv(buf, index=False)
    text = buf.getvalue()
    assert text.startswith(old_text), 'existing rows would change'
    with open(TIDY, 'w') as f:
        f.write(text)
    print(f'appended {len(new)} rows; the {pdd.TOTAL_C39} existing rows are byte-identical')


def triples():
    """accept, moment and loggain tidy rows on the same seeds, keyed by (family, alpha, rep)."""
    t = pd.read_csv(TIDY)
    acc = t[t.block == 'accept']
    mom = t[(t.block == 'main') & (t.cost_type == 'power_law') & t.alpha.isin(pdd.ACCEPT_ALPHAS)]
    c = pd.read_csv(os.path.join(ROOT, 'pilot_c', 'tidy.csv'))
    log = c[c.scenario_id.isin(acc.twin_id.astype(int))].copy()
    log['member_firm_share'] = np.nan
    log['floor_hit_rate_member_status'] = np.nan
    key = ['log_family', 'alpha', 'rep']
    out = {}
    for name, d in (('accept', acc), ('moment', mom), ('loggain', log)):
        assert len(d) == 45, (name, len(d))
        out[name] = d.set_index(key).sort_index()
    assert (out['moment'].seed == out['accept'].seed).all() and (out['loggain'].seed == out['accept'].seed).all()
    return out


def fam_sort(df):
    df['family'] = pd.Categorical(df.family, FAMILIES, ordered=True)
    return df.sort_values(['family', 'alpha']).reset_index(drop=True)


def table():
    tr = triples()
    rows = []
    for (fam, a), _ in tr['accept'].groupby(level=['log_family', 'alpha']):
        r = dict(family=fam, alpha=a)
        for m in METRICS:
            for name in RULES:
                r[f'{m}_{name}'] = tr[name].loc[(fam, a)][m].median()
            diff = tr['accept'].loc[(fam, a)][m] - tr['moment'].loc[(fam, a)][m]
            r[f'{m}_accept_minus_moment_median'] = diff.median()
            r[f'{m}_accept_minus_moment_p25'] = diff.quantile(.25)
            r[f'{m}_accept_minus_moment_p75'] = diff.quantile(.75)
        rows.append(r)
    out = fam_sort(pd.DataFrame(rows))
    out.to_csv(os.path.join(D, 'accept_comparison.csv'), index=False)
    shown = METRICS[:7]
    print('| family | α | ' + ' | '.join(f'{LAB[m]} accept / moment / loggain' for m in shown) + ' |')
    print('|---|---|' + '---|' * len(shown))
    for _, r in out.iterrows():
        cells = [' / '.join(f'{r[f"{m}_{n}"]:{FMT[m]}}' for n in RULES) for m in shown]
        print(f'| {r.family} | {r.alpha:g} | ' + ' | '.join(cells) + ' |')
    print('\nPaired differences accept − moment (median [p25, p75] over the 5 reps)\n')
    shown = METRICS[:6] + ['floor_hit_rate_member_status', 'member_firm_share']
    print('| family | α | ' + ' | '.join(LAB[m] for m in shown) + ' |')
    print('|---|---|' + '---|' * len(shown))
    for _, r in out.iterrows():
        cells = [f'{r[f"{m}_accept_minus_moment_median"]:+{FMT[m]}} [{r[f"{m}_accept_minus_moment_p25"]:+{FMT[m]}}, '
                 f'{r[f"{m}_accept_minus_moment_p75"]:+{FMT[m]}}]' for m in shown]
        print(f'| {r.family} | {r.alpha:g} | ' + ' | '.join(cells) + ' |')
    print('\nMember floor-hit rate (per member-period) and share of firms in units, accept / moment\n')
    print('| family | α | member floor-hit rate | share of firms in units |')
    print('|---|---|---|---|')
    for _, r in out.iterrows():
        print(f'| {r.family} | {r.alpha:g} | ' + ' | '.join(
            f'{r[f"{m}_accept"]:{FMT[m]}} / {r[f"{m}_moment"]:{FMT[m]}}'
            for m in ('floor_hit_rate_member_status', 'member_firm_share')) + ' |')


def acceptance():
    rows = []
    pooled = {}
    for i in IDS:
        res = load(RESULTS, i)
        s = res['scenario']
        k = (s['log_family'], s['alpha'])
        P, Mg = res['proposals_by_K'], res['mergers_by_K']
        if k not in pooled:
            pooled[k] = [np.zeros_like(P), np.zeros_like(Mg)]
        pooled[k][0] += P
        pooled[k][1] += Mg
    for (fam, a), (P, Mg) in pooled.items():
        for K in range(2, len(P)):
            rows.append(dict(family=fam, alpha=a, K=K, rule='accept', proposals=int(P[K]), mergers=int(Mg[K]),
                             acceptance=Mg[K] / P[K] if P[K] else np.nan))
    for rule, path in (('moment', os.path.join(D, 'acceptance_by_K_pooled.csv')),
                       ('loggain', os.path.join(ROOT, 'pilot_c', 'acceptance_by_K_pooled.csv'))):
        p = pd.read_csv(path)
        sel = (p.block == 'main') & p.alpha.isin(pdd.ACCEPT_ALPHAS) & (p.p == 0.05)
        if 'decision_rule' in p:
            sel &= p.decision_rule == rule
        for _, r in p[sel].iterrows():
            rows.append(dict(family=r.family, alpha=r.alpha, K=int(r.K), rule=rule, proposals=int(r.proposals),
                             mergers=int(r.mergers), acceptance=r.acceptance))
    out = pd.DataFrame(rows)
    out = out[out.proposals > 0]
    out.to_csv(os.path.join(D, 'accept_acceptance_by_K.csv'), index=False)
    n_per = 5 * (11000 - 8000)
    for a in pdd.ACCEPT_ALPHAS:
        print(f'\n**α = {a:g}**: mergers per period by K (accept / moment / loggain) and acceptance (moment / loggain)\n')
        print('| K | ' + ' | '.join(f'{f} mergers' for f in FAMILIES) + ' | ' +
              ' | '.join(f'{f} acceptance' for f in FAMILIES) + ' |')
        print('|---|' + '---|' * 6)
        q = out[out.alpha == a]
        for K in range(2, 21):
            cells, acc = [], []
            for fam in FAMILIES:
                def g(rule, col):
                    x = q[(q.family == fam) & (q.K == K) & (q.rule == rule)]
                    return x[col].iloc[0] if len(x) else np.nan
                cells.append(' / '.join(f'{g(rl, "mergers") / n_per:.2f}' if not np.isnan(g(rl, 'mergers')) else '–'
                                        for rl in RULES))
                acc.append(' / '.join(f'{g(rl, "acceptance"):.2f}' if not np.isnan(g(rl, 'acceptance')) else '–'
                                      for rl in ('moment', 'loggain')))
            print(f'| {K} | ' + ' | '.join(cells + acc) + ' |')


def comp_counts(results_dir, ids):
    """Post-burn-in proposals and mergers by type and entrant class, summed over the given scenarios; per period."""
    P = M = None
    n_periods = 0
    for i in ids:
        res = load(results_dir, i)
        s = res['scenario']
        p, m = res['proposals_by_K_w_type'], res['mergers_by_K_w_type']
        P = p.copy() if P is None else P + p
        M = m.copy() if M is None else M + m
        n_periods += s['T'] - s['burn_in']
    return P, M, n_periods


def classes(arr):
    """(K, w, type) counts -> {(type, class): count} and margins."""
    out = {}
    for ti, t in enumerate(TYPES):
        for c, cols in CLASSES:
            out[(t, c)] = arr[:, cols, ti].sum()
    return out


def composition(moment_dir):
    scen = pdd.generate_all_scenarios()
    rows = []
    for fam in FAMILIES:
        for a in pdd.ACCEPT_ALPHAS:
            acc_ids = [s['scenario_id'] for s in scen if s['block'] == 'accept' and s['log_family'] == fam
                       and s['alpha'] == a]
            mom_ids = [s['scenario_id'] for s in scen if s['block'] == 'main' and s['cost_type'] == 'power_law'
                       and s['log_family'] == fam and s['alpha'] == a]
            assert len(acc_ids) == len(mom_ids) == 5
            Pa, Ma, na = comp_counts(RESULTS, acc_ids)
            Pm, Mm, nm = comp_counts(moment_dir, mom_ids)
            assert np.array_equal(Pa, Ma), 'accept: every proposal is a merger'
            ca, cm, cp = classes(Ma), classes(Mm), classes(Pm)
            for (t, c) in ca:
                rows.append(dict(family=fam, alpha=a, type=t, entrant=c, accept_mergers_pp=ca[(t, c)] / na,
                                 moment_mergers_pp=cm[(t, c)] / nm, moment_proposals_pp=cp[(t, c)] / nm,
                                 moment_refused_pp=(cp[(t, c)] - cm[(t, c)]) / nm))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(D, 'accept_composition.csv'), index=False)
    for a in pdd.ACCEPT_ALPHAS:
        q = out[out.alpha == a]
        print(f'\n**α = {a:g}: accepted proposals by type**, per period (share of all mergers): accept / moment; '
              'moment refusals per period\n')
        print('| family | ' + ' | '.join(TYPES) + ' | refused by moment: ' + ' / '.join(TYPES) + ' |')
        print('|---|---|---|---|---|')
        for fam in FAMILIES:
            x = q[q.family == fam].groupby('type')[['accept_mergers_pp', 'moment_mergers_pp', 'moment_refused_pp']].sum()
            ta, tm = x.accept_mergers_pp.sum(), x.moment_mergers_pp.sum()
            cells = [f'{x.loc[t].accept_mergers_pp:.2f} ({x.loc[t].accept_mergers_pp / ta:.0%}) / '
                     f'{x.loc[t].moment_mergers_pp:.2f} ({x.loc[t].moment_mergers_pp / tm:.0%})' for t in TYPES]
            cells.append(' / '.join(f'{x.loc[t].moment_refused_pp:.2f}' for t in TYPES))
            print(f'| {fam} | ' + ' | '.join(cells) + ' |')
        print(f'\n**α = {a:g}: entrant class of the accepted solo entries** (solo+solo and solo+unit; share of those '
              'mergers) accept / moment, and unit+unit by entrant share\n')
        print('| family | ' + ' | '.join(c for c, _ in CLASSES) + ' | unit+unit: ' +
              ' / '.join(c for c, _ in CLASSES[1:]) + ' |')
        print('|---|' + '---|' * (len(CLASSES) + 1))
        for fam in FAMILIES:
            x = q[q.family == fam]
            solo = x[x.type != 'unit+unit'].groupby('entrant', sort=False)[['accept_mergers_pp', 'moment_mergers_pp']].sum()
            uu = x[x.type == 'unit+unit'].set_index('entrant')
            cells = [f'{solo.loc[c].accept_mergers_pp / solo.accept_mergers_pp.sum():.0%} / '
                     f'{solo.loc[c].moment_mergers_pp / solo.moment_mergers_pp.sum():.0%}' for c, _ in CLASSES]
            ua, um = uu.accept_mergers_pp.sum(), uu.moment_mergers_pp.sum()
            cells.append(' / '.join(f'{uu.loc[c].accept_mergers_pp / ua:.0%}' for c, _ in CLASSES[1:]) + ' vs ' +
                         ' / '.join(f'{uu.loc[c].moment_mergers_pp / um:.0%}' if um else '–' for c, _ in CLASSES[1:]))
            print(f'| {fam} | ' + ' | '.join(cells) + ' |')
        print(f'\n**α = {a:g}: excess mergers per period, accept − moment**, by type and (solo entries) entrant class\n')
        print('| family | total | ' + ' | '.join(TYPES) + ' | solo entries: floor | w ≤ 0.10 | 0.10–0.25 | > 0.25 |')
        print('|---|---|---|---|---|---|---|---|---|')
        for fam in FAMILIES:
            x = q[q.family == fam].copy()
            x['ex'] = x.accept_mergers_pp - x.moment_mergers_pp
            by_t = x.groupby('type').ex.sum()
            solo = x[x.type != 'unit+unit'].groupby('entrant').ex.sum()
            big = solo.get('0.25-0.50', 0) + solo.get('w>0.50', 0)
            print(f'| {fam} | {x.ex.sum():+.2f} | ' + ' | '.join(f'{by_t[t]:+.2f}' for t in TYPES) +
                  f' | {solo["floor"]:+.2f} | {solo["w<=0.10"]:+.2f} | {solo["0.10-0.25"]:+.2f} | {big:+.2f} |')


def decomposition():
    x = pd.read_csv(os.path.join(D, 'consent_gap_decomposition.csv'))
    acc = pd.read_csv(os.path.join(D, 'accept_comparison.csv'))[['family', 'alpha', 'K_mean_accept']]
    d = x.merge(acc, on=['family', 'alpha'], how='inner')
    assert len(d) == len(x[x.alpha.isin(pdd.ACCEPT_ALPHAS)]), len(d)
    d = d.rename(columns={'K_mean_accept': 'K_accept'})
    d['environment'] = d.K_star - d.K_accept
    d['consent_size_noise'] = d.K_accept - d.K_mom_hom
    d['transfer'] = d.b_transfer
    d['verifiability'] = d.c_verif_het
    assert np.allclose(d.environment + d.consent_size_noise + d.transfer + d.verifiability, d.total)
    assert np.allclose(d.environment + d.consent_size_noise, d.a_attrition)
    cols = ['family', 'alpha', 'K_star', 'K_accept', 'K_mom_hom', 'K_mom_het', 'K_log_het', 'total', 'environment',
            'consent_size_noise', 'transfer', 'verifiability', 'a_attrition']
    d = fam_sort(d[cols].copy())
    d.to_csv(os.path.join(D, 'consent_gap_decomposition_accept.csv'), index=False)
    print('| family | α | K* | K_accept | K_hom | K | K_loggain | environment K* − K_accept | consent (size, noise) '
          'K_accept − K_hom | transfer K_hom − K | verifiability K − K_loggain |')
    print('|---|---|---|---|---|---|---|---|---|---|---|')
    for r in d.itertuples():
        t = r.total
        print(f'| {r.family} | {r.alpha:g} | {r.K_star:.0f} | {r.K_accept:.2f} | {r.K_mom_hom:.2f} | {r.K_mom_het:.2f} | '
              f'{r.K_log_het:.2f} | {r.environment:.1f} ({r.environment / t:.0%}) | '
              f'{r.consent_size_noise:.1f} ({r.consent_size_noise / t:.0%}) | {r.transfer:.1f} ({r.transfer / t:.0%}) | '
              f'{r.verifiability:.1f} ({r.verifiability / t:.0%}) |')


if __name__ == '__main__':
    if '--append-tidy' in sys.argv:
        append_tidy()
    elif '--table' in sys.argv:
        table()
    elif '--acceptance' in sys.argv:
        acceptance()
    elif '--composition' in sys.argv:
        k = sys.argv.index('--composition')
        composition(sys.argv[k + 1] if len(sys.argv) > k + 1 else os.path.join(D, 'results_c42_identity'))
    elif '--decomposition' in sys.argv:
        decomposition()
    else:
        sys.exit(__doc__)
