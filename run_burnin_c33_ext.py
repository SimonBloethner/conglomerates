#!/usr/bin/env python3
"""
C33: the burn-in runs of run_burnin_c33.py extended to T = 16000 (same scenarios, same seeds).

Follows the stop in diagnostics/deviations.md. Shocks are drawn step by step, so each run repeats
its pilot run up to step 11000; run_one checks this against the stored result.

Decision rule, fixed before these runs:
- Compare each headline series over the pilot window (8000, 11000] with (11000, 16000].
- burn_in = 8000 stands if the two means differ by at most 2% in every cell, and the t3 capital-share
  trend over (8000, 16000] implies a change of less than 2% over those 8000 steps.

Usage:
    python run_burnin_c33_ext.py <run_index>   # one run, 0-19 (SLURM array)
    python run_burnin_c33_ext.py --report      # csv, figures and markdown from the saved runs
"""
import csv
import gzip
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from pilot_design import BURN_IN, N_REPS, T
from run_burnin_c14 import compute_convergence_step
from run_burnin_c33 import (CELLS, FINAL_WINDOW, HEADLINE, SERIES, TOLERANCE, WINDOW, Z_THRESHOLD, analyse_series,
                            burnin_scenarios, cell_label, extract_series, fmt_conv, rolling_max_deviation,
                            run_scenario)

T_EXT = 16000
RUN_DIR = Path('diagnostics/burn_in_c33_ext_runs')

MAX_WINDOW_DIFF = 2.0  # percent
MAX_TREND_CHANGE = 2.0  # percent over (BURN_IN, T_EXT]
N_SURROGATES = 300
SURROGATE_BLOCK = 10  # observations


def prefix_matches_pilot(result, scenario):
    """True if the first T steps of this run reproduce the stored pilot result."""
    path = Path('pilot_c/results') / f"scenario_{scenario['scenario_id']:04d}.pkl.gz"
    with gzip.open(path, 'rb') as f:
        pilot = pickle.load(f)
    hp = result[-1]
    n_obs = T // scenario['metric_every']
    return (np.array_equal(hp['hill_exponent'][:n_obs], pilot['hill_exponent'], equal_nan=True)
            and np.array_equal(hp['hhi_within'][:, :n_obs], pilot['hhi_within'], equal_nan=True)
            and np.array_equal(hp['cong_capital_share'][:n_obs], pilot['cong_capital_share'], equal_nan=True)
            and np.array_equal(hp['floor_hits_by_status'][:T], pilot['floor_hits_by_status']))


def run_one(idx):
    """Run scenario idx of burnin_scenarios() to T_EXT and save its series."""
    scenario = dict(burnin_scenarios()[idx], T=T_EXT)
    print(f"Run {idx}: scenario {scenario['scenario_id']}, {scenario['log_family']}, "
          f"alpha={scenario['alpha']}, rep={scenario['rep']}, seed={scenario['seed']}, T={T_EXT}", flush=True)

    result = run_scenario(scenario)
    series = extract_series(result, scenario['metric_every'])
    identical = prefix_matches_pilot(result, scenario)
    print(f"First {T} steps identical to pilot result: {identical}", flush=True)

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(RUN_DIR / f'run_{idx:02d}.npz', identical=identical, **series)


def window_comparison(runs, steps):
    """Mean over (T, T_EXT] against (BURN_IN, T]: difference in % of the pilot-window mean, 95% CI across reps."""
    pilot = runs[:, (steps > BURN_IN) & (steps <= T)].mean(axis=1)
    later = runs[:, steps > T].mean(axis=1)
    base = pilot.mean()
    diffs = (later - pilot) / base * 100
    half_width = stats.t.ppf(0.975, len(diffs) - 1) * diffs.std(ddof=1) / np.sqrt(len(diffs))
    return {'pilot_mean': base, 'later_mean': later.mean(), 'diff': diffs.mean(),
            'ci_low': diffs.mean() - half_width, 'ci_high': diffs.mean() + half_width}


def trend_after_burn_in(runs, steps):
    """OLS slope over (BURN_IN, T_EXT], rep by rep, in % of the cell mean per 1000 steps."""
    post = steps > BURN_IN
    base = runs[:, post].mean()
    slopes = np.array([np.polyfit(steps[post], run[post], 1)[0] for run in runs]) * 1000 / base * 100
    return {'slope': slopes.mean(), 'p': stats.ttest_1samp(slopes, 0.0).pvalue,
            'implied_change': slopes.mean() * (T_EXT - BURN_IN) / 1000}


def keeps_burn_in(windows, t3_ccs_trend):
    """Decision rule from the module docstring."""
    return (all(abs(w['diff']) <= MAX_WINDOW_DIFF for w in windows)
            and abs(t3_ccs_trend['implied_change']) < MAX_TREND_CHANGE)


def stationary_surrogate(run, n_obs, rng):
    """Circular blocks resampled from one rep's post-burn-in window."""
    post = run[BURN_IN // 100:]
    idx = []
    while len(idx) < n_obs:
        start = rng.integers(len(post))
        idx.extend((start + j) % len(post) for j in range(SURROGATE_BLOCK))
    return post[np.array(idx[:n_obs])]


def placebo(runs, metric_every, rng):
    """Share of stationary surrogates that pass by burn-in; share deviating at least as much as observed."""
    observed = rolling_max_deviation(runs.mean(axis=0), BURN_IN, metric_every)
    passed, exceed = 0, 0
    for _ in range(N_SURROGATES):
        series = np.stack([stationary_surrogate(run, runs.shape[1], rng) for run in runs]).mean(axis=0)
        conv = compute_convergence_step(series, window=WINDOW, tolerance=TOLERANCE,
                                        final_window=FINAL_WINDOW, metric_every=metric_every)
        passed += conv <= BURN_IN
        exceed += rolling_max_deviation(series, BURN_IN, metric_every) >= observed
    return {'pass_share': passed / N_SURROGATES, 'exceed_share': exceed / N_SURROGATES}


def report():
    """Write csv, figures and markdown from the saved runs."""
    scenarios = burnin_scenarios()
    metric_every = scenarios[0]['metric_every']
    n_obs = T_EXT // metric_every
    steps = np.arange(1, n_obs + 1) * metric_every
    rng = np.random.default_rng(33)

    loaded = [np.load(RUN_DIR / f'run_{i:02d}.npz') for i in range(len(scenarios))]
    n_identical = sum(bool(run['identical']) for run in loaded)

    cells = {}
    for c, cell in enumerate(CELLS):
        cells[cell] = {key: np.stack([loaded[c * N_REPS + r][key] for r in range(N_REPS)])
                       for key, _, _ in SERIES}
    verdicts = {cell: {key: analyse_series(cells[cell][key], metric_every) for key, _, _ in SERIES}
                for cell in CELLS}

    labels = {key: label for key, label, _ in SERIES}
    rows = []
    for cell in CELLS:
        for key, _, _ in SERIES:
            if key in HEADLINE and verdicts[cell][key]['applicable']:
                runs = cells[cell][key]
                rows.append({'cell': cell, 'key': key, 'window': window_comparison(runs, steps),
                             'trend': trend_after_burn_in(runs, steps),
                             'placebo': placebo(runs, metric_every, rng)})

    t3_ccs_trend = next(r['trend'] for r in rows if r['cell'] == ('t3', 0.3) and r['key'] == 'ccs')
    keep = keeps_burn_in([r['window'] for r in rows], t3_ccs_trend)

    csv_path = 'diagnostics/burn_in_c33_ext.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['log_family', 'alpha', 'rep', 'seed', 'step'] + [key for key, _, _ in SERIES])
        for c, cell in enumerate(CELLS):
            for r in range(N_REPS):
                seed = scenarios[c * N_REPS + r]['seed']
                for i, step in enumerate(steps):
                    writer.writerow([cell[0], cell[1], r, seed, step]
                                    + [cells[cell][key][r, i] for key, _, _ in SERIES])
    print(f"Wrote {csv_path}")

    for cell in CELLS:
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        fig.suptitle(f'C33 extended burn-in: {cell[0]}, power_law, alpha={cell[1]:g}, T={T_EXT}', fontsize=12)
        for (key, label, _), ax in zip(SERIES, axes.flat):
            if verdicts[cell][key]['applicable']:
                with np.errstate(invalid='ignore'):
                    ax.plot(steps, np.nanmean(cells[cell][key], axis=0), 'b-', linewidth=1)
            ax.axvline(BURN_IN, color='k', linestyle=':', label=f'burn_in={BURN_IN}')
            ax.axvline(T, color='r', linestyle=':', label=f'pilot T={T}')
            ax.set_xlabel('Step')
            ax.set_title(label)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig_path = f'diagnostics/figures/burnin_c33_ext_{cell_label(*cell)}.png'
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"Wrote {fig_path}")

    lines = [
        '# C33: Burn-in extended to T = 16000',
        '',
        'Follows `burn_in_c33.md`. Same 20 scenarios and seeds, T = 16000.',
        f'{n_identical} of {len(scenarios)} runs reproduce the stored pilot result up to step {T} exactly.',
        '',
        '## Decision rule (fixed before the runs)',
        '',
        f'burn_in = {BURN_IN} stands if the mean of each headline series over ({T}, {T_EXT}] is within '
        f'{MAX_WINDOW_DIFF:g}% of its mean over the pilot window ({BURN_IN}, {T}] in every cell, and the t3 '
        f'capital-share trend over ({BURN_IN}, {T_EXT}] implies a change of less than {MAX_TREND_CHANGE:g}% '
        f'over those {T_EXT - BURN_IN} steps.',
        '',
        f"**Result: burn_in = {BURN_IN} {'stands' if keep else 'DOES NOT stand'}.** "
        f"t3 capital-share trend: {t3_ccs_trend['slope']:+.3f}% per 1000 steps (p = {t3_ccs_trend['p']:.3f}), "
        f"implied change {t3_ccs_trend['implied_change']:+.2f}%.",
        '',
        '## Headline series',
        '',
        f'Difference: ({T}, {T_EXT}] against ({BURN_IN}, {T}], % of the pilot-window mean, 95% CI across {N_REPS} reps. '
        f'Trend: OLS over ({BURN_IN}, {T_EXT}], rep by rep. Placebo: {N_SURROGATES} stationary surrogates '
        f'block-resampled from each rep\'s post-{BURN_IN} window (80 observations).',
        '',
        '| Cell | Series | Pilot window | Later | Difference % | 95% CI | Trend %/1000 | p | Implied change % | Surrogates passing | P(dev ≥ observed) |',
        '|------|--------|--------------|-------|--------------|--------|--------------|---|------------------|--------------------|-------------------|',
    ]
    for r in rows:
        w, tr, pl = r['window'], r['trend'], r['placebo']
        lines.append(f"| {r['cell'][0]}, α = {r['cell'][1]:g} | {labels[r['key']]} | {w['pilot_mean']:.4f} | "
                     f"{w['later_mean']:.4f} | {w['diff']:+.2f} | [{w['ci_low']:+.2f}, {w['ci_high']:+.2f}] | "
                     f"{tr['slope']:+.3f} | {tr['p']:.3f} | {tr['implied_change']:+.2f} | "
                     f"{pl['pass_share']:.2f} | {pl['exceed_share']:.2f} |")

    lines += [
        '',
        '## Convergence Results',
        '',
        f'C18 criterion on T = {T_EXT}: {WINDOW}-step rolling mean within {TOLERANCE:.0%} of the final-{FINAL_WINDOW}-step '
        f'mean (steps {T_EXT - FINAL_WINDOW + metric_every}-{T_EXT}). Geweke on steps {BURN_IN + metric_every}-{T_EXT} '
        f'(80 observations): first 8 against last 40, pass if |z| < {Z_THRESHOLD}.',
        '',
        'The band test depends on run length: a longer run gives a series more chances to leave the band, so '
        'series with no trend fail it later here than at T = 11000. The window comparison above is the evidence '
        'for the burn-in; this table is reported for completeness.',
        '',
    ]
    for cell in CELLS:
        lines += [
            f'### {cell[0]}, alpha={cell[1]:g}',
            '',
            f'| Metric | Convergence Step | Final Mean | Before {BURN_IN} | Max dev. | Geweke z | Geweke | Rep passes |',
            '|--------|-----------------|------------|-------------|----------|----------|--------|------------|',
        ]
        for key, label, fmt in SERIES:
            verdict = verdicts[cell][key]
            if not verdict['applicable']:
                lines.append(f'| {label} | n/a | n/a | n/a | n/a | n/a | n/a | n/a |')
                continue
            lines.append(
                f"| {label} | {fmt_conv(verdict)} | {verdict['final_mean']:{fmt}} | "
                f"{'yes' if verdict['converged'] else 'NO'} | {verdict['max_dev']:.1%} | "
                f"{verdict['z_mean']:.2f} | {'pass' if verdict['geweke_pass'] else 'FAIL'} | "
                f"{verdict['rep_pass']}/{N_REPS} |")
        lines.append('')

    lines += [f'## Metric Tables ({T + 1000}-{T_EXT}, every 1000 steps)', '', 'Means across reps at each step.', '']
    for cell in CELLS:
        lines += [
            f'### {cell[0]}, alpha={cell[1]:g}',
            '',
            '| Step | Hill | Mean K | K_eff/K | Cong Share | FHR Stand | FHR Memb | Mergers | Vol Exits | Floor Exits | HHI |',
            '|------|------|--------|---------|------------|-----------|----------|---------|-----------|-------------|-----|',
        ]
        for step in range(T + 1000, T_EXT + 1, 1000):
            idx = step // metric_every - 1
            row = [f'{step}']
            for key, _, fmt in SERIES:
                column = cells[cell][key][:, idx]
                row.append('n/a' if np.all(np.isnan(column)) else f'{np.nanmean(column):{fmt}}')
            lines.append('| ' + ' | '.join(row) + ' |')
        lines.append('')

    lines += ['## Figures', '']
    for cell in CELLS:
        lines.append(f'- `figures/burnin_c33_ext_{cell_label(*cell)}.png`: all series for {cell[0]}, α = {cell[1]:g}')
    lines.append('')

    with open('diagnostics/burn_in_c33_ext.md', 'w') as f:
        f.write('\n'.join(lines))
    print('Wrote diagnostics/burn_in_c33_ext.md')
    print(f'Identical to pilot up to T={T}: {n_identical}/{len(scenarios)}')
    print(f"burn_in = {BURN_IN} {'stands' if keep else 'DOES NOT stand'}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python run_burnin_c33_ext.py <run_index> | --report")
        sys.exit(1)

    if sys.argv[1] == '--report':
        report()
    else:
        run_one(int(sys.argv[1]))


if __name__ == '__main__':
    main()
