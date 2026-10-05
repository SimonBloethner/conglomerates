#!/usr/bin/env python3
"""
C6b: Burn-in rerun with pooled Hill estimator.

Runs: main block, laplace, power_law, alpha=0.1, M=N=50, T=8000, 5 reps
Computes 500-step rolling means of pooled Hill, mean K, cong_capital_share.
Finds first t where all three stay within 2% of final-2000-step mean.
"""
import numpy as np
import json
import os
import matplotlib.pyplot as plt

import collaborative_growth as cg

# Parameters from pilot_design.py
M = 50
N = 50
T = 8000
ALPHA = 0.1
REPS = 5
FLOOR_C = 1.0 - 1.0 / 1.06  # c_for_exponent(1.06)
METRIC_EVERY = 100
WINDOW = 500 // METRIC_EVERY  # 5 metric steps
FINAL_WINDOW = 2000 // METRIC_EVERY  # 20 metric steps

# Cost function for power_law - calibrated values from pilot_design.py
COST_TYPE = 'power_law'
C0 = 0.00002032  # Calibrated for convergence at size=40
C1 = 1.2
C2 = 0.001

# Seed computation matching pilot_design.cell_seed()
# For main block, laplace (family_idx=1), power_law (cost_idx=3): cell_id = 1*4 + 3 = 7
BASE_SEED = 42
CELL_ID = 7  # laplace + power_law

def cell_seed(cell_id, rep):
    return BASE_SEED + cell_id * 100 + rep

print(f"Running burn-in analysis:")
print(f"  M={M}, N={N}, T={T}, alpha={ALPHA}, reps={REPS}")
print(f"  floor_c={FLOOR_C:.10f}")
print(f"  cost_type={COST_TYPE}, c0={C0}, c1={C1}, c2={C2}")
print()

# Storage for time series
all_hill = []
all_mean_K = []
all_cong_share = []

for rep in range(REPS):
    seed = cell_seed(CELL_ID, rep)
    print(f"Rep {rep+1}/{REPS} (seed={seed})...", flush=True)

    params = [
        M, N, T, ALPHA, M * N,
        0.05,  # merge_thresh (from pilot_design)
        4,     # K_max default
        0.0,   # minimum_benefit default
        False, # proportional
        50,    # lookback
        COST_TYPE, C0, C1, C2
    ]

    result = cg.model(
        params, seed=seed,
        growth_process='log_family',
        log_family='laplace',
        mu_range=(0.01, 0.1),
        sigma_range=(0.1, 0.3),
        floor_c=FLOOR_C,
        cross_corr=0.0,
        metric_every=METRIC_EVERY,
        burn_in=0,  # No burn-in for this analysis
        sharing_rule='proportional'
    )

    hp = result[-1]

    # Extract time series (pooled Hill is now 1D)
    hill = hp['hill_exponent']  # Shape: (n_metric_steps,)
    cong_share = hp['cong_capital_share']  # Shape: (n_metric_steps,)

    # Mean K per metric step - need to compute from effective_members
    n_metric_steps = len(hill)
    mean_K = np.zeros(n_metric_steps)

    # effective_members is list of (step, [(cid, K, K_eff), ...])
    eff_members = hp['effective_members']
    for step_val, step_eff in eff_members:
        metric_idx = step_val // METRIC_EVERY
        if metric_idx < n_metric_steps and len(step_eff) > 0:
            Ks = [K for cid, K, K_eff in step_eff]
            mean_K[metric_idx] = np.mean(Ks) if len(Ks) > 0 else 0

    all_hill.append(hill)
    all_mean_K.append(mean_K)
    all_cong_share.append(cong_share)

# Convert to arrays and average across reps
all_hill = np.array(all_hill)
all_mean_K = np.array(all_mean_K)
all_cong_share = np.array(all_cong_share)

mean_hill = np.nanmean(all_hill, axis=0)
mean_K = np.nanmean(all_mean_K, axis=0)
mean_cong_share = np.nanmean(all_cong_share, axis=0)

n_metric_steps = len(mean_hill)
print(f"\nNumber of metric steps: {n_metric_steps}")

# Compute 500-step rolling means (5 metric steps)
def rolling_mean(arr, window):
    result = np.full_like(arr, np.nan)
    for i in range(window - 1, len(arr)):
        result[i] = np.nanmean(arr[i - window + 1:i + 1])
    return result

roll_hill = rolling_mean(mean_hill, WINDOW)
roll_K = rolling_mean(mean_K, WINDOW)
roll_cong = rolling_mean(mean_cong_share, WINDOW)

# Final-2000-step means
final_hill = np.nanmean(mean_hill[-FINAL_WINDOW:])
final_K = np.nanmean(mean_K[-FINAL_WINDOW:])
final_cong = np.nanmean(mean_cong_share[-FINAL_WINDOW:])

print(f"\nFinal-2000-step means:")
print(f"  Pooled Hill: {final_hill:.4f}")
print(f"  Mean K: {final_K:.4f}")
print(f"  Cong capital share: {final_cong:.4f}")

# Check acceptance criterion
if not (0.7 < final_hill < 1.5):
    print(f"\nWARNING: Final Hill {final_hill:.4f} not in (0.7, 1.5)!")
    print("Writing deviations.md and stopping.")

    os.makedirs('diagnostics', exist_ok=True)
    with open('diagnostics/deviations.md', 'w') as f:
        f.write("# Burn-in C6b Deviations\n\n")
        f.write(f"Final-2000-step pooled Hill mean: {final_hill:.4f}\n")
        f.write(f"Expected range: (0.7, 1.5)\n\n")
        f.write("## Series data\n\n")
        f.write("### Pooled Hill\n")
        f.write(f"```\n{mean_hill}\n```\n\n")
        f.write("### Mean K\n")
        f.write(f"```\n{mean_K}\n```\n\n")
        f.write("### Cong capital share\n")
        f.write(f"```\n{mean_cong_share}\n```\n")

    raise ValueError(f"Acceptance criterion failed: Hill={final_hill:.4f} not in (0.7, 1.5)")

# Find first t where all three stay within 2%
def find_convergence_point(roll_series, final_val, tol=0.02):
    """Find first metric index where series stays within tol of final."""
    for i in range(len(roll_series)):
        if np.isnan(roll_series[i]):
            continue
        # Check if all subsequent values are within tolerance
        subsequent = roll_series[i:]
        if np.all(np.abs(subsequent - final_val) / max(abs(final_val), 1e-10) < tol):
            return i
    return len(roll_series) - 1

conv_hill = find_convergence_point(roll_hill, final_hill)
conv_K = find_convergence_point(roll_K, final_K)
conv_cong = find_convergence_point(roll_cong, final_cong)

# Take the max (latest convergence point)
conv_metric_idx = max(conv_hill, conv_K, conv_cong)
burn_in_t = conv_metric_idx * METRIC_EVERY

print(f"\nConvergence points (metric index):")
print(f"  Pooled Hill: {conv_hill} (t={conv_hill * METRIC_EVERY})")
print(f"  Mean K: {conv_K} (t={conv_K * METRIC_EVERY})")
print(f"  Cong share: {conv_cong} (t={conv_cong * METRIC_EVERY})")
print(f"\nChosen burn_in = {burn_in_t}")
print(f"Chosen T = {burn_in_t + 3000}")

# Create plot
fig, axes = plt.subplots(3, 1, figsize=(10, 10))

time_steps = np.arange(n_metric_steps) * METRIC_EVERY

# Plot 1: Pooled Hill
ax1 = axes[0]
ax1.plot(time_steps, mean_hill, 'b-', alpha=0.3, label='Raw mean')
ax1.plot(time_steps, roll_hill, 'b-', linewidth=2, label='500-step rolling mean')
ax1.axhline(final_hill, color='r', linestyle='--', label=f'Final mean: {final_hill:.3f}')
ax1.axhline(final_hill * 1.02, color='r', linestyle=':', alpha=0.5)
ax1.axhline(final_hill * 0.98, color='r', linestyle=':', alpha=0.5)
ax1.axvline(burn_in_t, color='g', linestyle='--', label=f'Burn-in: {burn_in_t}')
ax1.set_xlabel('Time step')
ax1.set_ylabel('Pooled Hill exponent')
ax1.set_title('Pooled Hill Exponent Convergence')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Mean K
ax2 = axes[1]
ax2.plot(time_steps, mean_K, 'g-', alpha=0.3, label='Raw mean')
ax2.plot(time_steps, roll_K, 'g-', linewidth=2, label='500-step rolling mean')
ax2.axhline(final_K, color='r', linestyle='--', label=f'Final mean: {final_K:.3f}')
ax2.axhline(final_K * 1.02, color='r', linestyle=':', alpha=0.5)
ax2.axhline(final_K * 0.98, color='r', linestyle=':', alpha=0.5)
ax2.axvline(burn_in_t, color='g', linestyle='--', label=f'Burn-in: {burn_in_t}')
ax2.set_xlabel('Time step')
ax2.set_ylabel('Mean K')
ax2.set_title('Mean Conglomerate Size Convergence')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Cong capital share
ax3 = axes[2]
ax3.plot(time_steps, mean_cong_share, 'm-', alpha=0.3, label='Raw mean')
ax3.plot(time_steps, roll_cong, 'm-', linewidth=2, label='500-step rolling mean')
ax3.axhline(final_cong, color='r', linestyle='--', label=f'Final mean: {final_cong:.3f}')
ax3.axhline(final_cong * 1.02, color='r', linestyle=':', alpha=0.5)
ax3.axhline(final_cong * 0.98, color='r', linestyle=':', alpha=0.5)
ax3.axvline(burn_in_t, color='g', linestyle='--', label=f'Burn-in: {burn_in_t}')
ax3.set_xlabel('Time step')
ax3.set_ylabel('Cong capital share')
ax3.set_title('Conglomerate Capital Share Convergence')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
os.makedirs('diagnostics', exist_ok=True)
plt.savefig('diagnostics/burn_in_c_convergence.png', dpi=150)
plt.close()

print(f"\nPlot saved to diagnostics/burn_in_c_convergence.png")

# Write burn_in_c.md
with open('diagnostics/burn_in_c.md', 'w') as f:
    f.write("# C6b: Burn-in Rerun\n\n")
    f.write("## Parameters\n\n")
    f.write(f"- Block: main\n")
    f.write(f"- log_family: laplace\n")
    f.write(f"- cost_type: power_law\n")
    f.write(f"- alpha: {ALPHA}\n")
    f.write(f"- M = N = {M}\n")
    f.write(f"- T: {T}\n")
    f.write(f"- Reps: {REPS}\n")
    f.write(f"- floor_c: {FLOOR_C:.10f} (c_for_exponent(1.06))\n")
    f.write(f"- metric_every: {METRIC_EVERY}\n\n")

    f.write("## Results\n\n")
    f.write("### Final-2000-step means\n\n")
    f.write(f"| Metric | Final Mean |\n")
    f.write(f"|--------|------------|\n")
    f.write(f"| Pooled Hill | {final_hill:.4f} |\n")
    f.write(f"| Mean K | {final_K:.4f} |\n")
    f.write(f"| Cong capital share | {final_cong:.4f} |\n\n")

    f.write("### Convergence points\n\n")
    f.write("First t at which 500-step rolling mean stays within 2% of final:\n\n")
    f.write(f"| Metric | Convergence t |\n")
    f.write(f"|--------|---------------|\n")
    f.write(f"| Pooled Hill | {conv_hill * METRIC_EVERY} |\n")
    f.write(f"| Mean K | {conv_K * METRIC_EVERY} |\n")
    f.write(f"| Cong capital share | {conv_cong * METRIC_EVERY} |\n\n")

    f.write(f"**Chosen burn_in = {burn_in_t}** (max of convergence points)\n\n")
    f.write(f"**Chosen T = {burn_in_t + 3000}**\n\n")

    f.write("## Acceptance criterion\n\n")
    f.write(f"Final-2000-step pooled Hill mean: {final_hill:.4f}\n\n")
    if 0.7 < final_hill < 1.5:
        f.write("**PASS**: Hill in (0.7, 1.5)\n\n")
    else:
        f.write("**FAIL**: Hill not in (0.7, 1.5)\n\n")

    f.write("## Convergence plots\n\n")
    f.write("![Convergence](burn_in_c_convergence.png)\n")

print(f"Written diagnostics/burn_in_c.md")

# Output values for scenarios.json update
print(f"\n=== Values for scenarios.json ===")
print(f"T = {burn_in_t + 3000}")
print(f"burn_in = {burn_in_t}")
