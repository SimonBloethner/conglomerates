#!/usr/bin/env python3
"""
Check if all parametrizations have the same number of timesteps
"""
import pickle
from pathlib import Path

results_dir = Path('results')
robustness_dirs = list(results_dir.glob('robustness_*'))

print(f"Checking {len(robustness_dirs)} parametrizations...\n")

timesteps_found = {}

for param_dir in sorted(robustness_dirs)[:20]:  # Check first 20
    result_file = param_dir / 'counterfactual_results_final.pkl'

    if result_file.exists():
        with open(result_file, 'rb') as f:
            data = pickle.load(f)

        # Get first share
        first_share = data[0.0]

        # Check number of timesteps in mean_members_avg
        if 'mean_members_avg' in first_share:
            n_timesteps = len(first_share['mean_members_avg'])

            if n_timesteps not in timesteps_found:
                timesteps_found[n_timesteps] = []
            timesteps_found[n_timesteps].append(param_dir.name)

            # Also check hyperparameters
            hyperparams = first_share.get('hyperparameters', {})
            steps = hyperparams.get('steps', 'unknown')

            if n_timesteps != steps:
                print(f"⚠️  {param_dir.name}")
                print(f"   Timesteps in data: {n_timesteps}")
                print(f"   Steps in hyperparams: {steps}")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)

for n_timesteps, param_list in sorted(timesteps_found.items()):
    print(f"\n{n_timesteps} timesteps: {len(param_list)} parametrizations")
    if len(timesteps_found) > 1:
        for param in param_list[:3]:
            print(f"  - {param}")
        if len(param_list) > 3:
            print(f"  ... and {len(param_list) - 3} more")

if len(timesteps_found) > 1:
    print("\n❌ ERROR: Different parametrizations have different numbers of timesteps!")
    print("   This will cause the inhomogeneous shape error when stacking.")
else:
    print("\n✅ All parametrizations have the same number of timesteps")
