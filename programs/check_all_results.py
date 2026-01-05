#!/usr/bin/env python3
"""
Check what results directories actually exist and will be loaded
"""
from pathlib import Path

results_dir = Path('results')

print("All result directories found:\n")

all_dirs = sorted([d for d in results_dir.glob('*') if d.is_dir()])

for d in all_dirs:
    result_file = d / 'counterfactual_results_final.pkl'
    exists = "✅" if result_file.exists() else "❌"
    print(f"{exists} {d.name}")

print(f"\nTotal: {len(all_dirs)} directories")
print(f"  Robustness: {len([d for d in all_dirs if 'robustness' in d.name])}")
print(f"  Non-robustness: {len([d for d in all_dirs if 'robustness' not in d.name])}")

# Check if any have different naming patterns
non_robustness = [d.name for d in all_dirs if 'robustness' not in d.name]
if non_robustness:
    print(f"\nNon-robustness directories:")
    for name in non_robustness:
        print(f"  - {name}")