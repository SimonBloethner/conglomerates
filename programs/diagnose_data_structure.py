#!/usr/bin/env python3
"""
Diagnose the actual data structure in saved results to find the shape mismatch
"""
import pickle
import numpy as np
from pathlib import Path
import os

# Check what result directories exist
results_dir = Path('results')
print(f"Checking results directory: {results_dir.absolute()}")
print(f"Exists: {results_dir.exists()}")

if results_dir.exists():
    robustness_dirs = list(results_dir.glob('robustness_*'))
    print(f"\nFound {len(robustness_dirs)} robustness directories:")
    for d in robustness_dirs[:5]:
        print(f"  - {d.name}")

    if robustness_dirs:
        # Use the first one
        result_file = robustness_dirs[0] / 'counterfactual_results_final.pkl'
        print(f"\nChecking file: {result_file}")
        print(f"File exists: {result_file.exists()}")

        if result_file.exists():
            print(f"File size: {result_file.stat().st_size / (1024*1024):.2f} MB")

            with open(result_file, 'rb') as f:
                data = pickle.load(f)

            print(f"\nData type: {type(data)}")
            print(f"Number of keys (shares): {len(data)}")

            # Get first share's data
            first_share_key = next(iter(data.keys()))
            first_share = data[first_share_key]

            print(f"\nFirst share key: {first_share_key}")
            print(f"First share data type: {type(first_share)}")
            print(f"First share keys: {list(first_share.keys())}")

            # Now check if sensitivity_analysis exists
            if 'sensitivity_analysis' in first_share:
                print("\n✅ Found sensitivity_analysis")
                sens = first_share['sensitivity_analysis']
                print(f"Sensitivity analysis type: {type(sens)}")
                print(f"Sensitivity analysis keys: {list(sens.keys())[:5]}")
            else:
                print("\n❌ No sensitivity_analysis key found")
                print("This means the sensitivity analysis hasn't been run yet on this data")
        else:
            print("❌ Result file doesn't exist")
    else:
        print("❌ No robustness directories found")
else:
    print("❌ Results directory doesn't exist")