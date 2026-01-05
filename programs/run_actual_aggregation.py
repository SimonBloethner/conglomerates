#!/usr/bin/env python3
"""
Run the ACTUAL aggregation method and catch the error to see what's failing
"""
import sys
import numpy as np
from compare_parametrizations import ParametrizationComparator

print("Running ACTUAL comparison analysis to trigger the error...\n")

comparator = ParametrizationComparator(results_base_dir='results')

try:
    # This will run the full analysis including aggregation
    results = comparator.run_comprehensive_analysis()
    print("✅ Analysis completed successfully!")

except ValueError as e:
    print(f"\n❌ ValueError caught: {e}\n")
    print("Error details:")
    import traceback
    traceback.print_exc()

except Exception as e:
    print(f"\n❌ Other error: {type(e).__name__}: {e}\n")
    import traceback
    traceback.print_exc()
