#!/usr/bin/env python3
"""
Add debugging code to compare_parametrizations.py at line 1121
"""

with open('compare_parametrizations.py', 'r') as f:
    lines = f.readlines()

# Find line 1121 (index 1120) and add debugging before it
debug_code = """                    # DEBUG: Check shapes before stacking
                    shapes = [np.array(arr).shape for arr in coefficient_arrays]
                    if len(set(shapes)) > 1:
                        print(f"\\n❌ ERROR in key: {key}")
                        print(f"   Number of arrays: {len(coefficient_arrays)}")
                        print(f"   Unique shapes: {set(shapes)}")
                        for i, (arr, shape) in enumerate(zip(coefficient_arrays[:5], shapes[:5])):
                            print(f"   [{i}] shape={shape}, type={type(arr).__name__}, scalar={np.isscalar(arr)}")
                        print(f"   Skipping this key to continue...")
                        continue

"""

# Insert debug code before line 1121 (array index 1120)
# Line 1121 is: "stacked = np.array(coefficient_arrays)..."
target_line_num = 1120  # 0-indexed
lines.insert(target_line_num, debug_code)

# Write back
with open('compare_parametrizations.py', 'w') as f:
    f.writelines(lines)

print("✅ Added debugging code before line 1121")
print("Now run: python run_robustness_analysis.py analyze")
print("This will show which key is causing the error and what shapes it contains")
