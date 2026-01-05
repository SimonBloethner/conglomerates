#!/usr/bin/env python3
"""
Check if compare_parametrizations.py has the panel_coefficients fix
"""

with open('compare_parametrizations.py', 'r') as f:
    lines = f.readlines()

# Check for the key fix at specific line
print("Checking for fixes in compare_parametrizations.py:\n")

# Check 1: panel_coefficients dictionary initialization
panel_coeff_init = any("'panel_coefficients': {}" in line for line in lines)
print(f"1. Panel coefficients dict init: {'✅' if panel_coeff_init else '❌'}")

# Check 2: panel_stats initialization
panel_stats_init = any("cost_data['panel_stats'] = {}" in line for line in lines)
print(f"2. Panel stats init: {'✅' if panel_stats_init else '❌'}")

# Check 3: Panel coefficient aggregation logic
panel_aggregation = any("# Process panel coefficients (scalars)" in line for line in lines)
print(f"3. Panel aggregation logic: {'✅' if panel_aggregation else '❌'}")

# Check 4: Look for the specific aggregation code
panel_scalar_processing = any("scalars = np.array(coefficient_scalars)" in line for line in lines)
print(f"4. Panel scalar processing: {'✅' if panel_scalar_processing else '❌'}")

# Check 5: Panel routing in quantile data
panel_routing_quantile = sum(1 for line in lines if "if 'panel' in quantile_data:" in line)
print(f"5. Panel routing (quantile): {panel_routing_quantile} occurrences")

# Check 6: Panel routing in scalar metrics
panel_routing_scalar = sum(1 for line in lines if "if 'panel' in metric_data:" in line)
print(f"6. Panel routing (scalar): {panel_routing_scalar} occurrences")

# Show line numbers for debugging
print("\nKey line locations:")
for i, line in enumerate(lines, 1):
    if "'panel_coefficients': {}" in line:
        print(f"  panel_coefficients init at line {i}")
    if "cost_data['panel_stats'] = {}" in line:
        print(f"  panel_stats init at line {i}")
    if "# Process panel coefficients (scalars)" in line:
        print(f"  panel aggregation comment at line {i}")
    if "cost_data['panel_stats'][key] = {" in line:
        print(f"  panel_stats assignment at line {i}")

# Final verdict
all_checks = [panel_coeff_init, panel_stats_init, panel_aggregation, panel_scalar_processing]
if all(all_checks):
    print("\n✅ ALL FIXES ARE PRESENT - File should work correctly")
else:
    print("\n❌ SOME FIXES ARE MISSING - Need to re-upload the file")
    print("Missing fixes:")
    if not panel_coeff_init:
        print("  - Panel coefficients dictionary initialization")
    if not panel_stats_init:
        print("  - Panel stats initialization")
    if not panel_aggregation:
        print("  - Panel aggregation logic")
    if not panel_scalar_processing:
        print("  - Panel scalar processing code")