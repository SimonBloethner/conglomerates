#!/usr/bin/env python3
"""
Check what's actually at line 1121 in the SLURM version
"""

with open('compare_parametrizations.py', 'r') as f:
    lines = f.readlines()

print("Lines 1115-1125 in compare_parametrizations.py:\n")
for i in range(1114, 1125):
    if i < len(lines):
        print(f"{i+1:4d}: {lines[i]}", end='')