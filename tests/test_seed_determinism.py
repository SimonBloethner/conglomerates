#!/usr/bin/env python3
"""
Test that seed generation is deterministic across Python processes.

Python's hash() function is salted with a random value per-process (PYTHONHASHSEED),
so hash-based seeds would differ between SLURM tasks. We use zlib.crc32 instead
to ensure deterministic seeds.
"""

import subprocess
import sys


def test_seed_determinism():
    """Verify that seed generation is identical across two fresh Python interpreters."""

    # Code to compute seed in a subprocess
    seed_code = '''
import zlib
scenario_name = "test_scenario"
exp_id = 42
seed = zlib.crc32(f"{scenario_name}:{exp_id}".encode()) & 0xFFFFFFFF
print(seed)
'''

    # Run in two separate subprocesses (fresh interpreters)
    # We don't pass env={} as that can break on cluster environments
    # PYTHONHASHSEED doesn't affect zlib.crc32 anyway
    import os
    env = os.environ.copy()
    env['PYTHONHASHSEED'] = 'random'  # Ensure hash randomization is on

    result1 = subprocess.run(
        [sys.executable, '-c', seed_code],
        capture_output=True,
        text=True,
        env=env
    )
    assert result1.returncode == 0, f"Subprocess 1 failed: {result1.stderr}"
    seed1 = int(result1.stdout.strip())

    result2 = subprocess.run(
        [sys.executable, '-c', seed_code],
        capture_output=True,
        text=True,
        env=env
    )
    assert result2.returncode == 0, f"Subprocess 2 failed: {result2.stderr}"
    seed2 = int(result2.stdout.strip())

    assert seed1 == seed2, f"Seeds differ across processes: {seed1} != {seed2}"

    # Also verify the value is what we expect from crc32
    import zlib
    expected = zlib.crc32(b"test_scenario:42") & 0xFFFFFFFF
    assert seed1 == expected, f"Seed {seed1} != expected {expected}"

    print(f"Seed determinism verified: {seed1}")


def test_hash_is_not_deterministic():
    """Demonstrate that Python's hash() is NOT deterministic across processes."""

    # Code using hash() - this should give different results
    hash_code = '''
scenario_name = "test_scenario"
exp_id = 42
seed = hash((scenario_name, exp_id)) % (2**32)
print(seed)
'''

    # Run multiple times and collect results
    seeds = []
    for _ in range(5):
        result = subprocess.run(
            [sys.executable, '-c', hash_code],
            capture_output=True,
            text=True,
            # Don't set PYTHONHASHSEED to allow randomization
        )
        if result.stdout.strip():
            seeds.append(int(result.stdout.strip()))

    # With hash randomization enabled (default in Python 3.3+),
    # seeds should differ across runs
    # Note: this test may pass if PYTHONHASHSEED is set in the environment
    unique_seeds = set(seeds)

    if len(unique_seeds) == 1:
        print(f"Warning: hash() produced same value {seeds[0]} across {len(seeds)} runs")
        print("This may happen if PYTHONHASHSEED is set to a fixed value")
    else:
        print(f"hash() produced {len(unique_seeds)} unique values across {len(seeds)} runs")
        print("This demonstrates why we use zlib.crc32 instead")


if __name__ == "__main__":
    test_seed_determinism()
    test_hash_is_not_deterministic()
    print("All tests passed!")
