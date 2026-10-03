#!/usr/bin/env python3
"""
Test online mobility tracking (§4).

- Verifies that mobility CSV is created and non-empty after a short run
- Tests that rank autocorrelation values are in valid range [-1, 1]
"""
import numpy as np
import tempfile
import os
import sys
sys.path.insert(0, '..')
from collaborative_growth import model


def test_mobility_csv_created():
    """
    Mobility CSV should be created when mobility_csv path is provided.
    """
    params = [10, 10, 50, 0.1, 100, 0.05, 4, 0.85, False, 20,
              'power_law', None, None, None]

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        csv_path = f.name

    try:
        result = model(params, seed=42, mobility_csv=csv_path)

        # CSV should exist
        assert os.path.exists(csv_path), "Mobility CSV file was not created"

        # CSV should be non-empty
        with open(csv_path, 'r') as f:
            lines = f.readlines()

        assert len(lines) > 1, f"CSV should have header + data rows, got {len(lines)} lines"

        # Check header
        assert lines[0].strip() == 'step,rank_autocorr', \
            f"Expected header 'step,rank_autocorr', got '{lines[0].strip()}'"

        # Should have steps-1 data rows (no autocorr for step 0)
        assert len(lines) == 50, f"Expected 50 lines (header + 49 data), got {len(lines)}"

        print(f"Mobility CSV created with {len(lines)} lines")

    finally:
        if os.path.exists(csv_path):
            os.unlink(csv_path)


def test_rank_autocorr_valid_range():
    """
    Rank autocorrelation should be in [-1, 1].
    """
    params = [10, 10, 100, 0.1, 100, 0.05, 4, 0.85, False, 20,
              'power_law', None, None, None]

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        csv_path = f.name

    try:
        result = model(params, seed=42, mobility_csv=csv_path)

        # Read values
        with open(csv_path, 'r') as f:
            lines = f.readlines()[1:]  # Skip header

        for line in lines:
            parts = line.strip().split(',')
            step = int(parts[0])
            autocorr = float(parts[1])

            assert -1.0 <= autocorr <= 1.0, \
                f"Rank autocorr at step {step} out of range: {autocorr}"

        print("All rank autocorrelation values in valid range [-1, 1]")

    finally:
        if os.path.exists(csv_path):
            os.unlink(csv_path)


def test_rank_autocorr_positive():
    """
    Rank autocorrelation should typically be positive (ranks are persistent).
    """
    params = [10, 10, 100, 0.1, 100, 0.05, 4, 0.85, False, 20,
              'power_law', None, None, None]

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        csv_path = f.name

    try:
        result = model(params, seed=42, mobility_csv=csv_path)

        # Read values
        with open(csv_path, 'r') as f:
            lines = f.readlines()[1:]  # Skip header

        autocorrs = []
        for line in lines:
            parts = line.strip().split(',')
            autocorrs.append(float(parts[1]))

        mean_autocorr = np.mean(autocorrs)

        # Ranks should be persistent (positive autocorrelation)
        assert mean_autocorr > 0.5, \
            f"Expected high positive rank autocorrelation, got mean {mean_autocorr:.4f}"

        print(f"Mean rank autocorrelation: {mean_autocorr:.4f}")

    finally:
        if os.path.exists(csv_path):
            os.unlink(csv_path)


def test_no_csv_without_arg():
    """
    When mobility_csv is not provided, no CSV file should be created.
    """
    params = [10, 10, 50, 0.1, 100, 0.05, 4, 0.85, False, 20,
              'power_law', None, None, None]

    # Run without mobility_csv
    result = model(params, seed=42)

    # Should complete without error
    assert result is not None

    print("Model runs correctly without mobility_csv argument")


if __name__ == '__main__':
    print("Testing mobility CSV created...")
    test_mobility_csv_created()
    print("PASS\n")

    print("Testing rank autocorr valid range...")
    test_rank_autocorr_valid_range()
    print("PASS\n")

    print("Testing rank autocorr positive...")
    test_rank_autocorr_positive()
    print("PASS\n")

    print("Testing no CSV without arg...")
    test_no_csv_without_arg()
    print("PASS\n")

    print("All online mobility tests passed!")
