"""Tests for trace file saving."""
import tempfile
import numpy as np
from collaborative_growth import model


def test_trace_floor_shape_and_cong_dtype():
    """Verify floor array shape (F, T) and cong dtype int16 in saved trace."""
    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
        trace_path = f.name

    # M=N=10, T=100 as specified
    params = [10, 10, 100, 0.1, 100, 0.05, 'internal', 0.0, 0.0, 50,
              'power_law', 0.00002032, 1.2, 0.001]
    model(params, seed=42, trace_path=trace_path, floor_c=0.12717,
          decision_rule='loggain', g=0.055, sharing_rule='proportional')

    z = np.load(trace_path)

    # floor shape: (firms, steps) = (100, 100)
    assert z['floor'].shape == (100, 100), f"floor shape {z['floor'].shape} != (100, 100)"
    assert z['floor'].dtype == np.bool_, f"floor dtype {z['floor'].dtype} != bool"

    # cong dtype must be int16
    assert z['cong'].dtype == np.int16, f"cong dtype {z['cong'].dtype} != int16"
