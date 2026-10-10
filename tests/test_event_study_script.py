#!/usr/bin/env python3
"""
C24: Tests for analysis/event_study.py script.
"""
import numpy as np
import tempfile
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from analysis.event_study import events_from_trace, summarize


def test_events_from_trace_basic():
    """Test basic event detection from synthetic trace."""
    T = 200
    N = 20
    markets = 4
    firms_per_market = 5
    l = 50

    logshare = np.random.randn(T + 1, N).astype(np.float32) * 0.01
    cong = np.full((T + 1, N), -1, dtype=np.int16)
    home = np.repeat(np.arange(markets), firms_per_market).astype(np.int16)

    cong[100:, 0] = 0

    events = events_from_trace(logshare, cong, 0.0, home, l=l, tol=1.0)

    assert isinstance(events, list)


def test_events_from_trace_with_entry():
    """Test event detection when an entry occurs."""
    T = 200
    N = 10
    markets = 2
    firms_per_market = 5
    l = 20

    logshare = np.zeros((T + 1, N), dtype=np.float32)
    for i in range(N):
        logshare[:, i] = -np.log(firms_per_market)

    cong = np.full((T + 1, N), -1, dtype=np.int16)
    home = np.repeat(np.arange(markets), firms_per_market).astype(np.int16)

    entry_time = 50
    joiner = 0
    cong[entry_time:entry_time + l + 10, joiner] = 0

    events = events_from_trace(logshare, cong, 0.0, home, l=l, tol=1.0)

    assert len(events) >= 1
    joiner_events = [e for e in events if e['joiner'] == joiner and e['t'] == entry_time]
    assert len(joiner_events) == 1


def test_summarize_empty():
    """Test summarize with no events."""
    summary = summarize([])
    assert summary['n_events'] == 0
    assert np.isnan(summary['did_median'])


def test_summarize_with_events():
    """Test summarize with some events."""
    events = [
        {'did': 0.01, 'joiner_before': -0.001, 'joiner_after': 0.002,
         'ctrl_before': -0.001, 'ctrl_after': -0.001},
        {'did': 0.02, 'joiner_before': -0.002, 'joiner_after': 0.003,
         'ctrl_before': -0.001, 'ctrl_after': -0.002},
        {'did': -0.01, 'joiner_before': 0.000, 'joiner_after': 0.001,
         'ctrl_before': 0.001, 'ctrl_after': 0.002},
    ]
    summary = summarize(events)
    assert summary['n_events'] == 3
    assert not np.isnan(summary['did_median'])
    assert summary['did_p25'] <= summary['did_median'] <= summary['did_p75']


def test_trace_roundtrip():
    """Test saving and loading a trace file."""
    T = 100
    N = 10
    markets = 2

    logshare = np.random.randn(T + 1, N).astype(np.float32)
    cong = np.full((T + 1, N), -1, dtype=np.int16)
    floor = 0.05
    home = np.repeat(np.arange(markets), N // markets).astype(np.int16)

    with tempfile.TemporaryDirectory() as tmpdir:
        trace_path = os.path.join(tmpdir, 'test_trace.npz')
        np.savez(trace_path, logshare=logshare, cong=cong,
                 floor=np.array([floor]), home=home)

        data = np.load(trace_path)
        assert data['logshare'].shape == (T + 1, N)
        assert data['cong'].shape == (T + 1, N)
        assert float(data['floor'][0]) == floor
        assert data['home'].shape == (N,)


if __name__ == '__main__':
    test_events_from_trace_basic()
    test_events_from_trace_with_entry()
    test_summarize_empty()
    test_summarize_with_events()
    test_trace_roundtrip()
    print("All tests passed!")
