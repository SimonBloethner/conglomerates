"""C34: every module the suite tests is the repository's own copy.

Before C34 tests/test_assortativity.py put a hard-coded /groups/.../IOxEE (a stale tree) first on sys.path,
so test_phase_b_identity, and any module first imported after collection, could resolve to that copy.
"""
import importlib
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODULES = ['collaborative_growth', 'pilot_design', 'run_pilot_c', 'summarize_pilot_c', 'create_alpha_scatter',
           'analytics.benchmarks', 'analytics.calibrate_floor', 'analysis.event_study', 'analysis.status_panel',
           'analysis.burstiness', 'tests.test_phase_b_identity']


@pytest.mark.parametrize('name', MODULES)
def test_module_loaded_from_repo(name):
    """Catches: a test importing a stale copy of a module from another tree on sys.path."""
    mod = importlib.import_module(name)
    assert os.path.realpath(mod.__file__).startswith(os.path.realpath(ROOT) + os.sep), mod.__file__
