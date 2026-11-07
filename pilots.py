"""
C37: which pilot a script reads and writes.

- Pilot C (`pilot_c/`, 2,145 scenarios): the loggain reference, frozen at C36.
- Pilot D (`pilot_d/`): the moment rule as the base case, the paper's numbers.

Scripts take `--pilot c|d` (default c, so existing calls are unchanged) and build their paths with pilot_dir().
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
PILOTS = ('c', 'd')


def pop_pilot(argv=None, default='c'):
    """Remove `--pilot X` from argv (sys.argv by default) and return X."""
    argv = sys.argv if argv is None else argv
    if '--pilot' not in argv:
        return default
    k = argv.index('--pilot')
    if k + 1 >= len(argv) or argv[k + 1] not in PILOTS:
        raise SystemExit(f'--pilot takes one of {PILOTS}')
    p = argv[k + 1]
    del argv[k:k + 2]
    return p


def pilot_dir(pilot, *parts, absolute=False):
    """pilot_<pilot>/<parts...>, relative to the repository root (absolute=True: under ROOT)."""
    rel = os.path.join(f'pilot_{pilot}', *parts)
    return os.path.join(ROOT, rel) if absolute else rel
