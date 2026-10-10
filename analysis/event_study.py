#!/usr/bin/env python3
"""
C24: External event study from trace files.

Reads .npz trace files with:
- logshare: (steps+1, total_firms) log market share
- cong: (steps+1, total_firms) conglomerate membership (-1 = standalone)
- floor: scalar floor_c parameter
- home: (total_firms,) home market for each firm

Detects entry events and computes DiD vs matched controls.
"""
import numpy as np
import argparse
from pathlib import Path


def events_from_trace(logshare, cong, floor, home, l=50, tol=0.25):
    """
    Extract entry events and compute DiD from trace arrays.

    Parameters
    ----------
    logshare : ndarray, shape (T+1, N)
        Log market share at each step for each firm.
    cong : ndarray, shape (T+1, N)
        Conglomerate membership at each step (-1 = standalone).
    floor : float
        Floor parameter (not used in current implementation).
    home : ndarray, shape (N,)
        Home market for each firm.
    l : int
        Lookback window (default 50).
    tol : float
        Match tolerance for control selection (default 0.25).

    Returns
    -------
    events : list of dict
        Each dict contains entry event data including DiD.
    """
    T, N = logshare.shape[0] - 1, logshare.shape[1]
    markets = int(home.max()) + 1
    firms_per_market = N // markets

    events = []

    for t in range(l, T - l + 1):
        prev_cong = cong[t - 1]
        curr_cong = cong[t]

        newly_entered = (prev_cong == -1) & (curr_cong >= 0)
        new_entrants = np.where(newly_entered)[0]

        for firm_id in new_entrants:
            market = home[firm_id]
            market_start = market * firms_per_market
            market_end = (market + 1) * firms_per_market

            joiner_before = logshare[t, firm_id] - logshare[t - l, firm_id]
            joiner_after = logshare[t + l, firm_id] - logshare[t, firm_id]

            stayed = True
            for s in range(t + 1, t + l + 1):
                if cong[s, firm_id] == -1:
                    stayed = False
                    break
            if not stayed:
                continue

            best_control = None
            best_dist = np.inf

            for ctrl_id in range(market_start, market_end):
                if ctrl_id == firm_id:
                    continue

                standalone_throughout = True
                for s in range(t - l, t + l + 1):
                    if cong[s, ctrl_id] != -1:
                        standalone_throughout = False
                        break
                if not standalone_throughout:
                    continue

                dist = abs(logshare[t, firm_id] - logshare[t, ctrl_id])
                if dist <= tol and dist < best_dist:
                    best_dist = dist
                    best_control = ctrl_id

            if best_control is None:
                continue

            ctrl_before = logshare[t, best_control] - logshare[t - l, best_control]
            ctrl_after = logshare[t + l, best_control] - logshare[t, best_control]

            did = (joiner_after - joiner_before) - (ctrl_after - ctrl_before)

            events.append({
                'joiner': firm_id,
                't': t,
                'market': market,
                'joiner_before': joiner_before / l,
                'joiner_after': joiner_after / l,
                'control': best_control,
                'ctrl_before': ctrl_before / l,
                'ctrl_after': ctrl_after / l,
                'did': did / l,
            })

    return events


def summarize(ev):
    """
    Summarize event study results.

    Parameters
    ----------
    ev : list of dict
        Events from events_from_trace.

    Returns
    -------
    summary : dict
        Summary statistics including n_events, did_median, did_p25, did_p75.
    """
    if not ev:
        return {
            'n_events': 0,
            'did_median': np.nan,
            'did_p25': np.nan,
            'did_p75': np.nan,
            'joiner_before_median': np.nan,
            'joiner_after_median': np.nan,
            'ctrl_before_median': np.nan,
            'ctrl_after_median': np.nan,
        }

    dids = np.array([e['did'] for e in ev])
    joiner_befores = np.array([e['joiner_before'] for e in ev])
    joiner_afters = np.array([e['joiner_after'] for e in ev])
    ctrl_befores = np.array([e['ctrl_before'] for e in ev])
    ctrl_afters = np.array([e['ctrl_after'] for e in ev])

    return {
        'n_events': len(ev),
        'did_median': np.median(dids),
        'did_p25': np.percentile(dids, 25),
        'did_p75': np.percentile(dids, 75),
        'joiner_before_median': np.median(joiner_befores),
        'joiner_after_median': np.median(joiner_afters),
        'ctrl_before_median': np.median(ctrl_befores),
        'ctrl_after_median': np.median(ctrl_afters),
    }


def main():
    parser = argparse.ArgumentParser(description='Event study from trace files')
    parser.add_argument('trace_files', nargs='+', help='Path(s) to .npz trace files')
    parser.add_argument('--output', '-o', default='events.csv', help='Output CSV path')
    parser.add_argument('-l', '--lookback', type=int, default=50, help='Lookback window')
    parser.add_argument('--tol', type=float, default=0.25, help='Match tolerance')
    args = parser.parse_args()

    all_events = []

    for trace_file in args.trace_files:
        data = np.load(trace_file, allow_pickle=True)
        logshare = data['logshare']
        cong = data['cong']
        floor_val = data['floor']
        floor = float(floor_val[0]) if hasattr(floor_val, '__len__') else float(floor_val)
        home = data['home']

        events = events_from_trace(logshare, cong, floor, home, l=args.lookback, tol=args.tol)
        for e in events:
            e['trace'] = Path(trace_file).stem
        all_events.extend(events)

    summary = summarize(all_events)
    print(f"Events: {summary['n_events']}")
    print(f"DiD median: {summary['did_median']:.6f}")
    print(f"DiD IQR: [{summary['did_p25']:.6f}, {summary['did_p75']:.6f}]")

    if all_events:
        import csv
        with open(args.output, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['trace', 'joiner', 't', 'market',
                                                   'joiner_before', 'joiner_after',
                                                   'control', 'ctrl_before', 'ctrl_after', 'did'])
            writer.writeheader()
            writer.writerows(all_events)
        print(f"Wrote {args.output}")


if __name__ == '__main__':
    main()
