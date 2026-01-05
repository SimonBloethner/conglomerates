"""
Numba-accelerated merger kernel for collaborative_growth.py

This module contains the full merger pipeline compiled with Numba for maximum performance.
"""
import numpy as np
import numba as nb
from numba import njit


@njit
def random_uniform_numba(size):
    """Generate uniform [0,1) numbers using Numba's RNG"""
    out = np.empty(size, np.float64)
    for i in range(size):
        out[i] = np.random.random()
    return out


@njit
def shuffle_numba(arr):
    """Fisher-Yates shuffle using Numba's RNG"""
    n = len(arr)
    for i in range(n-1, 0, -1):
        j = np.random.randint(0, i + 1)
        arr[i], arr[j] = arr[j], arr[i]


@njit
def get_historical_states(firm_log_states_buffer, start_step, end_step, firm_ids):
    """Extract historical log states from circular buffer (lookback+1, total_firms)"""
    lookback_plus_1 = firm_log_states_buffer.shape[0]
    n_steps = end_step - start_step
    n_firms = len(firm_ids)
    result = np.empty((n_steps, n_firms), np.float64)
    for t in range(n_steps):
        idx = (start_step + t) % lookback_plus_1
        for f in range(n_firms):
            result[t, f] = firm_log_states_buffer[idx, firm_ids[f]]
    return result


@njit
def get_historical_returns(firm_log_returns_buffer, start_step, end_step, firm_ids):
    """Extract historical log returns from circular buffer"""
    lookback_plus_1 = firm_log_returns_buffer.shape[0]
    n_steps = end_step - start_step
    n_firms = len(firm_ids)
    result = np.empty((n_steps, n_firms), np.float64)
    for t in range(n_steps):
        idx = (start_step + t) % lookback_plus_1
        for f in range(n_firms):
            result[t, f] = firm_log_returns_buffer[idx, firm_ids[f]]
    return result


@njit
def logsumexp_numba(x):
    """Stable logsumexp over last axis (axis=1 for 2D arrays)"""
    n_rows = x.shape[0]
    result = np.empty(n_rows, dtype=np.float64)

    for i in range(n_rows):
        row = x[i, :]
        mx = np.max(row)
        exp_sum = 0.0
        for j in range(len(row)):
            exp_sum += np.exp(row[j] - mx)
        result[i] = mx + np.log(exp_sum)

    return result


@njit(cache=True)
def merger_kernel_numba(
    step, lookback, share, proportional, merge_thresh,
    firm_conglom, firm_home_market, firm_entered,
    cong_firms, cong_size, cong_active, cong_occupies_market,
    cong_pool, management_costs_lookup,
    firm_log_states_buffer, firm_log_returns_buffer,
    total_firms, markets, firms_per_market
):
    """
    FULLY NUMBA-ACCELERATED merger step.
    Replaces ~180 lines of Python with one function call.

    Returns:
    --------
    mergers_this_step : int
        Number of successful mergers in this step
    """
    # 1. Generate merger draws
    draws = random_uniform_numba(total_firms) < merge_thresh
    candidates = np.where(draws)[0]

    if len(candidates) == 0:
        return 0  # no mergers

    shuffle_numba(candidates)

    mergers_this_step = 0

    for init_idx in range(len(candidates)):
        initiator = candidates[init_idx]
        initiator_cong = firm_conglom[initiator]

        # Skip if initiator's cong is already full
        if initiator_cong != -1 and cong_size[initiator_cong] == markets:
            continue

        # Determine available target markets
        if initiator_cong != -1:
            occupied = cong_occupies_market[initiator_cong]
            available = np.where(~occupied)[0]
        else:
            # Build available markets list (all except home)
            home = firm_home_market[initiator]
            available = np.empty(markets - 1, dtype=np.int64)
            idx = 0
            for m in range(markets):
                if m != home:
                    available[idx] = m
                    idx += 1

        if len(available) == 0:
            continue

        target_market = available[np.random.randint(0, len(available))]
        target_firm = target_market * firms_per_market + np.random.randint(0, firms_per_market)

        target_cong = firm_conglom[target_firm]

        # Extract firm lists
        if initiator_cong != -1:
            init_size = cong_size[initiator_cong]
            init_firms = cong_firms[initiator_cong, :init_size].copy()
        else:
            init_firms = np.array([initiator], dtype=np.int32)

        if target_cong != -1:
            targ_size = cong_size[target_cong]
            targ_firms = cong_firms[target_cong, :targ_size].copy()
        else:
            targ_firms = np.array([target_firm], dtype=np.int32)

        # OVERLAP CHECK (critical bug fix!)
        if initiator_cong != -1 and target_cong != -1:
            if np.any(cong_occupies_market[initiator_cong] & cong_occupies_market[target_cong]):
                continue
        elif initiator_cong == -1 and target_cong != -1:
            if cong_occupies_market[target_cong, firm_home_market[initiator]]:
                continue

        # Merge firm lists
        merged_firms = np.empty(len(init_firms) + len(targ_firms), dtype=np.int32)
        merged_firms[:len(init_firms)] = init_firms
        merged_firms[len(init_firms):] = targ_firms
        merged_firms = np.sort(merged_firms)
        merged_size = len(merged_firms)

        # Synthetic pool evaluation (only if >2 firms)
        synth_pool = np.zeros(lookback, dtype=np.float64)
        hist_len = 0

        if merged_size > 2:
            hist_start = max(0, step - lookback)
            hist_len = step - hist_start

            if hist_len > 0:
                past_states = get_historical_states(firm_log_states_buffer, hist_start, step, merged_firms)
                past_returns = get_historical_returns(firm_log_returns_buffer, hist_start, step, merged_firms)

                log_S = logsumexp_numba(past_states)

                # Compute weights
                w = np.empty_like(past_states)
                for i in range(past_states.shape[0]):
                    for j in range(past_states.shape[1]):
                        w[i, j] = np.exp(past_states[i, j] - log_S[i])

                # Compute r_minus1
                r_m1 = np.empty_like(past_returns)
                for i in range(past_returns.shape[0]):
                    for j in range(past_returns.shape[1]):
                        r_m1[i, j] = np.expm1(past_returns[i, j])

                # Weighted average gain
                avg_gain = np.zeros(past_states.shape[0], dtype=np.float64)
                for i in range(past_states.shape[0]):
                    for j in range(past_states.shape[1]):
                        avg_gain[i] += w[i, j] * r_m1[i, j]

                m = management_costs_lookup[merged_size]

                if proportional:
                    for i in range(len(log_S)):
                        cost_factor = 1.0 - m * np.exp(-log_S[i])
                        synth_pool[lookback - hist_len + i] = share * avg_gain[i] * cost_factor
                else:
                    for i in range(len(avg_gain)):
                        synth_pool[lookback - hist_len + i] = share * avg_gain[i] - m

        # Acceptance test
        accept = True
        if merged_size > 2:
            synth_total = np.sum(synth_pool)

            if initiator_cong == -1 or target_cong == -1:
                existing_cong = initiator_cong if target_cong == -1 else target_cong
                existing_pool = np.sum(cong_pool[existing_cong, :])
                if synth_total < existing_pool:
                    accept = False
            else:
                p1 = np.sum(cong_pool[initiator_cong, :])
                p2 = np.sum(cong_pool[target_cong, :])
                if synth_total < p1 or synth_total < p2:
                    accept = False

        if not accept:
            continue

        # PERFORM MERGER
        merged_markets = firm_home_market[merged_firms]

        if initiator_cong != -1 and target_cong != -1:
            # Two congs → keep target, deallocate initiator
            cong_firms[target_cong, :merged_size] = merged_firms
            cong_size[target_cong] = merged_size

            # Update occupancy for all merged markets
            for m in merged_markets:
                cong_occupies_market[target_cong, m] = True

            for f in merged_firms:
                firm_conglom[f] = target_cong
                firm_entered[f] = step

            # Store pool if >2
            if merged_size > 2:
                cong_pool[target_cong, :] = synth_pool

            # Deactivate initiator
            cong_active[initiator_cong] = False
            cong_size[initiator_cong] = 0
            cong_occupies_market[initiator_cong, :] = False

        elif initiator_cong == -1 and target_cong != -1:
            # Solo + cong
            cong_firms[target_cong, :merged_size] = merged_firms
            cong_size[target_cong] = merged_size
            cong_occupies_market[target_cong, firm_home_market[initiator]] = True
            firm_conglom[initiator] = target_cong
            firm_entered[initiator] = step

        elif initiator_cong != -1 and target_cong == -1:
            # Cong + solo
            cong_firms[initiator_cong, :merged_size] = merged_firms
            cong_size[initiator_cong] = merged_size
            cong_occupies_market[initiator_cong, firm_home_market[target_firm]] = True
            firm_conglom[target_firm] = initiator_cong
            firm_entered[target_firm] = step

        else:
            # Two solos → allocate new cong
            new_id = -1
            for cid in range(cong_firms.shape[0]):
                if not cong_active[cid]:
                    new_id = cid
                    break

            if new_id == -1:
                continue  # no free slot (should be rare)

            cong_active[new_id] = True
            cong_firms[new_id, :merged_size] = merged_firms
            cong_size[new_id] = merged_size

            for m in merged_markets:
                cong_occupies_market[new_id, m] = True

            cong_pool[new_id, :] = np.zeros(lookback, dtype=np.float64)

            for f in merged_firms:
                firm_conglom[f] = new_id
                firm_entered[f] = step

        mergers_this_step += 1

    return mergers_this_step