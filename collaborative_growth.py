import numpy as np
from scipy.stats import random_correlation, norm, laplace, t as student_t
from scipy.special import logsumexp, expm1
import numba as nb


@nb.njit(cache=True)
def seed_numba(s):
    """Seed Numba's random number generator for reproducibility."""
    np.random.seed(s)


def logistic_cost(x, k, x_0):
    y = 1 / (1 + np.exp(-k * (x - x_0)))
    return y


def power_law_cost(size, b0, b1):
    return b0 * size ** b1


def get_cost_function_defaults(cost_type):
    """Get default parameters for a specific cost function type"""
    defaults = {'power_law': {'c0': 0.00002032, 'c1': 1.2, 'c2': 0.001},
        'linear': {'c0': 0.00001, 'c1': 0.00004225, 'c2': 0.001},
        'quadratic': {'c0': 0.0001, 'c1': 0.000001, 'c2': 0.00000097},
        'exponential': {'c0': 0.001, 'c1': 0.01326571, 'c2': 0.001}}
    return defaults.get(cost_type, defaults['power_law'])  # Default to power_law if unknown


def management_cost_function(size, cost_type, c0=None, c1=None, c2=None):
    """
    Unified management cost function supporting multiple functional forms

    Parameters:
    size: conglomerate size (number of firms)
    cost_type: 'linear', 'quadratic', 'exponential', 'power_law'
    c0: base cost parameter (constant term) - if None, uses cost-function-specific default
    c1: linear scaling parameter - if None, uses cost-function-specific default
    c2: quadratic scaling parameter - if None, uses cost-function-specific default

    Default parameters are calibrated so all functions converge to ~0.0017 at size=40
    """

    # Cost-function-specific default parameters (calibrated for convergence at size=40)
    defaults = {'power_law': {'c0': 0.00002032, 'c1': 1.2, 'c2': 0.001},
        'linear': {'c0': 0.00001, 'c1': 0.00004225, 'c2': 0.001},
        'quadratic': {'c0': 0.0001, 'c1': 0.000001, 'c2': 0.00000097},
        'exponential': {'c0': 0.001, 'c1': 0.01326571, 'c2': 0.001}}

    if cost_type not in defaults:
        raise ValueError(
            f"Unknown cost_type: {cost_type}. Supported: 'linear', 'quadratic', 'exponential', 'power_law'")

    # Use provided parameters or defaults
    func_defaults = defaults[cost_type]
    c0 = c0 if c0 is not None else func_defaults['c0']
    c1 = c1 if c1 is not None else func_defaults['c1']
    c2 = c2 if c2 is not None else func_defaults['c2']

    if cost_type == 'linear':
        return c0 + c1 * size
    elif cost_type == 'quadratic':
        return c0 + c1 * size + c2 * size ** 2
    elif cost_type == 'exponential':
        return c0 * np.exp(c1 * size)
    elif cost_type == 'power_law':
        return c0 * size ** c1


def cost(size, progression, degree=2):
    """Legacy cost function - kept for backward compatibility"""
    if progression == 'linear':
        costs = size
    elif progression == 'polynomial':
        costs = size ** degree
    elif progression == 'logistic':
        costs = logistic_cost(size)
    else:
        costs = np.exp(size)

    return costs


def format_func(value, tick_number):
    return f'{value:.2f}'


def get_states_from_buffer(buffer, start_step, end_step, firms):
    """
    Extract states from circular buffer for given timesteps and firms.

    Parameters:
    - buffer: circular buffer (lookback+1, total_firms)
    - start_step: first timestep (inclusive)
    - end_step: last timestep (exclusive)
    - firms: int (single firm) or array (multiple firms)

    Returns:
    - 1D array (n_steps,) if firms is scalar
    - 2D array (n_steps, n_firms) if firms is array
    """
    lookback_plus_1 = buffer.shape[0]
    timesteps = np.arange(start_step, end_step)
    buffer_indices = timesteps % lookback_plus_1

    # Scalar firm: direct indexing
    if np.isscalar(firms):
        return buffer[buffer_indices, firms]

    # Array of firms: need broadcasting
    return buffer[buffer_indices[:, None], firms]


def exit_(firm_id, firm_conglom, firm_entered, firm_home_market, cong_firms, cong_size,
          cong_occupies_market, to_delete=None, step=None, exits_per_period=None):
    """
    SoA: Array-based exit function using Structure-of-Arrays with occupancy flags

    Parameters:
    -----------
    step : int, optional
        Current timestep (required for exit tracking)
    exits_per_period : ndarray, optional
        Array to track firm exits per period
    """
    cong_id = firm_conglom[firm_id]
    if cong_id != -1:
        # Track firm exit from conglomerate
        if step is not None and exits_per_period is not None:
            exits_per_period[step] += 1

        # SoA: Remove firm from array
        n_firms = cong_size[cong_id]

        # Get current firms
        current_firms = cong_firms[cong_id, :n_firms]

        # Remove firm
        new_firms = current_firms[current_firms != firm_id]
        new_n_firms = len(new_firms)

        # Update SoA arrays
        if new_n_firms > 0:
            cong_firms[cong_id, :new_n_firms] = new_firms
            # Update occupancy: rebuild from remaining firms
            cong_occupies_market[cong_id, :] = False
            cong_occupies_market[cong_id, firm_home_market[new_firms]] = True
        else:
            # No firms left - occupancy cleared in deallocate_cong_id
            pass

        cong_size[cong_id] = new_n_firms

        # Mark conglomerate for deletion if empty or only one firm remains
        if new_n_firms <= 1:
            # Clean up the remaining firm's state if there is one
            if new_n_firms == 1:
                remaining_firm_id = new_firms[0]
                firm_conglom[remaining_firm_id] = -1
                firm_entered[remaining_firm_id] = -1

            # Mark for deletion instead of deleting immediately
            if to_delete is not None:
                to_delete.add(cong_id)

    # Reset exiting firm's state
    firm_conglom[firm_id] = -1
    firm_entered[firm_id] = -1


@nb.njit(cache=True)
def logsumexp_numba_2d(x):
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


@nb.njit(cache=True)
def get_historical_states_numba(firm_log_states_buffer, start_step, end_step, firm_ids):
    """Extract historical log states from circular buffer"""
    lookback_plus_1 = firm_log_states_buffer.shape[0]
    n_steps = end_step - start_step
    n_firms = len(firm_ids)
    result = np.empty((n_steps, n_firms), np.float64)
    for t in range(n_steps):
        idx = (start_step + t) % lookback_plus_1
        for f in range(n_firms):
            result[t, f] = firm_log_states_buffer[idx, firm_ids[f]]
    return result


@nb.njit(cache=True)
def get_historical_returns_numba(firm_log_returns_buffer, start_step, end_step, firm_ids):
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


@nb.njit(cache=True)
def replay_member_growth(
    past_states, past_returns, step_states,
    alpha, management_cost, proportional, sharing_rule_code
):
    """
    Replay member growth under a given alpha and return per-member log growth.

    This is the factored-out counterfactual computation from merger_kernel_numba.

    Parameters:
    -----------
    past_states : ndarray, shape (h, K)
        Log states for h historical steps for K firms
    past_returns : ndarray, shape (h, K)
        Log returns for h historical steps for K firms
    step_states : ndarray, shape (1, K)
        Log states at the final step (for computing last growth)
    alpha : float
        Pooling fraction (share)
    management_cost : float
        Management cost Φ(K)
    proportional : bool
        Whether management cost is proportional
    sharing_rule_code : int
        0 = equal, 1 = proportional

    Returns:
    --------
    member_growth : ndarray, shape (K,)
        Mean log growth for each member under this alpha
    valid : bool
        False if any member would have invalid growth (exit condition)
    """
    h = past_states.shape[0]
    K = past_states.shape[1]

    member_growth = np.zeros(K, dtype=np.float64)

    # Precompute sum_Delta and sum_s for each tau
    sum_Delta_tau = np.empty(h, dtype=np.float64)
    sum_s_tau = np.empty(h, dtype=np.float64)
    for tau in range(h):
        sum_Delta = 0.0
        sum_s = 0.0
        for k in range(K):
            s_k_tau = np.exp(past_states[tau, k])
            r_k_tau = np.expm1(past_returns[tau, k])
            sum_Delta += s_k_tau * r_k_tau
            sum_s += s_k_tau
        sum_Delta_tau[tau] = sum_Delta
        sum_s_tau[tau] = sum_s

    # For each firm, compute synthetic log growth
    for j in range(K):
        g_hat = 0.0
        has_invalid = False

        for tau in range(h):
            s_i_tau = np.exp(past_states[tau, j])
            r_i_tau = np.expm1(past_returns[tau, j])
            Delta_i_tau = s_i_tau * r_i_tau

            sum_Delta = sum_Delta_tau[tau]
            sum_s = sum_s_tau[tau]

            if proportional:
                Omega_hat = alpha * sum_Delta * (1.0 - management_cost / sum_s) if sum_s > 0 else 0.0
            else:
                Omega_hat = alpha * sum_Delta - management_cost * sum_s

            if sharing_rule_code == 0:
                # equal: each firm gets equal dollars from pool
                Pi_hat_i = (1.0 - alpha) * Delta_i_tau + Omega_hat / K
            else:
                # proportional: each firm gets share proportional to its size
                w_i = s_i_tau / sum_s if sum_s > 0 else 0.0
                Pi_hat_i = (1.0 - alpha) * Delta_i_tau + w_i * Omega_hat

            growth_ratio = Pi_hat_i / s_i_tau if s_i_tau > 0 else -2.0
            if growth_ratio <= -1.0:
                has_invalid = True
                break

            g_hat += np.log1p(growth_ratio)

        if has_invalid:
            return member_growth, False

        member_growth[j] = g_hat / h

    return member_growth, True


@nb.njit(cache=True)
def compute_realized_growth(past_states, step_states):
    """
    Compute realized log growth for each member.

    Parameters:
    -----------
    past_states : ndarray, shape (h, K)
        Log states for h historical steps
    step_states : ndarray, shape (1, K)
        Log states at the final step

    Returns:
    --------
    realized_growth : ndarray, shape (K,)
        Mean log growth for each member
    """
    h = past_states.shape[0]
    K = past_states.shape[1]

    realized_growth = np.zeros(K, dtype=np.float64)

    for j in range(K):
        g_realized = 0.0
        for tau in range(h - 1):
            g_realized += past_states[tau + 1, j] - past_states[tau, j]
        g_realized += step_states[0, j] - past_states[h - 1, j]
        realized_growth[j] = g_realized / h

    return realized_growth


# Alpha grid for endogenous alpha search
ALPHA_GRID = np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
                       0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0])


@nb.njit(cache=True)
def find_optimal_alpha(
    past_states, past_returns, step_states,
    current_alpha, management_cost, proportional, sharing_rule_code
):
    """
    Find the optimal alpha that maximizes the minimum member gain relative to current alpha.

    Parameters:
    -----------
    past_states, past_returns, step_states : ndarrays
        Historical data for computing counterfactual growth
    current_alpha : float
        The conglomerate's current alpha
    management_cost : float
        Management cost Φ(K)
    proportional : bool
        Whether management cost is proportional
    sharing_rule_code : int
        0 = equal, 1 = proportional

    Returns:
    --------
    optimal_alpha : float
        The optimal alpha from the grid
    should_change : bool
        True if the optimal alpha improves on current (min gain > 0)
    """
    # Compute growth under current alpha
    current_growth, current_valid = replay_member_growth(
        past_states, past_returns, step_states,
        current_alpha, management_cost, proportional, sharing_rule_code
    )

    if not current_valid:
        # Current alpha is invalid, try to find any valid alpha
        current_growth = np.full(past_states.shape[1], -np.inf, dtype=np.float64)

    best_alpha = current_alpha
    best_min_gain = -np.inf
    should_change = False

    # Grid: {0, 0.05, ..., 1.0}
    for alpha_idx in range(21):
        alpha = alpha_idx * 0.05

        new_growth, valid = replay_member_growth(
            past_states, past_returns, step_states,
            alpha, management_cost, proportional, sharing_rule_code
        )

        if not valid:
            continue

        # Compute minimum gain relative to current alpha
        min_gain = np.inf
        for j in range(len(new_growth)):
            gain = new_growth[j] - current_growth[j]
            if gain < min_gain:
                min_gain = gain

        # Adopt if this alpha has the best min gain and it's positive
        if min_gain > best_min_gain:
            best_min_gain = min_gain
            best_alpha = alpha

    # Only change if min gain > 0
    if best_min_gain > 0 and best_alpha != current_alpha:
        should_change = True
    else:
        best_alpha = current_alpha
        should_change = False

    return best_alpha, should_change


@nb.njit(cache=True)
def merger_kernel_numba(
    step, lookback, share, proportional, merge_thresh,
    firm_conglom, firm_home_market, firm_entered,
    cong_firms, cong_size, cong_active, cong_occupies_market,
    cong_pool, management_costs_lookup,
    firm_log_states_buffer, firm_log_returns_buffer,
    total_firms, markets, firms_per_market,
    sharing_rule_code
):
    """
    FULLY NUMBA-ACCELERATED merger step.

    Returns tuple: (mergers_this_step, proposals_this_step)
    - proposals: feasible matches that reached the desirability test
    - mergers: proposals that were accepted
    """
    # Generate merger draws
    draws = np.random.random(total_firms) < merge_thresh
    candidates = np.where(draws)[0]

    if len(candidates) == 0:
        return 0, 0

    # Shuffle candidates
    n = len(candidates)
    for i in range(n-1, 0, -1):
        j = np.random.randint(0, i + 1)
        candidates[i], candidates[j] = candidates[j], candidates[i]

    mergers_this_step = 0
    proposals_this_step = 0

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

        # OVERLAP CHECK
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

        # Per-member merger test (same criterion as exit test)
        # Accept iff every firm i in the merged set would have higher synthetic
        # log growth inside the merged conglomerate than realized log growth
        # under their current arrangement.
        
        # Window: [t-h, t) where h = min(lookback, step)
        h = min(lookback, step)
        
        # Reject if no history available
        if h == 0:
            continue
        
        # This is a feasible proposal reaching the desirability test
        proposals_this_step += 1
        
        hist_start = step - h
        K_hat = merged_size  # Size of proposed merged set
        m = management_costs_lookup[K_hat]  # Management cost Phi(K_hat)
        
        # Get historical states and returns for merged firms
        # past_states[tau, j] = log_state of firm merged_firms[j] at time hist_start + tau
        past_states = get_historical_states_numba(firm_log_states_buffer, hist_start, step, merged_firms)
        past_returns = get_historical_returns_numba(firm_log_returns_buffer, hist_start, step, merged_firms)
        
        # Also need states at step (for computing growth from step-1 to step)
        # Actually we need states at hist_start to step (inclusive) for realized growth
        # past_states is [hist_start, step), we need [hist_start, step] for growth computation
        # Get the extra state at step
        step_states = get_historical_states_numba(firm_log_states_buffer, step, step + 1, merged_firms)
        
        accept = True

        # OPTIMIZATION: Precompute sum_Delta and sum_s for each tau ONCE before firm loop
        # This reduces complexity from O(K²·h) to O(K·h)
        sum_Delta_tau = np.empty(h, dtype=np.float64)
        sum_s_tau = np.empty(h, dtype=np.float64)
        for tau in range(h):
            sum_Delta = 0.0
            sum_s = 0.0
            for k in range(K_hat):
                s_k_tau = np.exp(past_states[tau, k])
                r_k_tau = np.expm1(past_returns[tau, k])
                sum_Delta += s_k_tau * r_k_tau
                sum_s += s_k_tau
            sum_Delta_tau[tau] = sum_Delta
            sum_s_tau[tau] = sum_s

        # For each firm in the merged set, compute synthetic vs realized growth
        for j in range(K_hat):
            firm_id = merged_firms[j]

            # Compute synthetic log growth g_hat_i
            g_hat = 0.0
            has_invalid = False

            for tau in range(h):
                # s_i_tau = exp(log_state)
                s_i_tau = np.exp(past_states[tau, j])
                # r_i_tau = expm1(log_return)
                r_i_tau = np.expm1(past_returns[tau, j])
                # Delta_i_tau = s_i_tau * r_i_tau
                Delta_i_tau = s_i_tau * r_i_tau

                # Use precomputed sums for synthetic pool Omega_hat_tau
                sum_Delta = sum_Delta_tau[tau]
                sum_s = sum_s_tau[tau]

                if proportional:
                    # Omega_hat = share * sum_Delta * (1 - m / sum_s)
                    Omega_hat = share * sum_Delta * (1.0 - m / sum_s) if sum_s > 0 else 0.0
                else:
                    # Omega_hat = share * sum_Delta - m * sum_s
                    Omega_hat = share * sum_Delta - m * sum_s

                # Synthetic per-member profit depends on sharing rule
                if sharing_rule_code == 0:
                    # equal: each firm gets equal dollars from pool
                    Pi_hat_i = (1.0 - share) * Delta_i_tau + Omega_hat / K_hat
                else:
                    # proportional: each firm gets share proportional to its size
                    w_i = s_i_tau / sum_s if sum_s > 0 else 0.0
                    Pi_hat_i = (1.0 - share) * Delta_i_tau + w_i * Omega_hat

                # Check for invalid growth (would cause exit)
                growth_ratio = Pi_hat_i / s_i_tau if s_i_tau > 0 else -2.0
                if growth_ratio <= -1.0:
                    has_invalid = True
                    break

                g_hat += np.log1p(growth_ratio)
            
            if has_invalid:
                accept = False
                break
            
            g_hat /= h  # Mean log growth
            
            # Compute realized log growth g_i under current arrangement
            # g_i = (1/h) * sum_tau (log_state[tau+1] - log_state[tau])
            g_realized = 0.0
            for tau in range(h - 1):
                g_realized += past_states[tau + 1, j] - past_states[tau, j]
            # Last step: from past_states[h-1] to step_states[0]
            g_realized += step_states[0, j] - past_states[h - 1, j]
            g_realized /= h
            
            # Reject if firm would not benefit
            if g_hat <= g_realized:
                accept = False
                break

        if not accept:
            continue

        # PERFORM MERGER
        merged_markets = firm_home_market[merged_firms]

        if initiator_cong != -1 and target_cong != -1:
            # Two congs → keep target, deallocate initiator
            cong_firms[target_cong, :merged_size] = merged_firms
            cong_size[target_cong] = merged_size

            # Update occupancy
            for m in merged_markets:
                cong_occupies_market[target_cong, m] = True

            for f in merged_firms:
                firm_conglom[f] = target_cong
                firm_entered[f] = step


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
                continue  # no free slot

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

    return mergers_this_step, proposals_this_step


@nb.njit(cache=True)
def compute_exit_candidates_numba(
    firm_outside_log_profits,      # shape (lookback, total_firms)
    firm_log_states_buffer,        # shape (lookback+1, total_firms)
    exit_candidates,               # 1D array of firm IDs
    lookback,
    step
):
    """
    Numba-compiled exit evaluation logic.

    Returns array of firm IDs that should exit (not a mask).
    """
    n = len(exit_candidates)
    if n == 0:
        return np.empty(0, dtype=np.int64)

    lookback_plus_1 = firm_log_states_buffer.shape[0]
    start_idx = (step - lookback) % lookback_plus_1

    # Track which firms should exit
    exit_list = np.empty(n, dtype=np.int64)
    exit_count = 0

    for firm_id in exit_candidates:
        # Outside profit: arithmetic mean of log profits → log of geometric mean
        log_outside = 0.0
        for t in range(lookback):
            log_outside += firm_outside_log_profits[t, firm_id]
        log_outside /= lookback

        # Extract lookback+1 states and compute returns using direct modulo arithmetic
        # This eliminates array allocation - just compute indices on the fly
        log_inside = 0.0
        for i in range(lookback):
            curr_idx = (start_idx + i) % lookback_plus_1
            next_idx = (start_idx + i + 1) % lookback_plus_1
            curr_state = firm_log_states_buffer[curr_idx, firm_id]
            next_state = firm_log_states_buffer[next_idx, firm_id]
            log_inside += (next_state - curr_state)
        log_inside /= lookback

        if log_outside > log_inside:
            exit_list[exit_count] = firm_id
            exit_count += 1

    return exit_list[:exit_count]


@nb.njit(cache=True)
def process_conglomerate_pooling_numba(
    cong_firms,
    cong_size,
    active_cong_ids,
    log_returns,
    log_states_curr,
    management_costs_lookup,
    share,
    proportional,
    lookback,
    step,
    sharing_rule_code,
    cong_alpha,
    alpha_endogenous
):
    """
    Numba-compiled pooling function for all active conglomerates.

    Pool formation is ALWAYS capital-weighted (eq. 9): Ω = α·Σ(w_i·r_i) - Φ/S

    Parameters:
    -----------
    sharing_rule_code : int
        0 = equal: pool distributed equally per dollar (δ_i = (1-α)r_i + Ω/(K·w_i))
        1 = proportional: pool distributed proportionally (δ_i = (1-α)r_i + Ω)
    cong_alpha : ndarray
        Per-conglomerate alpha values (used when alpha_endogenous=True)
    alpha_endogenous : bool
        If True, use cong_alpha[cong_id] instead of global share

    Returns:
    --------
    new_log_states : ndarray, shape (total_firms,)
        Updated log states for next timestep
    cong_pools : ndarray, shape (n_active_congs,)
        Pool values (pool_over_S) for each conglomerate
    firms_to_exit : ndarray
        IDs of firms that need to exit (d <= -1 or not finite)
    """
    n_congs = len(active_cong_ids)
    total_firms = log_states_curr.shape[0]

    # Output arrays
    new_log_states = np.copy(log_states_curr)
    cong_pools = np.zeros(n_congs, dtype=np.float64)

    # Track firms to exit (worst case: all firms could exit)
    exit_list = np.zeros(total_firms, dtype=np.int64)
    exit_count = 0

    # Process each active conglomerate
    for idx in range(n_congs):
        cong_id = active_cong_ids[idx]
        n_firms = cong_size[cong_id]

        if n_firms == 0:
            continue

        # Use per-conglomerate alpha if endogenous, otherwise global share
        if alpha_endogenous:
            alpha_c = cong_alpha[cong_id]
        else:
            alpha_c = share

        # Extract firms in this conglomerate
        conglomerate = cong_firms[cong_id, :n_firms]

        # Get log returns and states for this conglomerate
        log_ret = log_returns[conglomerate]
        log_st = log_states_curr[conglomerate]

        # Step 1: Compute log_S using manual logsumexp
        max_log_st = np.max(log_st)
        sum_exp = 0.0
        for i in range(n_firms):
            sum_exp += np.exp(log_st[i] - max_log_st)
        log_S = max_log_st + np.log(sum_exp)

        # Step 2: Compute normalized weights (for cap-weighted)
        w = np.exp(log_st - log_S)

        # Step 3: Compute r_minus1 = exp(log_returns) - 1
        r_m1 = np.expm1(log_ret)

        # Step 4: Weighted average gain (ALWAYS capital-weighted per eq. 9)
        # Pool = α·Σ(w_i·r_i) - Φ/S, regardless of sharing rule
        avg_gain_weighted = 0.0
        for i in range(n_firms):
            avg_gain_weighted += w[i] * r_m1[i]

        # Step 5: Management cost
        m = management_costs_lookup[n_firms]

        # Step 6: Pool calculation (eq. 9: Ω = α·Σ Δ_i - Φ·S)
        if proportional:
            mgmt_over_S = m * np.exp(-log_S)
            cost_factor = 1.0 - mgmt_over_S
            pool_over_S = alpha_c * avg_gain_weighted * cost_factor
        else:
            pool_over_S = alpha_c * avg_gain_weighted - m

        # Store pool value
        cong_pools[idx] = pool_over_S

        # Step 7: Compute delta for each firm (depends on sharing rule)
        K = n_firms
        for i in range(n_firms):
            firm = conglomerate[i]

            if sharing_rule_code == 0:
                # equal: each firm gets equal dollars from the pool
                # δ_i = (1-α)·r_i + Ω/(K·w_i)
                w_safe = max(w[i], 1e-300)
                delta = (1.0 - alpha_c) * r_m1[i] + pool_over_S / (K * w_safe)
            else:
                # proportional: each firm gets proportional to size
                # δ_i = (1-α)·r_i + Ω
                delta = (1.0 - alpha_c) * r_m1[i] + pool_over_S

            # Check for exit condition
            if not np.isfinite(delta) or delta <= -1.0:
                new_log_states[firm] = 0.0  # Reset to log(1)
                exit_list[exit_count] = firm
                exit_count += 1
            else:
                log_old = log_states_curr[firm]
                new_log_states[firm] = log_old + np.log1p(delta)

    # Return only the firms that need to exit
    firms_to_exit = exit_list[:exit_count]

    return new_log_states, cong_pools, firms_to_exit


def compute_hhi(shares):
    """
    Compute Herfindahl-Hirschman Index: sum of squared market shares.

    Parameters:
    -----------
    shares : array
        Market shares (should sum to 1, but will normalize if not)

    Returns:
    --------
    float : HHI value in [0, 1] where 1 = monopoly, 1/N = equal shares
    """
    shares = np.asarray(shares)
    total = shares.sum()
    if total <= 0:
        return 0.0
    normalized = shares / total
    return np.sum(normalized ** 2)


def compute_aggregate_hhi(firm_sizes, firm_conglom):
    """
    Compute aggregate HHI over control units.

    Each conglomerate is one unit with its members' total capital.
    Each standalone firm (firm_conglom == -1) is its own unit.

    Parameters:
    -----------
    firm_sizes : array
        Size (capital) of each firm
    firm_conglom : array
        Conglomerate ID for each firm (-1 = standalone)

    Returns:
    --------
    float : Aggregate HHI
    """
    firm_sizes = np.asarray(firm_sizes)
    firm_conglom = np.asarray(firm_conglom)

    total_capital = firm_sizes.sum()
    if total_capital <= 0:
        return 0.0

    # Aggregate capital by control unit
    unit_capitals = []

    # Standalone firms
    standalone = firm_conglom == -1
    for s in firm_sizes[standalone]:
        if s > 0:
            unit_capitals.append(s)

    # Conglomerates: sum capital of members
    cong_ids = np.unique(firm_conglom[firm_conglom >= 0])
    for cid in cong_ids:
        cong_capital = firm_sizes[firm_conglom == cid].sum()
        if cong_capital > 0:
            unit_capitals.append(cong_capital)

    if len(unit_capitals) == 0:
        return 0.0

    unit_capitals = np.array(unit_capitals)
    shares = unit_capitals / total_capital
    return np.sum(shares ** 2)


def compute_effective_members(sizes):
    """
    Compute effective number of members: 1 / sum(w_i^2).

    This is the inverse of the Herfindahl index on member weights,
    representing "how many equal-sized firms would give same concentration".

    Parameters:
    -----------
    sizes : array
        Sizes (capital) of conglomerate members

    Returns:
    --------
    float : Effective number of members
    """
    sizes = np.asarray(sizes)
    total = sizes.sum()
    if total <= 0 or len(sizes) == 0:
        return 0.0
    weights = sizes / total
    hhi = np.sum(weights ** 2)
    if hhi <= 0:
        return 0.0
    return 1.0 / hhi


def hill_estimator(sizes, k_fraction=0.1):
    """
    Hill estimator for tail index on top k_fraction of sizes.

    For Pareto distribution with P(X > x) ~ x^{-alpha}, this estimates alpha.

    Parameters:
    -----------
    sizes : array
        Sample of sizes (e.g., firm capitals)
    k_fraction : float
        Fraction of top order statistics to use (default 0.1 = top 10%)

    Returns:
    --------
    float : Estimated tail exponent alpha
    """
    sizes = np.asarray(sizes)
    sizes = sizes[sizes > 0]  # Remove zeros/negatives

    if len(sizes) < 2:
        return np.nan

    sorted_sizes = np.sort(sizes)[::-1]  # Descending order
    n = len(sorted_sizes)
    k = max(int(n * k_fraction), 1)

    if k >= n:
        k = n - 1
    if k < 1:
        return np.nan

    # Top k values: X_(1), X_(2), ..., X_(k)
    # Threshold: X_(k+1) (the k+1 th order statistic)
    top_k = sorted_sizes[:k]
    threshold = sorted_sizes[k]

    if threshold <= 0:
        return np.nan

    # Hill estimator: alpha_hat = k / sum(log(X_(i) / X_(k+1)))
    log_ratios = np.log(top_k / threshold)
    sum_log = log_ratios.sum()

    if sum_log <= 0:
        return np.nan

    return k / sum_log


def iqr_to_scale(family, iqr, nu=3.0):
    """
    Convert IQR to distribution scale parameter.

    Parameters:
    -----------
    family : str
        Distribution family: 'normal', 'laplace', 'student_t'
    iqr : float or array
        Inter-quartile range
    nu : float
        Degrees of freedom for Student-t (default 3.0)

    Returns:
    --------
    float or array : Scale parameter (σ for normal, b for Laplace, s for t)
    """
    if family == 'normal':
        # IQR = 2 * z_0.75 * σ = 2 * 0.6745 * σ ≈ 1.349 * σ
        return iqr / (2 * norm.ppf(0.75))
    elif family == 'laplace':
        # IQR = 2 * b * ln(2)
        return iqr / (2 * np.log(2))
    elif family == 'student_t' or family == 't3':
        # IQR = 2 * s * t_inv(0.75, df=nu)
        return iqr / (2 * student_t.ppf(0.75, df=nu))
    else:
        raise ValueError(f"Unknown family: {family}")


def model(params, seed=None, market_corr="identity",
          growth_process="normal_net", mu_range=(0.01, 0.1), sigma_range=(0.01, 0.05),
          sharing_rule="equal", rho=0.0, cross_corr=0.0,
          log_family="normal", nu=3.0, floor_c=0.0,
          metric_every=100, burn_in=0, alpha_endogenous=False):
    """
    Main simulation model.

    Parameters:
    -----------
    params : list
        Model parameters (10-14 elements depending on version)
    seed : int, optional
        Random seed for reproducibility
    market_corr : str
        Market correlation type: 'identity' (default) or 'random'
    growth_process : str
        Growth process type: 'normal_net' (Phase A default), 'lognormal', or 'log_family'
        - normal_net: r ~ N(μ, σ²), then log(1+r) is stored
        - lognormal: log δ = (μ - σ²/2) + σ·ε, so E[δ] = exp(μ)
        - log_family: log δ = μ + scale·ε, where scale from IQR (sigma_range)
    mu_range : tuple
        (min_mu, max_mu) bounds for mean growth rate
    sigma_range : tuple
        (min_sigma, max_sigma) bounds for volatility.
        Under log_family, interpreted as the IQR of log δ.
    sharing_rule : str
        Sharing rule: 'equal' (Phase A default) or 'proportional'
        - equal: pool distributed equally per dollar (δ_i = (1-α)r_i + Ω/(K·w_i))
        - proportional: pool distributed proportionally to size (δ_i = (1-α)r_i + Ω)
    rho : float
        Within-market correlation coefficient (default 0.0)
    cross_corr : float
        Cross-market correlation coefficient (default 0.0)
    log_family : str
        Distribution family for log_family process: 'normal', 'laplace', 'student_t'
        Only active when growth_process == 'log_family'. Default 'normal'.
    nu : float
        Degrees of freedom for Student-t distribution. Default 3.0.
        Only active when log_family == 'student_t'.
    floor_c : float
        Reflecting floor coefficient. Default 0.0 (off).
        When > 0, after state updates, firms below c × market_median are
        raised to that floor. Tracks floor_hits per firm and by status.
    metric_every : int
        Compute outcome metrics every N steps. Default 100.
    burn_in : int
        Steps to exclude from summary statistics. Default 0.
    alpha_endogenous : bool
        Enable endogenous alpha adaptation. Default False.
        When True, each conglomerate carries its own alpha, updated every
        lookback periods to maximize minimum member gain.
    """
    import time

    model_start_time = time.time()

    # Seed both NumPy and Numba RNGs if seed is provided
    if seed is not None:
        np.random.seed(seed)
        seed_numba(seed)

    # Unpack mu and sigma ranges from parameters
    min_mu, max_mu = mu_range
    min_sig, max_sig = sigma_range

    min_bound = np.array([min_mu, min_sig])
    max_bound = np.array([max_mu, max_sig])

    mu_sig_corr = 0.7
    means = [0.1, 0.05]

    # Extract parameters - now including cost function parameters
    # Removed debug prints for cleaner output

    if len(params) == 14:
        markets, firms_per_market, steps, share, total_firms, merge_thresh, comparison, break_thresh, proportional, lookback, cost_type, c0, c1, c2 = params
        use_custom_cost = True  # Parameter extraction complete
    elif len(params) == 13:
        # Backward compatibility with 13-parameter format (no c2)
        markets, firms_per_market, steps, share, total_firms, merge_thresh, comparison, break_thresh, proportional, lookback, cost_type, c0, c1 = params
        defaults = get_cost_function_defaults(cost_type)
        c2 = defaults['c2']  # Use cost-function-specific default for c2
        use_custom_cost = True
    elif len(params) == 12:
        # Backward compatibility with old b0, b1 parameters
        markets, firms_per_market, steps, share, total_firms, merge_thresh, comparison, break_thresh, proportional, lookback, b0, b1 = params
        cost_type = 'power_law'
        c0, c1 = b0, b1
        defaults = get_cost_function_defaults(cost_type)
        c2 = defaults['c2']  # Use cost-function-specific default for c2
        use_custom_cost = True
    else:
        markets, firms_per_market, steps, share, total_firms, merge_thresh, comparison, break_thresh, proportional, lookback = params
        cost_type = 'power_law'
        defaults = get_cost_function_defaults(cost_type)
        c0, c1, c2 = defaults['c0'], defaults['c1'], defaults['c2']  # Use all cost-function-specific defaults
        use_custom_cost = True

    cov_mat = np.ones((2, 2)) * mu_sig_corr + np.diag(np.tile(1 - mu_sig_corr, 2))
    growth_vars = np.random.multivariate_normal(means, cov_mat, markets)
    growth_vars = growth_vars + np.abs(np.minimum(growth_vars.min(axis=0), 0))
    growth_vars = min_bound + (growth_vars / growth_vars.max(axis=0)) * (max_bound - min_bound)

    # Market correlation: identity (uncorrelated, paper baseline) or random
    if market_corr == "identity":
        market_corr_matrix = np.eye(markets)
    elif market_corr == "random":
        # Generate random correlation matrix, seeded for reproducibility
        eigen_vals = np.random.uniform(0.1, 3, markets)
        eigen_vals = eigen_vals * markets / eigen_vals.sum()
        # Use seeded generator if seed was provided, otherwise use a fresh one
        rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        market_corr_matrix = random_correlation.rvs(tuple(eigen_vals), random_state=rng)
    else:
        raise ValueError(f"Unknown market_corr type: {market_corr}. Use 'identity' or 'random'.")

    # Sharing rule: convert string to int code for Numba
    # 0 = equal (distribute equally per dollar), 1 = proportional (distribute by size)
    sharing_rule_code = 0 if sharing_rule == "equal" else 1

    # Correlation structure: rho is now a float directly
    rho_val = float(rho)

    # Cross-market correlation: apply constant off-diagonal correlation
    if cross_corr != 0.0:
        # Create correlation matrix with cross_corr as off-diagonal elements
        cross_corr_matrix = np.full((markets, markets), cross_corr)
        np.fill_diagonal(cross_corr_matrix, 1.0)
        market_corr_matrix = cross_corr_matrix
    # else: keep existing market_corr_matrix (identity or random)

    market_cov = np.outer(growth_vars[:, 1], growth_vars[:, 1]) * market_corr_matrix

    # PERFORMANCE OPTIMIZATION: Precompute Cholesky decomposition for random number generation
    # This avoids recomputing the decomposition at every timestep (10,000x speedup for this operation)
    market_cov_cholesky = np.linalg.cholesky(market_cov)

    # MEMORY OPTIMIZATION: Store only what's needed for lookback windows (not full history)
    markets_structure = np.array([[x, y] for x in range(markets) for y in range(firms_per_market)])
    ids = np.arange(markets * firms_per_market).reshape(markets, firms_per_market)

    # Circular buffers: only store lookback+1 timesteps instead of full history
    firm_log_states_buffer = np.zeros((lookback + 1, total_firms), dtype=np.float64)
    firm_log_returns_buffer = np.zeros((lookback + 1, total_firms), dtype=np.float64)
    firm_outside_log_profits = np.zeros((lookback, total_firms), dtype=np.float64)

    # Floor tracking arrays (only used when floor_c > 0)
    floor_hits = np.zeros(total_firms, dtype=np.int32)  # Per-firm count
    floor_hits_by_status = np.zeros((steps, 2), dtype=np.int32)  # Per-step by status (0=standalone, 1=member)

    # Outcome metrics arrays (computed every metric_every steps)
    n_metric_steps = (steps + metric_every - 1) // metric_every  # Ceiling division
    hill_exponent = np.full((markets, n_metric_steps), np.nan, dtype=np.float64)
    hhi_within = np.zeros((markets, n_metric_steps), dtype=np.float64)
    hhi_aggregate = np.zeros(n_metric_steps, dtype=np.float64)
    top10_aggregate = np.zeros(n_metric_steps, dtype=np.float64)
    cong_capital_share = np.zeros(n_metric_steps, dtype=np.float64)
    # effective_members stored as list of (cong_id, K, K_eff) tuples per metric step
    effective_members_list = []

    # Alpha history for endogenous alpha tracking (sampled every metric_every steps)
    # Shape: (MAX_CONGLOMERATES, n_metric_steps) - NaN for inactive conglomerates
    MAX_CONGLOMERATES = total_firms // 2
    alpha_history = np.full((MAX_CONGLOMERATES, n_metric_steps), np.nan, dtype=np.float64)

    management_costs_lookup = np.array(
        [management_cost_function(size, cost_type, c0, c1, c2) if size > 0 else 0.0 for size in range(markets + 1)])

    firm_home_market = np.repeat(np.arange(markets, dtype=np.int16), firms_per_market)  # Home market for each firm
    firm_conglom = np.full(total_firms, -1, dtype=np.int32)  # -1 = solo firm, otherwise conglomerate ID
    firm_entered = np.full(total_firms, -1, dtype=np.int32)  # -1 = never entered, otherwise entry timestep

    # OPTIMIZATION: Structure-of-Arrays (SoA) for conglomerates - replaces dictionary
    # Pre-allocate fixed-size arrays for O(1) operations and better cache locality
    # Max conglomerates = total_firms/2 (each conglomerate has ≥2 firms, max 1 firm per market)
    MAX_CONGLOMERATES = total_firms // 2

    cong_firms = np.full((MAX_CONGLOMERATES, markets), -1, dtype=np.int32)
    cong_size = np.zeros(MAX_CONGLOMERATES, dtype=np.int16)
    cong_pool = np.zeros((MAX_CONGLOMERATES, lookback), dtype=np.float64)
    cong_active = np.zeros(MAX_CONGLOMERATES, dtype=bool)
    cong_occupies_market = np.zeros((MAX_CONGLOMERATES, markets), dtype=bool)  # Fast market occupancy lookup
    cong_alpha = np.full(MAX_CONGLOMERATES, share, dtype=np.float64)  # Per-conglomerate alpha (endogenous)

    # ID management for conglomerate slots
    free_ids = []
    next_cong_id = 0

    def allocate_cong_id():
        """Get next available conglomerate ID (reuse freed slots first)"""
        nonlocal next_cong_id
        if free_ids:
            return free_ids.pop()  # LIFO reuse (good for cache locality)
        else:
            if next_cong_id >= MAX_CONGLOMERATES:
                raise RuntimeError(f"Exceeded max conglomerates ({MAX_CONGLOMERATES})")
            cid = next_cong_id
            next_cong_id += 1
            return cid

    def deallocate_cong_id(cong_id):
        """Mark conglomerate as inactive and return ID to pool"""
        cong_active[cong_id] = False
        cong_size[cong_id] = 0
        cong_occupies_market[cong_id, :] = False  # Clear occupancy flags
        # Don't zero arrays - just use size=0 as sentinel for performance
        free_ids.append(cong_id)

    # Pre-allocate output arrays (computed during simulation, not post-processing)
    # MEMORY OPTIMIZATION: Only store current timestep snapshot, not full history
    market_share_current = np.zeros((markets, firms_per_market), dtype=np.float32)
    gini_coefficient = np.zeros((markets, steps), dtype=np.float32)

    # Online rank statistics (replaces full ranks array to save memory)
    # Track min/max/sum/sum_sq for each firm to compute rank_range and rank_std
    ranks_current = np.zeros((markets, firms_per_market), dtype=np.uint16)
    rank_min = np.full((markets, firms_per_market), np.iinfo(np.uint16).max, dtype=np.uint16)
    rank_max = np.zeros((markets, firms_per_market), dtype=np.uint16)
    rank_sum = np.zeros((markets, firms_per_market), dtype=np.float64)
    rank_sum_sq = np.zeros((markets, firms_per_market), dtype=np.float64)
    rank_count = np.zeros((markets, firms_per_market), dtype=np.uint32)

    # Pre-allocate quantiles array (computed on-the-fly)
    quantiles_shares = np.zeros((7, markets, steps), dtype=np.float32)  # 7 quantiles

    # Track merger frequency per period
    mergers_per_period = np.zeros(steps)
    proposals_per_period = np.zeros(steps)

    # Track firm exits from conglomerates per period
    exits_per_period = np.zeros(steps)

    # Pre-allocate lists for conglomerate metrics (computed on-the-fly)
    avg_shares = []
    avg_ranks = []
    num_cong = []
    members = []
    mean_members = np.zeros(steps, dtype=np.float32)
    quantiles_members = np.zeros((steps, 5), dtype=np.float32)  # 5 quantiles

    # Helper arrays for fast conglomerate indexing
    firm_to_market = markets_structure[:, 0]
    firm_to_local_idx = markets_structure[:, 1]


    for step in range(steps):
        # PERFORMANCE OPTIMIZATION: Use precomputed Cholesky decomposition for random generation
        # Generate standard normal random variates and transform using L @ z
        if rho_val == 0.0:
            # uncorr: independent firms within market (Phase A default)
            z = np.random.standard_normal((markets, firms_per_market))
        else:
            # Apply within-market correlation using factor model:
            # z_i = sqrt(|rho|) * sign(rho) * z_common + sqrt(1-|rho|) * z_idio
            z_common = np.random.standard_normal((markets, 1))
            z_idio = np.random.standard_normal((markets, firms_per_market))
            rho_abs = abs(rho_val)
            rho_sign = 1.0 if rho_val > 0 else -1.0
            z = rho_sign * np.sqrt(rho_abs) * z_common + np.sqrt(1 - rho_abs) * z_idio

        if growth_process == "normal_net":
            # Phase A default: r ~ N(μ, σ²), then log(1+r)
            realizations_step = growth_vars[:, 0][:, np.newaxis] + market_cov_cholesky @ z
            log_realizations_step = np.log1p(realizations_step.ravel()).astype(np.float64)
        elif growth_process == "log_family":
            # Log-family process: log δ = μ + scale·ε
            # E[log δ] = μ by construction (no -σ²/2 term)
            # sigma_range interpreted as IQR of log δ
            mu_m = growth_vars[:, 0]  # Market-specific location
            iqr_m = growth_vars[:, 1]  # Market-specific IQR
            scale_m = iqr_to_scale(log_family, iqr_m, nu)

            # Check if we need Gaussian copula (non-normal family with correlation)
            needs_copula = (log_family != 'normal') and (rho_val != 0.0 or cross_corr != 0.0)

            if needs_copula:
                # Gaussian copula: draw correlated normals, map through Φ, then family quantile
                # Step 1: Generate correlated standard normals
                # z already has within-market correlation from rho_val
                # Apply cross-market correlation via Cholesky of market correlation matrix
                z_corr = market_cov_cholesky @ z / growth_vars[:, 1][:, np.newaxis]  # Undo sigma scaling

                # Step 2: Map through Φ to get uniform marginals
                u = norm.cdf(z_corr)

                # Step 3: Map through family's quantile function (standardized to IQR=1)
                if log_family == 'laplace':
                    # Laplace quantile: sign(u-0.5) * b * ln(1 - 2|u-0.5|)
                    # For standard Laplace (b=1), IQR = 2*ln(2)
                    eps_raw = laplace.ppf(u)
                    eps = eps_raw / (2 * np.log(2))  # Standardize to IQR = 1
                elif log_family == 'student_t' or log_family == 't3':
                    eps_raw = student_t.ppf(u, df=nu)
                    eps = eps_raw / (2 * student_t.ppf(0.75, df=nu))  # Standardize to IQR = 1

                # log δ = μ + IQR * ε (where ε has IQR=1)
                log_realizations_step = (mu_m[:, np.newaxis] + iqr_m[:, np.newaxis] * eps).ravel().astype(np.float64)
            else:
                # No copula needed: either normal family, or no correlation
                if log_family == 'normal':
                    # For normal, z already has the right structure
                    # log δ = μ + scale * z, where scale = IQR / 1.349
                    log_realizations_step = (mu_m[:, np.newaxis] + scale_m[:, np.newaxis] * z).ravel().astype(np.float64)
                elif log_family == 'laplace':
                    # Independent Laplace draws (no correlation)
                    eps_raw = np.random.laplace(0, 1, (markets, firms_per_market))
                    eps = eps_raw / (2 * np.log(2))  # Standardize to IQR = 1
                    log_realizations_step = (mu_m[:, np.newaxis] + iqr_m[:, np.newaxis] * eps).ravel().astype(np.float64)
                elif log_family == 'student_t' or log_family == 't3':
                    # Independent Student-t draws (no correlation)
                    eps_raw = np.random.standard_t(df=nu, size=(markets, firms_per_market))
                    eps = eps_raw / (2 * student_t.ppf(0.75, df=nu))  # Standardize to IQR = 1
                    log_realizations_step = (mu_m[:, np.newaxis] + iqr_m[:, np.newaxis] * eps).ravel().astype(np.float64)
        else:
            # Lognormal: log δ = (μ - σ²/2) + σ·ε
            # This gives E[δ] = exp(μ), E[log δ] = μ - σ²/2
            drift = growth_vars[:, 0] - 0.5 * growth_vars[:, 1] ** 2
            log_realizations_step = (drift[:, np.newaxis] + market_cov_cholesky @ z).ravel().astype(np.float64)

        # Calculate circular buffer indices for current and next timestep
        curr_idx = step % (lookback + 1)
        next_idx = (step + 1) % (lookback + 1)

        # Store this step's log returns in circular buffer
        firm_log_returns_buffer[curr_idx] = log_realizations_step

        # ENDOGENOUS ALPHA: Update alpha for each active conglomerate every lookback periods
        # Must run BEFORE pooling so historical buffer data is still valid
        if alpha_endogenous and step > 0 and step % lookback == 0:
            h = min(lookback, step)
            hist_start = step - h

            # Get active conglomerates for alpha update
            active_cong_ids_for_alpha = np.where(cong_active)[0]

            for cid in active_cong_ids_for_alpha:
                K = cong_size[cid]
                if K < 2:
                    continue  # Need at least 2 members

                member_firms = cong_firms[cid, :K]
                m = management_costs_lookup[K]

                # Get historical data for this conglomerate
                # past_states and past_returns are from the circular buffer
                past_states = get_historical_states_numba(firm_log_states_buffer, hist_start, step, member_firms)
                past_returns = get_historical_returns_numba(firm_log_returns_buffer, hist_start, step, member_firms)
                step_states = get_historical_states_numba(firm_log_states_buffer, step, step + 1, member_firms)

                # Find optimal alpha
                optimal_alpha, should_change = find_optimal_alpha(
                    past_states, past_returns, step_states,
                    cong_alpha[cid], m, proportional, sharing_rule_code
                )

                if should_change:
                    cong_alpha[cid] = optimal_alpha

        # NUMBA OPTIMIZATION: Full merger pipeline in compiled code
        if merge_thresh > 0:
            num_mergers, num_proposals = merger_kernel_numba(
                step, lookback, share, proportional, merge_thresh,
                firm_conglom, firm_home_market, firm_entered,
                cong_firms, cong_size, cong_active, cong_occupies_market,
                cong_pool, management_costs_lookup,
                firm_log_states_buffer, firm_log_returns_buffer,
                total_firms, markets, firms_per_market,
                sharing_rule_code
            )
            mergers_per_period[step] = num_mergers
            proposals_per_period[step] = num_proposals

        # OPTIMIZATION: Vectorized mask for solo firms (faster than np.where for boolean operations)
        solo = firm_conglom == -1

        # SoA: Track conglomerates to delete after pooling loop
        conglomerates_to_delete = set()

        # NUMBA OPTIMIZATION: Process all conglomerate firms using JIT-compiled function
        active_cong_ids = np.where(cong_active)[0]

        if len(active_cong_ids) > 0:
            # Call Numba function for pooling logic
            new_log_states, cong_pools_step, firms_to_exit = process_conglomerate_pooling_numba(
                cong_firms,
                cong_size,
                active_cong_ids,
                log_realizations_step,
                firm_log_states_buffer[curr_idx],
                management_costs_lookup,
                share,
                proportional,
                lookback,
                step,
                sharing_rule_code,
                cong_alpha,
                alpha_endogenous
            )

            # Update log states in buffer
            firm_log_states_buffer[next_idx] = new_log_states

            # Store pools for active conglomerates
            for idx, cong_id in enumerate(active_cong_ids):
                cong_pool[cong_id, step % lookback] = cong_pools_step[idx]

            # Store outside log-profits for firms in conglomerates
            in_cong = firm_conglom != -1
            firm_outside_log_profits[step % lookback, in_cong] = log_realizations_step[in_cong]

            # Handle exits (must be done in Python due to exit_ function)
            for firm_id in firms_to_exit:
                exit_(firm_id, firm_conglom, firm_entered, firm_home_market, cong_firms, cong_size,
                      cong_occupies_market, to_delete=conglomerates_to_delete,
                      step=step, exits_per_period=exits_per_period)

        # SoA: Clean up conglomerates marked for deletion
        for cong_id in conglomerates_to_delete:
            deallocate_cong_id(cong_id)

        # OPTIMIZATION: Vectorized solo firm updates (eliminates loop over potentially 1000s of firms)
        firm_outside_log_profits[step % lookback, solo] = log_realizations_step[solo]
        firm_log_states_buffer[next_idx, solo] = firm_log_states_buffer[curr_idx, solo] + log_realizations_step[solo]

        # FLOOR: Apply reflecting floor at c × market mean (after pooling, before exit test)
        # The floor is floor_c × market_mean = (floor_c/N) × total_market_capital,
        # i.e. a minimum market share of floor_c/N. This is the Levy-Solomon barrier.
        # Iterate until convergence: raising firms changes the mean, which raises the floor.
        if floor_c > 0.0:
            current_log_states_flat = firm_log_states_buffer[next_idx]
            for m in range(markets):
                start_idx = m * firms_per_market
                end_idx = (m + 1) * firms_per_market

                # Iterate until no firms are below floor (raising firms raises mean and floor)
                for _ in range(100):  # Max iterations as safety
                    market_log_states = current_log_states_flat[start_idx:end_idx]

                    # Compute mean in levels, then log floor
                    mean_size = np.mean(np.exp(market_log_states))
                    log_floor = np.log(floor_c * mean_size)

                    # Find firms below floor
                    below_floor_mask = market_log_states < log_floor
                    if not below_floor_mask.any():
                        break  # Converged

                    # Get firm indices in global array
                    below_floor_local = np.where(below_floor_mask)[0]
                    below_floor_global = start_idx + below_floor_local

                    # Apply floor
                    firm_log_states_buffer[next_idx, below_floor_global] = log_floor

                    # Track hits per firm
                    floor_hits[below_floor_global] += 1

                    # Track by status: 0=standalone, 1=member
                    for firm_id in below_floor_global:
                        status = 0 if firm_conglom[firm_id] == -1 else 1
                        floor_hits_by_status[step, status] += 1

        # VECTORIZED: Exit checks using geometric means
        if step > lookback:
            exit_cleanup_to_delete = set()

            # Get firms in conglomerates that have been there long enough
            in_cong = firm_conglom != -1
            old_enough = firm_entered < step - lookback
            exit_candidates = np.where(in_cong & old_enough)[0]

            if len(exit_candidates) > 0:
                # NUMBA OPTIMIZATION: Compute exit decisions using JIT-compiled function
                # Returns firm IDs directly (not a mask)
                firms_to_exit = compute_exit_candidates_numba(
                    firm_outside_log_profits,
                    firm_log_states_buffer,
                    exit_candidates,
                    lookback,
                    step
                )

                # Process exits for firms that should leave
                for firm_id in firms_to_exit:
                    exit_(firm_id, firm_conglom, firm_entered, firm_home_market, cong_firms, cong_size,
                          cong_occupies_market, to_delete=exit_cleanup_to_delete,
                          step=step, exits_per_period=exits_per_period)

            # SoA: Clean up conglomerates marked for deletion
            for cong_id in exit_cleanup_to_delete:
                deallocate_cong_id(cong_id)

        # Reshape next_idx states for market-wise calculations
        current_log_states = firm_log_states_buffer[next_idx].reshape(markets, firms_per_market)

        # Market share calculation (current timestep only)
        log_market_totals = logsumexp(current_log_states, axis=1, keepdims=True)
        log_market_share = current_log_states - log_market_totals
        market_share_current[:, :] = np.exp(log_market_share).astype(np.float32)

        # VECTORIZED: Ranks and Gini for ALL markets at this timestep (not per-market loop)
        # Sort indices for all markets at once
        sorted_idx = np.argsort(market_share_current, axis=1).astype(np.uint8 if firms_per_market <= 256 else np.uint16)

        # Get sorted shares using fancy indexing
        sorted_shares = np.take_along_axis(market_share_current, sorted_idx, axis=1)

        # Gini calculation (vectorized across all markets)
        cum_shares = np.cumsum(sorted_shares, axis=1)
        sums = cum_shares[:, -1:]  # Total for each market

        with np.errstate(divide='ignore', invalid='ignore'):
            lorenz = cum_shares / sums
            lorenz = np.nan_to_num(lorenz)

        area = np.trapezoid(y=lorenz, axis=1, dx=1 / firms_per_market)
        gini_coefficient[:, step] = (1 - 2 * area).astype(np.float32)

        # Ranks (double argsort trick, vectorized) - online statistics
        ranks_current[:, :] = (np.argsort(sorted_idx, axis=1) + 1).astype(np.uint16)

        # Update online rank statistics
        rank_min = np.minimum(rank_min, ranks_current)
        rank_max = np.maximum(rank_max, ranks_current)
        rank_sum += ranks_current.astype(np.float64)
        rank_sum_sq += (ranks_current.astype(np.float64) ** 2)
        rank_count += 1

        # Quantiles of market share (computed per-market for this timestep)
        quantiles_shares[:, :, step] = np.quantile(market_share_current, q=[0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1], axis=1).astype(np.float32)

        # OPTIMIZATION: Leverage SoA structure to avoid expensive unique/where operations
        # Use cong_active and cong_size directly instead of scanning all firms
        active_cong_ids = np.where(cong_active)[0]
        num_cong.append(len(active_cong_ids))

        if len(active_cong_ids) > 0:
            # Direct access to sizes from SoA (no unique operation needed)
            active_sizes = cong_size[active_cong_ids]
            members.append(active_sizes.copy())

            # Compute mean and quantiles on-the-fly
            mean_members[step] = np.mean(active_sizes)
            quantiles_members[step] = np.quantile(active_sizes, q=[0.1, 0.25, 0.5, 0.75, 0.9])

            # PERFORMANCE: Vectorized avg_share and avg_rank using np.bincount
            # Replaces per-conglomerate loop with O(n) bincount aggregation
            in_cong_mask = firm_conglom != -1
            cids = firm_conglom[in_cong_mask]
            cnt = np.bincount(cids, minlength=MAX_CONGLOMERATES)
            sh_sum = np.bincount(cids, weights=market_share_current.ravel()[in_cong_mask], minlength=MAX_CONGLOMERATES)
            rk_sum = np.bincount(cids, weights=ranks_current.ravel()[in_cong_mask], minlength=MAX_CONGLOMERATES)
            avg_share = np.column_stack([active_sizes, sh_sum[active_cong_ids] / cnt[active_cong_ids]]).astype(np.float32)
            avg_rank = np.column_stack([active_sizes, rk_sum[active_cong_ids] / cnt[active_cong_ids]]).astype(np.float32)

            avg_shares.append(avg_share)
            avg_ranks.append(avg_rank)
        else:
            # No conglomerates this timestep
            members.append(np.array([]))
            avg_shares.append(np.array([]))
            avg_ranks.append(np.array([]))
            mean_members[step] = 0.0
            quantiles_members[step] = np.zeros(5)

        # OUTCOME METRICS: Compute every metric_every steps
        if (step + 1) % metric_every == 0 or step == steps - 1:
            metric_idx = step // metric_every
            if metric_idx < n_metric_steps:
                # Get current firm sizes in levels
                current_sizes = np.exp(firm_log_states_buffer[next_idx])

                # Hill exponent per market (top 10%)
                for m in range(markets):
                    market_sizes = current_sizes[m * firms_per_market:(m + 1) * firms_per_market]
                    hill_exponent[m, metric_idx] = hill_estimator(market_sizes, k_fraction=0.1)

                # HHI within each market
                for m in range(markets):
                    market_sizes = current_sizes[m * firms_per_market:(m + 1) * firms_per_market]
                    market_total = market_sizes.sum()
                    if market_total > 0:
                        market_shares = market_sizes / market_total
                        hhi_within[m, metric_idx] = compute_hhi(market_shares)

                # Aggregate metrics over control units
                hhi_aggregate[metric_idx] = compute_aggregate_hhi(current_sizes, firm_conglom)

                # Top 10% aggregate share
                unit_capitals = []
                standalone = firm_conglom == -1
                unit_capitals.extend(current_sizes[standalone].tolist())
                cong_ids_unique = np.unique(firm_conglom[firm_conglom >= 0])
                for cid in cong_ids_unique:
                    unit_capitals.append(current_sizes[firm_conglom == cid].sum())
                if len(unit_capitals) > 0:
                    unit_capitals = np.array(unit_capitals)
                    sorted_units = np.sort(unit_capitals)[::-1]
                    n_units = len(sorted_units)
                    top_k = max(1, n_units // 10)
                    total_capital = sorted_units.sum()
                    if total_capital > 0:
                        top10_aggregate[metric_idx] = sorted_units[:top_k].sum() / total_capital

                # Conglomerate capital share
                total_capital = current_sizes.sum()
                if total_capital > 0:
                    cong_capital = current_sizes[firm_conglom >= 0].sum()
                    cong_capital_share[metric_idx] = cong_capital / total_capital

                # Effective members for each active conglomerate
                step_eff_members = []
                for cid in active_cong_ids:
                    K = cong_size[cid]
                    if K > 0:
                        member_firms = cong_firms[cid, :K]
                        member_sizes = current_sizes[member_firms]
                        K_eff = compute_effective_members(member_sizes)
                        step_eff_members.append((cid, K, K_eff))
                effective_members_list.append((step, step_eff_members))

                # Track alpha history for active conglomerates
                if alpha_endogenous:
                    for cid in active_cong_ids:
                        if cong_active[cid]:
                            alpha_history[cid, metric_idx] = cong_alpha[cid]

    # Compute final rank mobility statistics from online accumulators
    # rank_range: max - min rank for each firm (measures total rank mobility)
    rank_range = (rank_max - rank_min).astype(np.float32)

    # rank_std: standard deviation of ranks for each firm
    # Var(X) = E[X²] - E[X]² = sum_sq/n - (sum/n)²
    with np.errstate(invalid='ignore'):
        rank_mean = rank_sum / rank_count
        rank_var = (rank_sum_sq / rank_count) - (rank_mean ** 2)
        rank_std = np.sqrt(np.maximum(rank_var, 0)).astype(np.float32)  # Clamp negative due to float precision

    # Compute summary statistics over t >= burn_in
    burn_in_metric_idx = burn_in // metric_every
    valid_metric_indices = slice(burn_in_metric_idx, n_metric_steps)

    # Median of each metric over valid steps
    summary = {
        'hill_exponent_median': np.nanmedian(hill_exponent[:, valid_metric_indices], axis=1),
        'hhi_within_median': np.nanmedian(hhi_within[:, valid_metric_indices], axis=1),
        'hhi_aggregate_median': np.nanmedian(hhi_aggregate[valid_metric_indices]),
        'top10_aggregate_median': np.nanmedian(top10_aggregate[valid_metric_indices]),
        'cong_capital_share_median': np.nanmedian(cong_capital_share[valid_metric_indices]),
    }

    # Median K and K_eff/K over valid steps
    all_K = []
    all_K_eff_over_K = []
    for step_val, step_eff in effective_members_list:
        if step_val >= burn_in:
            for cid, K, K_eff in step_eff:
                all_K.append(K)
                if K > 0:
                    all_K_eff_over_K.append(K_eff / K)
    summary['K_median'] = np.median(all_K) if len(all_K) > 0 else np.nan
    summary['K_eff_over_K_median'] = np.median(all_K_eff_over_K) if len(all_K_eff_over_K) > 0 else np.nan

    # Floor-hit rate per firm-period by status (for steps >= burn_in)
    if floor_c > 0.0 and steps > burn_in:
        valid_steps = steps - burn_in
        total_firm_periods = valid_steps * total_firms
        standalone_periods = np.sum(floor_hits_by_status[burn_in:, 0])
        member_periods = np.sum(floor_hits_by_status[burn_in:, 1])
        summary['floor_hit_rate_standalone'] = standalone_periods / total_firm_periods if total_firm_periods > 0 else 0.0
        summary['floor_hit_rate_member'] = member_periods / total_firm_periods if total_firm_periods > 0 else 0.0
    else:
        summary['floor_hit_rate_standalone'] = 0.0
        summary['floor_hit_rate_member'] = 0.0

    # Mergers, proposals, exits per period (for steps >= burn_in)
    if steps > burn_in:
        summary['mergers_per_period'] = np.mean(mergers_per_period[burn_in:])
        summary['proposals_per_period'] = np.mean(proposals_per_period[burn_in:])
        summary['exits_per_period'] = np.mean(exits_per_period[burn_in:])
    else:
        summary['mergers_per_period'] = 0.0
        summary['proposals_per_period'] = 0.0
        summary['exits_per_period'] = 0.0

    # Add hyperparameter metadata for result organization
    hyperparameters = {'markets': markets, 'firms_per_market': firms_per_market, 'steps': steps,
                       'merge_thresh': merge_thresh, 'comparison': comparison, 'break_thresh': break_thresh,
                       'proportional': proportional, 'lookback': lookback, 'cost_type': cost_type, 'c0': c0, 'c1': c1, 'c2': c2,
                       'growth_process': growth_process, 'mu_range': mu_range, 'sigma_range': sigma_range,
                       'sharing_rule': sharing_rule, 'rho': rho, 'cross_corr': cross_corr,
                       'log_family': log_family, 'nu': nu,
                       'floor_c': floor_c, 'floor_hits': floor_hits, 'floor_hits_by_status': floor_hits_by_status,
                       'final_log_states': firm_log_states_buffer[(steps - 1) % (lookback + 1)].copy(),
                       'metric_every': metric_every, 'burn_in': burn_in,
                       'hill_exponent': hill_exponent, 'hhi_within': hhi_within,
                       'hhi_aggregate': hhi_aggregate, 'top10_aggregate': top10_aggregate,
                       'cong_capital_share': cong_capital_share, 'effective_members': effective_members_list,
                       'alpha_endogenous': alpha_endogenous, 'alpha_history': alpha_history,
                       'summary': summary}

    # Model results: 13 elements
    # rank_range and rank_std replace the full ranks array (memory optimization)
    model_results = [mean_members, quantiles_members, num_cong, avg_shares, quantiles_shares,
                     gini_coefficient, avg_ranks, mergers_per_period, proposals_per_period,
                     exits_per_period, rank_range, rank_std, hyperparameters]
    total_time = time.time() - model_start_time
    print(f"Runtime: {total_time:.1f}s", flush=True)

    return model_results
