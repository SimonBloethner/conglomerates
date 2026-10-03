import numpy as np
from scipy.stats import random_correlation
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
    sharing_rule_code
):
    """
    Numba-compiled pooling function for all active conglomerates.

    Pool formation is ALWAYS capital-weighted (eq. 9): Ω = α·Σ(w_i·r_i) - Φ/S

    Parameters:
    -----------
    sharing_rule_code : int
        0 = equal: pool distributed equally per dollar (δ_i = (1-α)r_i + Ω/(K·w_i))
        1 = proportional: pool distributed proportionally (δ_i = (1-α)r_i + Ω)

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
            pool_over_S = share * avg_gain_weighted * cost_factor
        else:
            pool_over_S = share * avg_gain_weighted - m

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
                delta = (1.0 - share) * r_m1[i] + pool_over_S / (K * w_safe)
            else:
                # proportional: each firm gets proportional to size
                # δ_i = (1-α)·r_i + Ω
                delta = (1.0 - share) * r_m1[i] + pool_over_S

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


def model(params, seed=None, market_corr="identity",
          growth_process="normal_net", mu_range=(0.01, 0.1), sigma_range=(0.01, 0.05),
          sharing_rule="equal", pool_history="rolling", pool_window=None,
          rho="uncorr", cross_corr="none", mobility_csv=None):
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
        Growth process type: 'normal_net' (Phase A default) or 'lognormal'
        - normal_net: r ~ N(μ, σ²), then log(1+r) is stored
        - lognormal: log δ = (μ - σ²/2) + σ·ε, so E[δ] = exp(μ)
    mu_range : tuple
        (min_mu, max_mu) bounds for mean growth rate
    sigma_range : tuple
        (min_sigma, max_sigma) bounds for volatility
    sharing_rule : str
        Sharing rule: 'equal' (Phase A default) or 'proportional'
        - equal: pool distributed equally per dollar (δ_i = (1-α)r_i + Ω/(K·w_i))
        - proportional: pool distributed proportionally to size (δ_i = (1-α)r_i + Ω)
    pool_history : str
        Pool history mode: 'rolling' (default) or 'full'
    pool_window : int, optional
        Rolling window size (default: same as lookback)
    rho : str
        Within-market correlation: 'uncorr' (ρ=0), 'pos' (ρ=0.3), 'neg' (ρ=-0.3)
    cross_corr : str
        Cross-market correlation: 'block' (industry groups), 'ar1' (distance decay), 'none'
    mobility_csv : str, optional
        Path to write online mobility metrics (rank autocorrelation per step)
    """
    import time

    model_start_time = time.time()

    # Seed both NumPy and Numba RNGs if seed is provided
    if seed is not None:
        np.random.seed(seed)
        seed_numba(seed)

    # Open mobility CSV file if requested
    mobility_file = None
    if mobility_csv is not None:
        mobility_file = open(mobility_csv, 'w')
        mobility_file.write('step,rank_autocorr\n')

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

    # Pool window defaults to lookback if not specified
    if pool_window is None:
        pool_window = lookback

    # Correlation structure: convert rho to numeric value
    rho_values = {'uncorr': 0.0, 'pos': 0.3, 'neg': -0.3}
    rho_val = rho_values.get(rho, 0.0)

    # Cross-market correlation: modify market_corr_matrix
    if cross_corr == "block":
        # Industry block structure: divide markets into 4 groups with high within-group correlation
        n_blocks = min(4, markets)
        block_size = markets // n_blocks
        block_corr_matrix = np.eye(markets)
        for b in range(n_blocks):
            start = b * block_size
            end = start + block_size if b < n_blocks - 1 else markets
            for i in range(start, end):
                for j in range(start, end):
                    if i != j:
                        block_corr_matrix[i, j] = 0.5  # Within-block correlation
        market_corr_matrix = block_corr_matrix
    elif cross_corr == "ar1":
        # AR(1) decay: correlation decays with "distance" between markets
        ar1_matrix = np.zeros((markets, markets))
        decay = 0.7  # AR(1) coefficient
        for i in range(markets):
            for j in range(markets):
                ar1_matrix[i, j] = decay ** abs(i - j)
        market_corr_matrix = ar1_matrix
    # else: cross_corr == "none", keep existing market_corr_matrix (identity or random)

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
    ranks = np.zeros((steps, markets, firms_per_market), dtype=np.uint16)

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
                sharing_rule_code
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

        # Ranks (double argsort trick, vectorized)
        ranks[step, :, :] = (np.argsort(sorted_idx, axis=1) + 1).astype(np.uint16)

        # Online mobility tracking: rank autocorrelation between consecutive steps
        if mobility_file is not None and step > 0:
            # Spearman rank correlation: ρ = 1 - (6 * Σd²) / (n * (n² - 1))
            # where d is the difference in ranks between consecutive steps
            prev_ranks = ranks[step - 1].ravel().astype(np.float64)
            curr_ranks = ranks[step].ravel().astype(np.float64)
            n = len(prev_ranks)
            d_sq_sum = np.sum((curr_ranks - prev_ranks) ** 2)
            rank_autocorr = 1.0 - (6.0 * d_sq_sum) / (n * (n * n - 1))
            mobility_file.write(f'{step},{rank_autocorr:.6f}\n')

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
            rk_sum = np.bincount(cids, weights=ranks[step].ravel()[in_cong_mask], minlength=MAX_CONGLOMERATES)
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

    # Add hyperparameter metadata for result organization
    hyperparameters = {'markets': markets, 'firms_per_market': firms_per_market, 'steps': steps,
                       'merge_thresh': merge_thresh, 'comparison': comparison, 'break_thresh': break_thresh,
                       'proportional': proportional, 'lookback': lookback, 'cost_type': cost_type, 'c0': c0, 'c1': c1, 'c2': c2,
                       'growth_process': growth_process, 'mu_range': mu_range, 'sigma_range': sigma_range,
                       'sharing_rule': sharing_rule, 'pool_history': pool_history, 'pool_window': pool_window,
                       'rho': rho, 'cross_corr': cross_corr}

    # Hyperparameters stored successfully

    # MEMORY OPTIMIZATION: Removed market_share from results (no longer needed)
    # Now includes: avg_shares (firm-level size/market-share data) and exits_per_period
    # Close mobility CSV file if opened
    if mobility_file is not None:
        mobility_file.close()

    model_results = [mean_members, quantiles_members, num_cong, avg_shares, quantiles_shares,
                     gini_coefficient, ranks, avg_ranks, mergers_per_period, proposals_per_period,
                     exits_per_period, hyperparameters]
    total_time = time.time() - model_start_time
    print(f"Runtime: {total_time:.1f}s", flush=True)

    return model_results
