import numpy as np
from scipy.stats import random_correlation
from scipy.special import logsumexp, expm1


class Firm:
    def __init__(self, firm_id, home_market):
        self.id = int(firm_id)  # Ensure Python int, not numpy int64
        self.home_market = home_market
        self.entered = None  # Timestamp when firm entered a conglomerate
        self.conglomerate_id = None  # ID of conglomerate this firm belongs to (None if solo)  # Note: states and outside_profits are now stored in centralized arrays


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
          cong_occupies_market, to_delete=None):
    """
    SoA: Array-based exit function using Structure-of-Arrays with occupancy flags
    """
    cong_id = firm_conglom[firm_id]
    if cong_id != -1:
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


def model(params):
    """
    Main simulation model.

    Parameters:
    -----------
    params : list
        Model parameters (10-14 elements depending on version)
    """
    import time
    model_start_time = time.time()

    min_mu = 0.01
    max_mu = 0.1
    min_sig = 0.01
    max_sig = 0.05

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

    eigen_vals = np.random.uniform(0.1, 3, markets)
    eigen_vals = eigen_vals * markets / eigen_vals.sum()

    market_corr = random_correlation.rvs(tuple(eigen_vals), random_state=np.random.default_rng())

    # market_corr = np.diag(np.ones(markets))
    # market_corr[np.triu_indices(markets, k=1)] = np.random.uniform(0.1, 0.8, int(markets * (markets - 1) / 2))
    # market_corr = market_corr + market_corr.T - np.diag(np.ones(markets))
    market_cov = np.outer(growth_vars[:, 1], growth_vars[:, 1]) * market_corr

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

    effective_merge_thresh = 0 if share == 0 else merge_thresh

    for step in range(steps):
        # MEMORY OPTIMIZATION: Generate realizations for this step only (not all steps at once)
        realizations_step = np.random.multivariate_normal(
            growth_vars[:, 0], market_cov, size=firms_per_market
        )
        log_realizations_step = np.log(realizations_step.T.flatten() + 1).astype(np.float64)

        # Generate merger draws for this step
        merger_draws_step = np.random.uniform(0, 1, total_firms)
        draws = np.where(merger_draws_step < effective_merge_thresh)[0]

        # Calculate circular buffer indices for current and next timestep
        curr_idx = step % (lookback + 1)
        next_idx = (step + 1) % (lookback + 1)

        # Store this step's log returns in circular buffer
        firm_log_returns_buffer[curr_idx] = log_realizations_step

        if draws.shape[0] > 0:
            # OPTIMIZATION: Shuffle draws to remove ordering artifacts
            np.random.shuffle(draws)

            for firm in draws:
                # OPTIMIZATION: Fast occupancy-based merger targeting
                initiator_cong_id = firm_conglom[firm]

                # Fast full-conglomerate check using size
                if initiator_cong_id != -1 and cong_size[initiator_cong_id] == markets:
                    continue

                # OPTIMIZATION: O(markets) lookup instead of O(total_firms) filtering
                if initiator_cong_id != -1:
                    # Use occupancy bitmap for instant available market list
                    available = np.where(~cong_occupies_market[initiator_cong_id])[0]
                    if available.size == 0:
                        continue
                    target_market = np.random.choice(available)

                    # Get initiator firms
                    n_firms = cong_size[initiator_cong_id]
                    firm_firms = cong_firms[initiator_cong_id, :n_firms]
                else:
                    # Solo firm: any market except home market
                    firm_home = firm_home_market[firm]
                    available = np.arange(markets)
                    available = available[available != firm_home]
                    if available.size == 0:
                        continue
                    target_market = np.random.choice(available)
                    firm_firms = np.array([firm], dtype=np.int16)

                # Direct market-to-firm conversion (no np.where needed!)
                target_firm = target_market * firms_per_market + np.random.randint(firms_per_market)

                # Get target's firms
                target_cong_id = firm_conglom[target_firm]
                if target_cong_id != -1:
                    n_target_firms = cong_size[target_cong_id]
                    target_firms = cong_firms[target_cong_id, :n_target_firms]
                else:
                    target_firms = np.array([target_firm], dtype=np.int16)

                # Merge firms - already know no overlap by construction
                conglomerate = np.sort(np.concatenate([target_firms, firm_firms]))

                if len(conglomerate) > 2:
                    # VECTORIZED: Calculate synthetic pool for merger evaluation
                    hist_start = max(0, step - lookback)
                    hist_len = step - hist_start

                    # Store pool_over_S instead of absolute pool
                    synth_pool_over_S_time = np.zeros(lookback, dtype=np.float64)

                    if hist_len > 0:
                        # CIRCULAR BUFFER: Extract historical states and returns
                        past_log_states = get_states_from_buffer(
                            firm_log_states_buffer, hist_start, step, conglomerate
                        )
                        past_log_returns = get_states_from_buffer(
                            firm_log_returns_buffer, hist_start, step, conglomerate
                        )

                        # VECTORIZED: Compute pool for ALL historical periods at once
                        # Shape: past_log_states is (hist_len, n_firms)
                        log_S = logsumexp(past_log_states, axis=1)  # (hist_len,)
                        w = np.exp(past_log_states - log_S[:, None])  # (hist_len, n_firms)
                        r_minus1 = expm1(past_log_returns)  # (hist_len, n_firms)
                        avg_gain_weighted = np.sum(w * r_minus1, axis=1)  # (hist_len,)
                        m = management_costs_lookup[len(conglomerate)]

                        if proportional:
                            mgmt_over_S = m * np.exp(-log_S)  # (hist_len,)
                            cost_factor = 1.0 - mgmt_over_S
                            synth_pool_over_S_time[-hist_len:] = share * avg_gain_weighted * cost_factor
                        else:
                            synth_pool_over_S_time[-hist_len:] = share * avg_gain_weighted - m

                    # SoA: Compare synthetic pool to actual pool(s)
                    synth_pool_total = synth_pool_over_S_time.sum()

                    if initiator_cong_id == -1 or target_cong_id == -1:
                        # One conglomerate + one solo: beat the one existing pool
                        conglomerate_id = initiator_cong_id if target_cong_id == -1 else target_cong_id
                        true_pool_total = cong_pool[conglomerate_id].sum()

                        if synth_pool_total < true_pool_total:
                            continue
                    else:
                        # Two conglomerates: beat BOTH existing pools
                        true_pool0_total = cong_pool[initiator_cong_id].sum()
                        true_pool1_total = cong_pool[target_cong_id].sum()

                        if synth_pool_total < true_pool0_total or synth_pool_total < true_pool1_total:
                            continue

                # Get merged markets for occupancy update
                merged_markets = firm_home_market[conglomerate]

                # SoA: Handle four merger cases with occupancy flag maintenance
                if initiator_cong_id != -1 and target_cong_id != -1:
                    # Two conglomerates merge: keep target_cong_id, deallocate initiator
                    n_firms = len(conglomerate)

                    cong_firms[target_cong_id, :n_firms] = conglomerate
                    cong_size[target_cong_id] = n_firms
                    # Update occupancy: set all merged markets as occupied
                    cong_occupies_market[target_cong_id, merged_markets] = True

                    if len(conglomerate) > 2:
                        cong_pool[target_cong_id] = synth_pool_over_S_time

                    firm_conglom[conglomerate] = target_cong_id
                    firm_entered[conglomerate] = step

                    # Deallocate initiator (clears its occupancy flags)
                    deallocate_cong_id(initiator_cong_id)

                elif initiator_cong_id == -1 and target_cong_id != -1:
                    # Solo + conglomerate: expand target conglomerate
                    n_firms = len(conglomerate)

                    cong_firms[target_cong_id, :n_firms] = conglomerate
                    cong_size[target_cong_id] = n_firms
                    # Update occupancy: add new market
                    cong_occupies_market[target_cong_id, firm_home_market[firm]] = True

                    firm_conglom[firm] = target_cong_id
                    firm_entered[firm] = step

                elif initiator_cong_id != -1 and target_cong_id == -1:
                    # Conglomerate + solo: expand initiator conglomerate
                    n_firms = len(conglomerate)

                    cong_firms[initiator_cong_id, :n_firms] = conglomerate
                    cong_size[initiator_cong_id] = n_firms
                    # Update occupancy: add new market
                    cong_occupies_market[initiator_cong_id, firm_home_market[target_firm]] = True

                    firm_conglom[target_firm] = initiator_cong_id
                    firm_entered[target_firm] = step

                elif initiator_cong_id == -1 and target_cong_id == -1:
                    # Two solo firms: create new conglomerate
                    new_id = allocate_cong_id()
                    n_firms = len(conglomerate)

                    cong_firms[new_id, :n_firms] = conglomerate
                    cong_size[new_id] = n_firms
                    cong_pool[new_id] = np.zeros(lookback, dtype=np.float64)
                    cong_active[new_id] = True
                    # Set occupancy for both markets
                    cong_occupies_market[new_id, firm_home_market[firm]] = True
                    cong_occupies_market[new_id, firm_home_market[target_firm]] = True

                    firm_conglom[firm] = new_id
                    firm_conglom[target_firm] = new_id
                    firm_entered[firm] = step
                    firm_entered[target_firm] = step

                # Count successful merger
                mergers_per_period[step] += 1

        # OPTIMIZATION: Vectorized mask for solo firms (faster than np.where for boolean operations)
        solo = firm_conglom == -1

        # SoA: Track conglomerates to delete after pooling loop
        conglomerates_to_delete = set()

        # SoA: Process conglomerate firms (iterate over active conglomerates)
        active_cong_ids = np.where(cong_active)[0]
        for cong_id in active_cong_ids:
            # Extract firm list from SoA
            n_firms = cong_size[cong_id]
            conglomerate = cong_firms[cong_id, :n_firms]

            # CIRCULAR BUFFER: Get current states and returns
            log_returns = log_realizations_step[conglomerate]
            log_states = firm_log_states_buffer[curr_idx, conglomerate]

            # Compute log_S and weights
            log_S = logsumexp(log_states)
            w = np.exp(log_states - log_S)

            # r_minus1 vector
            r_minus1 = expm1(log_returns)

            # Average weighted gain
            avg_gain_weighted = np.sum(w * r_minus1)

            m = management_costs_lookup[len(conglomerate)]

            # Pool calculation
            if proportional:
                mgmt_over_S = m * np.exp(-log_S)
                cost_factor = 1.0 - mgmt_over_S
                pool_over_S = share * avg_gain_weighted * cost_factor
            else:
                pool_over_S = share * avg_gain_weighted - m

            # Compute delta_over_state for each firm
            K = len(conglomerate)
            w_safe = np.maximum(w, 1e-300)
            delta_over_state = (1.0 - share) * r_minus1 + pool_over_S / (K * w_safe)

            # Update log states
            for idx_local, firm in enumerate(conglomerate):
                log_old = firm_log_states_buffer[curr_idx, firm]
                d = delta_over_state[idx_local]
                if not np.isfinite(d) or d <= -1.0:
                    firm_log_states_buffer[next_idx, firm] = 0.0  # Reset to log(1)
                    exit_(firm, firm_conglom, firm_entered, firm_home_market, cong_firms, cong_size,
                          cong_occupies_market, to_delete=conglomerates_to_delete)
                else:
                    firm_log_states_buffer[next_idx, firm] = log_old + np.log1p(d)

                # Store outside log-profits
                firm_outside_log_profits[step % lookback, firm] = log_realizations_step[firm]

            # SoA: Store pool
            cong_pool[cong_id, step % lookback] = pool_over_S

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
                # VECTORIZED: Compute all outside profits at once
                log_outside_profits = np.mean(firm_outside_log_profits[:, exit_candidates], axis=0)

                # VECTORIZED: Get all recent states and compute inside returns
                recent_states = get_states_from_buffer(
                    firm_log_states_buffer, step - lookback, step + 1, exit_candidates
                )
                log_inside_returns = np.diff(recent_states, axis=0)
                log_inside_profits = np.mean(log_inside_returns, axis=0)

                # VECTORIZED: Find which firms should exit
                exit_mask = log_outside_profits > log_inside_profits

                # Process exits for firms that should leave
                for firm_id in exit_candidates[exit_mask]:
                    exit_(firm_id, firm_conglom, firm_entered, firm_home_market, cong_firms, cong_size,
                          cong_occupies_market, to_delete=exit_cleanup_to_delete)

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

        area = np.trapz(y=lorenz, axis=1, dx=1 / firms_per_market)
        gini_coefficient[:, step] = (1 - 2 * area).astype(np.float32)

        # Ranks (double argsort trick, vectorized)
        ranks[step, :, :] = (np.argsort(sorted_idx, axis=1) + 1).astype(np.uint16)

        # Quantiles of market share (computed per-market for this timestep)
        quantiles_shares[:, :, step] = np.quantile(market_share_current, q=[0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1], axis=1).astype(np.float32)

        # ON-THE-FLY: Compute conglomerate metrics for this timestep
        snapshot = firm_conglom
        in_cong = snapshot != -1

        if not in_cong.any():
            # No conglomerates this timestep
            num_cong.append(0)
            members.append(np.array([]))
            avg_shares.append(np.array([]))
            avg_ranks.append(np.array([]))
            mean_members[step] = 0.0
            quantiles_members[step] = np.zeros(5)
        else:
            # Get unique conglomerates and their sizes
            unique_congs, counts = np.unique(snapshot[in_cong], return_counts=True)
            num_cong.append(len(unique_congs))
            members.append(counts)  # Already a numpy array

            # Compute mean and quantiles on-the-fly
            mean_members[step] = np.mean(counts)
            quantiles_members[step] = np.quantile(counts, q=[0.1, 0.25, 0.5, 0.75, 0.9])

            avg_share = []
            avg_rank = []

            for cong_id in unique_congs:
                # Get all firms in this conglomerate
                firm_ids = np.where(snapshot == cong_id)[0]
                size = len(firm_ids)

                # Get market and local indices
                cong_markets = firm_to_market[firm_ids]
                local_idxs = firm_to_local_idx[firm_ids]

                # Direct indexing using this timestep's data (use current snapshot)
                avg_share_val = market_share_current[cong_markets, local_idxs].mean()
                avg_rank_val = ranks[step, cong_markets, local_idxs].mean()

                avg_share.append([size, avg_share_val])
                avg_rank.append([size, avg_rank_val])

            avg_shares.append(np.array(avg_share))
            avg_ranks.append(np.array(avg_rank))

    simulation_time = time.time() - model_start_time

    postprocessing_start_time = time.time()

    # Add hyperparameter metadata for result organization
    hyperparameters = {'markets': markets, 'firms_per_market': firms_per_market, 'steps': steps,
                       'merge_thresh': merge_thresh, 'comparison': comparison, 'break_thresh': break_thresh,
                       'proportional': proportional, 'lookback': lookback, 'cost_type': cost_type, 'c0': c0, 'c1': c1, 'c2': c2}

    # Hyperparameters stored successfully

    # MEMORY OPTIMIZATION: Removed market_share from results (no longer needed)
    model_results = [mean_members, quantiles_members, num_cong, avg_shares, quantiles_shares,
                     gini_coefficient, ranks, avg_ranks, mergers_per_period, hyperparameters]
    postprocessing_time = time.time() - postprocessing_start_time
    total_time = time.time() - model_start_time
    print(f"Post-processing: {postprocessing_time:.1f}s", flush=True)
    print(
        f"Total runtime: {total_time:.1f}s (simulation: {simulation_time:.1f}s, post-processing: {postprocessing_time:.1f}s)",
        flush=True)

    return model_results
