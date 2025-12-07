import numpy as np
from scipy.stats import random_correlation
from scipy.special import logsumexp, expm1

from programs.counterfactuals import firms_per_market


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


def exit_(firm_id, firm_conglom, firm_entered, firm_home_market, conglomerates, to_delete=None):
    """
    OPTIMIZED: Array-based exit function (no Firm objects needed)
    """
    cong_id = firm_conglom[firm_id]
    if cong_id != -1:
        # OPTIMIZATION: Use NumPy array operations for removal (faster than list.remove())
        cong_firms = conglomerates[cong_id]['firms']
        cong_markets = conglomerates[cong_id]['markets']

        conglomerates[cong_id]['firms'] = cong_firms[cong_firms != firm_id]
        conglomerates[cong_id]['markets'] = cong_markets[cong_markets != firm_home_market[firm_id]]

        # Mark conglomerate for deletion if empty or only one firm remains
        remaining_firms = conglomerates[cong_id]['firms']
        if len(remaining_firms) <= 1:
            # Clean up the remaining firm's state if there is one
            if len(remaining_firms) == 1:
                remaining_firm_id = remaining_firms[0]
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

    conglomerate_snapshots = np.full((steps, total_firms), -1, dtype=np.int32)
    conglomerates = {}

    # Pre-allocate output arrays (computed during simulation, not post-processing)
    market_share = np.zeros((markets, steps, firms_per_market), dtype=np.float32)
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
            for firm_idx, firm in enumerate(draws):
                # Get active markets from conglomerates dict or home market for solo firms
                initiator_cong_id = firm_conglom[firm]
                if initiator_cong_id != -1:
                    active_markets = conglomerates[initiator_cong_id]['markets']
                else:
                    active_markets = np.array([firm_home_market[firm]], dtype=np.int16)

                if len(active_markets) == markets:
                    continue

                target_markets = markets_structure[np.logical_not(np.in1d(markets_structure[:, 0], active_markets)), :]
                target = target_markets[np.random.choice(np.arange(target_markets.shape[0]), 1)]
                target = np.where((markets_structure == target).all(axis=1))[0][0]

                # Get target's markets and firms
                target_cong_id = firm_conglom[target]
                if target_cong_id != -1:
                    target_markets_list = conglomerates[target_cong_id]['markets']
                    target_firms = conglomerates[target_cong_id]['firms']
                else:
                    target_markets_list = np.array([firm_home_market[target]], dtype=np.int16)
                    target_firms = np.array([target], dtype=np.int16)

                if initiator_cong_id != -1:
                    firm_markets_list = conglomerates[initiator_cong_id]['markets']
                    firm_firms = conglomerates[initiator_cong_id]['firms']
                else:
                    firm_markets_list = np.array([firm_home_market[firm]], dtype=np.int16)
                    firm_firms = np.array([firm], dtype=np.int16)

                # Check for market overlap
                if not np.isin(target_markets_list, firm_markets_list).any():
                    joint_markets = np.sort(np.concatenate([target_markets_list, firm_markets_list]))
                    conglomerate = np.sort(np.concatenate([target_firms, firm_firms]))

                    if len(conglomerate) > 2:
                        # LOG-SPACE: Calculate synthetic pool for merger evaluation
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

                            # Compute pool for each historical period using log-space
                            for h_idx in range(hist_len):
                                log_states_row = past_log_states[h_idx]
                                log_S = logsumexp(log_states_row)
                                w = np.exp(log_states_row - log_S)  # Normalized weights
                                r_minus1 = expm1(past_log_returns[h_idx])
                                avg_gain_weighted = np.sum(w * r_minus1)
                                m = management_costs_lookup[len(conglomerate)]

                                if proportional:
                                    mgmt_over_S = m * np.exp(-log_S)
                                    cost_factor = 1.0 - mgmt_over_S
                                    synth_pool_over_S_time[h_idx] = share * avg_gain_weighted * cost_factor
                                else:
                                    synth_pool_over_S_time[h_idx] = share * avg_gain_weighted - m

                            # Shift to match lookback window
                            synth_pool_over_S_full = np.zeros(lookback, dtype=np.float64)
                            synth_pool_over_S_full[-hist_len:] = synth_pool_over_S_time[:hist_len]
                            synth_pool_over_S_time = synth_pool_over_S_full

                        # LOG-SPACE: Compare synthetic pool to actual pool(s)
                        synth_pool_total = synth_pool_over_S_time.sum()

                        if initiator_cong_id == -1 or target_cong_id == -1:
                            # One conglomerate + one solo: beat the one existing pool
                            conglomerate_id = initiator_cong_id if target_cong_id == -1 else target_cong_id
                            true_pool_total = conglomerates[conglomerate_id]['pool'].sum()

                            if synth_pool_total < true_pool_total:
                                continue
                        else:
                            # Two conglomerates: beat BOTH existing pools
                            true_pool0_total = conglomerates[initiator_cong_id]['pool'].sum()
                            true_pool1_total = conglomerates[target_cong_id]['pool'].sum()

                            if synth_pool_total < true_pool0_total or synth_pool_total < true_pool1_total:
                                continue

                    if target_cong_id == -1 and initiator_cong_id == -1:
                        existing = np.array(list(conglomerates.keys()))
                        if len(existing) == 0:
                            new_id = 0
                        else:
                            all_ = np.arange(len(conglomerates.keys()))
                            lowest_new = all_[np.logical_not(np.isin(all_, existing))]
                            if lowest_new.shape[0] == 0:
                                new_id = np.max(all_) + 1
                            else:
                                new_id = np.min(lowest_new)


                    if initiator_cong_id != -1 and target_cong_id != -1:
                        conglomerates[target_cong_id] = {'firms': conglomerate, 'markets': joint_markets,
                                                         'pool': synth_pool_over_S_time}
                        firm_conglom[conglomerate] = target_cong_id
                        firm_entered[conglomerate] = step
                        del conglomerates[initiator_cong_id]

                    elif initiator_cong_id == -1 and target_cong_id != -1:
                        conglomerates[target_cong_id]['firms'] = conglomerate
                        conglomerates[target_cong_id]['markets'] = joint_markets
                        firm_conglom[firm] = target_cong_id
                        firm_entered[firm] = step

                    elif initiator_cong_id != -1 and target_cong_id == -1:
                        conglomerates[initiator_cong_id]['firms'] = conglomerate
                        conglomerates[initiator_cong_id]['markets'] = joint_markets
                        firm_conglom[target] = initiator_cong_id
                        firm_entered[target] = step

                    elif initiator_cong_id == -1 and target_cong_id == -1:
                        firm_conglom[firm] = new_id
                        firm_conglom[target] = new_id
                        firm_entered[firm] = step
                        firm_entered[target] = step
                        conglomerates[new_id] = {'firms': conglomerate, 'markets': joint_markets,
                                                 'pool': np.zeros(lookback, dtype=np.float64)}

                    # Count successful merger
                    mergers_per_period[step] += 1

        conglomerate_snapshots[step] = firm_conglom

        # OPTIMIZATION: Vectorized mask for solo firms (faster than np.where for boolean operations)
        solo = firm_conglom == -1

        # Track conglomerates to delete after pooling loop
        conglomerates_to_delete = set()

        # LOG-SPACE: Process conglomerate firms
        for conglomerate_idx, conglomerate_ in enumerate(conglomerates.keys()):
            conglomerate = conglomerates[conglomerate_]['firms']

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
                    exit_(firm, firm_conglom, firm_entered, firm_home_market, conglomerates,
                          to_delete=conglomerates_to_delete)
                else:
                    firm_log_states_buffer[next_idx, firm] = log_old + np.log1p(d)

                # Store outside log-profits
                firm_outside_log_profits[step % lookback, firm] = log_realizations_step[firm]

            # Store pool
            conglomerates[conglomerate_]['pool'][step % lookback] = pool_over_S

        # Clean up conglomerates marked for deletion
        for cong_id in conglomerates_to_delete:
            if cong_id in conglomerates:
                del conglomerates[cong_id]

        # OPTIMIZATION: Vectorized solo firm updates (eliminates loop over potentially 1000s of firms)
        firm_outside_log_profits[step % lookback, solo] = log_realizations_step[solo]
        firm_log_states_buffer[next_idx, solo] = firm_log_states_buffer[curr_idx, solo] + log_realizations_step[solo]

        # LOG-SPACE: Exit checks using geometric means
        if step > lookback:
            exit_cleanup_to_delete = set()

            # Get firms in conglomerates that have been there long enough
            in_cong = firm_conglom != -1
            old_enough = firm_entered < step - lookback
            exit_candidates = np.where(in_cong & old_enough)[0]

            for firm_id in exit_candidates:
                # Geometric mean in log space: exp(mean(log_returns))
                log_outside_profit = np.mean(firm_outside_log_profits[:, firm_id])

                # CIRCULAR BUFFER: Get recent states to compute inside returns
                recent_states = get_states_from_buffer(
                    firm_log_states_buffer, step - lookback, step + 1, firm_id
                )
                log_inside_returns = np.diff(recent_states)
                log_inside_profit = np.mean(log_inside_returns)

                if log_outside_profit > log_inside_profit:
                    exit_(firm_id, firm_conglom, firm_entered, firm_home_market, conglomerates,
                          to_delete=exit_cleanup_to_delete)

            # Clean up conglomerates marked for deletion
            for cong_id in exit_cleanup_to_delete:
                if cong_id in conglomerates:
                    del conglomerates[cong_id]

        # Reshape next_idx states for market-wise calculations
        current_log_states = firm_log_states_buffer[next_idx].reshape(markets, firms_per_market)

        # Market share calculation
        log_market_totals = logsumexp(current_log_states, axis=1, keepdims=True)
        log_market_share = current_log_states - log_market_totals
        market_share[:, step, :] = np.exp(log_market_share).astype(np.float32)

        # VECTORIZED: Ranks and Gini for ALL markets at this timestep (not per-market loop)
        shares_all_markets = market_share[:, step, :]  # Shape: (markets, firms_per_market)

        # Sort indices for all markets at once
        sorted_idx = np.argsort(shares_all_markets, axis=1).astype(np.uint8 if firms_per_market <= 256 else np.uint16)

        # Get sorted shares using fancy indexing
        sorted_shares = np.take_along_axis(shares_all_markets, sorted_idx, axis=1)

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
        quantiles_shares[:, :, step] = np.quantile(shares_all_markets, q=[0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1], axis=1).astype(np.float32)

        # ON-THE-FLY: Compute conglomerate metrics for this timestep
        snapshot = firm_conglom  # Already stored in conglomerate_snapshots[step]
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

                # Direct indexing using this timestep's data
                avg_share_val = market_share[cong_markets, step, local_idxs].mean()
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

    model_results = [mean_members, quantiles_members, num_cong, avg_shares, quantiles_shares, market_share,
                     gini_coefficient, ranks, avg_ranks, mergers_per_period, hyperparameters]
    postprocessing_time = time.time() - postprocessing_start_time
    total_time = time.time() - model_start_time
    print(f"Post-processing: {postprocessing_time:.1f}s", flush=True)
    print(
        f"Total runtime: {total_time:.1f}s (simulation: {simulation_time:.1f}s, post-processing: {postprocessing_time:.1f}s)",
        flush=True)

    return model_results
