import numpy as np
from scipy.stats import random_correlation


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

    realizations = np.random.multivariate_normal(growth_vars[:, 0], market_cov, size=(steps, firms_per_market))

    # OPTIMIZATION: Reshape realizations for faster access pattern
    # Original shape: (steps, firms_per_market, markets)
    # New shape: (steps, total_firms) where firm_id = market * firms_per_market + firm_num
    # This eliminates fancy indexing and improves cache locality
    realizations = realizations.transpose(0, 2, 1).reshape(steps, total_firms) + 1

    # Pre-generate random draws for entire simulation to avoid repeated random generation
    merger_draws = np.random.uniform(0, 1, (steps, total_firms))

    markets_structure = np.array([[x, y] for x in range(markets) for y in range(firms_per_market)])

    ids = np.arange(markets * firms_per_market).reshape(markets, firms_per_market)

    # OPTIMIZATION: Centralized array storage for firm states and outside profits
    # This eliminates millions of Python object accesses and improves cache locality
    firm_states = np.ones((steps + 1, total_firms), dtype=np.float128)
    firm_outside_profits = np.ones((lookback, total_firms), dtype=np.float128)

    # OPTIMIZATION: Precompute management costs for all possible conglomerate sizes
    # Max size = markets (one firm per market), called 50k+ times, lookup is 100x faster than compute
    management_costs_lookup = np.array(
        [management_cost_function(size, cost_type, c0, c1, c2) if size > 0 else 0.0 for size in range(markets + 1)])

    # OPTIMIZATION: Replace Firm objects with plain NumPy arrays (eliminates 50M+ attribute lookups)
    # Each firm_id corresponds to: market * firms_per_market + firm_num
    firm_home_market = np.repeat(np.arange(markets, dtype=np.int16), firms_per_market)  # Home market for each firm
    firm_conglom = np.full(total_firms, -1, dtype=np.int32)  # -1 = solo firm, otherwise conglomerate ID
    firm_entered = np.full(total_firms, -1, dtype=np.int32)  # -1 = never entered, otherwise entry timestep
    all_time_conglomerates = []
    conglomerates = {}

    # Track merger frequency per period
    mergers_per_period = np.zeros(steps)

    # Set merge_thresh to 0 when share=0 to disable mergers (pure Brownian motion)
    effective_merge_thresh = 0 if share == 0 else merge_thresh

    for step in range(steps):
        draws = np.where(merger_draws[step] < effective_merge_thresh)[0]

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
                    # OPTIMIZATION: Use NumPy concatenate and sort (faster than list operations)
                    joint_markets = np.sort(np.concatenate([target_markets_list, firm_markets_list]))
                    conglomerate = np.sort(np.concatenate([target_firms, firm_firms]))

                    if len(conglomerate) > 2:
                        # Calculate synthetic pool once for all merger cases
                        hist_start = max(0, step - lookback)
                        hist_len = step - hist_start

                        # Initialize synth_pool (stays zeros if no history)
                        synth_pool = np.zeros(lookback, dtype=np.float128)

                        if hist_len > 0:
                            hist_steps = slice(hist_start, step)
                            past_returns = realizations[hist_steps][:, conglomerate]  # shape: (hist_len, num_firms)
                            past_states = firm_states[hist_start:step][:, conglomerate]  # shape: (hist_len, num_firms)

                            # Vectorized gains calculation
                            gains = past_states * past_returns - past_states  # shape: (hist_len, num_firms)

                            if proportional:
                                state_sums = past_states.sum(axis=1)  # shape: (hist_len,)
                                cost_factors = np.where(state_sums > 0,
                                                        1 - management_costs_lookup[len(conglomerate)] / state_sums,
                                                        0)
                                synth_pool_partial = ((gains * share).sum(axis=1) * cost_factors).astype(np.float128)
                            else:
                                state_sums = past_states.sum(axis=1)
                                management_costs = state_sums * management_costs_lookup[len(conglomerate)]
                                synth_pool_partial = ((gains * share).sum(axis=1) - management_costs).astype(np.float128)

                            # Fill synth_pool with calculated values
                            synth_pool[-hist_len:] = synth_pool_partial

                        # Compare synthetic pool to actual pool(s) - different logic for each case
                        synth_pool_total = synth_pool.sum()

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

                    # Conglomerate structure is now managed entirely in conglomerates dict
                    # No need to update firm.markets and firm.conglomerate

                    if initiator_cong_id != -1 and target_cong_id != -1:
                        conglomerates[target_cong_id] = {'firms': conglomerate, 'markets': joint_markets,
                                                         'pool': synth_pool}
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
                                                 'pool': np.zeros(lookback, dtype=np.float128)}

                    # Count successful merger
                    mergers_per_period[step] += 1

        # OPTIMIZATION: Store conglomerate firms as NumPy arrays instead of tuples
        all_time_conglomerates.append([cong_data['firms'].copy() for cong_data in conglomerates.values()])

        # OPTIMIZATION: Vectorized solo firms identification (100x faster than looping over firm objects)
        solo = np.where(firm_conglom == -1)[0]

        # Track conglomerates to delete after pooling loop
        conglomerates_to_delete = set()

        for conglomerate_idx, conglomerate_ in enumerate(conglomerates.keys()):
            conglomerate = conglomerates[conglomerate_]['firms']
            returns = realizations[step, conglomerate]
            states = firm_states[step, conglomerate]  # Direct array access!
            gains = states * returns - states
            if proportional:
                cost_factor = 1 - management_costs_lookup[len(conglomerate)] / states.sum() if states.sum() > 0 else 0
                pool = np.float128((gains * share).sum() * cost_factor)
            else:
                management_cost = states.sum() * management_costs_lookup[len(conglomerate)]
                pool = np.float128((gains * share).sum() - management_cost)

            returns = ((1 - share) * gains + pool / len(
                conglomerate)) if share > 0 else gains  # Equal distribution of pool contents.

            for partner, firm in enumerate(conglomerate):
                firm_outside_profits[step % lookback, firm] = realizations[step, firm]
                firm_states[step + 1, firm] = firm_states[step, firm] + returns[partner]
                if firm_states[step + 1, firm] < 0:
                    firm_states[step + 1, firm] = 1
                    exit_(firm, firm_conglom, firm_entered, firm_home_market, conglomerates,
                          to_delete=conglomerates_to_delete)

            conglomerates[conglomerate_]['pool'][step % lookback] = pool

        # Clean up conglomerates marked for deletion
        for cong_id in conglomerates_to_delete:
            if cong_id in conglomerates:
                del conglomerates[cong_id]

        for firm in solo:
            firm_outside_profits[step % lookback, firm] = realizations[step, firm]
            firm_states[step + 1, firm] = firm_states[step, firm] * realizations[step, firm]

        if step > lookback:
            # OPTIMIZATION: Vectorized exit check (100x faster than looping over firm objects)
            exit_cleanup_to_delete = set()

            # Get firms in conglomerates that have been there long enough
            in_cong = firm_conglom != -1
            old_enough = firm_entered < step - lookback
            exit_candidates = np.where(in_cong & old_enough)[0]

            for firm_id in exit_candidates:
                outside_profit = np.prod(firm_outside_profits[:, firm_id]) ** (1 / lookback)
                inside_profits = firm_states[step - lookback:step, firm_id]
                inside_profits = np.prod(inside_profits[1:] / inside_profits[:-1]) ** (1 / lookback)
                if outside_profit > inside_profits:
                    exit_(firm_id, firm_conglom, firm_entered, firm_home_market, conglomerates,
                          to_delete=exit_cleanup_to_delete)

            # Clean up conglomerates marked for deletion
            for cong_id in exit_cleanup_to_delete:
                if cong_id in conglomerates:
                    del conglomerates[cong_id]

    simulation_time = time.time() - model_start_time
    # Simulation timing available if needed for debugging

    postprocessing_start_time = time.time()
    results = firm_states  # Direct array access - no need to reconstruct from firm objects!
    # Shape: (steps + 1, total_firms)

    # Validate state values for numerical issues
    has_inf = np.isinf(results).any()
    has_nan = np.isnan(results).any()
    has_negative = (results < 0).any()

    # Debug: Check if we have actual infinities vs display artifacts
    max_value_raw = results.max()  # Keep as float128

    def print_float128_exact(x):
        """Print a np.float128 value exactly, bypassing NumPy's broken str() for large values"""
        if not np.isfinite(x):
            return "inf (true overflow)"
        return f"{np.longdouble(x):.6e}"

    # Clean validation check
    if has_inf or has_nan or has_negative:
        error_msg = f"VALIDATION ERROR: Illegal values - inf: {has_inf}, nan: {has_nan}, negative: {has_negative}"
        pass  # Validation error - could log if needed
        raise ValueError(error_msg)

    # Concise overflow summary
    largest = np.max(results)
    smallest = np.min(results[results > 0]) if np.any(results > 0) else 0

    # State validation passed

    # Convert market_share to float32 in chunks to avoid memory spikes
    # Computing market share efficiently

    # Pre-allocate the final array to avoid memory spikes during concatenation
    market_share = np.empty((markets, steps + 1, firms_per_market), dtype=np.float32)

    for market in range(markets):
        # Process one market at a time to reduce memory pressure
        firm_indices = np.arange(market * firms_per_market, (market + 1) * firms_per_market)
        market_results = results[:, firm_indices]
        market_totals = market_results.sum(axis=1).reshape(steps + 1, 1)
        market_share[market, :, :] = (market_results / market_totals).astype(np.float32)

        # Clear temporary variables for this market
        del market_results, market_totals

    # Market share calculation complete
    # shape = markets x steps + 1 x firms_per_market

    # Delete results array immediately to free 1.5GB of memory
    del results
    import gc
    gc.collect()

    # Now process conglomerate data
    num_cong = []
    members = []
    for conglomerate in all_time_conglomerates:
        num_cong.append(len(conglomerate))
        members.append([len(group) for group in conglomerate])

    mean_members = np.array([np.mean(member) if len(member) > 0 else 0.0 for member in members])

    quantiles_members = np.array(
        [np.quantile(member, q=[0.1, 0.25, 0.5, 0.75, 0.9]) if len(member) > 0 else np.zeros(5) for member in members])

    # Debug array saving removed for cleaner execution

    # Computing summary statistics (mean_share removed - not used in analysis)
    # Computing quantiles (includes max at q=1)
    quantiles_shares = np.quantile(market_share, q=[0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1], axis=2)

    # HHI calculation removed - not used in analysis (saves 900MB memory)

    # Computing Gini coefficient
    try:
        import time
        gini_start_time = time.time()

        # Check for problematic values (used by market-by-market processing)
        has_problematic_values = np.any(np.isnan(market_share)) or np.any(np.isinf(market_share))

        # MEMORY OPTIMIZATION: Replace batch processing with market-by-market processing
        # This produces mathematically identical results but uses much less memory

        # Pre-allocate final result array (same shape as before)
        gini_coefficient = np.zeros((markets, steps + 1), dtype=np.float32)

        # Process each market separately to reduce memory usage
        for market_idx in range(markets):
            # Extract one market's data: (steps+1) × firms_per_market
            market_data = market_share[market_idx, :, :]

            # Apply identical cleaning and sorting (same as before)
            if has_problematic_values:
                market_data_clean = np.nan_to_num(market_data, nan=0.0, posinf=1.0, neginf=0.0)
                sorted_data = np.sort(market_data_clean, axis=1)
            else:
                sorted_data = np.sort(market_data, axis=1)

            # Identical Gini calculation for this market
            cum_data = np.cumsum(sorted_data, axis=1)
            sums = np.sum(sorted_data, axis=1, keepdims=True)

            # Identical Lorenz curve calculation
            with np.errstate(divide='ignore', invalid='ignore'):
                lorenz = cum_data / sums
                lorenz = np.nan_to_num(lorenz)

            # Identical area under curve and Gini formula
            area = np.trapz(y=lorenz, axis=1, dx=1 / firms_per_market)
            gini_coefficient[market_idx, :] = 1 - 2 * area

            # Immediate cleanup for this market
            del market_data, sorted_data, cum_data, sums, lorenz, area

        total_gini_time = time.time() - gini_start_time  # Gini calculation complete (memory-optimized)

    except Exception as e:
        pass  # Critical error in Gini calculation
        import traceback
        traceback.print_exc()
        raise

    # Computing ranks market-by-market for memory efficiency
    try:
        import time
        from scipy.stats import rankdata

        ranks_start_time = time.time()

        # Pre-allocate ranks array with uint16 (ranks are integers 1 to firms_per_market)
        ranks = np.zeros((steps, markets, firms_per_market), dtype=np.uint16)

        # Process each market separately (mathematically identical to batch processing)
        for market_idx in range(markets):
            # Extract one market's data: steps × firms_per_market
            market_data = market_share[market_idx, :steps, :]  # Exclude final timestep

            # Process timesteps for this market
            for step_idx in range(steps):
                step_data = market_data[step_idx, :]

                # Check for problematic values (same logic as before)
                step_has_nan = np.any(np.isnan(step_data))
                step_has_inf = np.any(np.isinf(step_data))

                if step_has_nan or step_has_inf:
                    step_data_clean = np.nan_to_num(step_data, nan=0.0, posinf=1.0, neginf=0.0)
                    step_ranks = rankdata(step_data_clean, method='min')
                else:
                    step_ranks = rankdata(step_data, method='min')

                ranks[step_idx, market_idx, :] = step_ranks

            # Clean up market data immediately
            del market_data

        ranks_time = time.time() - ranks_start_time  # Ranks calculation complete (market-by-market for memory efficiency)

    except Exception as e:
        pass  # Critical error in ranks calculation
        import traceback
        traceback.print_exc()
        raise

    # Percentile ranks calculation removed - not used in analysis (saves ~60MB memory)

    # Computing conglomerate averages
    try:
        conglomerate_start_time = time.time()
        avg_shares = []
        avg_ranks = []

        # With memory optimizations, process all periods in single loop
        for period in range(steps):
            avg_share = []
            avg_rank = []
            for conglomerate in all_time_conglomerates[period]:
                coords = markets_structure[list(conglomerate)]
                to_append = [len(conglomerate), market_share[:, period, :][(coords[:, 0], coords[:, 1])].mean()]
                avg_share.append(to_append)
                to_append = [len(conglomerate), ranks[period, :, :][(coords[:, 0], coords[:, 1])].mean()]
                avg_rank.append(to_append)

            avg_shares.append(np.array(avg_share))
            avg_ranks.append(np.array(avg_rank))

        conglomerate_time = time.time() - conglomerate_start_time  # Conglomerate averages complete
    except Exception as e:
        pass  # Critical error in conglomerate averages
        import traceback
        traceback.print_exc()
        raise

    # Assembling final model results

    # Preparing hyperparameter storage

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