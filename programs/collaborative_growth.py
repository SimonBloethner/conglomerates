import numpy as np
from scipy.stats import random_correlation
from tqdm import tqdm
import time


class Firm:
    def __init__(self, market, number, steps, lookback):
        self.id = int(number)  # Ensure Python int, not numpy int64
        self.home_market = market
        self.states = np.ones(steps + 1, dtype=np.float128)  # High precision for wealth
        self.markets = [market]
        self.conglomerate = [self.id]
        self.outside_profits = np.ones(lookback, dtype=np.float128)  # High precision for profits
        self.entered = None
        self.conglomerate_id = None


def logistic_cost(x, k, x_0):
    y = 1 / (1 + np.exp(-k * (x - x_0)))
    return y


def power_law_cost(size, b0, b1):
    return b0 * size ** b1


def cost(size, progression, degree=2):
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


def exit_(firms, firm, conglomerates, to_delete=None):
    partners = firm.conglomerate.copy()
    partners.remove(firm.id)
    for partner in partners:
        firms[partner].markets.remove(firm.home_market)
        firms[partner].conglomerate.remove(firm.id)

    conglomerates[firm.conglomerate_id]['firms'].remove(firm.id)

    # Mark conglomerate for deletion if empty or only one firm remains
    remaining_firms = conglomerates[firm.conglomerate_id]['firms']
    if len(remaining_firms) <= 1:
        # Clean up the remaining firm's state if there is one
        if len(remaining_firms) == 1:
            remaining_firm_id = remaining_firms[0]
            firms[remaining_firm_id].conglomerate_id = None
            firms[remaining_firm_id].conglomerate = [remaining_firm_id]
            firms[remaining_firm_id].markets = [firms[remaining_firm_id].home_market]  # Reset to home only
            firms[remaining_firm_id].entered = None  # Reset entry time

        # Mark for deletion instead of deleting immediately
        if to_delete is not None:
            to_delete.add(firm.conglomerate_id)

    firm.conglomerate = [firm.id]
    firm.markets = [firm.home_market]
    firm.entered = None
    firm.conglomerate_id = None


def model(params):
    print("DEBUG: Starting model function", flush=True)
    min_mu = 0.01
    max_mu = 0.1
    min_sig = 0.01
    max_sig = 0.05

    min_bound = np.array([min_mu, min_sig])
    max_bound = np.array([max_mu, max_sig])

    mu_sig_corr = 0.7
    means = [0.1, 0.05]

    print("DEBUG: Extracting parameters", flush=True)
    # Extract parameters - now including power law parameters
    if len(params) == 12:
        markets, firms_per_market, steps, share, total_firms, merge_thresh, comparison, break_thresh, proportional, lookback, b0, b1 = params
        use_power_law = True
    else:
        markets, firms_per_market, steps, share, total_firms, merge_thresh, comparison, break_thresh, proportional, lookback = params
        b0, b1 = 1.0, 1.0  # Default values
        use_power_law = False

    print("DEBUG: Creating covariance matrix", flush=True)
    cov_mat = np.ones((2, 2)) * mu_sig_corr + np.diag(np.tile(1 - mu_sig_corr, 2))
    growth_vars = np.random.multivariate_normal(means, cov_mat, markets)
    growth_vars = growth_vars + np.abs(np.minimum(growth_vars.min(axis=0), 0))
    growth_vars = min_bound + (growth_vars / growth_vars.max(axis=0)) * (max_bound - min_bound)

    print("DEBUG: Creating eigenvalues", flush=True)
    eigen_vals = np.random.uniform(0.1, 3, markets)
    eigen_vals = eigen_vals * markets / eigen_vals.sum()

    print("DEBUG: Creating market correlation matrix", flush=True)
    market_corr = random_correlation.rvs(tuple(eigen_vals), random_state=np.random.default_rng())

    # market_corr = np.diag(np.ones(markets))
    # market_corr[np.triu_indices(markets, k=1)] = np.random.uniform(0.1, 0.8, int(markets * (markets - 1) / 2))
    # market_corr = market_corr + market_corr.T - np.diag(np.ones(markets))
    print("DEBUG: Creating market covariance", flush=True)
    market_cov = np.outer(growth_vars[:, 1], growth_vars[:, 1]) * market_corr

    print("DEBUG: Generating realizations (this could be slow for large arrays)", flush=True)
    realizations = np.random.multivariate_normal(growth_vars[:, 0], market_cov, size=(steps, firms_per_market)) + 1
    
    print("DEBUG: Creating market structure", flush=True)
    markets_structure = np.array([[x, y] for x in range(markets) for y in range(firms_per_market)])

    print("DEBUG: Creating firm IDs", flush=True)
    ids = np.arange(markets * firms_per_market).reshape(markets, firms_per_market)

    print("DEBUG: Initializing firms (this creates large float128 arrays)", flush=True)
    firms = [Firm(market=market, number=ids[market, firm], steps=steps, lookback=lookback) for market in range(markets)
             for firm in range(firms_per_market)]
    all_time_conglomerates = []
    conglomerates = {}

    print("DEBUG: Starting main simulation loop", flush=True)
    # Track merger frequency per period
    mergers_per_period = np.zeros(steps)

    # for step in tqdm(range(steps)):
    for step in range(steps):
        if step == 0:
            print(f"DEBUG: CRITICAL - Starting step {step}", flush=True)
        # Progress logging every 1000 steps for debugging
        if step % 1000 == 0:
            print(f"DEBUG: Step {step}/{steps} at {time.time():.2f}", flush=True)
        # Set merge_thresh to 0 when share=0 to disable mergers (pure Brownian motion)
        effective_merge_thresh = 0 if share == 0 else merge_thresh
        draws = np.where(np.random.uniform(0, 1, total_firms) < effective_merge_thresh)[0]

        if step == 0:
            print(f"DEBUG: Step 0 - effective_merge_thresh={effective_merge_thresh}, draws.shape[0]={draws.shape[0]}", flush=True)

        if draws.shape[0] > 0:
            if step == 0:
                print(f"DEBUG: Step 0 - Starting merger loop with {draws.shape[0]} firms", flush=True)
            for firm_idx, firm in enumerate(draws):
                if step == 0 and firm_idx % 100 == 0:
                    print(f"DEBUG: Step 0 - Processing firm {firm_idx}/{draws.shape[0]}", flush=True)
                active_markets = firms[firm].markets
                if len(active_markets) == markets:
                    continue

                if step == 0 and firm_idx % 100 == 0:
                    print(f"DEBUG: Step 0 - Finding target markets for firm {firm}", flush=True)
                
                target_markets = markets_structure[np.logical_not(np.in1d(markets_structure[:, 0], active_markets)), :]
                
                if step == 0 and firm_idx % 100 == 0:
                    print(f"DEBUG: Step 0 - target_markets.shape={target_markets.shape}", flush=True)
                
                target = target_markets[np.random.choice(np.arange(target_markets.shape[0]), 1)]
                target = np.where((markets_structure == target).all(axis=1))[0][0]

                if step == 0 and firm_idx % 100 == 0:
                    print(f"DEBUG: Step 0 - Found target {target}, checking compatibility", flush=True)
                
                # Critical debug for the hanging case
                if step == 0 and target == 4007:
                    print(f"DEBUG: CRITICAL - firm={firm}, target={target}, about to check compatibility", flush=True)

                if not np.isin(firms[target].markets, firms[firm].markets).any():
                    if step == 0 and firm == 9125 and target == 4007:
                        print(f"DEBUG: CRITICAL - Passed compatibility check, starting merger evaluation", flush=True)
                    
                    home_id = firms[firm].conglomerate_id
                    cong_id = firms[target].conglomerate_id

                    if step == 0 and firm == 9125 and target == 4007:
                        print(f"DEBUG: CRITICAL - home_id={home_id}, cong_id={cong_id}", flush=True)

                    joint_markets = firms[target].markets + firms[firm].markets
                    joint_markets.sort()

                    conglomerate = firms[target].conglomerate + firms[firm].conglomerate
                    
                    if step == 0 and firm == 9125 and target == 4007:
                        print(f"DEBUG: CRITICAL - About to sort conglomerate={conglomerate}", flush=True)
                    
                    conglomerate.sort()
                    
                    if step == 0 and firm == 9125 and target == 4007:
                        print(f"DEBUG: CRITICAL - Finished sorting conglomerate={conglomerate}", flush=True)
                    
                    if step == 0 and firm == 9125 and target == 4007:
                        print(f"DEBUG: CRITICAL - joint_markets={joint_markets}, conglomerate={conglomerate}", flush=True)
                        print(f"DEBUG: CRITICAL - About to check len(conglomerate)={len(conglomerate)} > 2", flush=True)

                    if len(conglomerate) > 2:
                        if step == 0 and firm == 9125 and target == 4007:
                            print(f"DEBUG: CRITICAL - len(conglomerate)={len(conglomerate)} > 2, entering complex evaluation", flush=True)
                        
                        if home_id is None or cong_id is None:
                            conglomerate_id = home_id if cong_id is None else cong_id
                            true_pool = conglomerates[conglomerate_id]['pool']
                            synth_pool = np.zeros(lookback, dtype=np.float128)
                            indices = markets_structure[conglomerate, :]
                            for enumer, synth in enumerate(range(max(step - lookback, 0), step)):
                                returns = realizations[synth, indices[:, 1], indices[:, 0]]
                                states = np.array([firms[firm].states[synth] for firm in conglomerate], dtype=np.float128)
                                gains = states * returns - states
                                if proportional:
                                    if use_power_law:
                                        cost_factor = 1 - power_law_cost(len(conglomerate), b0,
                                                                         b1) / states.sum() if states.sum() > 0 else 0
                                    else:
                                        cost_factor = 1 - logistic_cost(len(conglomerate), k=1, x_0=markets)
                                    pool = np.float128((gains * share).sum() * cost_factor)
                                else:
                                    if use_power_law:
                                        management_cost = states.sum() * power_law_cost(len(conglomerate), b0, b1)
                                    else:
                                        management_cost = states.sum() * logistic_cost(len(conglomerate), k=1,
                                                                                       x_0=markets * 0.4)
                                    pool = np.float128((gains * share).sum() - management_cost)

                                synth_pool[enumer] = pool

                            if synth_pool.sum() < true_pool.sum():
                                continue
                        else:
                            true_pool0 = conglomerates[home_id]['pool']
                            true_pool1 = conglomerates[cong_id]['pool']
                            synth_pool = np.zeros(lookback, dtype=np.float128)
                            indices = markets_structure[conglomerate, :]
                            for enumer, synth in enumerate(range(max(step - lookback, 0), step)):
                                returns = realizations[synth, indices[:, 1], indices[:, 0]]
                                states = np.array([firms[firm].states[synth] for firm in conglomerate], dtype=np.float128)
                                gains = states * returns - states
                                if proportional:
                                    if use_power_law:
                                        cost_factor = 1 - power_law_cost(len(conglomerate), b0,
                                                                         b1) / states.sum() if states.sum() > 0 else 0
                                    else:
                                        cost_factor = 1 - logistic_cost(len(conglomerate), k=1, x_0=markets)
                                    pool = np.float128((gains * share).sum() * cost_factor)
                                else:
                                    if use_power_law:
                                        management_cost = states.sum() * power_law_cost(len(conglomerate), b0, b1)
                                    else:
                                        management_cost = states.sum() * logistic_cost(len(conglomerate), k=1,
                                                                                       x_0=markets * 0.4)
                                    pool = np.float128((gains * share).sum() - management_cost)

                                synth_pool[enumer] = pool

                            if synth_pool.sum() < true_pool0.sum() or synth_pool.sum() < true_pool1.sum():
                                continue

                    if cong_id is None and home_id is None:
                        if step == 0 and firm == 9125 and target == 4007:
                            print(f"DEBUG: CRITICAL - Both IDs are None, generating new ID", flush=True)
                        
                        existing = np.array(list(conglomerates.keys()))
                        
                        if step == 0 and firm == 9125 and target == 4007:
                            print(f"DEBUG: CRITICAL - existing conglomerates: {existing}", flush=True)
                        if len(existing) == 0:
                            new_id = 0
                        else:
                            if step == 0 and firm == 9125 and target == 4007:
                                print(f"DEBUG: CRITICAL - Creating ID range for {len(conglomerates.keys())} existing", flush=True)
                            
                            all_ = np.arange(len(conglomerates.keys()))
                            
                            if step == 0 and firm == 9125 and target == 4007:
                                print(f"DEBUG: CRITICAL - About to compute logical_not", flush=True)
                            
                            lowest_new = all_[np.logical_not(np.isin(all_, existing))]
                            
                            if step == 0 and firm == 9125 and target == 4007:
                                print(f"DEBUG: CRITICAL - lowest_new.shape[0]={lowest_new.shape[0]}", flush=True)
                            
                            if lowest_new.shape[0] == 0:
                                if step == 0 and firm == 9125 and target == 4007:
                                    print(f"DEBUG: CRITICAL - About to compute np.max(all_) where all_={all_}", flush=True)
                                
                                new_id = np.max(all_) + 1
                                
                                if step == 0 and firm == 9125 and target == 4007:
                                    print(f"DEBUG: CRITICAL - new_id={new_id}", flush=True)
                            else:
                                new_id = np.min(lowest_new)

                    if step == 0 and firm == 9125 and target == 4007:
                        print(f"DEBUG: CRITICAL - About to update {len(conglomerate)} partners: {conglomerate}", flush=True)
                    
                    for partner_idx, partner in enumerate(conglomerate):
                        if step == 0 and firm == 9125 and target == 4007:
                            print(f"DEBUG: CRITICAL - Updating partner {partner_idx}/{len(conglomerate)}: {partner}", flush=True)
                        
                        firms[partner].markets = joint_markets.copy()
                        
                        if step == 0 and firm == 9125 and target == 4007:
                            print(f"DEBUG: CRITICAL - Updated markets for {partner}, about to update conglomerate", flush=True)
                        
                        firms[partner].conglomerate = conglomerate.copy()
                        
                        if step == 0 and firm == 9125 and target == 4007:
                            print(f"DEBUG: CRITICAL - Updated conglomerate for {partner}", flush=True)
                    
                    if step == 0 and firm == 9125 and target == 4007:
                        print(f"DEBUG: CRITICAL - Partner updates complete, now handling conglomerate management", flush=True)
                        print(f"DEBUG: CRITICAL - home_id={home_id}, cong_id={cong_id}", flush=True)

                    if home_id is not None and cong_id is not None:
                        if step == 0 and firm == 9125 and target == 4007:
                            print(f"DEBUG: CRITICAL - Case 1: both home_id and cong_id exist", flush=True)
                        conglomerates[cong_id] = {'firms': conglomerate.copy(), 'pool': synth_pool}
                        if step == 0 and firm == 9125 and target == 4007:
                            print(f"DEBUG: CRITICAL - Updated conglomerates[{cong_id}], about to update {len(conglomerate)} members", flush=True)

                        for member_idx, member in enumerate(conglomerate):
                            if step == 0 and firm == 9125 and target == 4007:
                                print(f"DEBUG: CRITICAL - Updating member {member_idx}/{len(conglomerate)}: {member} conglomerate_id to {cong_id}", flush=True)
                            firms[member].conglomerate_id = cong_id
                            firms[firm].entered = step
                            if step == 0 and firm == 9125 and target == 4007:
                                print(f"DEBUG: CRITICAL - Updated member {member}", flush=True)

                        if step == 0 and firm == 9125 and target == 4007:
                            print(f"DEBUG: CRITICAL - About to delete conglomerates[{home_id}]", flush=True)
                        del conglomerates[home_id]
                        if step == 0 and firm == 9125 and target == 4007:
                            print(f"DEBUG: CRITICAL - Deleted conglomerates[{home_id}]", flush=True)

                    elif home_id is None and cong_id is not None:
                        if step == 0 and firm == 9125 and target == 4007:
                            print(f"DEBUG: CRITICAL - Case 2: only cong_id exists", flush=True)
                        conglomerates[cong_id]['firms'] = conglomerate.copy()
                        firms[firm].conglomerate_id = cong_id
                        firms[firm].entered = step

                    elif home_id is not None and cong_id is None:
                        if step == 0 and firm == 9125 and target == 4007:
                            print(f"DEBUG: CRITICAL - Case 3: only home_id exists", flush=True)
                        conglomerates[home_id]['firms'] = conglomerate.copy()
                        firms[target].conglomerate_id = home_id
                        firms[target].entered = step

                    elif home_id is None and cong_id is None:
                        if step == 0 and firm == 9125 and target == 4007:
                            print(f"DEBUG: CRITICAL - Case 4: neither home_id nor cong_id exists, using new_id={new_id}", flush=True)
                        firms[firm].conglomerate_id = new_id
                        firms[target].conglomerate_id = new_id
                        firms[firm].entered = step
                        firms[target].entered = step
                        conglomerates[new_id] = {'firms': conglomerate.copy(), 'pool': np.zeros(lookback, dtype=np.float128)}
                        if step == 0 and firm == 9125 and target == 4007:
                            print(f"DEBUG: CRITICAL - Completed case 4 setup", flush=True)
                    
                    if step == 0 and firm == 9125 and target == 4007:
                        print(f"DEBUG: CRITICAL - All conglomerate management complete, about to count merger", flush=True)

                    # Count successful merger
                    mergers_per_period[step] += 1
                    if step == 0 and firm == 9125 and target == 4007:
                        print(f"DEBUG: CRITICAL - Merger counted, now computing period_conglomerates", flush=True)

        if step == 0:
            print(f"DEBUG: CRITICAL - About to compute period_conglomerates for step {step}", flush=True)
            print(f"DEBUG: CRITICAL - Processing {len(firms)} firms", flush=True)
            
            # Check for problematic conglomerate lists first
            for i, firm in enumerate(firms):
                if len(firm.conglomerate) > 1:
                    if i < 10 or firm.id in [4007, 9125]:  # Debug first 10 and our problematic firms
                        print(f"DEBUG: CRITICAL - firm {firm.id}: conglomerate={firm.conglomerate}", flush=True)
        
        print(f"DEBUG: CRITICAL - Step {step}: Starting period_conglomerates computation", flush=True)
        period_conglomerates = list(
            set(tuple(lst) for lst in [firm.conglomerate for firm in firms if len(firm.conglomerate) > 1]))
        print(f"DEBUG: CRITICAL - Step {step}: Finished period_conglomerates computation", flush=True)
        
        if step == 0:
            print(f"DEBUG: CRITICAL - Computed period_conglomerates: {len(period_conglomerates)} conglomerates", flush=True)
        
        # Performance investigation for the slow seed
        if step > 0 and step % 100 == 0:  # Every 100 steps
            # Analyze conglomerate complexity
            cong_sizes = [len(cong) for cong in period_conglomerates]
            max_cong_size = max(cong_sizes) if cong_sizes else 0
            avg_cong_size = sum(cong_sizes) / len(cong_sizes) if cong_sizes else 0
            total_cong_members = sum(cong_sizes)
            
            # Count firms in conglomerates vs solo
            firms_in_conglomerates = len([f for f in firms if len(f.conglomerate) > 1])
            solo_firms = len(firms) - firms_in_conglomerates
            
            # Check for extreme wealth values that might slow float128 operations
            max_wealth = max(firm.states[step] for firm in firms)
            min_wealth = min(firm.states[step] for firm in firms)
            
            print(f"PERF: Step {step}: {len(period_conglomerates)} conglomerates, max_size={max_cong_size}, avg_size={avg_cong_size:.1f}, total_members={total_cong_members}, solo={solo_firms}, wealth_range=[{min_wealth:.2e}, {max_wealth:.2e}]", flush=True)
        
        all_time_conglomerates.append(period_conglomerates)

        solo = [firm.id for firm in firms if len(firm.conglomerate) == 1]

        # Track conglomerates to delete after pooling loop
        conglomerates_to_delete = set()

        if step == 0:
            print(f"DEBUG: CRITICAL - About to enter pooling loop with {len(conglomerates)} conglomerates: {list(conglomerates.keys())}", flush=True)

        for conglomerate_idx, conglomerate_ in enumerate(conglomerates.keys()):
            if step == 0:
                print(f"DEBUG: CRITICAL - Processing conglomerate {conglomerate_idx}/{len(conglomerates)}: {conglomerate_}", flush=True)
            conglomerate = conglomerates[conglomerate_]['firms']
            indices = markets_structure[conglomerate, :]
            returns = realizations[step, indices[:, 1], indices[:, 0]]
            states = np.array([firms[firm].states[step] for firm in conglomerate], dtype=np.float128)
            gains = states * returns - states
            if proportional:
                if use_power_law:
                    cost_factor = 1 - power_law_cost(len(conglomerate), b0,
                                                     b1) / states.sum() if states.sum() > 0 else 0
                else:
                    cost_factor = 1 - logistic_cost(len(conglomerate), k=1, x_0=markets)
                pool = np.float128((gains * share).sum() * cost_factor)
            else:
                if use_power_law:
                    management_cost = states.sum() * power_law_cost(len(conglomerate), b0, b1)
                else:
                    management_cost = states.sum() * logistic_cost(len(conglomerate), k=1, x_0=markets * 0.4)
                pool = np.float128((gains * share).sum() - management_cost)

            returns = ((1 - share) * gains + pool / len(
                conglomerate)) if share > 0 else gains  # Equal distribution of pool contents.

            for partner, firm in enumerate(conglomerate):
                firms[firm].outside_profits[step % lookback] = realizations[
                    (step,) + tuple(markets_structure[firm, ::-1])]
                firms[firm].states[step + 1] = firms[firm].states[step] + returns[partner]
                if firms[firm].states[step + 1] < 0:
                    firms[firm].states[step + 1] = 1
                    exit_(firms, firms[firm], conglomerates=conglomerates, to_delete=conglomerates_to_delete)

            conglomerates[conglomerate_]['pool'][step % lookback] = pool

        # Clean up conglomerates marked for deletion
        for cong_id in conglomerates_to_delete:
            if cong_id in conglomerates:
                del conglomerates[cong_id]

        for firm in solo:
            firms[firm].outside_profits[step % lookback] = realizations[(step,) + tuple(markets_structure[firm, ::-1])]
            firms[firm].states[step + 1] = firms[firm].states[step] * realizations[
                (step,) + tuple(markets_structure[firm, ::-1])]

        if step > lookback:
            # Track conglomerates to delete after exit checks
            exit_cleanup_to_delete = set()

            for firm in firms:
                if len(firm.conglomerate) > 1:
                    if firm.entered < step - lookback:
                        outside_profit = np.prod(firm.outside_profits) ** (1 / lookback)
                        inside_profits = firm.states[step - lookback:step]
                        inside_profits = np.prod(inside_profits[1:] / inside_profits[:-1]) ** (1 / lookback)
                        if outside_profit > inside_profits:
                            exit_(firms, firm, conglomerates=conglomerates, to_delete=exit_cleanup_to_delete)

            # Clean up conglomerates marked for deletion
            for cong_id in exit_cleanup_to_delete:
                if cong_id in conglomerates:
                    del conglomerates[cong_id]
        
        if step == 0:
            print(f"DEBUG: CRITICAL - Completed step {step}", flush=True)

    results = np.array([firm.states for firm in firms], dtype=np.float128).T

    num_cong = []
    members = []
    for conglomerate in all_time_conglomerates:
        num_cong.append(len(conglomerate))
        members.append([len(members) for members in conglomerate])

    mean_members = np.array([np.mean(member) if len(member) > 0 else 0.0 for member in members])

    quantiles_members = np.array(
        [np.quantile(member, q=[0.1, 0.25, 0.5, 0.75, 0.9]) if len(member) > 0 else np.zeros(5) for member in members])

    market_share = np.array([results[:,
                             np.arange(market * firms_per_market, (market + 1) * firms_per_market)] / results[:,
                                                                                                      np.arange(
                                                                                                          market * firms_per_market,
                                                                                                          (
                                                                                                                      market + 1) * firms_per_market)].sum(
        axis=1).reshape(steps + 1, 1) for market in range(markets)])
    # shape = markets x steps + 1 x firms_per_market
    mean_share = market_share.mean(axis=2)
    quantiles_shares = np.quantile(market_share, q=[0.1, 0.25, 0.5, 0.75, 0.9, 0.99], axis=2)
    max_shares = np.max(market_share, axis=2)

    squared_market_share = market_share ** 2
    hhi = np.sum(squared_market_share, axis=2)

    sorted_market_share = np.sort(market_share, axis=2)
    cum_market_share = np.cumsum(sorted_market_share, axis=2)
    sums = np.sum(sorted_market_share, axis=2)
    sums = sums[..., np.newaxis]
    Lorenz_curve = cum_market_share / sums
    area_under_curve = np.trapz(y=Lorenz_curve, axis=2, dx=1 / firms_per_market)
    gini_coefficient = 1 - 2 * area_under_curve

    ranks = np.array([np.argsort(np.argsort(market_share[:, step_, :], axis=1)) for step_ in
                      range(steps)])  # Have to sort twice to get ranks.

    percentile_thresh = firms_per_market * np.array([0.1, 0.9, ranks.max() / firms_per_market])

    percentile_ranks = [(ranks >= thresh).sum(axis=0) for thresh in percentile_thresh]

    avg_shares = []
    avg_ranks = []
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

    model_results = [mean_members, quantiles_members, num_cong, avg_shares, quantiles_shares, max_shares, market_share,
                     hhi, gini_coefficient, ranks, percentile_ranks, avg_ranks, mergers_per_period]

    return model_results