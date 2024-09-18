import numpy as np
from tqdm import tqdm
import torch
import os
import PricingModels
from scipy.stats import random_correlation
from scipy import stats

np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

device = 'cpu'

path_ = os.getcwd()
local = path_.find('Simon') > 0
if local:
    path_ = '/Users/Simon/Documents/Projects/EWF/Research/PhD/Ergodicity Economics/IOxEE'
else:
    path_ = 'collusion'

action_std = 0.1  # starting std for action distribution (Multivariate Normal)
action_std_decay_rate = 0.05  # linearly decay action_std (action_std = action_std - action_std_decay_rate)
action_std_init = 0.6

random_seed = 0  # set random seed if required (0 = no random seed)
checkpoint_path = '{}/{}/models'.format(path_, 'programs' if local else 'pythonFiles')


def logistic_cost(x, k, x_0):
    y = 1 / (1 + np.exp(-k * (x - x_0)))
    return y


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


def exit_(firms, firm):
    partners = firm.conglomerate.copy()
    partners.remove(firm.id)
    for partner in partners:
        firms[partner].markets.remove(firm.home_market)
        firms[partner].conglomerate.remove(firm.id)

    firm.conglomerate = [firm.id]
    firm.markets = [firm.home_market]
    firm.entered = None
    firm.conglomerate_id = None


def model(params):
    markets, firms_per_market, steps, share, total_firms, merge_thresh, comparison, break_thresh, proportional, cost_pooling, lookback, eps = params
    np_random_state = np.random.RandomState(1)
    prices = np.zeros((steps, firms_per_market, markets))
    demands = np.zeros((steps, firms_per_market, markets))
    profits = np.ones((steps, firms_per_market, markets))
    sizes = np.ones((steps, firms_per_market, markets))
    market_share = np.tile(1 / firms_per_market, (steps, firms_per_market, markets))

    markets_structure = np.array([[x, y] for x in range(markets) for y in range(firms_per_market)])
    min_mu = 0.01
    max_mu = 0.1
    min_sig = 0.01
    max_sig = 0.05

    min_bound = np.array([min_mu, min_sig])
    max_bound = np.array([max_mu, max_sig])

    mu_sig_corr = 0.7
    means = [0.1, 0.05]

    cov_mat = np.ones((2, 2)) * mu_sig_corr + np.diag(np.tile(1 - mu_sig_corr, 2))
    growth_vars = np.random.multivariate_normal(means, cov_mat, markets)
    growth_vars = growth_vars + np.abs(np.minimum(growth_vars.min(axis=0), 0))
    growth_vars = min_bound + (growth_vars / growth_vars.max(axis=0)) * (max_bound - min_bound)

    eigen_vals = np.random.uniform(0.1, 3, markets)
    eigen_vals = eigen_vals * markets / eigen_vals.sum()

    market_corr = random_correlation.rvs(tuple(eigen_vals), random_state=np_random_state)

    # market_corr = np.diag(np.ones(markets))
    # market_corr[np.triu_indices(markets, k=1)] = np.random.uniform(0.1, 0.8, int(markets * (markets - 1) / 2))
    # market_corr = market_corr + market_corr.T - np.diag(np.ones(markets))
    market_cov = np.outer(growth_vars[:, 1], growth_vars[:, 1]) * market_corr

    market_growth = np.random.multivariate_normal(growth_vars[:, 0], market_cov, size=(steps, firms_per_market)) + 1
    market_growth = np.ones((steps, markets))
    # market_growth = np.random.normal(1.04, 0.04, size=(steps, markets))

    ids = np.arange(markets * firms_per_market).reshape(markets, firms_per_market)

    firms = [PricingModels.Firm(market=market, number=ids[market, firm], market_id=firm, steps=steps, lookback=lookback) for market in range(markets) for firm in range(firms_per_market)]
    all_time_conglomerates = []
    conglomerates = {}
    for step in tqdm(range(steps)):
    # for step in range(steps):
        to_del = []
        draws = np.where(np.random.uniform(0, 1, total_firms) < merge_thresh)[0]

        if draws.shape[0] > 0:
            for firm in draws:
                active_markets = firms[firm].markets
                if len(active_markets) == markets:
                    continue

                target_markets = markets_structure[np.logical_not(np.in1d(markets_structure[:, 0], active_markets)), :]
                target = target_markets[np.random.choice(np.arange(target_markets.shape[0]), 1)]
                target = np.where((markets_structure == target).all(axis=1))[0][0]

                if not np.isin(firms[target].markets, firms[firm].markets).any():
                    home_id = firms[firm].conglomerate_id
                    cong_id = firms[target].conglomerate_id

                    joint_markets = firms[target].markets + firms[firm].markets
                    joint_markets.sort()

                    conglomerate = firms[target].conglomerate + firms[firm].conglomerate
                    conglomerate.sort()

                    if len(conglomerate) > 2:
                        if home_id is None or cong_id is None:
                            conglomerate_id = home_id if cong_id is None else cong_id
                            true_pool = conglomerates[conglomerate_id]['pool']
                            synth_pool = np.zeros(lookback).astype(float)
                            indices = markets_structure[conglomerate][:, [1, 0]]
                            for enumer, synth in enumerate(range(max(step - lookback, 0), step)):
                                solo_profits = np.array([firms[firm].outside_profits[synth % lookback] for firm in conglomerate])
                                states = sizes[synth, indices[:, 0], indices[:, 1]]
                                if proportional:
                                    pool = (solo_profits * share).sum() * (1 - logistic_cost(len(conglomerate), k=1, x_0=markets))
                                else:
                                    pool = (solo_profits * share).sum() - states.sum() * logistic_cost(len(conglomerate), k=1, x_0=markets * 0.4)  # len(conglomerate) / markets

                                synth_pool[enumer] = pool

                            if synth_pool.sum() < true_pool.sum():
                                continue
                        else:
                            true_pool0 = conglomerates[home_id]['pool']
                            true_pool1 = conglomerates[cong_id]['pool']
                            synth_pool = np.zeros(lookback).astype(float)
                            indices = markets_structure[conglomerate][:, [1, 0]]
                            for enumer, synth in enumerate(range(max(step - lookback, 0), step)):
                                solo_profits = np.array([firms[firm].outside_profits[synth % lookback] for firm in conglomerate])
                                states = sizes[synth, indices[:, 0], indices[:, 1]]
                                if proportional:
                                    pool = (solo_profits * share).sum() * (1 - logistic_cost(len(conglomerate), k=1, x_0=markets))
                                else:
                                    pool = (solo_profits * share).sum() - states.sum() * logistic_cost(len(conglomerate), k=1, x_0=markets * 0.4)  # len(conglomerate) / markets

                                synth_pool[enumer] = pool

                            if synth_pool.sum() < true_pool0.sum() or synth_pool.sum() < true_pool1.sum():
                                continue

                    if cong_id is None and home_id is None:
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

                    for partner in conglomerate:
                        firms[partner].markets = joint_markets.copy()
                        firms[partner].conglomerate = conglomerate.copy()

                    if home_id is not None and cong_id is not None:
                        conglomerates[cong_id] = {'firms': conglomerate.copy(), 'pool': synth_pool}

                        for member in conglomerate:
                            firms[member].conglomerate_id = cong_id
                            firms[firm].entered = step

                        del conglomerates[home_id]

                    elif home_id is None and cong_id is not None:
                        conglomerates[cong_id]['firms'] = conglomerate.copy()
                        firms[firm].conglomerate_id = cong_id
                        firms[firm].entered = step

                    elif home_id is not None and cong_id is None:
                        conglomerates[home_id]['firms'] = conglomerate.copy()
                        firms[target].conglomerate_id = home_id
                        firms[target].entered = step

                    elif home_id is None and cong_id is None:
                        firms[firm].conglomerate_id = new_id
                        firms[target].conglomerate_id = new_id
                        firms[firm].entered = step
                        firms[target].entered = step
                        conglomerates[new_id] = {'firms': conglomerate.copy(), 'pool': np.zeros(lookback).astype(float)}

        period_conglomerates = list(set(tuple(lst) for lst in [firm.conglomerate for firm in firms if len(firm.conglomerate) > 1]))
        all_time_conglomerates.append(period_conglomerates)

        solo = [firm.id for firm in firms if len(firm.conglomerate) == 1]

        for firm in firms:
            firm_id = firm.id % firms_per_market
            mask = np.ones(firms_per_market, dtype=bool)
            mask[firm_id] = False
            p_ = np.max(prices[step - 1, mask, firm.home_market])
            p__ = np.max(prices[step - 2, mask, firm.home_market])
            state = [prices[step - 1, firm_id, firm.home_market], p_, p__,
                     market_share[step - 1, firm_id, firm.home_market]]
            # prices[step, firm_id, firm.home_market] = firm.select_action(state)
            prices[step, firm_id, firm.home_market] = firm.select_action_matrix(state)

        # demand_growth = market_growth[:step + 1, :, :].prod(axis=0)
        for market_ in range(markets):
            demands[step, :, market_] = demand(prices[step, :, market_], share=market_share[step - 1, :, market_]) * market_growth[:, market_].prod() * market_share[step - 1, :, market_]
            profits[step, :, market_] = profit(prices[step, :, market_], share=market_share[step - 1, :, market_]) * market_growth[:, market_].prod() * market_share[step - 1, :, market_]

        for conglomerate_ in conglomerates.keys():
            conglomerate = conglomerates[conglomerate_]['firms']
            indices = markets_structure[conglomerate, :]
            cong_profits = profits[step, indices[:, 1], indices[:, 0]]
            # costs = np.random.lognormal(0, 1, len(conglomerate)) / 50
            costs = np.abs(np.random.normal(0, 0.1, len(conglomerate)))
            # cong_profits = cong_profits - costs * sizes[step - 1, indices[:, 1], indices[:, 0]]
            cong_profits = cong_profits * (1 - costs)   # - costs * sizes[step - 1, indices[:, 1], indices[:, 0]]
            if cost_pooling:    # TODO: Make the building and comparison of a cost dependent pool feasible. How to do the synthetic pool?
                costs = (share * costs).sum() / len(conglomerate) + (1 - share) * costs
                returns = cong_profits - costs
            else:
                # cong_profits -= costs
                if proportional:
                    pool = (cong_profits * share).sum() * (1 - logistic_cost(len(conglomerate), k=1, x_0=markets))
                else:
                    # pool = (cong_profits * share).sum() - profits[step, indices[:, 1], indices[:, 0]].sum() * logistic_cost(len(conglomerate), k=1, x_0=markets * 0.4) * (0 if share == 0 else 1)  # len(conglomerate) / markets
                    pool = (cong_profits * share).sum() - sizes[step - 1, indices[:, 1], indices[:, 0]].sum() * logistic_cost(len(conglomerate), k=1, x_0=markets * 0.4) * (0 if share == 0 else 1)  # len(conglomerate) / markets
                    returns = (1 - share) * cong_profits + pool / len(conglomerate) if share > 0 else cong_profits  # Equal distribution of pool contents.
            profits[step, indices[:, 1], indices[:, 0]] = returns

            conglomerates[conglomerate_]['pool'][step % lookback] = pool    # TODO: Under cost sharing, there is no pool to define. Find metric to compare instead here.

            del_firms = []
            for partner, firm in enumerate(conglomerate):
                profit_ = returns[partner]
                firms[firm].outside_profits[step % lookback] = cong_profits[partner]
                firms[firm].states[step + 1] = profit_
                tup = tuple([markets_structure[firm][1], markets_structure[firm][0]])
                if profit_ / sizes[(step - 1,) + tup] < np.exp(-1) - 1 or sizes[(step - 1,) + tup] < 0:
                    firms[firm].states[step + 1] = 1
                    profits[(step,) + tup] = 0
                    exit_(firms, firms[firm])
                    del_firms.append(firm)

            for firm in del_firms:
                conglomerates[conglomerate_]['firms'].remove(firm)

            if len(conglomerates[conglomerate_]['firms']) < 2:
                to_del.append(conglomerate_)

        for firm in solo:
            tup = tuple([markets_structure[firm][1], markets_structure[firm][0]])
            cost_ = np.abs(np.random.normal(0, 0.01))
            profit_ = profits[(step,) + tup] * (1 - cost_)   # - cost_ * sizes[(step - 1,) + tup]
            firms[firm].outside_profits[step % lookback] = profit_
            firms[firm].states[step + 1] = profit_
            if profit_ / sizes[(step - 1,) + tup] < np.exp(-1) - 1 or sizes[(step - 1,) + tup] < 0:
                firms[firm].states[step + 1] = 1
                profits[(step,) + tup] = 0

        sizes[step, :, :] = sizes[step - 1, :, :] * (1 + np.log(1 + profits[step, :, :] / sizes[step - 1, :, :]))
        if np.isnan(sizes[step, :, :]).any():
            locs = np.where(np.isnan(sizes[step, :, :]))
            print(locs)
            print(profits[step, locs[0], locs[1]])
            print(sizes[step - 1, locs[0], locs[1]])
        market_share[step, :, :] = sizes[step, :, :] / np.sum(sizes[step, :, :], axis=0)
        if (market_share < 0).any():
            print('!')
        market_growth[step, :] = sizes[step, :, :].sum(axis=0) / sizes[step - 1, :, :].sum(axis=0)

        if step > lookback:
            for firm in firms:
                if len(firm.conglomerate) > 1:
                    if firm.entered < step - lookback:
                        synth_size = sizes[step - lookback, firm.market_id, firm.home_market]
                        for synth_step in range(lookback):
                            synth_size = synth_size * (1 + np.log(1 + firm.outside_profits[synth_step - lookback] / synth_size))
                        if synth_size > sizes[step, firm.market_id, firm.home_market]:
                            conglomerates[firm.conglomerate_id]['firms'].remove(firm.id)
                            if len(conglomerates[firm.conglomerate_id]['firms']) < 2:
                                to_del.append(firm.conglomerate_id)
                            exit_(firms, firm)

        for deletion in to_del:
            for firm in conglomerates[deletion]['firms']:
                exit_(firms, firms[firm])
            del conglomerates[deletion]

    num_cong = []
    members = []
    for conglomerate in all_time_conglomerates:
        num_cong.append(len(conglomerate))
        members.append([len(members) for members in conglomerate])

    mean_members = np.array([np.mean(member) for member in members])

    quantiles_members = np.array([np.quantile(member, q=[0.1, 0.25, 0.5, 0.75, 0.9]) for member in members])

    # shape = markets x steps + 1 x firms_per_market
    quantiles_shares = np.quantile(market_share, q=[0.1, 0.25, 0.5, 0.75, 0.9, 0.99], axis=1)
    max_shares = np.max(market_share, axis=1)

    squared_market_share = market_share ** 2
    hhi = np.sum(squared_market_share, axis=1)

    sorted_market_share = np.sort(market_share, axis=1)
    cum_market_share = np.cumsum(sorted_market_share, axis=1)
    sums = np.sum(sorted_market_share, axis=1)
    sums = sums[..., np.newaxis]
    lorenz_curve = cum_market_share / sums
    area_under_curve = np.trapz(y=lorenz_curve, axis=1, dx=1 / firms_per_market)
    gini_coefficient = 1 - 2 * area_under_curve

    ranks = np.array([np.argsort(np.argsort(market_share[step_, :, :], axis=0)) for step_ in range(steps)])

    percentile_thresh = firms_per_market * np.array([0.1, 0.9, ranks.max() / firms_per_market])

    percentile_ranks = [(ranks >= thresh).sum(axis=0) for thresh in percentile_thresh]

    avg_shares = []
    avg_ranks = []
    for period in range(steps):
        avg_share = []
        avg_rank = []
        for conglomerate in all_time_conglomerates[period]:
            coords = markets_structure[list(conglomerate)]
            to_append = [len(conglomerate), market_share[period, :, :][(coords[:, 0], coords[:, 1])].mean()]
            avg_share.append(to_append)
            to_append = [len(conglomerate), ranks[period, :, :][(coords[:, 0], coords[:, 1])].mean()]
            avg_rank.append(to_append)

        avg_shares.append(np.array(avg_share))
        avg_ranks.append(np.array(avg_rank))

    model_results = [mean_members, quantiles_members, num_cong, avg_shares, quantiles_shares, max_shares, market_share,
                     hhi, gini_coefficient, ranks, percentile_ranks, avg_ranks, sizes, profits, prices]

    return model_results


def demand(p, share):
    n_firms = p.shape[0]
    minus_i = (np.ones((n_firms, n_firms)) - np.eye(n_firms)).astype(bool)
    price = np.tile(p, (n_firms, 1))
    minus_i = price[minus_i].reshape(n_firms, n_firms - 1)
    minus_i = np.max(minus_i, axis=1)

    d = 1 - p * (1 - share) + 0.5 * minus_i
    return d


def profit(p, share):
    pi = np.array(p) * demand(p, share)
    return pi
