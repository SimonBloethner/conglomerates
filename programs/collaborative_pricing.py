import numpy as np
from tqdm import tqdm
import torch
import os
import PricingModels

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


def exit_(firms, firm, conglomerates):
    partners = firm.conglomerate.copy()
    partners.remove(firm.id)
    for partner in partners:
        firms[partner].markets.remove(firm.home_market)
        firms[partner].conglomerate.remove(firm.id)

    conglomerates[firm.conglomerate_id]['firms'].remove(firm.id)

    del_cong = firm.conglomerate_id if len(conglomerates[firm.conglomerate_id]['firms']) == 0 else None

    firm.conglomerate = [firm.id]
    firm.markets = [firm.home_market]
    firm.entered = None
    firm.conglomerate_id = None

    return del_cong


def model(params):
    markets, firms_per_market, steps, share, total_firms, merge_thresh, comparison, break_thresh, proportional, cost_pooling, lookback = params

    prices = np.zeros((steps, firms_per_market, markets))
    demands = np.zeros((steps, firms_per_market, markets))
    profits = np.ones((steps, firms_per_market, markets))
    sizes = np.ones((steps, firms_per_market, markets))
    market_share = np.zeros((steps, firms_per_market, markets))

    market_share[:, :, :] = 1 / firms_per_market

    markets_structure = np.array([[x, y] for x in range(markets) for y in range(firms_per_market)])

    ids = np.arange(markets * firms_per_market).reshape(markets, firms_per_market)

    firms = [PricingModels.Firm(market=market, number=ids[market, firm], steps=steps, lookback=lookback) for market in range(markets) for firm in range(firms_per_market)]
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
                            for enumer, synth in enumerate(range(max(step - lookback, 0), step)):
                                solo_profits = np.array([firms[firm].outside_profits[synth % lookback] for firm in conglomerate])
                                states = np.array([firms[firm].states[synth] for firm in conglomerate])
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
                            for enumer, synth in enumerate(range(max(step - lookback, 0), step)):
                                solo_profits = np.array([firms[firm].outside_profits[synth % lookback] for firm in conglomerate])
                                states = np.array([firms[firm].states[synth] for firm in conglomerate])
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
            p_ = np.max(prices[step - 1, mask])
            p__ = np.max(prices[step - 2, mask])
            state = [prices[step - 1, firm_id, firm.home_market], p_, p__,
                     market_share[step - 1, firm_id, firm.home_market]]
            # prices[step, firm_id, firm.home_market] = firm.select_action(state)
            prices[step, firm_id, firm.home_market] = firm.select_action_matrix(state)

        demand_growth = 1.01 ** step
        for market_ in range(markets):
            demands[step, :, market_] = demand(prices[step, :, market_], share=market_share[step - 1, :, market_]) * demand_growth
            profits[step, :, market_] = profit(prices[step, :, market_], share=market_share[step - 1, :, market_]) * demand_growth

        for conglomerate_ in conglomerates.keys():
            conglomerate = conglomerates[conglomerate_]['firms']
            indices = markets_structure[conglomerate, :]
            cong_profits = profits[step, indices[:, 1], indices[:, 0]]
            costs = np.random.lognormal(0, 1, len(conglomerate)) / 50
            costs = costs * 0
            costs = costs * profits[:step + 1, indices[:, 1], indices[:, 0]].sum(axis=0)    # Assume that costs are always calculated relative to the firm market value, not the period revenue.
            if cost_pooling:    # TODO: Make the building and comparison of a cost dependent pool feasible. How to do the synthetic pool?
                costs = (share * costs).sum() / len(conglomerate) + (1 - share) * costs
                returns = cong_profits - costs
            else:
                cong_profits -= costs
                if proportional:
                    pool = (cong_profits * share).sum() * (1 - logistic_cost(len(conglomerate), k=1, x_0=markets))
                else:
                    pool = (cong_profits * share).sum() - profits[step, indices[:, 1], indices[:, 0]].sum() * logistic_cost(len(conglomerate), k=1, x_0=markets * 0.4) * (0 if share == 0 else 1)  # len(conglomerate) / markets
                    returns = (1 - share) * cong_profits + pool / len(conglomerate) if share > 0 else cong_profits  # Equal distribution of pool contents.

            profits[step, indices[:, 1], indices[:, 0]] = returns

            conglomerates[conglomerate_]['pool'][step % lookback] = pool    # TODO: Under cost sharing, there is no pool to define. Find metric to compare instead here.

            for partner, firm in enumerate(conglomerate):
                firms[firm].outside_profits[step % lookback] = cong_profits[partner]
                firms[firm].states[step + 1] = firms[firm].states[step] + returns[partner]
                if firms[firm].states[step + 1] < 0:
                    firms[firm].states[step + 1] = 1
                    to_del.append(exit_(firms, firms[firm], conglomerates=conglomerates))

        # market_share[step, :, :] = np.sum(profits[:step + 1, :, :], axis=0) / np.sum(profits[:step + 1, :, :], axis=(0, 1))  # Compute and store the period market shares.

        for firm in solo:
            firms[firm].outside_profits[step % lookback] = profits[(step,) + tuple(markets_structure[firm, :])]
            firms[firm].states[step + 1] = profits[(step,) + tuple(markets_structure[firm, :])]

        sizes[step, :, :] = sizes[step - 1, :, :] * (np.exp((profits[step, :, :] - profits[step - 1, :, :]) / profits[step - 1, :, :]) if step > 0 else 1)
        market_share[step, :, :] = sizes[step, :, :] / np.sum(sizes[step, :, :], axis=0)
        if step > lookback:
            for firm in firms:
                if len(firm.conglomerate) > 1:
                    if firm.entered < step - lookback:
                        t0 = firm.states[step - lookback]
                        t_end = t0 + firm.outside_profits.sum()
                        frac = t_end / t0
                        outside_profit = np.sign(frac) * np.abs(frac) ** (1 / lookback)
                        inside_profits = firm.states[step - lookback:step]
                        frac = inside_profits[-1] / inside_profits[0]
                        inside_profits = np.sign(frac) * np.abs(frac) ** (1 / lookback)
                        if outside_profit > inside_profits:
                            to_del.append(exit_(firms, firm, conglomerates=conglomerates))

        to_del = [x for x in to_del if x is not None]
        for deletion in to_del:
            del conglomerates[deletion]

    num_cong = []
    members = []
    for conglomerate in all_time_conglomerates:
        num_cong.append(len(conglomerate))
        members.append([len(members) for members in conglomerate])

    mean_members = np.array([np.mean(member) for member in members])

    quantiles_members = np.array([np.quantile(member, q=[0.1, 0.25, 0.5, 0.75, 0.9]) for member in members])

    # shape = markets x steps + 1 x firms_per_market
    quantiles_shares = np.quantile(market_share, q=[0.1, 0.25, 0.5, 0.75, 0.9, 0.99], axis=2)
    max_shares = np.max(market_share, axis=2)

    squared_market_share = market_share ** 2
    hhi = np.sum(squared_market_share, axis=2)

    sorted_market_share = np.sort(market_share, axis=2)
    cum_market_share = np.cumsum(sorted_market_share, axis=2)
    sums = np.sum(sorted_market_share, axis=2)
    sums = sums[..., np.newaxis]
    lorenz_curve = cum_market_share / sums
    area_under_curve = np.trapz(y=lorenz_curve, axis=2, dx=1 / firms_per_market)
    gini_coefficient = 1 - 2 * area_under_curve

    ranks = np.array([np.argsort(np.argsort(market_share[step_, :, :], axis=1)) for step_ in range(steps)])

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
