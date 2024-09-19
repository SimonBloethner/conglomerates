import numpy as np
import os
from tqdm import tqdm

np.random.seed(1)

path_ = os.getcwd()
local = path_.find('Simon') > 0
if local:
    path_ = '/Users/Simon/Documents/Projects/EWF/Research/PhD/Ergodicity Economics/IOxEE'
else:
    path_ = 'collusion'


class Firm:
    def __init__(self, market, number, market_id, steps, lookback):
        self.id = number
        self.market_id = market_id
        self.home_market = market
        self.states = np.ones(steps + 1)
        self.markets = [market]
        self.conglomerate = [self.id]
        self.outside_profits = np.ones(lookback)
        self.entered = None
        self.conglomerate_id = None


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
    markets, firms_per_market, steps, share, total_firms, merge_thresh, comparison, break_thresh, proportional, lookback = params
    prices = np.zeros((steps, firms_per_market, markets))
    demands = np.zeros((steps, firms_per_market, markets))
    profits = np.ones((steps, firms_per_market, markets))
    sizes = np.ones((steps, firms_per_market, markets))
    market_share = np.tile(1 / firms_per_market, (steps, firms_per_market, markets))
    lower_bound = 0.02
    upper_bound = 0.07
    costs = np.clip(np.random.normal(0.01, 0.001, (steps, firms_per_market, markets)), 0.0001, 0.1)
    cost_pooling = False
    markets_structure = np.array([[x, y] for x in range(markets) for y in range(firms_per_market)])

    market_growth = np.ones((steps, markets))

    ids = np.arange(markets * firms_per_market).reshape(markets, firms_per_market)

    firms = [Firm(market=market, number=ids[market, firm], market_id=firm, steps=steps, lookback=lookback) for market in range(markets) for firm in range(firms_per_market)]
    all_time_conglomerates = []
    conglomerates = {}
    # for step in tqdm(range(steps)):
    for step in range(steps):
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

        prices[step, :, :] = price_opt(market_share[step - 1, :, :], cost=costs[step, :, :])
        profits[step, :, :] = profit(prices[step, :, :], market_share[step - 1, :, :], cost=costs[step, :, :], scale=market_growth.prod(axis=0))

        for conglomerate_ in conglomerates.keys():
            conglomerate = conglomerates[conglomerate_]['firms']
            indices = markets_structure[conglomerate, :]
            cong_profits = profits[step, indices[:, 1], indices[:, 0]]
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
            profit_ = profits[(step,) + tup]
            firms[firm].outside_profits[step % lookback] = profit_
            firms[firm].states[step + 1] = profit_
            if profit_ / sizes[(step - 1,) + tup] < np.exp(-1) - 1 or sizes[(step - 1,) + tup] < 0:
                firms[firm].states[step + 1] = 1
                profits[(step,) + tup] = 0

        reinvestment = profits[step, :, :] / sizes[step - 1, :, :]

        normalized_reinvestment = (reinvestment - np.min(reinvestment, axis=0)) / (np.max(reinvestment, axis=0) - np.min(reinvestment, axis=0))     # Assumption: axis=0 means that there are normalizations for each market, not across all firms!

        means = lower_bound + normalized_reinvestment * (upper_bound - lower_bound)

        sds = 0.2 + 0.1 * means
        increments = np.random.normal(means, sds)
        increments[increments < -1] = -0.99
        sizes[step, :, :] = sizes[step - 1, :, :] + sizes[step - 1, :, :] * increments
        market_share[step, :, :] = sizes[step, :, :] / np.sum(sizes[step, :, :], axis=0)

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
    # sums = sums[..., np.newaxis]
    sums = np.expand_dims(sums, axis=1)
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
            to_append = [len(conglomerate), market_share[period, :, :][(coords[:, 1], coords[:, 0])].mean()]
            avg_share.append(to_append)
            to_append = [len(conglomerate), ranks[period, :, :][(coords[:, 1], coords[:, 0])].mean()]
            avg_rank.append(to_append)

        avg_shares.append(np.array(avg_share))
        avg_ranks.append(np.array(avg_rank))

    model_results = [mean_members, quantiles_members, num_cong, avg_shares, quantiles_shares, max_shares, market_share,
                     hhi, gini_coefficient, ranks, percentile_ranks, avg_ranks, sizes, profits, prices]

    return model_results


def price_opt(share, cost):
    p_opt = np.exp(cost) / (1 - share + 1e-6)
    return p_opt


def profit(p, share, cost, scale=10):
    pi = (np.array(p) - np.exp(cost)) * scale * p ** (- 1 / share + 1e-6)
    return pi

