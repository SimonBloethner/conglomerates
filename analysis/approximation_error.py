"""
C38 Task 3: size of the moment rule's second-order approximation, on the 27 pilot D traces.

    python analysis/approximation_error.py <trace_dir> [--n 2000] [--pilot d]

For each traced run (main, power law, 3 families x alpha {0.05, 0.1, 0.3} x reps 0-2) it draws n decision points
(t, i): t uniform on the post-burn-in steps [burn_in + l, T], i a firm standalone throughout [t - l, t], and a
prospective unit of i and K - 1 partners, each a firm standalone throughout [t - l, t] in a different market
(one per market, not i's), the partners of K in {2, 3, 5, 8} nested (the first K - 1 of seven draws). On the
window [t - l, t) it computes, for i, with the model's own functions (collaborative_growth):
    Delta-hat   member_gain_from_prepared (loggain: the sample mean of the replayed log growth gain)
    Delta-tilde member_gain_moment        (moment: its second-order expansion in the window's moments)
at the run's alpha, Phi(K) of the run's cost function, the run's g and proportional sharing.

Window inputs. The trace does not store the rule's inputs, so they are rebuilt from it:
    return  r_jt = logshare[t+1, j] - logshare[t, j] - jump[t+1, j]   (standalone firm, floor jump removed)
    size    s_jt = exp(logshare[t, j])                                (market mean size is 1 under
                                                                       market_size_fixed, so s is N x share)
For a standalone firm the model's relative return r~_jt equals this r_jt plus the market's renormalisation
term log(sum_k s_k / N) of that step (common to the market, from members' pooled growth and floor top-ups;
summary['renorm_correction_mean'] reports its mean absolute size), so the windows are the model's up to that
term. Both estimates are computed on the same inputs, so their difference is exactly the omitted
higher-order terms for those inputs.

Writes
  <pilot>/approximation_error.csv            by family x K (alpha = 'all', reps pooled) and family x alpha x K:
      n, n_dropped (either estimate -inf), median / p25 / p75 / p5 / p95 of diff = Delta-tilde - Delta-hat and of
      ratio = diff / Delta-hat, median of Delta-hat, Delta-tilde, |Delta-hat|, |diff|, rel_abs = median |diff| /
      median |Delta-hat|, sign_disagree = share with (Delta-tilde > 0) != (Delta-hat > 0);
      and a decomposition of diff by recomputing both with the sizes frozen at the window's first period
      (*_fixed), which removes the in-window weight dynamics:
        higher_order      = Delta-tilde_fixed - Delta-hat_fixed   (the expansion's omitted terms proper)
        weight_term_hat   = Delta-hat_fixed - Delta-hat           (minus the replay's first-order term
                            (1/h) sum_t a sum_j c_jt (u_jt - u_it), which the moment rule drops as mean zero but
                            which is not zero in a window where each u_j is demeaned and the weights c_jt follow
                            the cumulated returns)
        weight_term_tilde = Delta-tilde - Delta-tilde_fixed       (the weights' effect inside the moment rule)
      each as median, p25, p75, mean and median ratio to Delta-hat, plus higher_order / Delta-hat_fixed (p5, median,
      p95)
  <pilot>/approximation_error_samples.csv.gz one row per draw and K (not committed)
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'analysis'))
import collaborative_growth as cg  # noqa: E402
from pilots import pop_pilot  # noqa: E402
from run_traces import select  # noqa: E402

KS = (2, 3, 5, 8)
QS = {'p5': 0.05, 'p25': 0.25, 'median': 0.5, 'p75': 0.75, 'p95': 0.95}


def gains(R, S, alpha, phi, g):
    """Delta-hat and Delta-tilde of member 0 of the set whose window returns / sizes are R, S (h, K)."""
    K = R.shape[1]
    eps, sz, sum_s, sum_s_eps = cg.prepare_window_loggain(np.arange(K), R, S, g)
    sum_su, sum_su2 = cg.prepare_window_moment(eps, sz, g)
    d_hat = cg.member_gain_from_prepared(0, K, eps, sz, sum_s, sum_s_eps, alpha, phi, 1)
    d_til = cg.member_gain_moment(0, K, eps, sz, sum_s, sum_s_eps, sum_su, sum_su2, alpha, phi, 1, g)
    return d_hat, d_til


def draws_for(s, tp, n):
    assert s['sharing_rule'] == 'proportional' and s['market_size_fixed'], s['scenario_id']
    z = np.load(tp)
    ls, cong, jump, home = z['logshare'], z['cong'], z['jump'], z['home']
    T, l = ls.shape[0] - 1, s['lookback']
    M = int(home.max()) + 1
    rng = np.random.default_rng(s['seed'])
    phi = {K: cg.coordination_cost(K, s['cost_type'], s['c0'], s['c1'], s['c2']) for K in KS}
    rows = []
    while len(rows) < n * len(KS):
        t = int(rng.integers(s['burn_in'] + l, T + 1))
        ok = (cong[t - l:t + 1] == -1).all(axis=0)                 # standalone throughout [t-l, t]
        i = rng.choice(np.nonzero(ok)[0])
        markets = rng.permutation(np.setdiff1d(np.arange(M), [home[i]]))
        partners = []
        for m in markets:
            cand = np.nonzero(ok & (home == m))[0]
            if len(cand):
                partners.append(rng.choice(cand))
            if len(partners) == max(KS) - 1:
                break
        if len(partners) < max(KS) - 1:
            continue
        firms = np.array([i] + partners)
        w = ls[t - l:t + 1, firms].astype(np.float64)
        R = np.diff(w, axis=0) - jump[t - l + 1:t + 1, firms].astype(np.float64)
        S = np.exp(w[:-1])
        S0 = np.tile(S[0], (l, 1))                                  # sizes frozen at the window's start
        hits = int((jump[t - l + 1:t + 1, firms] != 0).any(axis=0)[0])
        for K in KS:
            Rk = np.ascontiguousarray(R[:, :K])
            d_hat, d_til = gains(Rk, np.ascontiguousarray(S[:, :K]), s['alpha'], phi[K], s['g'])
            c_hat, c_til = gains(Rk, np.ascontiguousarray(S0[:, :K]), s['alpha'], phi[K], s['g'])
            rows.append(dict(family=s['log_family'], alpha=s['alpha'], rep=s['rep'], t=t, firm=int(i), K=K,
                             focal_hit=hits, delta_hat=d_hat, delta_tilde=d_til,
                             delta_hat_fixed=c_hat, delta_tilde_fixed=c_til))
    return rows


def stats(g):
    fin = g[np.isfinite(g.delta_hat) & np.isfinite(g.delta_tilde)]
    d = fin.delta_tilde - fin.delta_hat
    r = d / fin.delta_hat
    out = dict(n=len(fin), n_dropped=len(g) - len(fin))
    for k, q in QS.items():
        out[f'diff_{k}'] = d.quantile(q)
    for k, q in QS.items():
        out[f'ratio_{k}'] = r.quantile(q)
    out.update(delta_hat_median=fin.delta_hat.median(), delta_tilde_median=fin.delta_tilde.median(),
               abs_delta_hat_median=fin.delta_hat.abs().median(), abs_diff_median=d.abs().median(),
               rel_abs=d.abs().median() / fin.delta_hat.abs().median(),
               sign_disagree=((fin.delta_tilde > 0) != (fin.delta_hat > 0)).mean())
    # decomposition of diff with the sizes frozen at the window's start (no in-window weight dynamics):
    #   diff = higher_order + weight_term_hat + weight_term_tilde
    #   higher_order      = Delta-tilde_fixed - Delta-hat_fixed  (the expansion's omitted terms, same fixed weights)
    #   weight_term_hat   = Delta-hat_fixed - Delta-hat          (minus the replay's in-window weight-return term)
    #   weight_term_tilde = Delta-tilde - Delta-tilde_fixed      (the same in the moment rule's weights)
    fx = fin[np.isfinite(fin.delta_hat_fixed) & np.isfinite(fin.delta_tilde_fixed)]
    comps = {'higher_order': fx.delta_tilde_fixed - fx.delta_hat_fixed,
             'weight_term_hat': fx.delta_hat_fixed - fx.delta_hat,
             'weight_term_tilde': fx.delta_tilde - fx.delta_tilde_fixed}
    for name, v in comps.items():
        out[f'{name}_median'] = v.median()
        out[f'{name}_p25'] = v.quantile(.25)
        out[f'{name}_p75'] = v.quantile(.75)
        out[f'{name}_mean'] = v.mean()
        out[f'{name}_ratio_median'] = (v / fx.delta_hat).median()
    out['higher_order_ratio_fixed_p5'] = (comps['higher_order'] / fx.delta_hat_fixed).quantile(.05)
    out['higher_order_ratio_fixed_median'] = (comps['higher_order'] / fx.delta_hat_fixed).median()
    out['higher_order_ratio_fixed_p95'] = (comps['higher_order'] / fx.delta_hat_fixed).quantile(.95)
    out['diff_mean'] = d.mean()
    return out


def main():
    pdir = 'pilot_' + pop_pilot(default='d')
    trace_dir = sys.argv[1]
    n = int(sys.argv[sys.argv.index('--n') + 1]) if '--n' in sys.argv else 2000
    scenarios = json.load(open(os.path.join(ROOT, pdir, 'scenarios.json')))
    rows = []
    for s in select(scenarios):
        tag = f"{s['log_family']}_a{s['alpha']}_r{s['rep']}"
        t0 = time.time()
        rows += draws_for(s, os.path.join(trace_dir, tag + '.npz'), n)
        print(tag, f'{time.time() - t0:.0f}s', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ROOT, pdir, 'approximation_error_samples.csv.gz'), index=False)
    out = []
    for (fam, K), g in df.groupby(['family', 'K'], sort=False):
        out.append(dict(family=fam, alpha='all', K=K, **stats(g)))
    for (fam, a, K), g in df.groupby(['family', 'alpha', 'K'], sort=False):
        out.append(dict(family=fam, alpha=a, K=K, **stats(g)))
    pd.DataFrame(out).to_csv(os.path.join(ROOT, pdir, 'approximation_error.csv'), index=False)
    print(pd.DataFrame(out).head(12).to_string())


if __name__ == '__main__':
    main()
