import numpy as np
import pandas as pd

"Permutation test template"
def permutation_shuffler_test(metric_func, permuter, M=1000, **kwargs):
    unpermuted = metric_func(**kwargs)
    criterions = []
    for _ in range(M):
        artefacts = permuter(**kwargs)
        criterions.append(metric_func(**artefacts))
    k = sum(1 for c in criterions if c >= unpermuted)
    p_value = (1 + k) / (M + 1)
    return p_value

"Example metric function"
def metric(levs, rets, weights, **kwargs):
    capital_ret = [
        l * np.dot(w, r)
        for l, w, r in zip(levs.values, weights.values, rets.values)
    ]
    return round(float(np.mean(capital_ret) / np.std(capital_ret) * np.sqrt(365)),3)

"Generic shuffler"
def shuffle_weights_on_eligs(weights_df, eligs_df, method="time"):
    assert method in ["time", "xs"]
    if method == "time":
        cols = []
        for wc, ec in zip(weights_df.T.values, eligs_df.T.values):
            msk = np.where(ec)[0]
            prm = np.random.permutation(wc[msk])
            nwc = np.zeros(len(wc))
            np.put(nwc, msk, prm)
            cols.append(pd.Series(nwc))
        nweight = pd.concat(cols, axis=1)
        nweight.columns = weights_df.columns
        nweight.index = weights_df.index
        nweight = nweight.div(np.abs(nweight).sum(axis=1), axis=0).fillna(0.0)
        return nweight

    if method == "xs":
        rows = []
        for wr, er in zip(weights_df.values, eligs_df.values):
            msk = np.where(er)[0]
            prm = np.random.permutation(wr[msk])
            nwr = np.zeros(len(wr))
            np.put(nwr, msk, prm)
            rows.append(pd.Series(nwr))
        nweight = pd.concat(rows, axis=1).T
        nweight.columns = weights_df.columns
        nweight.index = weights_df.index
        return nweight

"Specific permuters for timer/picker"
def time_shuffler(levs, rets, weights, eligibles):
    weights = shuffle_weights_on_eligs(weights, eligibles, method="time")
    return {"levs": levs, "rets": rets, "weights": weights}

def pick_shuffler(levs, rets, weights, eligibles):
    weights = shuffle_weights_on_eligs(weights, eligibles, method="xs")
    return {"levs": levs, "rets": rets, "weights": weights}

"Stepdown procedure for multiple strategies"
def stepdown_algorithm(unpermuted_criterions, round_criterions, alpha):
    """
    unpermuted_criterions: N observed test statistics (e.g., Sharpe ratios)
    round_criterions: M lists of test statistics from M rounds of permutations, each of length N
    alpha: significance level (e.g., 0.05)
    """
    pvalues = np.array([None] * len(unpermuted_criterions))
    exact = np.array([False] * len(unpermuted_criterions), dtype=bool)
    indices = np.array(list(range(len(unpermuted_criterions))))
    while not all(exact):
        stepwise_indices = indices[~exact]
        stepwise_criterions = np.array(unpermuted_criterions)[stepwise_indices]
        member_count = np.zeros(len(stepwise_criterions))
        for round in range(len(round_criterions)):
            round_max = np.max(np.array(round_criterions[round])[stepwise_indices])
            member_count += (0.0 + round_max >= np.array(stepwise_criterions))
        bounded_pvals = (1 + member_count) / (len(round_criterions) + 1)
        best_member = np.argmin(bounded_pvals)
        exact[stepwise_indices[best_member]] = True
        pvalues[stepwise_indices[best_member]] = np.min(bounded_pvals)
        if np.min(bounded_pvals) >= alpha:
            for bounded_p, index in zip(bounded_pvals, stepwise_indices):
                pvalues[index] = bounded_p
            break
    
    return pvalues, exact