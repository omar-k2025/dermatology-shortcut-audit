"""
utils/bootstrap.py
Bootstrap confidence interval for domain degradation (ΔAUC).
"""

import numpy as np
from sklearn.metrics import roc_auc_score


def bootstrap_delta_auc_ci(labels_ref, probs_ref,
                            labels_ood, probs_ood,
                            n_boot=2000, alpha=0.05, seed=42):
    """
    Bootstrap CI for ΔAUC = AUC_ref - AUC_ood.

    Parameters
    ----------
    labels_ref : array-like, ground truth for in-distribution dataset
    probs_ref  : array-like, predicted probabilities for in-distribution
    labels_ood : array-like, ground truth for out-of-distribution dataset
    probs_ood  : array-like, predicted probabilities for out-of-distribution
    n_boot     : int, number of bootstrap iterations (default 2000)
    alpha      : float, significance level (default 0.05)
    seed       : int, random seed

    Returns
    -------
    delta_obs : float, observed ΔAUC
    ci_lo     : float, lower bound of CI
    ci_hi     : float, upper bound of CI
    """
    rng = np.random.default_rng(seed)
    auc_ref = roc_auc_score(labels_ref, probs_ref)
    auc_ood = roc_auc_score(labels_ood, probs_ood)
    delta_obs = auc_ref - auc_ood

    deltas = []
    for _ in range(n_boot):
        idx_r = rng.choice(len(labels_ref), len(labels_ref), replace=True)
        lr = np.array(labels_ref)[idx_r]
        pr = np.array(probs_ref)[idx_r]

        idx_o = rng.choice(len(labels_ood), len(labels_ood), replace=True)
        lo = np.array(labels_ood)[idx_o]
        po = np.array(probs_ood)[idx_o]

        try:
            a_r = roc_auc_score(lr, pr)
            a_o = roc_auc_score(lo, po)
            deltas.append(a_r - a_o)
        except ValueError:
            pass

    deltas = np.array(deltas)
    ci_lo  = np.percentile(deltas, alpha / 2 * 100)
    ci_hi  = np.percentile(deltas, (1 - alpha / 2) * 100)
    return delta_obs, ci_lo, ci_hi
