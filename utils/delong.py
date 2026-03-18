"""
utils/delong.py
DeLong method for AUC confidence intervals and two-sample comparison.

Reference:
    DeLong ER, DeLong DM, Clarke-Pearson DL. Comparing the areas under two
    or more correlated receiver operating characteristic curves: a nonparametric
    approach. Biometrics. 1988;44(3):837-845.
"""

import numpy as np
import scipy.stats as stats
from sklearn.metrics import roc_auc_score


def delong_ci(y_true, y_score, alpha=0.05):
    """
    Compute AUC and 95% confidence interval using the DeLong method.

    Parameters
    ----------
    y_true  : array-like, binary ground truth labels
    y_score : array-like, predicted probabilities
    alpha   : float, significance level (default 0.05)

    Returns
    -------
    auc : float
    ci_lo : float, lower bound of CI
    ci_hi : float, upper bound of CI
    se    : float, standard error
    """
    y_true  = np.array(y_true)
    y_score = np.array(y_score)
    auc = roc_auc_score(y_true, y_score)
    n1 = int(y_true.sum())
    n0 = len(y_true) - n1
    q1 = auc / (2 - auc)
    q2 = 2 * auc ** 2 / (1 + auc)
    var = (auc * (1 - auc) +
           (n1 - 1) * (q1 - auc ** 2) +
           (n0 - 1) * (q2 - auc ** 2)) / (n1 * n0)
    se = np.sqrt(max(var, 1e-10))
    z  = stats.norm.ppf(1 - alpha / 2)
    return auc, max(0.0, auc - z * se), min(1.0, auc + z * se), se


def delong_compare(y_true, scores1, scores2):
    """
    Two-sample DeLong test: H0: AUC1 == AUC2.

    Parameters
    ----------
    y_true  : array-like, shared binary ground truth labels
    scores1 : array-like, predicted probabilities from model 1
    scores2 : array-like, predicted probabilities from model 2

    Returns
    -------
    auc1 : float
    auc2 : float
    z    : float, test statistic
    p    : float, two-sided p-value
    """
    y_true  = np.array(y_true)
    s1      = np.array(scores1)
    s2      = np.array(scores2)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    n1, n0  = len(pos_idx), len(neg_idx)

    def structural_components(scores):
        V10 = np.array([np.mean(scores[pos_idx[i]] > scores[neg_idx])
                        for i in range(n1)])
        V01 = np.array([np.mean(scores[neg_idx[j]] < scores[pos_idx])
                        for j in range(n0)])
        return V10, V01

    V10_1, V01_1 = structural_components(s1)
    V10_2, V01_2 = structural_components(s2)

    S10  = np.cov(np.stack([V10_1, V10_2]))
    S01  = np.cov(np.stack([V01_1, V01_2]))
    S    = S10 / n1 + S01 / n0
    var_diff = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    se_diff  = np.sqrt(max(var_diff, 1e-12))

    auc1 = roc_auc_score(y_true, s1)
    auc2 = roc_auc_score(y_true, s2)
    z    = (auc1 - auc2) / se_diff
    p    = 2 * (1 - stats.norm.cdf(abs(z)))
    return auc1, auc2, z, p


def auc_vs_chance(auc, se):
    """
    One-sided z-test: H0: AUC = 0.5 (random performance).

    Returns
    -------
    z : float
    p : float, one-sided p-value
    """
    z = (auc - 0.5) / se
    p = 1 - stats.norm.cdf(z)
    return z, p
