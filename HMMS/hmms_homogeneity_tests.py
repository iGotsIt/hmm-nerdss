"""
Test whether two fitted HMMs share the same transition matrix
(Anderson-Goodman chi-squared test of homogeneity, applied row by row).

INPUT: posterior expected transition counts from each fit, i.e. the
       m x m matrix  xi[i, j] = sum_t P(s_t = i, s_{t+1} = j | obs).
       These are the sufficient statistics accumulated in the EM E-step,
       NOT the normalized transition matrix.

IMPORTANT: align the states across the two models (match by emission
           distribution) BEFORE calling this, or the result is meaningless.
"""

import numpy as np
from scipy.stats import chi2
import HMMS.hmm as hmm
from pathlib import Path



def chi_2_transition_homogeneity_test(xi1, xi2, min_expected=5.0):
    """
    Parameters
    ----------
    xi1, xi2 : (m, m) array_like
        Posterior expected transition counts from model 1 and model 2,
        with states already aligned to the same ordering.
    min_expected : float
        Warn if any expected cell count falls below this (the chi-squared
        approximation degrades for sparse cells).

    Returns
    -------
    stat : float   aggregated Pearson chi-squared statistic
    dof  : int     total degrees of freedom
    pval : float   p-value for H0: the two chains share a transition matrix

    For this transition matrix, there are some assumptions that are not 
    satisfied by the data, so the p-value is not reliable. In particular, 
    the expected counts of certain cells are 0 (no transitions), and the datapoints are not 
    really independent. P-value might be less significant than it appears.
    """
    xi1 = np.asarray(xi1, dtype=float)
    xi2 = np.asarray(xi2, dtype=float)
    assert xi1.shape == xi2.shape and xi1.shape[0] == xi1.shape[1], "need two square m x m matrices"
    m = xi1.shape[0]

    stat = 0.0
    dof = 0
    low_cells = 0

    for i in range(m):  # one 2 x m homogeneity test per from-state
        row1, row2 = xi1[i], xi2[i]
        n1, n2 = row1.sum(), row2.sum()
        if n1 == 0 or n2 == 0:
            continue  # a state never visited in one model -> skip its row

        pooled = (row1 + row2) / (n1 + n2)          # common transition probs under H0
        support = pooled > 0                        # reachable destination states
        e1, e2 = n1 * pooled, n2 * pooled           # expected counts

        for obs, exp in ((row1, e1), (row2, e2)):
            s = support & (exp > 0)
            stat += np.sum((obs[s] - exp[s]) ** 2 / exp[s])
            low_cells += np.sum(exp[s] < min_expected)

        dof += int(support.sum()) - 1               # (cols with support - 1)

    pval = float(chi2.sf(stat, dof)) if dof > 0 else float("nan")

    if low_cells:
        print(f"warning: {low_cells} expected cell(s) < {min_expected}; "
              "chi-squared p-value may be unreliable -- consider a bootstrap or G-test.")

    return float(stat), dof, pval



if __name__ == "__main__":
    # checking if the transition matrices from two fits are the same. 
    x1 = np.genfromtxt("/Users/jefferyzhou/Documents/johnson-lab/Jake_DNA_protein/GAFsmFRETdata/hel3_trace_9/A_counts.txt", delimiter=",")
    x2 = np.load("/Users/jefferyzhou/Documents/johnson-lab/runs/trace_9_counts.npy")

    stat, dof, pval = chi_2_transition_homogeneity_test(x1, x2)
    print(f"chi2 = {stat:.3f}, df = {dof}, p = {pval:.4f}")
    print("same transition matrix" if pval > 0.05 else "transition matrices differ")