from typing import Iterable
import sys

import numpy as np
from pathlib import Path
from glob import glob
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal


class MultivariateGaussian:
    def __init__(self, mean: np.ndarray, cov: np.ndarray, seed: int | None = None):
        self.seed = seed
        self.mean = mean
        self.cov = cov
        self.dim = mean.shape[0]
        self.precision = np.linalg.inv(cov)
        self.det_cov = np.linalg.det(cov)
        self.generator = multivariate_normal(self.mean, self.cov, self.seed)

    def draw(self, num_samples: int = 1) -> np.ndarray:
        """Draw a sample from the multivariate Gaussian distribution."""
        return self.generator.rvs(size=num_samples)

    def update(self, mean: np.ndarray, cov: np.ndarray):
        self.mean = mean
        self.cov = cov
        self.precision = np.linalg.inv(cov)
        self.det_cov = np.linalg.det(cov)


def gaussian_emission(u, sigma, trajectory, d):

    ## pdf of a multivariate gaussian distribution
    n = u.shape[1]
    p = np.zeros(n)
    for i in range(0, n):
        covm = sigma[:, :, i]
        detcovm = np.linalg.det(covm)
        convm_inv = np.linalg.inv(covm)
        exv = np.exp(
            -0.5 * (trajectory - u[:, i]).T @ convm_inv @ (trajectory - u[:, i])
        )
        p[i] = (1 / ((2 * np.pi) ** (d / 2) * detcovm**0.5)) * exv

    return p


def build_states(stateindex, ug, ub, ur):
    u = stateindex * 0
    print(f"Building states u from stateindex: {stateindex}")
    for i in range(0, stateindex.shape[1]):
        u[0, i] = ug[stateindex[0, i] - 1]
        u[1, i] = ub[stateindex[1, i] - 1]
        u[2, i] = ur[stateindex[2, i] - 1]

    print(f"Built states u: {u}")
    return u


def build_select_matrices(stateindex):

    state_num = stateindex.shape[1]
    d = stateindex.shape[0]

    # green
    Mg = np.zeros((d, max(stateindex[0, :]), state_num))
    for i in range(0, state_num):
        index = stateindex[0, i] - 1
        Mg[0, index, i] = 1

    # blue
    Mb = np.zeros((d, max(stateindex[1, :]), state_num))
    for i in range(0, state_num):
        index = stateindex[1, i] - 1
        Mb[1, index, i] = 1

    # red
    Mr = np.zeros((d, max(stateindex[2, :]), state_num))
    for i in range(0, state_num):
        index = stateindex[2, i] - 1
        Mr[2, index, i] = 1

    return Mg, Mb, Mr


## extend to multiple sequences by accumulating the sufficient statistics
def expectation_maximization_single_trace(
    stateindex,
    ug: np.ndarray,
    ub: np.ndarray,
    ur: np.ndarray,
    transition_mat: np.ndarray,
    sigma: np.ndarray,
    trajectory: np.ndarray,  # either a single trajectory of shape (d, T) or multiple trajectories of shape (num_traces, d, T)
):

    # need to check that everything is matmul or elementwise
    Mg, Mb, Mr = build_select_matrices(stateindex)
    u = build_states(stateindex, ug, ub, ur)

    likelihood = []
    transition_history = []
    sigma0 = sigma
    criterion = 1e-3
    isConverged = False
    iter = 100
    time = 0
    Nstate = u.shape[1]
    eps = np.spacing(1.0)

    print("Fitting HMM with EM algorithm...")

    d = trajectory.shape[0]
    T = trajectory.shape[1]

    while not isConverged and iter > 0:
        time = time + 1
        PI_acc = np.zeros(Nstate)
        trans_num = np.zeros((Nstate, Nstate))
        trans_den = np.zeros(Nstate)
        gammaSumT = np.zeros(Nstate)
        gammaSumY = np.zeros((d, Nstate))
        cov_acc = np.zeros((d, d, Nstate))
        totalLogLik = 0

        pi = np.ones(Nstate) / Nstate

        u_old = u

        # start loop for multiple traces if there are multiple

        obs_dist = np.zeros((Nstate, T))
        alphas = np.zeros((Nstate, T))
        betas = np.zeros((Nstate, T))
        gamma = np.zeros((Nstate, T))
        xi = np.zeros((Nstate, Nstate, T - 1))

        ## alphas (forward pass)
        obs_dist[:, 0] = gaussian_emission(u, sigma, trajectory[:, 0], d)
        for i in range(0, Nstate):
            alphas[i, 0] = pi[i] * obs_dist[i, 0]

        alphasum = sum(alphas[:, 0])
        alphas[:, 0] = alphas[:, 0] / alphasum

        for t in range(0, T - 1):
            obs_dist[:, t + 1] = gaussian_emission(u, sigma, trajectory[:, t + 1], d)
            for j in range(0, Nstate):
                alphas[j, t + 1] = (
                    alphas[:, t].T @ transition_mat[:, j] * obs_dist[j, t + 1]
                )
            alphasum = sum(alphas[:, t + 1])
            alphas[:, t + 1] = alphas[:, t + 1] / alphasum

        ## betas (backward pass)
        betas[:, T - 1] = np.ones(Nstate)
        betasum = sum(betas[:, T - 1])
        betas[:, T - 1] = betas[:, T - 1] / betasum

        for t in range(T - 2, -1, -1):
            for i in range(0, Nstate):
                betatmp = 0
                for j in range(0, Nstate):
                    betatmp = (
                        betatmp
                        + betas[j, t + 1] * obs_dist[j, t + 1] * transition_mat[i, j]
                    )
                betas[i, t] = betatmp
            betasum = sum(betas[:, t])
            betas[:, t] = betas[:, t] / betasum

        p_y = (
            alphas[:, 0].T @ betas[:, 0]
        )  # probability of seeing the data given the model parameters

        totalLogLik = totalLogLik + np.log(p_y)

        # gamma
        gamma[:, T - 1] = alphas[:, T - 1]
        for t in range(T - 2, -1, -1):
            for i in range(0, Nstate):
                gamma[i, t] = alphas[i, t] * betas[i, t] / p_y

            gamma[:, t] = gamma[:, t] / np.sum(gamma[:, t])

        # xi
        for t in range(0, T - 1):
            for i in range(0, Nstate):
                for j in range(0, Nstate):
                    xi[i, j, t] = (
                        alphas[i, t]
                        * obs_dist[j, t + 1]
                        * transition_mat[i, j]
                        * betas[j, t + 1]
                        / p_y
                    )
            xi[:, :, t] = xi[:, :, t] / np.sum(np.sum(xi[:, :, t]))

        # accumulate sufficient statistics
        PI_acc = PI_acc + gamma[:, 0]
        trans_num = trans_num + np.sum(xi, axis=2)
        trans_den = trans_den + np.sum(gamma[:, 0 : T - 1], axis=1)

        gammaSumT = gammaSumT + np.sum(gamma, axis=1)
        gammaSumY = gammaSumY + trajectory @ gamma.T

        for i in range(0, Nstate):
            for t in range(0, T):
                diff = trajectory[:, t] - u_old[:, i]
                cov_acc[:, :, i] = cov_acc[:, :, i] + gamma[i, t] * np.outer(diff, diff)

        likelihood = np.append(likelihood, totalLogLik)

        if time >= 2:
            if abs(likelihood[time - 1] - likelihood[time - 2]) < criterion:
                isConverged = True  # we have converged; just break and return
                print(f"EM algorithm converged in {100 - iter} iterations.")
                return u, sigma, transition_mat, pi, likelihood, transition_history

        # make updates (M step)
        pi = PI_acc / sum(PI_acc)  # update initial state probabilities

        # this is the main difference with the matlab code -- keep the previous row for dead states to prevent NaN values

        for i in range(0, Nstate):
            # if state i is (near-)unoccupied, trans_den[i] -> 0 and the row
            # becomes 0/0 = NaN, which poisons the next forward pass. Keep the
            # previous row for dead states instead.
            if (
                trans_den[i] < eps
            ):  ## guard against division by zero for unoccupied states (not in original code)
                continue
            for j in range(0, Nstate):
                transition_mat[i, j] = trans_num[i, j] / trans_den[i]

            transition_mat[i, :] = transition_mat[i, :] / sum(transition_mat[i, :])

        # record the transition matrix produced by this iteration
        transition_history.append(transition_mat.copy())

        """
        This is the other difference with the matlab code 

        these should be precision matrices, not sigmas. Differs from the matlab implementation. This is more 
        correct. See https://people.eecs.berkeley.edu/~jordan/courses/281A-fall04/lectures/lec-10-26.pdf
        for derivation of the log-likelihood of a Gaussian HMM. The terms in the supplemental should also
        be with respect to the log-likelihood, not the likelihood. 
        """

        precision = np.linalg.inv(
            sigma
        )  # precision is the inverse of the covariance matrix

        # update constrained means
        a11 = np.zeros((Mg.shape[1], Mg.shape[1]))
        a12 = np.zeros((Mg.shape[1], Mb.shape[1]))
        a13 = np.zeros((Mg.shape[1], Mr.shape[1]))

        a21 = np.zeros((Mb.shape[1], Mg.shape[1]))
        a22 = np.zeros((Mb.shape[1], Mb.shape[1]))
        a23 = np.zeros((Mb.shape[1], Mr.shape[1]))

        a31 = np.zeros((Mr.shape[1], Mg.shape[1]))
        a32 = np.zeros((Mr.shape[1], Mb.shape[1]))
        a33 = np.zeros((Mr.shape[1], Mr.shape[1]))

        for i in range(0, Nstate):
            a11 = a11 + Mg[:, :, i].T @ precision[:, :, i] @ Mg[:, :, i] * gammaSumT[i]
            a12 = a12 + Mg[:, :, i].T @ precision[:, :, i] @ Mb[:, :, i] * gammaSumT[i]
            a13 = a13 + Mg[:, :, i].T @ precision[:, :, i] @ Mr[:, :, i] * gammaSumT[i]
            a21 = a21 + Mb[:, :, i].T @ precision[:, :, i] @ Mg[:, :, i] * gammaSumT[i]
            a22 = a22 + Mb[:, :, i].T @ precision[:, :, i] @ Mb[:, :, i] * gammaSumT[i]
            a23 = a23 + Mb[:, :, i].T @ precision[:, :, i] @ Mr[:, :, i] * gammaSumT[i]
            a31 = a31 + Mr[:, :, i].T @ precision[:, :, i] @ Mg[:, :, i] * gammaSumT[i]
            a32 = a32 + Mr[:, :, i].T @ precision[:, :, i] @ Mb[:, :, i] * gammaSumT[i]
            a33 = a33 + Mr[:, :, i].T @ precision[:, :, i] @ Mr[:, :, i] * gammaSumT[i]

        c1 = np.zeros(Mg.shape[1])
        c2 = np.zeros(Mb.shape[1])
        c3 = np.zeros(Mr.shape[1])

        for i in range(0, Nstate):
            c1 = c1 + Mg[:, :, i].T @ precision[:, :, i] @ gammaSumY[:, i]
            c2 = c2 + Mb[:, :, i].T @ precision[:, :, i] @ gammaSumY[:, i]
            c3 = c3 + Mr[:, :, i].T @ precision[:, :, i] @ gammaSumY[:, i]

        coeff = np.block([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]])
        b = np.concatenate([c1, c2, c3])
        results = np.linalg.lstsq(coeff.T, b.T)[0].T
        # results = np.linalg.pinv(coeff) @ b

        ug = results[0 : c1.shape[0]]
        print(f"C1 shape: {c1.shape}")
        ub = results[c1.shape[0] : c1.shape[0] + c2.shape[0]]
        ur = results[
            c1.shape[0] + c2.shape[0] : c1.shape[0] + c2.shape[0] + c3.shape[0]
        ]
        print(f"Updated means: ug={ug}, ub={ub}, ur={ur}")

        u = build_states(stateindex, ug, ub, ur)

        for i in range(0, Nstate):
            if gammaSumT[i] < eps:
                sigma[:, :, i] = sigma0[:, :, i]
                continue
            sigmatmp = cov_acc[:, :, i] / gammaSumT[i]
            if (
                np.isnan(sigmatmp).any()
                or np.isinf(sigmatmp).any()
                or np.sqrt(sigmatmp[0, 0]) < 0.05
                or np.sqrt(sigmatmp[1, 1]) < 0.05
                or np.sqrt(sigmatmp[2, 2]) < 0.05
                or np.linalg.det(sigmatmp) < 1e-9
            ):
                sigma[:, :, i] = sigma0[:, :, i]
            else:
                sigma[:, :, i] = sigmatmp

        iter = iter - 1

    if not isConverged:
        print(f"EM algorithm did not converge within the maximum number of iterations.")

    return u, sigma, transition_mat, pi, likelihood, transition_history


## extend to multiple sequences by accumulating the sufficient statistics
def expectation_maximization_multi_trace(
    stateindex,
    ug: np.ndarray,
    ub: np.ndarray,
    ur: np.ndarray,
    transition_mat: np.ndarray,
    sigma: np.ndarray,
    trajectory: list | np.ndarray,
):

    # need to check that everything is matmul or elementwise
    Mg, Mb, Mr = build_select_matrices(stateindex)
    u = build_states(stateindex, ug, ub, ur)

    likelihood = []
    transition_history = []
    sigma0 = sigma
    criterion = 1e-3
    isConverged = False
    iter = 100
    time = 0
    Nstate = u.shape[1]
    eps = np.spacing(1.0)

    # initial-state distribution; updated each M step and reused in the next
    # iteration's forward pass
    pi = np.ones(Nstate) / Nstate

    print("Fitting HMM with EM algorithm...")

    while not isConverged and iter > 0:
        time = time + 1

        # sufficient statistics accumulated across ALL traces for this EM
        # iteration. Zeroed once here, added to inside the per-trace loop, and
        # consumed by a single M step after the loop.
        d = trajectory[0].shape[0]
        PI_acc = np.zeros(Nstate)
        trans_num = np.zeros((Nstate, Nstate))
        trans_den = np.zeros(Nstate)
        gammaSumT = np.zeros(Nstate)
        gammaSumY = np.zeros((d, Nstate))
        cov_acc = np.zeros((d, d, Nstate))
        totalLogLik = 0

        # means from the previous iteration, used for the covariance update
        u_old = u

        # ---------------- E step: one pass per trace ----------------
        for trace in trajectory:
            d = trace.shape[0]
            T = trace.shape[1]

            obs_dist = np.zeros((Nstate, T))
            alphas = np.zeros((Nstate, T))
            betas = np.zeros((Nstate, T))
            gamma = np.zeros((Nstate, T))
            xi = np.zeros((Nstate, Nstate, T - 1))

            ## alphas (forward pass)
            obs_dist[:, 0] = gaussian_emission(u, sigma, trace[:, 0], d)
            for i in range(0, Nstate):
                alphas[i, 0] = pi[i] * obs_dist[i, 0]

            alphasum = sum(alphas[:, 0])
            alphas[:, 0] = alphas[:, 0] / alphasum

            for t in range(0, T - 1):
                obs_dist[:, t + 1] = gaussian_emission(u, sigma, trace[:, t + 1], d)
                for j in range(0, Nstate):
                    alphas[j, t + 1] = (
                        alphas[:, t].T @ transition_mat[:, j] * obs_dist[j, t + 1]
                    )
                alphasum = sum(alphas[:, t + 1])
                alphas[:, t + 1] = alphas[:, t + 1] / alphasum

            ## betas (backward pass)
            betas[:, T - 1] = np.ones(Nstate)
            betasum = sum(betas[:, T - 1])
            betas[:, T - 1] = betas[:, T - 1] / betasum

            for t in range(T - 2, -1, -1):
                for i in range(0, Nstate):
                    betatmp = 0
                    for j in range(0, Nstate):
                        betatmp = (
                            betatmp
                            + betas[j, t + 1]
                            * obs_dist[j, t + 1]
                            * transition_mat[i, j]
                        )
                    betas[i, t] = betatmp
                betasum = sum(betas[:, t])
                betas[:, t] = betas[:, t] / betasum

            p_y = (
                alphas[:, 0].T @ betas[:, 0]
            )  # probability of seeing the data given the model parameters

            totalLogLik = totalLogLik + np.log(p_y)

            # gamma
            gamma[:, T - 1] = alphas[:, T - 1]
            for t in range(T - 2, -1, -1):
                for i in range(0, Nstate):
                    gamma[i, t] = alphas[i, t] * betas[i, t] / p_y

                gamma[:, t] = gamma[:, t] / np.sum(gamma[:, t])

            # xi
            for t in range(0, T - 1):
                for i in range(0, Nstate):
                    for j in range(0, Nstate):
                        xi[i, j, t] = (
                            alphas[i, t]
                            * obs_dist[j, t + 1]
                            * transition_mat[i, j]
                            * betas[j, t + 1]
                            / p_y
                        )
                xi[:, :, t] = xi[:, :, t] / np.sum(np.sum(xi[:, :, t]))

            # accumulate this trace's sufficient statistics into the shared totals
            PI_acc = PI_acc + gamma[:, 0]
            trans_num = trans_num + np.sum(xi, axis=2)
            trans_den = trans_den + np.sum(gamma[:, 0 : T - 1], axis=1)

            gammaSumT = gammaSumT + np.sum(gamma, axis=1)
            gammaSumY = gammaSumY + trace @ gamma.T

            for i in range(0, Nstate):
                for t in range(0, T):
                    diff = trace[:, t] - u_old[:, i]
                    cov_acc[:, :, i] = cov_acc[:, :, i] + gamma[i, t] * np.outer(
                        diff, diff
                    )

        # ---------------- M step: once, from pooled statistics ----------------
        likelihood = np.append(likelihood, totalLogLik)

        if time >= 2:
            if abs(likelihood[time - 1] - likelihood[time - 2]) < criterion:
                isConverged = True  # we have converged; just break and return
                print(f"EM algorithm converged in {100 - iter} iterations.")
                return u, sigma, transition_mat, pi, likelihood, transition_history

        pi = PI_acc / sum(PI_acc)  # update initial state probabilities

        for i in range(0, Nstate):
            # if state i is (near-)unoccupied, trans_den[i] -> 0 and the row
            # becomes 0/0 = NaN, which poisons the next forward pass. Keep the
            # previous row for dead states instead.
            if (
                trans_den[i] < eps
            ):  ## guard against division by zero for unoccupied states (not in original code)
                continue
            for j in range(0, Nstate):
                transition_mat[i, j] = trans_num[i, j] / trans_den[i]

            transition_mat[i, :] = transition_mat[i, :] / sum(transition_mat[i, :])

        # record the transition matrix produced by this iteration
        transition_history.append(transition_mat.copy())

        precision = np.linalg.inv(
            sigma
        )  # precision is the inverse of the covariance matrix

        # update constrained means
        a11 = np.zeros((Mg.shape[1], Mg.shape[1]))
        a12 = np.zeros((Mg.shape[1], Mb.shape[1]))
        a13 = np.zeros((Mg.shape[1], Mr.shape[1]))

        a21 = np.zeros((Mb.shape[1], Mg.shape[1]))
        a22 = np.zeros((Mb.shape[1], Mb.shape[1]))
        a23 = np.zeros((Mb.shape[1], Mr.shape[1]))

        a31 = np.zeros((Mr.shape[1], Mg.shape[1]))
        a32 = np.zeros((Mr.shape[1], Mb.shape[1]))
        a33 = np.zeros((Mr.shape[1], Mr.shape[1]))

        for i in range(0, Nstate):
            a11 = a11 + Mg[:, :, i].T @ precision[:, :, i] @ Mg[:, :, i] * gammaSumT[i]
            a12 = a12 + Mg[:, :, i].T @ precision[:, :, i] @ Mb[:, :, i] * gammaSumT[i]
            a13 = a13 + Mg[:, :, i].T @ precision[:, :, i] @ Mr[:, :, i] * gammaSumT[i]
            a21 = a21 + Mb[:, :, i].T @ precision[:, :, i] @ Mg[:, :, i] * gammaSumT[i]
            a22 = a22 + Mb[:, :, i].T @ precision[:, :, i] @ Mb[:, :, i] * gammaSumT[i]
            a23 = a23 + Mb[:, :, i].T @ precision[:, :, i] @ Mr[:, :, i] * gammaSumT[i]
            a31 = a31 + Mr[:, :, i].T @ precision[:, :, i] @ Mg[:, :, i] * gammaSumT[i]
            a32 = a32 + Mr[:, :, i].T @ precision[:, :, i] @ Mb[:, :, i] * gammaSumT[i]
            a33 = a33 + Mr[:, :, i].T @ precision[:, :, i] @ Mr[:, :, i] * gammaSumT[i]

        c1 = np.zeros(Mg.shape[1])
        c2 = np.zeros(Mb.shape[1])
        c3 = np.zeros(Mr.shape[1])

        for i in range(0, Nstate):
            c1 = c1 + Mg[:, :, i].T @ precision[:, :, i] @ gammaSumY[:, i]
            c2 = c2 + Mb[:, :, i].T @ precision[:, :, i] @ gammaSumY[:, i]
            c3 = c3 + Mr[:, :, i].T @ precision[:, :, i] @ gammaSumY[:, i]

        coeff = np.block([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]])
        b = np.concatenate([c1, c2, c3])

        ## overdetermined matrix
        results = np.linalg.lstsq(coeff.T, b.T)[0].T
        # results = np.linalg.pinv(coeff) @ b

        ug = results[0 : c1.shape[0]]
        print(f"C1 shape: {c1.shape}")
        ub = results[c1.shape[0] : c1.shape[0] + c2.shape[0]]
        ur = results[
            c1.shape[0] + c2.shape[0] : c1.shape[0] + c2.shape[0] + c3.shape[0]
        ]
        print(f"Updated means: ug={ug}, ub={ub}, ur={ur}")

        u = build_states(stateindex, ug, ub, ur)

        for i in range(0, Nstate):
            if gammaSumT[i] < eps:
                sigma[:, :, i] = sigma0[:, :, i]
                continue
            sigmatmp = cov_acc[:, :, i] / gammaSumT[i]
            if (
                np.isnan(sigmatmp).any()
                or np.isinf(sigmatmp).any()
                or np.sqrt(sigmatmp[0, 0]) < 0.05
                or np.sqrt(sigmatmp[1, 1]) < 0.05
                or np.sqrt(sigmatmp[2, 2]) < 0.05
                or np.linalg.det(sigmatmp) < 1e-9
            ):
                sigma[:, :, i] = sigma0[:, :, i]
            else:
                sigma[:, :, i] = sigmatmp

        iter = iter - 1

    if not isConverged:
        print(f"EM algorithm did not converge within the maximum number of iterations.")

    return u, sigma, transition_mat, pi, likelihood, transition_history


def semi_pooled_expectation_maximization_multi_trace(
    stateindex,
    ug: np.ndarray,
    ub: np.ndarray,
    ur: np.ndarray,
    transition_mat: np.ndarray,
    sigma: np.ndarray,
    trajectory: list | np.ndarray,
):

    # need to check that everything is matmul or elementwise
    Mg, Mb, Mr = build_select_matrices(stateindex)
    u = build_states(stateindex, ug, ub, ur)

    likelihood = []
    transition_history = []
    sigma0 = sigma
    criterion = 1e-3
    isConverged = False
    iter = 100
    time = 0
    Nstate = u.shape[1]
    eps = np.spacing(1.0)

    # initial-state distribution; updated each M step and reused in the next
    # iteration's forward pass
    pi = np.ones(Nstate) / Nstate

    print("Fitting HMM with EM algorithm...")

    MVGenerator = MultivariateGaussian(mean=u, cov=sigma)

    while not isConverged and iter > 0:
        time = time + 1

        # sufficient statistics accumulated across ALL traces for this EM
        # iteration. Zeroed once here, added to inside the per-trace loop, and
        # consumed by a single M step after the loop.
        d = trajectory[0].shape[0]
        PI_acc = np.zeros(Nstate)
        trans_num = np.zeros((Nstate, Nstate))
        trans_den = np.zeros(Nstate)
        gammaSumT = np.zeros(Nstate)
        gammaSumY = np.zeros((d, Nstate))
        cov_acc = np.zeros((d, d, Nstate))
        totalLogLik = 0

        # means from the previous iteration, used for the covariance update
        u_old = u

        per_trace_means = []
        per_trace_covs = []

        # ---------------- E step: one pass per trace ----------------
        for trace in trajectory:
            d = trace.shape[0]
            T = trace.shape[1]

            obs_dist = np.zeros((Nstate, T))
            alphas = np.zeros((Nstate, T))
            betas = np.zeros((Nstate, T))
            gamma = np.zeros((Nstate, T))
            xi = np.zeros((Nstate, Nstate, T - 1))

            ## alphas (forward pass)
            obs_dist[:, 0] = gaussian_emission(u, sigma, trace[:, 0], d)
            for i in range(0, Nstate):
                alphas[i, 0] = pi[i] * obs_dist[i, 0]

            alphasum = sum(alphas[:, 0])
            alphas[:, 0] = alphas[:, 0] / alphasum

            for t in range(0, T - 1):
                obs_dist[:, t + 1] = gaussian_emission(u, sigma, trace[:, t + 1], d)
                for j in range(0, Nstate):
                    alphas[j, t + 1] = (
                        alphas[:, t].T @ transition_mat[:, j] * obs_dist[j, t + 1]
                    )
                alphasum = sum(alphas[:, t + 1])
                alphas[:, t + 1] = alphas[:, t + 1] / alphasum

            ## betas (backward pass)
            betas[:, T - 1] = np.ones(Nstate)
            betasum = sum(betas[:, T - 1])
            betas[:, T - 1] = betas[:, T - 1] / betasum

            for t in range(T - 2, -1, -1):
                for i in range(0, Nstate):
                    betatmp = 0
                    for j in range(0, Nstate):
                        betatmp = (
                            betatmp
                            + betas[j, t + 1]
                            * obs_dist[j, t + 1]
                            * transition_mat[i, j]
                        )
                    betas[i, t] = betatmp
                betasum = sum(betas[:, t])
                betas[:, t] = betas[:, t] / betasum

            p_y = (
                alphas[:, 0].T @ betas[:, 0]
            )  # probability of seeing the data given the model parameters

            totalLogLik = totalLogLik + np.log(p_y)

            # gamma
            gamma[:, T - 1] = alphas[:, T - 1]
            for t in range(T - 2, -1, -1):
                for i in range(0, Nstate):
                    gamma[i, t] = alphas[i, t] * betas[i, t] / p_y

                gamma[:, t] = gamma[:, t] / np.sum(gamma[:, t])

            # xi
            for t in range(0, T - 1):
                for i in range(0, Nstate):
                    for j in range(0, Nstate):
                        xi[i, j, t] = (
                            alphas[i, t]
                            * obs_dist[j, t + 1]
                            * transition_mat[i, j]
                            * betas[j, t + 1]
                            / p_y
                        )
                xi[:, :, t] = xi[:, :, t] / np.sum(np.sum(xi[:, :, t]))

            # accumulate this trace's sufficient statistics into the shared totals
            PI_acc = PI_acc + gamma[:, 0]
            trans_num = trans_num + np.sum(xi, axis=2)
            trans_den = trans_den + np.sum(gamma[:, 0 : T - 1], axis=1)

            gammaSumT = gammaSumT + np.sum(gamma, axis=1)
            gammaSumY = gammaSumY + trace @ gamma.T

            for i in range(0, Nstate):
                for t in range(0, T):
                    diff = trace[:, t] - u_old[:, i]
                    cov_acc[:, :, i] = cov_acc[:, :, i] + gamma[i, t] * np.outer(
                        diff, diff
                    )

            # ---------------- M step: once per trace ----------------

            likelihood = np.append(likelihood, totalLogLik)

            if time >= 2:
                if abs(likelihood[time - 1] - likelihood[time - 2]) < criterion:
                    isConverged = True  # we have converged; just break and return
                    print(f"EM algorithm converged in {100 - iter} iterations.")
                    return u, sigma, transition_mat, pi, likelihood, transition_history

            pi = PI_acc / sum(PI_acc)  # update initial state probabilities

            for i in range(0, Nstate):
                # if state i is (near-)unoccupied, trans_den[i] -> 0 and the row
                # becomes 0/0 = NaN, which poisons the next forward pass. Keep the
                # previous row for dead states instead.
                if (
                    trans_den[i] < eps
                ):  ## guard against division by zero for unoccupied states (not in original code)
                    continue
                for j in range(0, Nstate):
                    transition_mat[i, j] = trans_num[i, j] / trans_den[i]

                transition_mat[i, :] = transition_mat[i, :] / sum(transition_mat[i, :])

            # record the transition matrix produced by this iteration
            transition_history.append(transition_mat.copy())

            precision = np.linalg.inv(
                sigma
            )  # precision is the inverse of the covariance matrix

            # update constrained means
            a11 = np.zeros((Mg.shape[1], Mg.shape[1]))
            a12 = np.zeros((Mg.shape[1], Mb.shape[1]))
            a13 = np.zeros((Mg.shape[1], Mr.shape[1]))

            a21 = np.zeros((Mb.shape[1], Mg.shape[1]))
            a22 = np.zeros((Mb.shape[1], Mb.shape[1]))
            a23 = np.zeros((Mb.shape[1], Mr.shape[1]))

            a31 = np.zeros((Mr.shape[1], Mg.shape[1]))
            a32 = np.zeros((Mr.shape[1], Mb.shape[1]))
            a33 = np.zeros((Mr.shape[1], Mr.shape[1]))

            for i in range(0, Nstate):
                a11 = (
                    a11
                    + Mg[:, :, i].T @ precision[:, :, i] @ Mg[:, :, i] * gammaSumT[i]
                )
                a12 = (
                    a12
                    + Mg[:, :, i].T @ precision[:, :, i] @ Mb[:, :, i] * gammaSumT[i]
                )
                a13 = (
                    a13
                    + Mg[:, :, i].T @ precision[:, :, i] @ Mr[:, :, i] * gammaSumT[i]
                )
                a21 = (
                    a21
                    + Mb[:, :, i].T @ precision[:, :, i] @ Mg[:, :, i] * gammaSumT[i]
                )
                a22 = (
                    a22
                    + Mb[:, :, i].T @ precision[:, :, i] @ Mb[:, :, i] * gammaSumT[i]
                )
                a23 = (
                    a23
                    + Mb[:, :, i].T @ precision[:, :, i] @ Mr[:, :, i] * gammaSumT[i]
                )
                a31 = (
                    a31
                    + Mr[:, :, i].T @ precision[:, :, i] @ Mg[:, :, i] * gammaSumT[i]
                )
                a32 = (
                    a32
                    + Mr[:, :, i].T @ precision[:, :, i] @ Mb[:, :, i] * gammaSumT[i]
                )
                a33 = (
                    a33
                    + Mr[:, :, i].T @ precision[:, :, i] @ Mr[:, :, i] * gammaSumT[i]
                )

            c1 = np.zeros(Mg.shape[1])
            c2 = np.zeros(Mb.shape[1])
            c3 = np.zeros(Mr.shape[1])

            for i in range(0, Nstate):
                c1 = c1 + Mg[:, :, i].T @ precision[:, :, i] @ gammaSumY[:, i]
                c2 = c2 + Mb[:, :, i].T @ precision[:, :, i] @ gammaSumY[:, i]
                c3 = c3 + Mr[:, :, i].T @ precision[:, :, i] @ gammaSumY[:, i]

            coeff = np.block([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]])
            # adding the population precision hyperparameter error term
            coeff += MVGenerator.precision

            # adding the population mean hyperparameter error term
            b = np.concatenate([c1, c2, c3])
            b += MVGenerator.precision @ MVGenerator.mean

            ## overdetermined matrix
            results = np.linalg.lstsq(coeff.T, b.T)[0].T
            # results = np.linalg.pinv(coeff) @ b

            ug = results[0 : c1.shape[0]]
            print(f"C1 shape: {c1.shape}")
            ub = results[c1.shape[0] : c1.shape[0] + c2.shape[0]]
            ur = results[
                c1.shape[0] + c2.shape[0] : c1.shape[0] + c2.shape[0] + c3.shape[0]
            ]
            print(f"Updated means: ug={ug}, ub={ub}, ur={ur}")

            u = build_states(stateindex, ug, ub, ur)

            per_trace_means.append(u)
            per_trace_covs.append(sigma)

            for i in range(0, Nstate):
                if gammaSumT[i] < eps:
                    sigma[:, :, i] = sigma0[:, :, i]
                    continue
                sigmatmp = cov_acc[:, :, i] / gammaSumT[i]
                if (
                    np.isnan(sigmatmp).any()
                    or np.isinf(sigmatmp).any()
                    or np.sqrt(sigmatmp[0, 0]) < 0.05
                    or np.sqrt(sigmatmp[1, 1]) < 0.05
                    or np.sqrt(sigmatmp[2, 2]) < 0.05
                    or np.linalg.det(sigmatmp) < 1e-9
                ):
                    sigma[:, :, i] = sigma0[:, :, i]
                else:
                    sigma[:, :, i] = sigmatmp

        updated_hyper_mean = np.mean(per_trace_means, axis=0)
        updated_hyper_cov = np.sum(
            [
                (per_trace_means[index] - MVGenerator.mean)
                @ (per_trace_means[index] - MVGenerator.mean).T
                + np.linalg.inv(coeff + MVGenerator.precision)
                for index in range(len(per_trace_means))
            ],
            axis=0,
        ) / len(per_trace_means)

        MVGenerator.update(updated_hyper_mean, updated_hyper_cov)

        ## inverse-wishart update for covariances
        d = 3  # number of dimensions (green, blue, red)
        for channel in range(d):
            upsilon = (
                d
                + 3
                + 2
                * (cov_acc[channel, channel, :].sum() / len(per_trace_covs)) ** 2
                / np.var(cov_acc[channel, channel, :])
            )  # degrees of freedom

            psi = (
                (upsilon - d - 1)
                * cov_acc[channel, channel, :].sum()
                / len(per_trace_covs)
            )  # scale matrix

        iter = iter - 1

    if not isConverged:
        print(f"EM algorithm did not converge within the maximum number of iterations.")

    return u, sigma, transition_mat, pi, likelihood, transition_history


def build_trajectory(file_path):
    data = np.genfromtxt(file_path)
    time = data[:, 0]
    green = data[:, 1]
    red = data[:, 2]
    blue = data[:, 3]
    trajectory = np.stack((green, blue, red))  # shape (3, T)
    print(f"Trajectory shape: {trajectory.shape}")
    return trajectory


def fit_gmm_levels(trajectory, n_levels=3, covariance_type="diag", random_state=0):
    """Fit a 3-component 1-D Gaussian mixture to each channel to get the
    starting level vectors, mirroring fit_gmm_levels() in hmm.py.

    trajectory : (d, T) array with rows ordered (green, blue, red) as built by
        build_trajectory(). Each row is fit independently and its components
        are sorted by mean (descending) so the ordering lines up with the
        1-based level indices in stateindex (1=high, 2=mid, 3=low).

    Returns six length-`n_levels` vectors:
        ug, ub, ur                  per-channel means, sorted high -> mid -> low
        sigma_g, sigma_b, sigma_r   matching standard deviations
    """
    means = np.zeros((3, n_levels))
    sigmas = np.zeros((3, n_levels))
    for c in range(3):
        col = trajectory[c, :].reshape(-1, 1)
        gmm = GaussianMixture(
            n_components=n_levels,
            covariance_type=covariance_type,
            random_state=random_state,
        )
        gmm.fit(col)
        mu = gmm.means_.ravel()  # ty: ignore
        sd = np.sqrt(gmm.covariances_.ravel())  # ty: ignore
        order = np.argsort(mu)[::-1]  # high -> mid -> low
        means[c] = mu[order]
        sigmas[c] = sd[order]

    ug, ub, ur = means
    sigma_g, sigma_b, sigma_r = sigmas
    return ug, ub, ur, sigma_g, sigma_b, sigma_r


def viterbi_decode(trajectory, u, sigma2, transition_mat, pi):
    """Viterbi decoding of a single trajectory, returning the most likely state
    sequence.

    trajectory : (d, T) array of observed intensities.

    u : (d, Nstate) array of state means.

    sigma : (d, d, Nstate) array of state covariances.

    transition_mat : (Nstate, Nstate) transition matrix.

    pi : (Nstate,) initial state distribution.
    """

    state_num = u.shape[1]
    timesteps = trajectory.shape[1]
    prob_sequence = np.zeros(timesteps)
    state_sequence = np.zeros(timesteps)
    prob_matrix = np.zeros((timesteps, state_num))
    index_backtrack = np.zeros((timesteps, state_num))

    for i in range(timesteps):
        for j in range(state_num):
            ydata = trajectory[:, i]
            udata = u[:, j]
            sigma = sigma2[:, :, j]
            log_emit = (
                -np.log(2 * np.pi)
                - 0.5 * np.log(np.linalg.det(sigma))
                - 0.5 * (ydata - udata).T @ np.linalg.inv(sigma) @ (ydata - udata)
            )

            if i == 0:
                prob_matrix[i, j] = np.log(pi[j]) + log_emit
                index_backtrack[i, j] = j
            else:
                prob_max_tmp = -np.inf
                prob_max_index = 0
                for k in range(state_num):
                    prob_tmp = (
                        log_emit + prob_matrix[i - 1, k] + np.log(transition_mat[k, j])
                    )
                    if prob_tmp > prob_max_tmp:
                        prob_max_tmp = prob_tmp
                        prob_max_index = k

                prob_matrix[i, j] = prob_max_tmp
                index_backtrack[i, j] = prob_max_index

    prob_max = -np.inf
    for i in range(timesteps - 1, 0, -1):
        if i == timesteps - 1:
            prob_tmp = prob_matrix[i, :]
            prob_max = max(prob_tmp)
            state_sequence[i] = np.argmax(prob_tmp)
        else:
            state_sequence[i] = index_backtrack[i + 1, int(state_sequence[i + 1])]
            prob_tmp = prob_matrix[i, :]

        m = max(prob_tmp)
        log_norm = m + np.log(np.sum(np.exp(prob_tmp - m)))
        prob_sequence[i] = np.exp(prob_matrix[i, int(state_sequence[i])] - log_norm)

    return (
        state_sequence,
        prob_max,
        prob_sequence,
    )


def main():
    trajectory = build_trajectory(
        "/Users/jefferyzhou/Documents/johnson-lab/Jake_DNA_protein/GAFsmFRETdata/hel3_trace_9.dat"
    )
    stateindex = np.array(
        [
            [
                3,
                1,
                2,
                2,
                2,
                1,
            ],  ## each column is a state; numbers are 1-based level indices for intensity
            [3, 3, 1, 2, 3, 3],
            [3, 3, 3, 2, 1, 3],
        ]
    )
    # Gaussian-mixture starting condition: fit a 3-component GMM per channel to
    # this trajectory and use its sorted means/sigmas as the EM start point.

    # old guess

    # ug0 = np.array([410, 150, 0])
    # ub0 = np.array([120, 56, 5])
    # ur0 = np.array([250, 110, 3])

    # sigma_g = np.array([70, 60, 40])
    # sigma_b = np.array([25, 30, 18])
    # sigma_r = np.array([40, 55, 30])

    ug0, ub0, ur0, sigma_g, sigma_b, sigma_r = fit_gmm_levels(trajectory)
    print("GMM-fitted starting levels (high, mid, low):")
    print(f"  ug0 = {ug0}  sigma_g = {sigma_g}")
    print(f"  ub0 = {ub0}  sigma_b = {sigma_b}")
    print(f"  ur0 = {ur0}  sigma_r = {sigma_r}")

    sigma2 = np.zeros((3, 3, stateindex.shape[1]))

    for i in range(stateindex.shape[1]):
        sigma2[0, 0, i] = sigma_g[stateindex[0, i] - 1] ** 2
        sigma2[1, 1, i] = sigma_b[stateindex[1, i] - 1] ** 2
        sigma2[2, 2, i] = sigma_r[stateindex[2, i] - 1] ** 2

    transition_mat = np.array(
        [
            [0.9394, 0.0001, 0.0108, 0.0495, 0.0001, 0.0001],
            [0.0472, 0.5279, 0.4121, 0.0128, 0, 0],
            [0.0147, 0.0597, 0.5057, 0.1005, 0.3194, 0],
            [0.0001, 0.0555, 0.3652, 0.2535, 0.3252, 0.0005],
            [0.0001, 0, 0.2417, 0.1869, 0.5712, 0.0001],
            [0.0001, 0, 0, 0.0001, 0.0001, 0.9997],
        ]
    )

    u, sigma2, transition_mat, pi, likelihood, transition_history = (
        expectation_maximization_single_trace(
            stateindex, ug0, ub0, ur0, transition_mat, sigma2, trajectory
        )
    )

    np.savetxt(
        "transition_matrix_results.txt",
        transition_mat,
        fmt="%0.4f",
    )

    # all per-iteration transition matrices, shape (n_iters, Nstate, Nstate)
    transition_history = np.array(transition_history)
    np.save("transition_matrix_history.npy", transition_history)

    with open("transition_matrix_history.txt", "w") as f:
        for it, mat in enumerate(transition_history):
            f.write(f"# iteration {it + 1}\n")
            np.savetxt(f, mat, fmt="%0.4f")
            f.write("\n")

    viterbi_path = viterbi_decode(trajectory, u, sigma2, transition_mat, pi)
    print(f"Viterbi path: {viterbi_path}")


def main_multi():
    exp_condition = glob(
        "/Users/jefferyzhou/Documents/johnson-lab/Jake_DNA_protein/GAFsmFRETdata/expData_3colorFRET/expCondition_461/group1/*"
    )

    trajectories = []

    for trace in exp_condition:
        trajectories.append(build_trajectory(trace))

    # aggregate values across traces
    pooled_trajectories = np.hstack(trajectories)

    stateindex = np.array(
        [
            [
                3,
                1,
                2,
                2,
                2,
                1,
            ],  ## each column is a state; numbers are 1-based level indices for intensity
            [3, 3, 1, 2, 3, 3],
            [3, 3, 3, 2, 1, 3],
        ]
    )

    ug0, ub0, ur0, sigma_g, sigma_b, sigma_r = fit_gmm_levels(pooled_trajectories)

    sigma2 = np.zeros((3, 3, stateindex.shape[1]))

    for i in range(stateindex.shape[1]):
        sigma2[0, 0, i] = sigma_g[stateindex[0, i] - 1] ** 2
        sigma2[1, 1, i] = sigma_b[stateindex[1, i] - 1] ** 2
        sigma2[2, 2, i] = sigma_r[stateindex[2, i] - 1] ** 2

    transition_mat = np.array(
        [
            [0.9394, 0.0001, 0.0108, 0.0495, 0.0001, 0.0001],
            [0.0472, 0.5279, 0.4121, 0.0128, 0, 0],
            [0.0147, 0.0597, 0.5057, 0.1005, 0.3194, 0],
            [0.0001, 0.0555, 0.3652, 0.2535, 0.3252, 0.0005],
            [0.0001, 0, 0.2417, 0.1869, 0.5712, 0.0001],
            [0.0001, 0, 0, 0.0001, 0.0001, 0.9997],
        ]
    )

    u, sigma2, transition_mat, pi, likelihood, transition_history = (
        expectation_maximization_multi_trace(
            stateindex,
            ug0,
            ub0,
            ur0,
            transition_mat,
            sigma2,
            trajectories,
        )
    )

    np.savetxt(
        "transition_matrix_results_multi.txt",
        transition_mat,
        fmt="%0.4f",
    )

    # all per-iteration transition matrices, shape (n_iters, Nstate, Nstate)
    transition_history = np.array(transition_history)
    np.save("transition_matrix_history_multi.npy", transition_history)

    with open("transition_matrix_history_multi.txt", "w") as f:
        for it, mat in enumerate(transition_history):
            f.write(f"# iteration {it + 1}\n")
            np.savetxt(f, mat, fmt="%0.4f")
            f.write("\n")


if __name__ == "__main__":
    if sys.argv[1] == "single":
        main()
    elif sys.argv[1] == "multi":
        main_multi()
