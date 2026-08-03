from typing import Iterable
import sys

import numpy as np
from pathlib import Path
from glob import glob
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.backends.backend_pdf import PdfPages

# channels are stored in (green, blue, red) order, matching build_trajectory()
CHANNEL_COLORS = [(0.47, 0.67, 0.19), (0.07, 0.62, 1.0), (0.64, 0.08, 0.18)]
CHANNEL_NAMES = ["green", "blue", "red"]
# distinct color per HMM state, extended lazily if there are more than 6 states
STATE_COLORS = [
    "#4e79a7",
    "#f28e2b",
    "#59a14f",
    "#e15759",
    "#b07aa1",
    "#9c755f",
]


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
    u = stateindex * 0.0
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
    solve: str = "precision",
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

        traceLogLik = 0

        ## alphas (forward pass)
        obs_dist[:, 0] = gaussian_emission(u, sigma, trajectory[:, 0], d)
        for i in range(0, Nstate):
            alphas[i, 0] = pi[i] * obs_dist[i, 0]

        alphasum = sum(alphas[:, 0])
        traceLogLik += np.log(alphasum)  # c_0
        alphas[:, 0] = alphas[:, 0] / alphasum

        for t in range(0, T - 1):
            obs_dist[:, t + 1] = gaussian_emission(u, sigma, trajectory[:, t + 1], d)
            for j in range(0, Nstate):
                alphas[j, t + 1] = (
                    alphas[:, t].T @ transition_mat[:, j] * obs_dist[j, t + 1]
                )
            alphasum = sum(alphas[:, t + 1])
            traceLogLik += np.log(alphasum)  # c_{t+1}
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

        totalLogLik = totalLogLik + traceLogLik

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

        if solve == "precision":
            precision = np.zeros((sigma.shape[0], sigma.shape[1], sigma.shape[2]))
            for i in range(0, Nstate):
                precision[:, :, i] = np.linalg.inv(
                    sigma[:, :, i]
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
            b = np.concatenate([c1, c2, c3])
            results = np.linalg.lstsq(coeff.T, b.T)[0].T

        elif solve == "covariance":
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
                a11 = a11 + Mg[:, :, i].T @ sigma[:, :, i] @ Mg[:, :, i] * gammaSumT[i]
                a12 = a12 + Mg[:, :, i].T @ sigma[:, :, i] @ Mb[:, :, i] * gammaSumT[i]
                a13 = a13 + Mg[:, :, i].T @ sigma[:, :, i] @ Mr[:, :, i] * gammaSumT[i]
                a21 = a21 + Mb[:, :, i].T @ sigma[:, :, i] @ Mg[:, :, i] * gammaSumT[i]
                a22 = a22 + Mb[:, :, i].T @ sigma[:, :, i] @ Mb[:, :, i] * gammaSumT[i]
                a23 = a23 + Mb[:, :, i].T @ sigma[:, :, i] @ Mr[:, :, i] * gammaSumT[i]
                a31 = a31 + Mr[:, :, i].T @ sigma[:, :, i] @ Mg[:, :, i] * gammaSumT[i]
                a32 = a32 + Mr[:, :, i].T @ sigma[:, :, i] @ Mb[:, :, i] * gammaSumT[i]
                a33 = a33 + Mr[:, :, i].T @ sigma[:, :, i] @ Mr[:, :, i] * gammaSumT[i]

            c1 = np.zeros(Mg.shape[1])
            c2 = np.zeros(Mb.shape[1])
            c3 = np.zeros(Mr.shape[1])

            for i in range(0, Nstate):
                c1 = c1 + Mg[:, :, i].T @ sigma[:, :, i] @ gammaSumY[:, i]
                c2 = c2 + Mb[:, :, i].T @ sigma[:, :, i] @ gammaSumY[:, i]
                c3 = c3 + Mr[:, :, i].T @ sigma[:, :, i] @ gammaSumY[:, i]

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
    solve: str = "precision"
):

    # need to check that everything is matmul or elementwise
    Mg, Mb, Mr = build_select_matrices(stateindex)
    u = build_states(stateindex, ug, ub, ur)

    likelihood = []
    transition_history = []
    sigma0 = sigma
    criterion = 1e-3
    isConverged = False
    iter = 1000
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

            traceLogLik = 0

            ## alphas (forward pass)
            obs_dist[:, 0] = gaussian_emission(u, sigma, trace[:, 0], d)
            for i in range(0, Nstate):
                alphas[i, 0] = pi[i] * obs_dist[i, 0]

            alphasum = sum(alphas[:, 0])
            traceLogLik += np.log(alphasum)  # c_0
            alphas[:, 0] = alphas[:, 0] / alphasum

            for t in range(0, T - 1):
                obs_dist[:, t + 1] = gaussian_emission(u, sigma, trace[:, t + 1], d)
                for j in range(0, Nstate):
                    alphas[j, t + 1] = (
                        alphas[:, t].T @ transition_mat[:, j] * obs_dist[j, t + 1]
                    )
                alphasum = sum(alphas[:, t + 1])
                traceLogLik += np.log(alphasum)  # c_{t+1}
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

            """
            another correction: loglikelihood should be the sum of the scaling factors, 
            not sum(alpha @ beta)
            """
            totalLogLik = totalLogLik + traceLogLik

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
                print(f"EM algorithm converged in {1000 - iter} iterations.")
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

        precision = np.zeros((sigma.shape[0], sigma.shape[1], sigma.shape[2]))
        for i in range(0, Nstate):
            precision[:, :, i] = np.linalg.inv(
                sigma[:, :, i]
            )  # precision is the inverse of the covariance matrix

        if solve == "precision":
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


        if solve == "covariance":
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
                a11 = a11 + Mg[:, :, i].T @ sigma[:, :, i] @ Mg[:, :, i] * gammaSumT[i]
                a12 = a12 + Mg[:, :, i].T @ sigma[:, :, i] @ Mb[:, :, i] * gammaSumT[i]
                a13 = a13 + Mg[:, :, i].T @ sigma[:, :, i] @ Mr[:, :, i] * gammaSumT[i]
                a21 = a21 + Mb[:, :, i].T @ sigma[:, :, i] @ Mg[:, :, i] * gammaSumT[i]
                a22 = a22 + Mb[:, :, i].T @ sigma[:, :, i] @ Mb[:, :, i] * gammaSumT[i]
                a23 = a23 + Mb[:, :, i].T @ sigma[:, :, i] @ Mr[:, :, i] * gammaSumT[i]
                a31 = a31 + Mr[:, :, i].T @ sigma[:, :, i] @ Mg[:, :, i] * gammaSumT[i]
                a32 = a32 + Mr[:, :, i].T @ sigma[:, :, i] @ Mb[:, :, i] * gammaSumT[i]
                a33 = a33 + Mr[:, :, i].T @ sigma[:, :, i] @ Mr[:, :, i] * gammaSumT[i]

            c1 = np.zeros(Mg.shape[1])
            c2 = np.zeros(Mb.shape[1])
            c3 = np.zeros(Mr.shape[1])

            for i in range(0, Nstate):
                c1 = c1 + Mg[:, :, i].T @ sigma[:, :, i] @ gammaSumY[:, i]
                c2 = c2 + Mb[:, :, i].T @ sigma[:, :, i] @ gammaSumY[:, i]
                c3 = c3 + Mr[:, :, i].T @ sigma[:, :, i] @ gammaSumY[:, i]

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


def _build_u_float(stateindex, ug, ub, ur):
    """Float-safe build_states. build_states() does `u = stateindex * 0`, which
    is an INTEGER array, so assigning normalised [0, 1] means into it truncates
    them to 0. This builds a float means matrix instead."""
    Nstate = stateindex.shape[1]
    u = np.zeros((3, Nstate), dtype=float)
    for i in range(Nstate):
        u[0, i] = ug[stateindex[0, i] - 1]
        u[1, i] = ub[stateindex[1, i] - 1]
        u[2, i] = ur[stateindex[2, i] - 1]
    return u


def _levels_from_u(stateindex, u):
    """Invert build_states: read the per-color level vectors back out of the
    (d, Nstate) means matrix. For each channel, level L's value is u[c, i] for
    any state i whose stateindex[c, i] == L."""
    levels = []
    for c in range(stateindex.shape[0]):
        nlev = int(stateindex[c].max())
        vec = np.zeros(nlev)
        for L in range(1, nlev + 1):
            i = int(np.argmax(stateindex[c] == L))  # first state at this level
            vec[L - 1] = u[c, i]
        levels.append(vec)
    return levels  # [ug, ub, ur]


def expectation_maximization_multi_trace_meansolve(
    stateindex,
    ug: np.ndarray,
    ub: np.ndarray,
    ur: np.ndarray,
    transition_mat: np.ndarray,
    sigma: np.ndarray,
    trajectory: list | np.ndarray,
    mean_solve: str = "precision",
    update_transition: bool = True,
    update_sigma: bool = True,
    max_iter: int = 200,
    criterion: float = 1e-5,
):
    """Pooled multi-trace EM, identical to expectation_maximization_multi_trace
    except the constrained-means solve can be weighted two ways:

        mean_solve="precision" : weight by Sigma_i^{-1}  (proper EM update)
        mean_solve="variance"  : weight by Sigma_i       (MATLAB reference form)

    update_transition / update_sigma pin the transition matrix / covariances at
    their initial (here: ground-truth) values so the mean solve can be tested in
    isolation. Returns (u, ug, ub, ur, sigma, transition_mat, pi, likelihood).
    """
    if mean_solve not in ("precision", "variance"):
        raise ValueError("mean_solve must be 'precision' or 'variance'")

    Mg, Mb, Mr = build_select_matrices(stateindex)
    u = _build_u_float(stateindex, ug, ub, ur)

    likelihood = []
    sigma = sigma.copy()
    sigma0 = sigma.copy()
    transition_mat = transition_mat.copy()
    Nstate = u.shape[1]
    eps = np.spacing(1.0)
    pi = np.ones(Nstate) / Nstate
    isConverged = False

    for it in range(max_iter):
        d = trajectory[0].shape[0]
        PI_acc = np.zeros(Nstate)
        trans_num = np.zeros((Nstate, Nstate))
        trans_den = np.zeros(Nstate)
        gammaSumT = np.zeros(Nstate)
        gammaSumY = np.zeros((d, Nstate))
        cov_acc = np.zeros((d, d, Nstate))
        totalLogLik = 0.0
        u_old = u

        # ---------------- E step: one pass per trace ----------------
        for trace in trajectory:
            T = trace.shape[1]
            obs_dist = np.zeros((Nstate, T))
            alphas = np.zeros((Nstate, T))
            betas = np.zeros((Nstate, T))
            gamma = np.zeros((Nstate, T))
            xi = np.zeros((Nstate, Nstate, T - 1))

            obs_dist[:, 0] = gaussian_emission(u, sigma, trace[:, 0], d)
            alphas[:, 0] = pi * obs_dist[:, 0]
            alphasum = alphas[:, 0].sum()
            totalLogLik += np.log(alphasum)
            alphas[:, 0] /= alphasum

            for t in range(T - 1):
                obs_dist[:, t + 1] = gaussian_emission(u, sigma, trace[:, t + 1], d)
                for j in range(Nstate):
                    alphas[j, t + 1] = (
                        alphas[:, t] @ transition_mat[:, j] * obs_dist[j, t + 1]
                    )
                alphasum = alphas[:, t + 1].sum()
                totalLogLik += np.log(alphasum)
                alphas[:, t + 1] /= alphasum

            betas[:, T - 1] = 1.0 / Nstate
            for t in range(T - 2, -1, -1):
                for i in range(Nstate):
                    betas[i, t] = np.sum(
                        betas[:, t + 1] * obs_dist[:, t + 1] * transition_mat[i, :]
                    )
                betas[:, t] /= betas[:, t].sum()

            p_y = alphas[:, 0] @ betas[:, 0]

            gamma[:, T - 1] = alphas[:, T - 1]
            for t in range(T - 2, -1, -1):
                gamma[:, t] = alphas[:, t] * betas[:, t] / p_y
                gamma[:, t] /= gamma[:, t].sum()

            for t in range(T - 1):
                for i in range(Nstate):
                    xi[i, :, t] = (
                        alphas[i, t]
                        * obs_dist[:, t + 1]
                        * transition_mat[i, :]
                        * betas[:, t + 1]
                        / p_y
                    )
                xi[:, :, t] /= xi[:, :, t].sum()

            PI_acc += gamma[:, 0]
            trans_num += np.sum(xi, axis=2)
            trans_den += np.sum(gamma[:, : T - 1], axis=1)
            gammaSumT += np.sum(gamma, axis=1)
            gammaSumY += trace @ gamma.T
            for i in range(Nstate):
                diff = trace - u_old[:, i : i + 1]  # (d, T)
                cov_acc[:, :, i] += (gamma[i, :] * diff) @ diff.T

        # ---------------- convergence check ----------------
        likelihood.append(totalLogLik)
        if it >= 1 and abs(likelihood[-1] - likelihood[-2]) < criterion:
            isConverged = True
            break

        # ---------------- M step ----------------
        pi = PI_acc / PI_acc.sum()

        if update_transition:
            for i in range(Nstate):
                if trans_den[i] < eps:
                    continue
                transition_mat[i, :] = trans_num[i, :] / trans_den[i]
                transition_mat[i, :] /= transition_mat[i, :].sum()

        # per-state weighting matrix: precision (inverse) or covariance itself
        weight = np.zeros_like(sigma)
        for i in range(Nstate):
            weight[:, :, i] = (
                np.linalg.inv(sigma[:, :, i])
                if mean_solve == "precision"
                else sigma[:, :, i]
            )

        a11 = np.zeros((Mg.shape[1], Mg.shape[1]))
        a12 = np.zeros((Mg.shape[1], Mb.shape[1]))
        a13 = np.zeros((Mg.shape[1], Mr.shape[1]))
        a21 = np.zeros((Mb.shape[1], Mg.shape[1]))
        a22 = np.zeros((Mb.shape[1], Mb.shape[1]))
        a23 = np.zeros((Mb.shape[1], Mr.shape[1]))
        a31 = np.zeros((Mr.shape[1], Mg.shape[1]))
        a32 = np.zeros((Mr.shape[1], Mb.shape[1]))
        a33 = np.zeros((Mr.shape[1], Mr.shape[1]))
        for i in range(Nstate):
            W = weight[:, :, i]
            a11 += Mg[:, :, i].T @ W @ Mg[:, :, i] * gammaSumT[i]
            a12 += Mg[:, :, i].T @ W @ Mb[:, :, i] * gammaSumT[i]
            a13 += Mg[:, :, i].T @ W @ Mr[:, :, i] * gammaSumT[i]
            a21 += Mb[:, :, i].T @ W @ Mg[:, :, i] * gammaSumT[i]
            a22 += Mb[:, :, i].T @ W @ Mb[:, :, i] * gammaSumT[i]
            a23 += Mb[:, :, i].T @ W @ Mr[:, :, i] * gammaSumT[i]
            a31 += Mr[:, :, i].T @ W @ Mg[:, :, i] * gammaSumT[i]
            a32 += Mr[:, :, i].T @ W @ Mb[:, :, i] * gammaSumT[i]
            a33 += Mr[:, :, i].T @ W @ Mr[:, :, i] * gammaSumT[i]

        c1 = np.zeros(Mg.shape[1])
        c2 = np.zeros(Mb.shape[1])
        c3 = np.zeros(Mr.shape[1])
        for i in range(Nstate):
            W = weight[:, :, i]
            c1 += Mg[:, :, i].T @ W @ gammaSumY[:, i]
            c2 += Mb[:, :, i].T @ W @ gammaSumY[:, i]
            c3 += Mr[:, :, i].T @ W @ gammaSumY[:, i]

        coeff = np.block([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]])
        b = np.concatenate([c1, c2, c3])
        results = np.linalg.lstsq(coeff, b, rcond=None)[0]

        ug = results[: c1.shape[0]]
        ub = results[c1.shape[0] : c1.shape[0] + c2.shape[0]]
        ur = results[c1.shape[0] + c2.shape[0] :]
        u = _build_u_float(stateindex, ug, ub, ur)

        if update_sigma:
            for i in range(Nstate):
                if gammaSumT[i] < eps:
                    sigma[:, :, i] = sigma0[:, :, i]
                    continue
                sigmatmp = cov_acc[:, :, i] / gammaSumT[i]
                if (
                    np.isnan(sigmatmp).any()
                    or np.isinf(sigmatmp).any()
                    or np.sqrt(sigmatmp[0, 0]) < 0.03
                    or np.sqrt(sigmatmp[1, 1]) < 0.03
                    or np.sqrt(sigmatmp[2, 2]) < 0.03
                    or np.linalg.det(sigmatmp) < 1e-12
                ):
                    sigma[:, :, i] = sigma0[:, :, i]
                else:
                    sigma[:, :, i] = sigmatmp

    ug, ub, ur = _levels_from_u(stateindex, u)
    return u, ug, ub, ur, sigma, transition_mat, pi, np.asarray(likelihood)


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

            traceLogLik = 0

            ## alphas (forward pass)
            obs_dist[:, 0] = gaussian_emission(u, sigma, trace[:, 0], d)
            for i in range(0, Nstate):
                alphas[i, 0] = pi[i] * obs_dist[i, 0]

            alphasum = sum(alphas[:, 0])
            traceLogLik += np.log(alphasum)  # c_0
            alphas[:, 0] = alphas[:, 0] / alphasum

            for t in range(0, T - 1):
                obs_dist[:, t + 1] = gaussian_emission(u, sigma, trace[:, t + 1], d)
                for j in range(0, Nstate):
                    alphas[j, t + 1] = (
                        alphas[:, t].T @ transition_mat[:, j] * obs_dist[j, t + 1]
                    )
                alphasum = sum(alphas[:, t + 1])
                traceLogLik += np.log(alphasum)  # c_{t+1}
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

            totalLogLik = totalLogLik + traceLogLik

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

            precision = np.zeros((sigma.shape[0], sigma.shape[1], sigma.shape[2]))
            for i in range(0, Nstate):
                precision[:, :, i] = np.linalg.inv(
                    sigma[:, :, i]
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

            cov_acc[channel, channel, :] = (cov_acc[channel, channel, :] + psi) / (
                gammaSumT[:, :, i] + upsilon + d + 1
            )

        iter = iter - 1

    if not isConverged:
        print(f"EM algorithm did not converge within the maximum number of iterations.")

    return u, sigma, transition_mat, pi, likelihood, transition_history


def _theta_to_states(stateindex, theta, Lg, Lb, Lr):
    """Build the (3, Nstate) state-mean matrix from a level vector
    theta = [ug; ub; ur]. Float-safe (build_states truncates via int dtype)."""
    ug, ub, ur = theta[:Lg], theta[Lg : Lg + Lb], theta[Lg + Lb : Lg + Lb + Lr]
    Nstate = stateindex.shape[1]
    u = np.zeros((3, Nstate))
    for i in range(Nstate):
        u[0, i] = ug[stateindex[0, i] - 1]
        u[1, i] = ub[stateindex[1, i] - 1]
        u[2, i] = ur[stateindex[2, i] - 1]
    return u


# claude implementation of semi-pooled hierarchical bayes using naive EM
def expectation_maximization_multi_trace_hb(
    stateindex,
    ug: np.ndarray,
    ub: np.ndarray,
    ur: np.ndarray,
    transition_mat: np.ndarray,
    sigma: np.ndarray,
    trajectory: list,
    S_frac0: float = 0.10,
    var_floor: float = 0.05**2,
    max_iter: int = 1000,
    criterion: float = 1e-3,
    apply_v_corrections: bool = True,
):
    """Variational-EM hierarchical (empirical-Bayes) multi-trace HMM.

    Each trace n has its OWN level vector theta^(n) = [ug; ub; ur] in R^L, drawn
    from a shared population theta^(n) ~ N(m, S). Both the state paths z^(n) and
    the levels theta^(n) are LATENT; (m, S, A, pi, Sigma) are the parameters.
    The algorithm is coordinate ascent on the variational free energy under the
    mean field q_n(z, theta) = q_n(z) q_n(theta), which is exact given the
    factorisation because the model is conditionally Gaussian and linear in
    theta. See eb-hmm-derivation.md for the full derivation.

    q_n(theta) = N(theta^(n), V^(n)) comes out in closed form:

        V^(n)     = (coeff^(n) + S^-1)^-1
        theta^(n) = V^(n) (b^(n) + S^-1 m)

    Note this solve is an E-STEP -- it is the MEAN of a Gaussian posterior, and
    only coincides with the maximiser of a penalised likelihood because a
    Gaussian's mode equals its mean. That is why V^(n) is available at all, and
    V^(n) is what the (m, S) update needs.

    Three places require V^(n) beyond the theta solve. Dropping any of them
    silently reverts to "theta known exactly" and biases S downward, which
    compounds into over-shrinkage (see derivation section 5.1):

      1. the emissions used by forward-backward carry a factor
         exp(-1/2 tr(Sigma_i^-1 M_i V^(n) M_i^T));
      2. the Sigma_i update carries an extra N_i^(n) M_i V^(n) M_i^T;
      3. the monitored objective is the free energy
         F = sum_n [log Ztilde^(n) - KL(q_n(theta) || N(m, S))],
         NOT the conditional log-likelihood at a point.

    Shared across traces: transition matrix, pi, per-state covariance.

    apply_v_corrections=False drops all three V^(n) terms, recovering the
    plug-in / "theta as a parameter" variant. It is provided ONLY for ablation
    -- that variant is known to over-shrink and to drive S toward zero, so it
    should not be used for production fits. The free energy is still reported,
    but it is not the objective that variant ascends and need not be monotone.

    Returns m (population mean levels), S (population covariance), thetas
    (per-trace level vectors, shape (N, L)), sigma, transition_mat, pi, the
    per-iteration free energy, and the transition-matrix history.
    """
    Mg, Mb, Mr = build_select_matrices(stateindex)
    Lg, Lb, Lr = Mg.shape[1], Mb.shape[1], Mr.shape[1]
    L = Lg + Lb + Lr
    Nstate = stateindex.shape[1]
    N = len(trajectory)
    eps = np.spacing(1.0)

    # Stacked selection matrix per state, (d, L), so that u_i = M_all[i] @ theta.
    # Same content as the Mg/Mb/Mr block assembly, just in one piece -- the
    # emission correction and the Sigma correction both need M_i as a whole.
    M_all = [np.hstack([Mg[:, :, i], Mb[:, :, i], Mr[:, :, i]]) for i in range(Nstate)]

    # ---- population (hyper) parameters, in LEVEL space R^L ----
    m = np.concatenate(
        [np.asarray(ug, float), np.asarray(ub, float), np.asarray(ur, float)]
    )
    # Initial spread: a fraction of each level, floored relative to the overall
    # level scale so the initialisation is unit-agnostic. (A hard floor of 1.0
    # is negligible for raw intensities but enormous for [0, 1]-normalised
    # levels, where it would make the initial prior essentially flat.)
    scale = np.sqrt(np.mean(m**2)) if np.any(m) else 1.0
    S = np.diag(np.maximum((S_frac0 * np.abs(m)) ** 2, (S_frac0 * scale) ** 2))

    # per-trace level vectors; all start at the population mean
    thetas = np.tile(m, (N, 1))

    # Posterior covariances of q_n(theta). These MUST persist across iterations:
    # the emission correction at iteration `it` needs the V from the previous
    # theta solve, which is the correct coordinate-ascent ordering (q(z) is
    # updated given the current q(theta)). Initialised to the prior S, since
    # before a trace's data is used q_n(theta) is just p(theta | m, S).
    V = [S.copy() for _ in range(N)]

    sigma = sigma.copy()
    sigma0 = sigma.copy()
    pi = np.ones(Nstate) / Nstate

    free_energy = []
    transition_history = []
    isConverged = False

    print("Fitting hierarchical HMM with variational EM...")

    for it in range(max_iter):
        Sinv = np.linalg.inv(S)
        # precision recomputed from the CURRENT covariance each iteration
        precision = np.stack(
            [np.linalg.inv(sigma[:, :, i]) for i in range(Nstate)], axis=-1
        )

        # ---- pooled accumulators (shared params) ----
        PI_acc = np.zeros(Nstate)
        trans_num = np.zeros((Nstate, Nstate))
        trans_den = np.zeros(Nstate)
        cov_acc = np.zeros((3, 3, Nstate))
        cov_den = np.zeros(Nstate)
        # F = sum_n log Ztilde^(n) - sum_n KL(q_n(theta) || N(m, S)), both
        # evaluated at the q(theta) and (m, S) in force at the START of this
        # iteration, so the recorded sequence is a consistent point in the
        # coordinate-ascent cycle and is monotone non-decreasing.
        logZ_total = 0.0
        kl_total = 0.0

        _, logdetS = np.linalg.slogdet(S)

        # ===================== per-trace E step + theta solve =====================
        for n, trace in enumerate(trajectory):
            u = _theta_to_states(stateindex, thetas[n], Lg, Lb, Lr)
            T = trace.shape[1]

            # KL(N(theta^(n), V^(n)) || N(m, S)) for the free energy, using the
            # q(theta) that the forward pass below is about to condition on.
            dm = thetas[n] - m
            _, logdetV = np.linalg.slogdet(V[n])
            kl_total += 0.5 * (
                np.trace(Sinv @ V[n]) + dm @ Sinv @ dm - L + logdetS - logdetV
            )

            obs_dist = np.zeros((Nstate, T))
            alphas = np.zeros((Nstate, T))
            betas = np.zeros((Nstate, T))
            gamma = np.zeros((Nstate, T))
            xi = np.zeros((Nstate, Nstate, T - 1))

            # ---- effective emissions ----
            # E_q(theta)[log N(y_t | M_i theta, Sigma_i)]
            #   = log N(y_t | M_i theta^(n), Sigma_i) - 1/2 tr(Sigma_i^-1 M_i V M_i^T)
            # The correction has no t dependence, so it is one scalar per state.
            # It is state-DEPENDENT, so it survives the per-step renormalisation
            # of the forward pass and genuinely reweights the responsibilities:
            # states whose levels are poorly determined get down-weighted.
            log_corr = (
                np.array(
                    [
                        -0.5
                        * np.trace(precision[:, :, i] @ M_all[i] @ V[n] @ M_all[i].T)
                        for i in range(Nstate)
                    ]
                )
                if apply_v_corrections
                else np.zeros(Nstate)
            )
            # Factor out the max before exponentiating so a large V cannot
            # underflow every state to zero. Pulling out exp(shift) once per
            # frame scales Ztilde by exp(T * shift), added back below.
            shift = log_corr.max()
            emit_corr = np.exp(log_corr - shift)

            obs_dist[:, 0] = gaussian_emission(u, sigma, trace[:, 0], 3) * emit_corr
            alphas[:, 0] = pi * obs_dist[:, 0]
            asum = max(alphas[:, 0].sum(), eps)
            logZ_n = np.log(asum)
            alphas[:, 0] /= asum

            for t in range(T - 1):
                obs_dist[:, t + 1] = (
                    gaussian_emission(u, sigma, trace[:, t + 1], 3) * emit_corr
                )
                for j in range(Nstate):
                    alphas[j, t + 1] = (
                        alphas[:, t] @ transition_mat[:, j] * obs_dist[j, t + 1]
                    )
                asum = max(alphas[:, t + 1].sum(), eps)
                logZ_n += np.log(asum)  # log of the scaling constant c_{t+1}
                alphas[:, t + 1] /= asum

            # undo the max-shift factored out of every frame's emissions
            logZ_total += logZ_n + T * shift

            betas[:, T - 1] = 1.0 / Nstate
            for t in range(T - 2, -1, -1):
                for i in range(Nstate):
                    betas[i, t] = np.sum(
                        betas[:, t + 1] * obs_dist[:, t + 1] * transition_mat[i, :]
                    )
                betas[:, t] /= betas[:, t].sum()

            # gamma and xi are renormalised per timestep, so the old p_y divisor
            # was a no-op here; the log-likelihood it used to feed is now
            # logZ_n above, accumulated from the forward scaling constants.
            gamma[:, T - 1] = alphas[:, T - 1]
            for t in range(T - 2, -1, -1):
                gamma[:, t] = alphas[:, t] * betas[:, t]
                gamma[:, t] /= max(gamma[:, t].sum(), eps)

            for t in range(T - 1):
                for i in range(Nstate):
                    xi[i, :, t] = (
                        alphas[i, t]
                        * obs_dist[:, t + 1]
                        * transition_mat[i, :]
                        * betas[:, t + 1]
                    )
                xi[:, :, t] /= max(xi[:, :, t].sum(), eps)

            # ---- pooled stats (transition + covariance are shared) ----
            PI_acc += gamma[:, 0]
            trans_num += np.sum(xi, axis=2)
            trans_den += np.sum(gamma[:, : T - 1], axis=1)

            # ---- THIS trace's sufficient statistics (for its own theta) ----
            gammaSumT = np.sum(gamma, axis=1)  # (Nstate,)
            gammaSumY = trace @ gamma.T  # (3, Nstate)

            # coeff^(n) = sum_i N_i M_i^T Sigma_i^-1 M_i,  b^(n) = sum_i M_i^T Sigma_i^-1 r_i
            # Identical to the Mg/Mb/Mr block assembly -- the (r, c) block of
            # M_all[i].T @ P @ M_all[i] is exactly Mr_i.T @ P @ Mc_i.
            a = np.zeros((L, L))
            b = np.zeros(L)
            for i in range(Nstate):
                MtP = M_all[i].T @ precision[:, :, i]  # (L, 3)
                a += (MtP @ M_all[i]) * gammaSumT[i]
                b += MtP @ gammaSumY[:, i]

            # ridge toward the population, then solve for this trace's levels
            coeff_n = a + Sinv
            thetas[n] = np.linalg.solve(coeff_n, b + Sinv @ m)
            V[n] = np.linalg.inv(coeff_n)

            # Covariance accumulator, under the freshly-solved q_n(theta):
            #   E_q[(y - M_i theta)(y - M_i theta)^T]
            #     = (y - M_i thetabar)(y - M_i thetabar)^T + M_i V M_i^T
            # Dropping the second term biases Sigma low, which inflates
            # coeff^(n), which shrinks V, which shrinks S -- the same collapse
            # the V term in the S update exists to prevent.
            u_new = _theta_to_states(stateindex, thetas[n], Lg, Lb, Lr)
            for i in range(Nstate):
                diff = trace - u_new[:, i : i + 1]  # (3, T)
                cov_acc[:, :, i] += (gamma[i, :] * diff) @ diff.T
                if apply_v_corrections:
                    cov_acc[:, :, i] += gammaSumT[i] * (M_all[i] @ V[n] @ M_all[i].T)
            cov_den += gammaSumT

        # Free energy at a consistent point in the cycle: log Ztilde comes from
        # the forward pass just run, and the KL uses the same q(theta) and
        # (m, S) that pass conditioned on. This sequence must be monotone
        # non-decreasing; it is the objective both the theta solve and the
        # (m, S) update ascend. The old log(alpha . beta) was neither.
        F = logZ_total - kl_total
        free_energy.append(F)
        if it >= 1 and abs(free_energy[-1] - free_energy[-2]) < criterion:
            isConverged = True
            print(f"Converged in {it + 1} iterations (F = {F:.4f}).")
            break

        # ===================== M step: shared + hyperparameters =====================
        # shared dynamics (pooled, unchanged from your pooled model)
        pi = PI_acc / PI_acc.sum()
        for i in range(Nstate):
            if trans_den[i] < eps:
                continue
            transition_mat[i, :] = trans_num[i, :] / trans_den[i]
            transition_mat[i, :] /= transition_mat[i, :].sum()
        transition_history.append(transition_mat.copy())

        # population mean:  m = average of the per-trace level vectors
        m = thetas.mean(axis=0)

        # Population covariance: scatter of the thetas PLUS the posterior
        # covariances V^(n), since E_q[(theta-m)(theta-m)^T] = V + (thetabar-m)(...)^T.
        # Without V this is the scatter of already-shrunk point estimates, whose
        # only fixed point is S = 0 (derivation section 5.1).
        S = np.zeros((L, L))
        for n in range(N):
            dtheta = thetas[n] - m
            S += np.outer(dtheta, dtheta)
            if apply_v_corrections:
                S += V[n]
        S /= N
        S += 1e-6 * np.eye(L)  # keep strictly positive-definite

        # shared per-state covariance: pooled weighted MLE, floored
        for i in range(Nstate):
            if cov_den[i] < eps:
                sigma[:, :, i] = sigma0[:, :, i]
                continue
            sigmatmp = cov_acc[:, :, i] / cov_den[i]
            for c in range(3):
                sigmatmp[c, c] = max(sigmatmp[c, c], var_floor)
            sigma[:, :, i] = sigmatmp

    if not isConverged:
        print("EM did not converge within the maximum number of iterations.")

    return (
        m,
        S,
        thetas,
        sigma,
        transition_mat,
        pi,
        np.array(free_energy),
        transition_history,
    )


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

    # Forbidden transitions / unreachable start states have probability 0, and
    # log(0) = -inf both spams a divide-by-zero warning and can propagate into
    # nan in the normalization below. Take logs once, flooring at the smallest
    # positive double so a 0-probability move becomes a large finite penalty
    # (~ -708) that Viterbi will avoid unless nothing else is possible.
    tiny = np.finfo(float).tiny
    log_pi = np.log(np.clip(pi, tiny, None))
    log_trans = np.log(np.clip(transition_mat, tiny, None))

    # this would probably need to be different for a EB Gaussian HMM
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
                prob_matrix[i, j] = log_pi[j] + log_emit
                index_backtrack[i, j] = j
            else:
                prob_max_tmp = -np.inf
                prob_max_index = 0
                for k in range(state_num):
                    prob_tmp = log_emit + prob_matrix[i - 1, k] + log_trans[k, j]
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


def viterbi_decode_hb(
    trajectory: list,
    stateindex,
    thetas: np.ndarray,
    sigma: np.ndarray,
    transition_mat: np.ndarray,
    pi: np.ndarray,
):
    """Viterbi-decode every trace of a hierarchical-Bayes fit using that trace's
    OWN level vector, as returned by expectation_maximization_multi_trace_hb().

    The hierarchy already produced a per-trace posterior mean over the levels
    (thetas[n]); decoding uses it directly rather than drawing from the
    population prior N(m, S). The emission covariance, transition matrix and
    initial distribution are shared across traces, so they are passed straight
    through to viterbi_decode().

    trajectory : list of (d, T) arrays, same order as the fit's `trajectory`.

    stateindex : (d, Nstate) 1-based level index matrix used for the fit.

    thetas : (N, L) per-trace level vectors [ug; ub; ur] (the `thetas` return
        value of the HB fit).

    sigma : (d, d, Nstate) shared per-state covariance.

    transition_mat : (Nstate, Nstate) shared transition matrix.

    pi : (Nstate,) shared initial-state distribution.

    Returns a list of (state_sequence, prob_max, prob_sequence) tuples, one per
    trace, plus the list of per-trace mean matrices `u_n` (handy for plotting).
    """
    Mg, Mb, Mr = build_select_matrices(stateindex)
    Lg, Lb, Lr = Mg.shape[1], Mb.shape[1], Mr.shape[1]

    if len(thetas) != len(trajectory):
        raise ValueError(
            f"thetas has {len(thetas)} rows but there are {len(trajectory)} traces"
        )

    decodes = []
    means = []
    for n, trace in enumerate(trajectory):
        u_n = _theta_to_states(stateindex, thetas[n], Lg, Lb, Lr)
        means.append(u_n)
        decodes.append(viterbi_decode(trace, u_n, sigma, transition_mat, pi))

    return decodes, means


def plot_viterbi_trace(
    trajectory,
    u,
    state_sequence,
    dt: float = 0.05,
    title: str | None = None,
    ax=None,
    save_path: str | Path | None = None,
):
    """Plot a single trajectory with its Viterbi-decoded state fit.

    Shows the raw 3-color FRET signal and the fitted per-state means on a main
    axis, with a colored bar underneath marking the decoded state at each frame,
    mirroring the layout in Jake_DNA_protein/HMM/analyze.py.

    trajectory : (d, T) array of observed intensities, rows (green, blue, red).

    u : (d, Nstate) array of state means (as returned by the EM routines).

    state_sequence : (T,) decoded state index per frame (from viterbi_decode).

    dt : seconds between frames; sets the time axis.

    title : optional plot title.

    ax : optional existing axis; if given, the state bar is drawn as an inset
        below it. If None, a new figure with stacked axes is created.

    save_path : if given, the figure is saved there.
    """
    trajectory = np.asarray(trajectory)
    state_seq = np.asarray(state_sequence).astype(int)
    d, T = trajectory.shape
    Nstate = u.shape[1]

    time = np.arange(T) * dt
    # fitted mean signal: pick each frame's state mean -> (d, T)
    yfit = u[:, state_seq]

    if ax is None:
        fig = plt.figure(figsize=(11, 5.5))
        gs = fig.add_gridspec(2, 1, height_ratios=[6, 1], hspace=0.05)
        ax_top = fig.add_subplot(gs[0])
        ax_bar = fig.add_subplot(gs[1], sharex=ax_top)
    else:
        fig = ax.figure
        ax_top = ax
        divider = ax_top.inset_axes([0, -0.18, 1, 0.12])
        ax_bar = divider

    for i in range(d):
        ax_top.plot(
            time,
            trajectory[i],
            "-",
            color=CHANNEL_COLORS[i],
            linewidth=1.0,
            alpha=0.35,
            label=f"{CHANNEL_NAMES[i]} (raw)",
        )
        ax_top.plot(
            time,
            yfit[i],
            "-",
            color=CHANNEL_COLORS[i],
            linewidth=1.8,
            label=f"{CHANNEL_NAMES[i]} (fit)",
        )

    ax_top.set_ylabel("intensity (counts)")
    if title is not None:
        ax_top.set_title(title)
    ax_top.legend(loc="upper right", fontsize=7, ncol=d)
    ax_top.tick_params(labelbottom=False)
    ax_top.set_xlim(time[0], time[-1])

    # extend the palette if the model has more states than preset colors
    colors: list = list(STATE_COLORS)
    if Nstate > len(colors):
        extra = plt.get_cmap("tab20")(np.linspace(0, 1, Nstate - len(colors)))
        colors = colors + [tuple(c) for c in extra]
    cmap = ListedColormap(colors[:Nstate])
    norm = BoundaryNorm(np.arange(-0.5, Nstate + 0.5, 1.0), cmap.N)

    ax_bar.imshow(
        state_seq.reshape(1, -1),
        aspect="auto",
        cmap=cmap,
        norm=norm,
        extent=(time[0], time[-1], 0, 1),
        interpolation="nearest",
    )
    ax_bar.set_yticks([])
    ax_bar.set_xlabel("time (s)")
    ax_bar.set_ylabel("state", rotation=0, ha="right", va="center")

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Saved Viterbi trace plot to {save_path}")

    return fig, (ax_top, ax_bar)


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

    with open("sigma_results.txt", "w") as f:
        for i in range(3):
            np.savetxt(
                f,
                sigma2[:, :, i],
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

    state_sequence, prob_max, prob_sequence = viterbi_decode(
        trajectory, u, sigma2, transition_mat, pi
    )
    print(f"Viterbi state sequence: {state_sequence}")

    plot_viterbi_trace(
        trajectory,
        u,
        state_sequence,
        title="hel3_trace_9 — Viterbi decode",
        save_path="viterbi_trace_single.pdf",
    )


def main_old():
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
            stateindex,
            ug0,
            ub0,
            ur0,
            transition_mat,
            sigma2,
            trajectory,
            solve="covariance",
        )
    )

    np.savetxt(
        "sigma_transition_matrix_results.txt",
        transition_mat,
        fmt="%0.4f",
    )

    with open("sigma_results)old.txt", "w") as f:
        for i in range(3):
            np.savetxt(
                f,
                sigma2[:, :, i],
                fmt="%0.4f",
            )

    # all per-iteration transition matrices, shape (n_iters, Nstate, Nstate)
    transition_history = np.array(transition_history)
    np.save("sigma_transition_matrix_history.npy", transition_history)

    with open("sigma_transition_matrix_history.txt", "w") as f:
        for it, mat in enumerate(transition_history):
            f.write(f"# iteration {it + 1}\n")
            np.savetxt(f, mat, fmt="%0.4f")
            f.write("\n")

    state_sequence, prob_max, prob_sequence = viterbi_decode(
        trajectory, u, sigma2, transition_mat, pi
    )
    print(f"Viterbi state sequence: {state_sequence}")

    plot_viterbi_trace(
        trajectory,
        u,
        state_sequence,
        title="hel3_trace_9 — Viterbi decode",
        save_path="sigma_viterbi_trace_single.pdf",
    )


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
            solve="covariance"
        )
    )

    np.savetxt(
        "transition_matrix_results_multi_covariance.txt",
        transition_mat,
        fmt="%0.4f",
    )

    # all per-iteration transition matrices, shape (n_iters, Nstate, Nstate)
    transition_history = np.array(transition_history)
    np.save("transition_matrix_history_multi_covariance.npy", transition_history)

    with open("transition_matrix_history_multi_covariance.txt", "w") as f:
        for it, mat in enumerate(transition_history):
            f.write(f"# iteration {it + 1}\n")
            np.savetxt(f, mat, fmt="%0.4f")
            f.write("\n")


def main_multi_hb():
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

    m, S, thetas, sigma, transition_mat, pi, free_energy, transition_history = (
        expectation_maximization_multi_trace_hb(
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
        "transition_matrix_results_multi_hb.txt",
        transition_mat,
        fmt="%0.4f",
    )

    # all per-iteration transition matrices, shape (n_iters, Nstate, Nstate)
    transition_history = np.array(transition_history)
    np.save("transition_matrix_history_multi_hb.npy", transition_history)

    with open("transition_matrix_history_multi_hb.txt", "w") as f:
        for it, mat in enumerate(transition_history):
            f.write(f"# iteration {it + 1}\n")
            np.savetxt(f, mat, fmt="%0.4f")
            f.write("\n")

    # decode every trace with its own per-trace level vector (shared sigma,
    # transition_mat, pi), then write one page per trace to a single PDF.
    decodes, means = viterbi_decode_hb(
        trajectories, stateindex, thetas, sigma, transition_mat, pi
    )

    with PdfPages("viterbi_trace_multi_hb.pdf") as pdf:
        for n, trajectory in enumerate(trajectories):
            state_sequence, prob_max, prob_sequence = decodes[n]
            print(f"Trace {n} Viterbi state sequence: {state_sequence}")

            fig, _ = plot_viterbi_trace(
                trajectory,
                means[n],
                state_sequence,
                title=f"Viterbi decoded trace {n} (hierarchical Bayes)",
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"Saved {len(trajectories)} trace pages to viterbi_trace_multi_hb.pdf")


# TODO
def main_multi_normalized():
    """todo"""


if __name__ == "__main__":
    if sys.argv[1] == "single":
        main()
    elif sys.argv[1] == "multi":
        main_multi()
    elif sys.argv[1] == "single_old":
        main_old()
    elif sys.argv[1] == "multi_hb":
        main_multi_hb()
    elif sys.argv[1] == "multi_normalized":