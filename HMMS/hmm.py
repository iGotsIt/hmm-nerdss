import sys
import glob
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.mixture import GaussianMixture

# Add the project root (one level above HMMS/) to sys.path so the
# Jake_DNA_protein namespace package is importable when this file is run
# directly as `python HMMS/hmm.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Jake_DNA_protein.HMM.analyze import analyze_model

UPPER_PERCENTILE = 0.95
LOWER_PERCENTILE = 0.05

def build_select_matrices(stateindex):
    """
    stateindex : (3, n_states) int array, rows = (green, blue, red), entries
                 are 1-based level indices (1=high, 2=mid, 3=low) per the
                 MATLAB convention. Levels actually used per color are
                 1..max(stateindex[c, :]) — typically 3.

    Returns Mg, Mb, Mr each of shape (n_features, n_levels, n_states),
    where n_features = 3 (green/blue/red).
    """
    d, n_states = stateindex.shape
    n_levels_g = stateindex[0].max()
    n_levels_b = stateindex[1].max()
    n_levels_r = stateindex[2].max()

    Mg = np.zeros((d, n_levels_g, n_states))
    Mb = np.zeros((d, n_levels_b, n_states))
    Mr = np.zeros((d, n_levels_r, n_states))
    for i in range(n_states):
        Mg[0, stateindex[0, i] - 1, i] = 1.0
        Mb[1, stateindex[1, i] - 1, i] = 1.0
        Mr[2, stateindex[2, i] - 1, i] = 1.0
    return Mg, Mb, Mr


def build_means_from_levels(stateindex, ug, ub, ur):
    """Reconstruct the (n_states, n_features) means matrix from level vectors."""
    n_states = stateindex.shape[1]
    means = np.zeros((n_states, 3))
    for i in range(n_states):
        means[i, 0] = ug[stateindex[0, i] - 1]
        means[i, 1] = ub[stateindex[1, i] - 1]
        means[i, 2] = ur[stateindex[2, i] - 1]
    return means


class ConstrainedGaussianHMM(GaussianHMM):
    """
    GaussianHMM with the shared-FRET-level constraint.

    Parameters
    ----------
    stateindex : (3, n_components) int array
        Per-color level index (1-based) for each state.
    All other params: forwarded to GaussianHMM.
    """

    def __init__(self, stateindex, **kwargs):
        kwargs.setdefault("n_components", stateindex.shape[1])
        super().__init__(**kwargs)
        self.stateindex = np.asarray(stateindex, dtype=int)
        self._Mg, self._Mb, self._Mr = build_select_matrices(self.stateindex)

    def _solve_constrained_means(self, stats):
        """
        Solve Eq. (9) from the supplement:

            [ A11 A12 A13 ] [ u^g ]   [ C1 ]
            [ A21 A22 A23 ] [ u^b ] = [ C2 ]
            [ A31 A32 A33 ] [ u^r ]   [ C3 ]

        where
            A_{αβ} = sum_i (M_i^α)^T  Σ_i  M_i^β  (sum_n γ_{ni})
            C_α   = sum_i (M_i^α)^T  Σ_i  (sum_n γ_{ni} x_n)

        Note: the MATLAB reference uses Σ_i directly in this solve, but the
        proper EM derivation uses the precision matrix Σ_i^{-1}. We use the
        precision form, which is what guarantees monotonic likelihood
        increase. (With near-equal diagonal Σ_i across states, the two forms
        give similar answers, which is why the MATLAB version converges in
        practice on its data.)

        stats['obs']  : (n_states, n_features) = sum_t γ_i(t) * x_t  per state
        stats['post'] : (n_states,)            = sum_t γ_i(t)        per state
        """
        Mg, Mb, Mr = self._Mg, self._Mb, self._Mr
        n_states = self.n_components
        gamaSumT = stats["post"]                  # (n_states,)
        gamaSumY = stats["obs"].T                 # (n_features, n_states)

        # self.covars_ is the property -- it always returns full (n_states, d, d)
        # regardless of covariance_type. Invert to get precision matrices.
        sigma = np.linalg.inv(self.covars_)

        kg = Mg.shape[1]
        kb = Mb.shape[1]
        kr = Mr.shape[1]

        A11 = np.zeros((kg, kg)); A12 = np.zeros((kg, kb)); A13 = np.zeros((kg, kr))
        A21 = np.zeros((kb, kg)); A22 = np.zeros((kb, kb)); A23 = np.zeros((kb, kr))
        A31 = np.zeros((kr, kg)); A32 = np.zeros((kr, kb)); A33 = np.zeros((kr, kr))
        C1 = np.zeros(kg); C2 = np.zeros(kb); C3 = np.zeros(kr)

        for i in range(n_states):
            S = sigma[i]                           # (d, d)
            w = gamaSumT[i]                        # scalar
            Mgi, Mbi, Mri = Mg[:, :, i], Mb[:, :, i], Mr[:, :, i]
            A11 += Mgi.T @ S @ Mgi * w
            A12 += Mgi.T @ S @ Mbi * w
            A13 += Mgi.T @ S @ Mri * w
            A21 += Mbi.T @ S @ Mgi * w
            A22 += Mbi.T @ S @ Mbi * w
            A23 += Mbi.T @ S @ Mri * w
            A31 += Mri.T @ S @ Mgi * w
            A32 += Mri.T @ S @ Mbi * w
            A33 += Mri.T @ S @ Mri * w
            yi = gamaSumY[:, i]                    # (d,)
            C1 += Mgi.T @ S @ yi
            C2 += Mbi.T @ S @ yi
            C3 += Mri.T @ S @ yi

        coeff = np.block([[A11, A12, A13],
                          [A21, A22, A23],
                          [A31, A32, A33]])
        rhs = np.concatenate([C1, C2, C3])
        # pinv handles rank-deficiency (state with ~0 occupancy)
        sol = np.linalg.pinv(coeff) @ rhs

        ug = sol[:kg]
        ub = sol[kg:kg + kb]
        ur = sol[kg + kb:kg + kb + kr]
        return build_means_from_levels(self.stateindex, ug, ub, ur), (ug, ub, ur)

    def _do_mstep(self, stats):
        # Update transmat_ / startprob_ only (skip GaussianHMM's mean+cov
        # update by jumping past it in the MRO -- goes to BaseHMM._do_mstep).
        super(GaussianHMM, self)._do_mstep(stats)

        # Constrained means using the *current* covariances (those from the
        # previous iteration's M-step), then update covariances against the
        # new constrained means. This ordering keeps mean/covar consistent
        # within a single iteration.
        if "m" in self.params:
            self.means_, (self._ug, self._ub, self._ur) = \
                self._solve_constrained_means(stats)

        if "c" in self.params:
            self._update_covars(stats)

    def _update_covars(self, stats):
        """Standard Gaussian-HMM covariance update against self.means_."""
        post = stats["post"][:, None]              # (n_states, 1)
        floor = 1e-5
        denom = np.maximum(post, floor)
        if self.covariance_type == "diag":
            cn = (stats["obs**2"]
                  - 2 * self.means_ * stats["obs"]
                  + self.means_ ** 2 * post)
            self.covars_ = np.maximum(cn / denom, floor)
        elif self.covariance_type == "full":
            new_covars = np.empty_like(self.covars_)
            for c in range(self.n_components):
                obs = stats["obs"][c]
                obs2 = stats["obs*obs.T"][c]
                mu = self.means_[c]
                cv = (obs2
                      - np.outer(obs, mu) - np.outer(mu, obs)
                      + np.outer(mu, mu) * post[c]) / denom[c]
                new_covars[c] = cv + floor * np.eye(cv.shape[0])
            self.covars_ = new_covars
        else:
            raise NotImplementedError(
                f"covariance_type={self.covariance_type!r} not supported "
                "by ConstrainedGaussianHMM in this example.")


# Ensure pickle always stores this class as constrained_gaussian_hmm.ConstrainedGaussianHMM
# rather than __main__.ConstrainedGaussianHMM when the script is run directly.
ConstrainedGaussianHMM.__module__ = "constrained_gaussian_hmm"


# ---------------------------------------------------------------------------
# Example / smoke test
# ---------------------------------------------------------------------------

def ingest_trajectory(file_path: str, normalize: bool):
    """
    Trying a different normalization approach; previously,
    we normalized to the 95th and 5th percentiles of each trajectory.
    Here we just normalize to the max and min of each trajectory;
    this still results in FRET values between 0 and 1 without clipping.
    """
    data = np.genfromtxt(
        file_path,
    )
    time = data[:, 0]
    green = data[:, 1]
    red = data[:, 2]
    blue = data[:, 3]

    ## do a standard normalization on each trajectory to get 0 - 1 values
    if normalize:
        green = (green - np.min(green)) / (np.max(green) - np.min(green))
        red = (red - np.min(red)) / (np.max(red) - np.min(red))
        blue = (blue - np.min(blue)) / (np.max(blue) - np.min(blue))

    return time, green, red, blue


def load_trace(path,
               low_percentile:  float = LOWER_PERCENTILE,
               high_percentile: float = UPPER_PERCENTILE):
    """Load a .dat file. Columns: time, green, red, blue.
    Reorder to (green, blue, red) to match the MATLAB convention used here.

    Each channel is normalised independently to [0, 1]:
      - the low_percentile value maps to 0  (robust dark-baseline anchor)
      - the high_percentile value maps to 1 (robust bright-state anchor)
    Values outside [floor, ceiling] are clipped to [0, 1].
    """
    data = np.loadtxt(path)
    time = data[:, 0] - data[0, 0]
    y = np.column_stack([data[:, 1], data[:, 3], data[:, 2]])  # green, blue, red

    # Per-channel two-sided normalisation. The percentile constants are
    # fractions (0.05 / 0.95); np.percentile expects 0-100, hence the *100.


    # floors   = np.percentile(y, low_percentile * 100,  axis=0)  # shape (3,)
    # ceilings = np.percentile(y, high_percentile * 100, axis=0)  # shape (3,)
    # y = (y - floors) / (ceilings - floors)
    # y = np.clip(y, 0.0, 1.0)

    return time, y


# Number of independent restarts.  Restart 0 always uses the hand-crafted
# starting point; restarts 1..N_RESTARTS-1 use randomly sampled starts.
N_RESTARTS = 25

# RNG seed for reproducible random starts.
MULTISTART_SEED = 42


def _build_start(stateindex, ug0, ub0, ur0, sigma_g, sigma_b, sigma_r):
    """Assemble means0 and covars0 from level vectors and per-level sigmas."""
    n_states = stateindex.shape[1]
    means0 = build_means_from_levels(stateindex, ug0, ub0, ur0)
    covars0 = np.zeros((n_states, 3))
    for i in range(n_states):
        covars0[i, 0] = sigma_g[stateindex[0, i] - 1] ** 2
        covars0[i, 1] = sigma_b[stateindex[1, i] - 1] ** 2
        covars0[i, 2] = sigma_r[stateindex[2, i] - 1] ** 2
    return means0, covars0


def fit_gmm_levels(X, n_levels=3, covariance_type="diag", random_state=0):
    """Fit a 3-component 1-D Gaussian mixture to each channel of X.

    X : (n_frames, 3) array, columns ordered (green, blue, red) to match
        load_trace() / the stateindex convention used throughout this file.

    Returns six length-`n_levels` vectors:
        ug, ub, ur          per-channel means, sorted high -> mid -> low
        sigma_g, sigma_b, sigma_r   matching standard deviations (sqrt of var)

    This mirrors the per-channel 3-component GMM explored in raw_fret.ipynb:
    fit each intensity distribution independently, then sort the components
    by mean (descending) so the ordering lines up with the 1-based level
    indices in stateindex (1=high, 2=mid, 3=low).
    """
    means = np.zeros((3, n_levels))
    sigmas = np.zeros((3, n_levels))
    for c in range(3):
        col = X[:, c].reshape(-1, 1)
        gmm = GaussianMixture(
            n_components=n_levels,
            covariance_type=covariance_type,
            random_state=random_state,
        )
        gmm.fit(col)
        mu = gmm.means_.ravel()
        sd = np.sqrt(gmm.covariances_.ravel())
        order = np.argsort(mu)[::-1]          # high -> mid -> low
        means[c] = mu[order]
        sigmas[c] = sd[order]

    ug, ub, ur = means
    sigma_g, sigma_b, sigma_r = sigmas
    return ug, ub, ur, sigma_g, sigma_b, sigma_r


def gmm_start(X, stateindex, **gmm_kwargs):
    """Per-condition starting point: fit the 3-component GMM on X, then build
    means0 / covars0 from the resulting level vectors."""
    ug, ub, ur, sigma_g, sigma_b, sigma_r = fit_gmm_levels(X, **gmm_kwargs)
    means0, covars0 = _build_start(
        stateindex, ug, ub, ur, sigma_g, sigma_b, sigma_r)
    return means0, covars0, (ug, ub, ur), (sigma_g, sigma_b, sigma_r)


def _sample_start(rng, stateindex):
    """Sample a random starting point.

    Level vectors: draw 3 values uniformly in (0, 1) per channel and sort
    them descending so high > mid > low.  Sigmas are drawn uniformly in
    [0.05, 0.25] — a reasonable range for normalised data.
    """

    ## sample start samples from a normalized distribution
    def ordered_levels():
        vals = rng.uniform(0.0, 1.0, size=3)
        vals.sort()
        return vals[::-1].copy()   # high, mid, low

    ug = ordered_levels()
    ub = ordered_levels()
    ur = ordered_levels()
    sigma_g = rng.uniform(0.05, 0.5, size=3)
    sigma_b = rng.uniform(0.05, 0.5, size=3)
    sigma_r = rng.uniform(0.05, 0.5, size=3)
    return _build_start(stateindex, ug, ub, ur, sigma_g, sigma_b, sigma_r)


def _sample_start_raw_counts(rng, stateindex):
    """Sample a random starting point on the raw-count scale, as in the original MATLAB code."""
    def ordered_levels():
        vals = rng.uniform(0.0, 1.0, size=3)
        vals.sort()
        return vals[::-1].copy()   # high, mid, low

    ug = ordered_levels() * 430
    ub = ordered_levels() * 120
    ur = ordered_levels() * 250
    sigma_g = rng.uniform(20, 80, size=3)
    sigma_b = rng.uniform(10, 40, size=3)
    sigma_r = rng.uniform(20, 60, size=3)
    return _build_start(stateindex, ug, ub, ur, sigma_g, sigma_b, sigma_r)

def expected_transition_counts(model, X, lengths=None):
    """Soft expected i->j transition counts (xi) from a *fitted* hmmlearn model."""
    captured = {}
    Base = type(model)
    class _Tap(Base):
        def _accumulate_sufficient_statistics(self, stats, *a, **k):
            super()._accumulate_sufficient_statistics(stats, *a, **k)
            captured["trans"] = stats["trans"].copy()
    tap = _Tap(n_components=model.n_components, stateindex=model.stateindex)
    tap.__dict__.update(model.__dict__)   # copy the fitted parameters
    tap.n_iter = 1
    tap.init_params = ""                   # don't reinitialize; use fitted params
    tap.fit(X, lengths)                    # one E-step fires the accumulator
    return captured["trans"]

def fit_once(X, lengths, stateindex, means0, covars0, restart_id=0):
    """Run EM from a single starting point.  Returns (model, train_log_likelihood)."""
    n_states = stateindex.shape[1]
    model = ConstrainedGaussianHMM(
        stateindex=stateindex,
        covariance_type="diag",
        n_iter=100, # originally 1000
        tol=1e-3, # originally 1e-4
        init_params="",   # we set all params manually below
        params="stmc",    # update startprob, transmat, means, covars
        verbose=False,
    )
    model.startprob_ = np.full(n_states, 1.0 / n_states)
    model.transmat_  = 0.9 * np.eye(n_states) + 0.1 / n_states
    model.means_     = means0
    model.covars_    = covars0

    model.fit(X, lengths=lengths)
    ll = model.score(X, lengths=lengths)
    print(f"  restart {restart_id:2d}: converged={model.monitor_.converged}  "
          f"train log-lik/frame = {ll / X.shape[0]:.4f}")
    return model, ll


def multistart_fit(X, lengths, stateindex, default_means0, default_covars0,
                   n_restarts=N_RESTARTS, seed=MULTISTART_SEED):
    """Run EM from n_restarts starting points; return the best model."""
    rng = np.random.default_rng(seed)
    best_model, best_ll = None, -np.inf

    for r in range(n_restarts):
        if r == 0:
            means0, covars0 = default_means0, default_covars0
        else:
            # means0, covars0 = _sample_start(rng, stateindex)
            means0, covars0 = _sample_start_raw_counts(rng, stateindex)

        try:
            model, ll = fit_once(X, lengths, stateindex, means0, covars0,
                                 restart_id=r)
        except Exception as exc:
            print(f"  restart {r:2d}: failed ({exc})")
            continue

        if ll > best_ll:
            best_ll, best_model = ll, model

    return best_model, best_ll


def run_condition(cond_dir: Path, condition_name: str, stateindex,
                  default_means0, default_covars0, out_dir: Path,
                  use_gmm_start: bool = False):
    files = sorted(glob.glob(str(cond_dir / "*.dat")))[:60]
    if not files:
        print(f"[{condition_name}] no .dat files found, skipping")
        return

    Xs, times, lengths = [], [], []
    for f in files:
        t, y = load_trace(f)
        Xs.append(y); times.append(t); lengths.append(y.shape[0])
    X = np.concatenate(Xs, axis=0)
    print(f"\n=== {condition_name} ===")
    print(f"Fitting on {len(files)} traces, {X.shape[0]} total frames, "
          f"{N_RESTARTS} restarts\n")

    # Pre-processing: derive this condition's restart-0 starting point from a
    # per-channel 3-component GMM fit to *its own* pooled intensities, so each
    # condition gets its own mu/sigma rather than a shared hand-crafted start.
    if use_gmm_start:
        default_means0, default_covars0, (ug0, ub0, ur0), (sg, sb, sr) = \
            gmm_start(X, stateindex)
        print("GMM-fitted starting levels (high, mid, low):")
        print(f"  ug0 = {np.array2string(ug0, precision=2)}  sigma_g = {np.array2string(sg, precision=2)}")
        print(f"  ub0 = {np.array2string(ub0, precision=2)}  sigma_b = {np.array2string(sb, precision=2)}")
        print(f"  ur0 = {np.array2string(ur0, precision=2)}  sigma_r = {np.array2string(sr, precision=2)}\n")

    model, best_ll = multistart_fit(
        X, lengths, stateindex, default_means0, default_covars0)

    print(f"\nBest train log-lik/frame: {best_ll / X.shape[0]:.4f}")
    print("Learned per-color level vectors:")
    print(f"  ug = {model._ug}")
    print(f"  ub = {model._ub}")
    print(f"  ur = {model._ur}")
    print("\nLearned means_ (n_states x 3):")
    print(np.array2string(model.means_, precision=2, suppress_small=True))
    print("\nLearned covars_ (n_states x 3, diagonal):")
    print(np.array2string(np.sqrt(model.covars_), precision=2, suppress_small=True))

    traces = [(Path(f).stem, times[k], Xs[k]) for k, f in enumerate(files)]
    analyze_model(model, traces, out_dir / f"{condition_name}_python.pdf",
                  condition_name=condition_name)

    pkl_path = out_dir / f"{condition_name}_python.pkl"
    with open(pkl_path, "wb") as fh:
        pickle.dump(model, fh)
    print(f"saved model to {pkl_path}")


def main():
    # 6 states, with the same stateindex used in maincode.m
    stateindex = np.array([
        [3, 1, 2, 2, 2, 1], ## each column is a state; numbers are 1-based level indices for intensity
        [3, 3, 1, 2, 3, 3],
        [3, 3, 3, 2, 1, 3],
    ])

    # Hand-crafted starting point (restart 0) on the normalised [0, 1] scale.
    # Original raw-count values were ~[430, 230, 0] / [120, 56, 5] / [250, 110, 3].
    ug0 = np.array([1.00, 0.53, 0.00])
    ub0 = np.array([1.00, 0.47, 0.04])
    ur0 = np.array([1.00, 0.44, 0.01])
    sigma_g = np.array([0.16, 0.14, 0.09])
    sigma_b = np.array([0.21, 0.25, 0.15])
    sigma_r = np.array([0.16, 0.22, 0.12])
    default_means0, default_covars0 = _build_start(
        stateindex, ug0, ub0, ur0, sigma_g, sigma_b, sigma_r)

    here = Path(__file__).resolve().parent          # HMM/
    exp_root = here.parent / "GAFsmFRETdata" / "expData_3colorFRET"
    out_dir = here / "analyze_output"
    out_dir.mkdir(exist_ok=True)

    conditions = [
        (exp_root / "expCondition_461" / "group1", "461_group1"),
        (exp_root / "expCondition_461" / "group2", "461_group2"),
        (exp_root / "expCondition_SHL7",            "SHL7"),
        (exp_root / "expCondition_lowMg2",           "lowMg2"),
    ]

    for cond_dir, condition_name in conditions:
        run_condition(cond_dir, condition_name, stateindex,
                      default_means0, default_covars0, out_dir)


def unnormalized_main():
    # 6 states, with the same stateindex used in maincode.m
    stateindex = np.array([
        [3, 1, 2, 2, 2, 1], ## each column is a state; numbers are 1-based level indices for intensity
        [3, 3, 1, 2, 3, 3],
        [3, 3, 3, 2, 1, 3],
    ])

    # use mu and sigma from a 3 component GMM fit to the raw data
    ug0 = np.array([347.43421222, 186.44653972, -0.58831247])
    ub0 = np.array([112.01630249, 73.27431567, 3.36572341])
    ur0 = np.array([261.20513266, 94.04038627, 1.03951787])

    sigma_g = np.array([123.78109316, 61.76524152, 24.86273684])
    sigma_b = np.array([62.25965747, 62.00245082, 14.219537  ])
    sigma_r = np.array([44.52976885, 30.47145886, 17.85639208])
    
    default_means0, default_covars0 = _build_start(
        stateindex, ug0, ub0, ur0, sigma_g, sigma_b, sigma_r)

    here = Path(__file__).resolve().parent          # HMM/
    exp_root = Path("/Users/jefferyzhou/Documents/johnson-lab/Jake_DNA_protein/GAFsmFRETdata/expData_3colorFRET")
    
    out_dir = here / "analyze_output"
    out_dir.mkdir(exist_ok=True)

    # need to build separate ones for each group

    conditions = [
        (exp_root / "expCondition_461" / "group1", "461_group1"),
        (exp_root / "expCondition_461" / "group2", "461_group2"),
        (exp_root / "expCondition_SHL7",            "SHL7"),
        (exp_root / "expCondition_lowMg2",           "lowMg2"),
    ]

    for cond_dir, condition_name in conditions:
        run_condition(cond_dir, condition_name, stateindex,
                      default_means0, default_covars0, out_dir)
        

def gmm_main():
    """Like unnormalized_main, but the restart-0 starting mu/sigma are fit
    per-condition with a 3-component GMM (one fit per channel) instead of
    being hardcoded.  See fit_gmm_levels / gmm_start."""
    stateindex = np.array([
        [3, 1, 2, 2, 2, 1], ## each column is a state; numbers are 1-based level indices for intensity
        [3, 3, 1, 2, 3, 3],
        [3, 3, 3, 2, 1, 3],
    ])

    here = Path(__file__).resolve().parent          # HMM/
    exp_root = Path("/Users/jefferyzhou/Documents/johnson-lab/Jake_DNA_protein/GAFsmFRETdata/expData_3colorFRET")

    out_dir = here / "analyze_output"
    out_dir.mkdir(exist_ok=True)

    conditions = [
        (exp_root / "expCondition_461" / "group1", "461_group1"),
        (exp_root / "expCondition_461" / "group2", "461_group2"),
        (exp_root / "expCondition_SHL7",            "SHL7"),
        (exp_root / "expCondition_lowMg2",           "lowMg2"),
    ]

    # default_means0/default_covars0 are unused when use_gmm_start=True (the
    # start is fit per-condition inside run_condition), so pass None.
    for cond_dir, condition_name in conditions:
        run_condition(cond_dir, condition_name, stateindex,
                      None, None, out_dir, use_gmm_start=True)


def single_trace():
    stateindex = np.array([
        [3, 1, 2, 2, 2, 1], ## each column is a state; numbers are 1-based level indices for intensity
        [3, 3, 1, 2, 3, 3],
        [3, 3, 3, 2, 1, 3],
    ])

    # Hand-crafted starting point (restart 0) on the normalised [0, 1] scale.
    # Original raw-count values were ~[430, 230, 0] / [120, 56, 5] / [250, 110, 3].

    # ug0 = np.array([1.00, 0.53, 0.00])
    # ub0 = np.array([1.00, 0.47, 0.04])
    # ur0 = np.array([1.00, 0.44, 0.01])
    # sigma_g = np.array([0.16, 0.14, 0.09])
    # sigma_b = np.array([0.21, 0.25, 0.15])
    # sigma_r = np.array([0.16, 0.22, 0.12])

    ug0 = np.array([410, 150, 0])
    ub0 = np.array([120, 56, 5])
    ur0 = np.array([250, 110, 3])

    sigma_g = np.array([70, 60, 40])
    sigma_b = np.array([25, 30, 18])
    sigma_r = np.array([40, 55, 30])
    
    default_means0, default_covars0 = _build_start(
        stateindex, ug0, ub0, ur0, sigma_g, sigma_b, sigma_r)

    path = "/Users/jefferyzhou/Documents/johnson-lab/Jake_DNA_protein/GAFsmFRETdata/hel3_trace_9.dat"

    t, y = load_trace(path)
    lengths = [y.shape[0]]   # single sequence of length T

    model, best_ll = multistart_fit(
        y, lengths, stateindex, default_means0, default_covars0)

    out_dir = Path("/Users/jefferyzhou/Documents/johnson-lab/runs")
    out_dir.mkdir(exist_ok=True)

    traces = [(Path(path).stem, t, y)]
    analyze_model(model, traces, out_dir / f"trace_9_python.pdf",
                  condition_name="trace_9")
    
    xi1 = expected_transition_counts(model, y, lengths)
    np.save(out_dir / "trace_9_counts.npy", xi1)

if __name__ == "__main__":
    if sys.argv[1] == "full":
        main()
    elif sys.argv[1] == "single_trace":
        single_trace()
    elif sys.argv[1] == "unnormalized":
        unnormalized_main()
    elif sys.argv[1] == "gmm":
        gmm_main()