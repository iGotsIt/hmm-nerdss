import pymc
import hmmlearn


class HierarchicalBayesHMM(hmmlearn.hmm.GaussianHMM):
    """
    Defines a hierarchical-Bayes HMM for analyzing multiple
    trajectories simultaneously.
    """

    def __init__(
        self,
        n_components=1,
        covariance_type="diag",
        n_iter=10,
        tol=1e-2,
        verbose=False,
        params="stmc",
        init_params="stmc",
    ):
        super().__init__(
            n_components=n_components,
            covariance_type=covariance_type,
            n_iter=n_iter,
            tol=tol,
            verbose=verbose,
            params=params,
            init_params=init_params,
        )


    def sample_gain()