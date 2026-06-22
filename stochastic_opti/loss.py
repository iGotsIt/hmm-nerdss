import numpy as np
import pandas as pd
import scipy 

def negative_log_likelihood(sequence, hmm):
    """
    Compute the negative log-likelihood of a sequence given an HMM
    using the forward algorithm:

    alpha(Xt) = alpha(Xt-1) * sum_(Xt-1) P(Xt | Xt-1) * P(yt | Xt)

    Parameters:
    sequence (array-like): The observed sequence.
    hmm (HMM): The hidden Markov model.

    Returns:
    float: The negative log-likelihood of the sequence given the HMM.
    """
    # Use the forward algorithm to compute the likelihood
    alpha = np.zeros((len(sequence), len(hmm.states)))
    
    # Initialization
    alpha[0, :] = hmm.initial_probabilities * hmm.emission_probabilities[:, sequence[0]]
    
    # Recursion
    for t in range(1, len(sequence)):
        for j in range(len(hmm.states)):
            alpha[t, j] = np.sum(alpha[t-1, :] * hmm.transition_probabilities[:, j]) * hmm.emission_probabilities[j, sequence[t]]
    
    # Termination
    likelihood = np.sum(alpha[-1, :])
    return -np.log(likelihood)



def mean_squared_error(sequence, hmm):

    # we need to decouple some of these experimental results from the experimental conditions
    # Diffusion constant for example is not a fundamental property -- relies on salt conditions and stuff
    # should we fit the diffusion constant as a parameter of the model?

    return