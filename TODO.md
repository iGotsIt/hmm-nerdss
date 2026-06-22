6/11

- Hierarchical hidden markov model for estimating multiple traces at once
- maybe normalize each trace with the HB hmm so we can make 1 with a single transition matrix
- continue profiling NERDSS
    - not sure what the bottleneck is; most of the time is coming from check_biomolecular_reactions
    - seems like free is taking a long long time