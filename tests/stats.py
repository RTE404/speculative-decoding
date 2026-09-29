"""Chi-square goodness-of-fit with rare cells merged, shared by the distribution tests."""

import numpy as np
from scipy.stats import chisquare

MIN_EXPECTED = 5  # the usual validity condition for the chi-square approximation
P_VALUE_FLOOR = 1e-3  # tests use fixed seeds, so a pass or fail is reproducible


def chi_square_p(observed: np.ndarray, probs: np.ndarray) -> float:
    """p-value for `observed` counts drawn from `probs`, merging cells expected below MIN_EXPECTED."""
    observed = np.asarray(observed, dtype=float).ravel()
    probs = np.asarray(probs, dtype=float).ravel()
    probs = probs / probs.sum()
    expected = probs * observed.sum()
    if observed[expected == 0].sum() > 0:
        return 0.0  # a sample landed where the target distribution has no mass
    big = expected >= MIN_EXPECTED
    obs = np.append(observed[big], observed[~big].sum())
    exp = np.append(expected[big], expected[~big].sum())
    keep = exp > 0
    return float(chisquare(obs[keep], exp[keep]).pvalue)
