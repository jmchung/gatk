import numpy as np
import logging

import pymc3.distributions.dist_math as pm_dist_math

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


def get_normalized_prob_vector(prob_vector: np.ndarray, prob_sum_tol: float) -> np.ndarray:
    """
    todo
    :param prob_vector:
    :param prob_sum_tol:
    :return:
    """
    assert all(prob_vector >= 0), "probabilities must be non-negative"
    prob_sum = np.sum(prob_vector)
    if np.abs(prob_sum - 1.0) < prob_sum_tol:
        return prob_vector
    else:
        _logger.warning("The given probability vector ({0}) was not normalized to unity within the provided "
                        "tolerance ({1}); sum = {2}; normalizing and proceeding.".format(
            prob_vector, prob_sum_tol, prob_sum))
        return prob_vector / prob_sum


def poisson_logp(mu, value):
    """
    Poisson log probability

    Note:
        Removed all assertions on the input parameters for speed. Be careful!

    :param mu: poisson mean
    :param value: observed
    :return: theano tensor
    """
    return pm_dist_math.logpow(mu, value) - pm_dist_math.factln(value) - mu


def negative_binomial_logp(mu, alpha, value):
    """
    Negative binomial log probability

    Note:
        Removed all assertions on the input parameters for speed. Be careful!

    :param mu: mean
    :param alpha: inverse over-dispersion
    :param value: observed
    :return: theano tensor
    """
    return (pm_dist_math.binomln(value + alpha - 1, value)
            + pm_dist_math.logpow(mu / (mu + alpha), value)
            + pm_dist_math.logpow(alpha / (mu + alpha), alpha))
