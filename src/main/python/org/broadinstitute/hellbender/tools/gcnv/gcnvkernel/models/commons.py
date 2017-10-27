import numpy as np
import logging
import theano.tensor as tt
import pymc3.distributions.dist_math as pm_dist_math
from .. import config

_logger = logging.getLogger(__name__)


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

    :param mu: poisson mean
    :param value: observed
    :return: theano tensor
    """
    return pm_dist_math.bound(
        pm_dist_math.logpow(mu, value) - pm_dist_math.factln(value) - mu,
        mu > 0, value >= 0)


def negative_binomial_logp(mu, alpha, value):
    """
    Negative binomial log probability

    :param mu: mean
    :param alpha: inverse over-dispersion
    :param value: observed
    :return: theano tensor
    """
    return pm_dist_math.bound(pm_dist_math.binomln(value + alpha - 1, value)
                              + pm_dist_math.logpow(mu / (mu + alpha), value)
                              + pm_dist_math.logpow(alpha / (mu + alpha), alpha),
                              mu > 0, value >= 0, alpha > 0)


# todo does this have a name?
def centered_heavy_tail_logp(mu, value):
    """
    This distribution is obtained by taking X ~ Exp and performing a Bose transformation
    Y = (exp(X) - 1)^{-1}. The result is:

        p(y) = (1 + 2 \mu) y^{2\mu} (1 + y)^{-2(1 + \mu)}

    It is a heavy-tail distribution with non-existent first moment.

    :param mu: mode of the distribution
    :param value: observed
    :return: theano tensor
    """
    return pm_dist_math.bound(tt.log(1.0 + 2.0 * mu) + 2.0 * mu * tt.log(value)
                              - 2.0 * (1.0 + mu) * tt.log(1.0 + value),
                              mu >= 0, value > 0)


def get_jensen_shannon_divergence(log_p_1, log_p_2):
    """
    todo
    :param log_p_1:
    :param log_p_2:
    :return:
    """
    p_1 = tt.exp(log_p_1)
    p_2 = tt.exp(log_p_2)
    return 0.5 * tt.sum((p_1 * (log_p_1 - log_p_2) + p_2 * (log_p_2 - log_p_1)), axis=-1)


def get_hellinger_distance(log_p_1, log_p_2):
    p_1 = tt.exp(log_p_1)
    p_2 = tt.exp(log_p_2)
    return tt.sqrt(tt.sum(tt.square(tt.sqrt(p_1) - tt.sqrt(p_2)), axis=-1)) / tt.sqrt(2)
