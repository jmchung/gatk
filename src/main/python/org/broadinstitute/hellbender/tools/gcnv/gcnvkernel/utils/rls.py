import numpy as np
from collections import deque
from typing import Deque, Tuple


class NonStationaryLinearRegression:
    """ This class performs maximum-likelihood linear regression for sequentially observed data
    on an equally spaced grid. The data is assumed to be non-stationary. A window size needs
    to be provided that determined the forgetting factor. This is a non-recursive implementation of
    the recursive least squares (RLS) algorithm.
    """

    def __init__(self, window: int = 50):
        assert window >= 1, "Window size must be >= 1"
        self._n_obs: int = 0
        self._window: int = window
        self._obs_buffer: Deque[float] = deque([], window)
        self._lambda: float = np.exp(-np.log(2.0) / window)  # forgetting factor
        self._slope: float = None
        self._intercept: float = None
        self._variance: float = None
        self._weights, self._summed_weights, self._x, self._kern = None, None, None, None

    @staticmethod
    def _generate_auxiliary_vars(lam: float, n: int) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
        assert n >= 2, "At least two observations are required"
        weights = np.asarray([lam ** (n - i) for i in range(1, n + 1)])
        summed_weights: float = np.sum(weights)
        x = np.arange(1, n + 1)
        kern: np.ndarray = np.linalg.inv(np.asarray(
            [[np.sum(weights), np.sum(weights * x)],
             [np.sum(weights * x), np.sum(weights * x * x)]]))
        return weights, summed_weights, x, kern

    def add_observation(self, y: float):
        self._n_obs += 1
        self._obs_buffer.append(y)
        if self._n_obs < 2:
            return
        elif 2 <= self._n_obs <= self._window:
            self._weights, self._summed_weights, self._x, self._kern = self._generate_auxiliary_vars(
                self._lambda, self._n_obs)
        self._update_regression()

    def _update_regression(self):
        y = np.asarray(self._obs_buffer)
        wy = np.sum(self._weights * y)
        wxy = np.sum(self._weights * self._x * y)
        sol = np.dot(self._kern, np.asarray([wy, wxy]).reshape((2, 1)))
        self._intercept = sol[0, 0]
        self._slope = sol[1, 0]
        dev = y - self._intercept - self._slope * self._x
        self._variance = np.sum(self._weights * (dev ** 2)) / self._summed_weights

    def get_slope(self):
        return self._slope

    def get_intercept(self):
        return self._intercept

    def get_variance(self):
        return self._variance
