import numpy as np
from pymc3.variational.callbacks import Callback
from ..utils.rls import NonStationaryLinearRegression


class NoisyELBOConvergenceTracker(Callback):
    """ Convergence stopping criterion based on the linear trend of the noisy ELBO observations """
    MIN_WINDOW_SIZE = 10

    def __init__(self,
                 window: int = 100,
                 snr_stop_trigger_threshold: float = 0.5,
                 stop_countdown_window: int = 10):
        """ Constructor.
        :param window: window size for performing linear regression
        :param snr_stop_trigger_threshold: signal-to-noise ratio threshold for triggering countdown to stop
        :param stop_countdown_window: once the trigger is pulled, the snr must remain under
        the given threshold for at least stop_countdown_window subsequent ELBO observations to raise StopIteration;
        the countdown will be reset if at any point the snr goes about snr_stop_trigger_threshold
        """
        self.window = window
        self.snr_stop_trigger_threshold = snr_stop_trigger_threshold
        self.stop_countdown_window = stop_countdown_window
        self._assert_params()
        self._lin_reg = NonStationaryLinearRegression(window=self.window)
        self._n_obs: int = 0
        self._n_obs_snr_under_threshold: int = 0
        self.egpi: float = None  # effective gain per iteration
        self.snr: float = None  # signal-to-noise ratio
        self.variance: float = None  # variance of elbo in the window
        self.drift: float = None  # absolute elbo change in the window

    def __call__(self, approx, loss, i):
        self._lin_reg.add_observation(loss)
        self._n_obs += 1
        self.egpi = self._lin_reg.get_slope()
        self.variance = self._lin_reg.get_variance()
        if self.egpi is not None and self.variance is not None:
            self.egpi *= -1
            self.drift = np.abs(self.egpi) * self.window
            self.snr = self.drift / np.sqrt(2 * self.variance)
            if self.snr < self.snr_stop_trigger_threshold:
                self._n_obs_snr_under_threshold += 1
            else:  # reset countdown
                self._n_obs_snr_under_threshold = 0
            if self._n_obs_snr_under_threshold == self.stop_countdown_window:
                raise StopIteration("Convergence criterion satisfied: SNR remained below {0} for "
                                    "{1} iterations.".format(self.snr_stop_trigger_threshold,
                                                             self.stop_countdown_window))

    def _assert_params(self):
        assert self.window > self.MIN_WINDOW_SIZE, \
            "ELBO linear regression window size is too small (minimum is {0})".format(self.MIN_WINDOW_SIZE)
        assert self.snr_stop_trigger_threshold > 0, "bad SNR stop trigger threshold (must be positive)"
        assert self.stop_countdown_window >= 1, "bad SNR-under-threshold countdown window (must be >= 1)"
