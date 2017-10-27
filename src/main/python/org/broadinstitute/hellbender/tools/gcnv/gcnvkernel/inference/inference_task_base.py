import numpy as np
import pymc3 as pm
import logging
import time
import tqdm
from pymc3 import Model
from typing import List, Callable, Optional
from abc import abstractmethod
from .covergence_tracker import NoisyELBOConvergenceTracker
from .param_tracker import ParamTrackerConfig, ParamTracker

_logger = logging.getLogger(__name__)


class Sampler:
    def __init__(self, hybrid_inference_params: 'HybridInferenceParameters'):
        self.hybrid_inference_params = hybrid_inference_params

    @abstractmethod
    def update_approximation(self, approx: pm.approximations.MeanField):
        raise NotImplementedError

    @abstractmethod
    def draw(self) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def erase(self):
        raise NotImplementedError

    @abstractmethod
    def increment(self, update):
        raise NotImplementedError

    @abstractmethod
    def get_latest_log_emission_expectation_estimator(self) -> np.ndarray:
        raise NotImplementedError


class Caller:
    @abstractmethod
    def call(self) -> 'CallerUpdateSummary':
        raise NotImplementedError


class CallerUpdateSummary:
    @abstractmethod
    def reduce_to_scalar(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def __repr__(self):  # the summary must be representable
        raise NotImplementedError


class InferenceTask:
    # Lay in a course for starbase one two warp nine point five...
    @abstractmethod
    def engage(self):
        raise NotImplementedError("Core breach imminent!")


class HybridInferenceTask(InferenceTask):
    """
    The "hybrid" inference framework is applicable to a PGM structured as follows:

        +--------------+           +----------------+
        | discrete RVs + --------► + continuous RVs |
        +--------------+           +----------------+

    Note that discrete RVs do not have any continuous parents. The inference is approximately
    performed by factorizing the true posterior into an uncorrelated product of discrete RVs (DRVs)
    and continuous RVs (CRVs):

        p(CRVs, DRVs | observed) ~ q(CRVs) q(DRVs)

    The user must supply the following components:

        (1) a pm.Model that yields the DRV-posterior-expectation of the log joint,
            E_{DRVs ~ q(DRVs)} [log_P(CRVs, DRVs, observed)]

        (2) a "sampler" that provides samples from the log emission, defined as:
            log_emission(DRVs) = E_{CRVs ~ q(CRVs)} [log_P (observed | CRVs, DRV)]

        (3) a "caller" that updates q(DRVs) given log_emission(DRV); it could be as simple as using
            the Bayes rule, or as complicated as doing iterative hierarchical HMM if correlations about
            DRVs are important.

    The general implementation motif is:

        (a) to store q(CRVs) as a shared theano tensor such that the the pymc3 model can access it,
        (b) to store log_emission(DRVs) as a shared theano tensor (or ndarray) such that the caller
            can access it, and:
        (c) let the caller directly update the shared q(CRVs).

    This class performs mean-field ADVI to obtain q(CRVs); q(DRV) is handled by the external
    "caller" and is out the scope of this class. This class, however, requires a CallerUpdateSummary
    from the Caller in order to check for convergence.
    """

    task_modes = ['advi', 'hybrid']

    def __init__(self,
                 hybrid_inference_params: 'HybridInferenceParameters',
                 continuous_model: Model,
                 sampler: Optional[Sampler],
                 caller: Optional[Caller],
                 **kwargs):
        self.hybrid_inference_params = hybrid_inference_params
        self.continuous_model = continuous_model
        self.sampler = sampler
        self.caller = caller

        if self.hybrid_inference_params.track_model_params:
            _logger.info("Instantiating the parameter tracker...")
            self.param_tracker = self._create_param_tracker()
        else:
            self.param_tracker = None

        _logger.info("Instantiating the convergence tracker...")
        self.convergence_tracker = NoisyELBOConvergenceTracker(
            self.hybrid_inference_params.convergence_snr_averaging_window,
            self.hybrid_inference_params.convergence_snr_trigger_threshold,
            self.hybrid_inference_params.convergence_snr_countdown_window)

        _logger.info("Setting up ADVI...")
        with self.continuous_model:
            self.continuous_model_advi = pm.ADVI()
            self.continuous_model_opt = pm.adamax(learning_rate=self.hybrid_inference_params.learning_rate)
            self.continuous_model_step_func = self.continuous_model_advi.objective.step_function(
                score=True,
                obj_optimizer=self.continuous_model_opt,
                total_grad_norm_constraint=self.hybrid_inference_params.total_grad_norm_constraint,
                obj_n_mc=self.hybrid_inference_params.obj_n_mc)

        if 'elbo_normalization_factor' in kwargs.keys():
            self.elbo_normalization_factor = kwargs['elbo_normalization_factor']
        else:
            self.elbo_normalization_factor = 1.0

        if 'advi_task_name' in kwargs.keys():
            self.advi_task_name = kwargs['advi_task_name']
        else:
            self.advi_task_name = "ADVI"

        if 'sampling_task_name' in kwargs.keys():
            self.sampling_task_name = kwargs['sampling_task_name']
        else:
            self.sampling_task_name = "sampling"

        if 'calling_task_name' in kwargs.keys():
            self.calling_task_name = kwargs['calling_task_name']
        else:
            self.calling_task_name = "calling_task_name"

        self._t0 = None
        self._t1 = None
        self.elbo_hist: List[float] = []
        self.snr_hist: List[float] = []

        if sampler is None or caller is None:
            _logger.warning("No discrete emission sampler and/or caller given -- running in plain continuous RV mode")
            self._engage = self._engage_advi
        else:
            self._engage = self._engage_hybrid

    def engage(self):
        self._engage()

    def _engage_advi(self):
        try:
            for i_epoch in range(self.hybrid_inference_params.max_training_epochs):
                _logger.info("Starting epoch {0}...".format(i_epoch))
                converged_continuous = self._update_continuous_posteriors(i_epoch)
                _logger.info("End of epoch {0}; converged: {1}".format(i_epoch, converged_continuous))
                if converged_continuous:
                    break

        except KeyboardInterrupt:
            pass

    def _engage_hybrid(self):
        try:
            for i_epoch in range(self.hybrid_inference_params.max_training_epochs):
                _logger.info("Starting epoch {0}...".format(i_epoch))
                converged_continuous = self._update_continuous_posteriors(i_epoch)
                converged_sampling = self._update_log_emission_posterior_expectation(i_epoch)
                converged_discrete = self._update_discrete_posteriors(i_epoch)
                _logger.info("End of epoch {0} -- converged continuous: {1},  converged sampling: {2}, "
                             "converged discrete: {3}".format(i_epoch, converged_continuous, converged_sampling,
                                                              converged_discrete))
                if converged_continuous and converged_sampling and converged_discrete:
                    break

        except KeyboardInterrupt:
            pass

    def _log_start(self, task_name: str, i_epoch: int):
        self._t0 = time.time()
        _logger.info("Starting {0} for epoch {1}...".format(task_name, i_epoch))

    def _log_stop(self, task_name: str, i_epoch: int):
        self._t1 = time.time()
        _logger.info('The {0} for epoch {1} successfully finished in {2:.2f}s'.format(
            task_name, i_epoch, self._t1 - self._t0))

    def _log_interrupt(self, task_name: str, i_epoch: int):
        _logger.warning('The {0} for epoch {1} was interrupted'.format(task_name, i_epoch))

    def _create_param_tracker(self):
        assert all([param_name in self.continuous_model.vars or
                    param_name in self.continuous_model.deterministics
                    for param_name in self.hybrid_inference_params.param_tracker_config.param_names]),\
            "Some of the parameters chosen to be tracker are not present in the model"
        return ParamTracker(self.hybrid_inference_params.param_tracker_config)

    def _update_continuous_posteriors(self, i_epoch) -> bool:
        self._log_start(self.advi_task_name, i_epoch)
        max_advi_iters = self.hybrid_inference_params.max_advi_iter_subsequent_epochs if i_epoch > 0 \
            else self.hybrid_inference_params.max_advi_iter_first_epoch
        converged = False
        with tqdm.trange(max_advi_iters, desc="({0}) starting...".format(self.advi_task_name)) as progress_bar:
            try:
                for i in progress_bar:
                    loss = self.continuous_model_step_func() / self.elbo_normalization_factor
                    self.convergence_tracker(self.continuous_model_advi.approx, loss, i)
                    snr = self.convergence_tracker.snr
                    egpi = self.convergence_tracker.egpi
                    if snr is not None:
                        self.snr_hist.append(snr)
                    self.elbo_hist.append(-loss)
                    progress_bar.set_description("({0}) ELBO: {1:2.6}, SNR: {2}, EGPI: {3}".format(
                        self.advi_task_name,
                        -loss,
                        "{0:2.2}".format(snr) if snr is not None else "N/A",
                        "{0:2.2}".format(egpi) if egpi is not None else "N/A"))
                    if self.param_tracker is not None \
                            and i % self.hybrid_inference_params.track_model_params_every == 0:
                        self.param_tracker(self.continuous_model_advi.approx, loss, i)

            except StopIteration as ex:
                if i_epoch > 0:
                    progress_bar.close()
                    _logger.info(ex)
                    converged = True
                    self._log_stop(self.advi_task_name, i_epoch)

            except KeyboardInterrupt:
                progress_bar.close()
                self._log_interrupt(self.advi_task_name, i_epoch)
                raise KeyboardInterrupt

        return converged

    def _update_log_emission_posterior_expectation(self, i_round):
        self._log_start(self.sampling_task_name, i_round)
        self.sampler.erase()
        self.sampler.update_approximation(self.continuous_model_advi.approx)
        converged = False
        median_rel_err = np.nan
        with tqdm.trange(self.hybrid_inference_params.log_emission_sampling_rounds,
                         desc="({0})".format(self.sampling_task_name)) as progress_bar:
            try:
                for i_round in progress_bar:
                    mean_new_samples = np.mean(self.sampler.draw(), axis=0)
                    latest_estimator = self.sampler.get_latest_log_emission_expectation_estimator()
                    update_to_estimator = (mean_new_samples - latest_estimator) / (i_round + 1)
                    self.sampler.increment(update_to_estimator)
                    median_rel_err = np.median(np.abs(update_to_estimator
                        / self.sampler.get_latest_log_emission_expectation_estimator()).flatten())
                    progress_bar.set_description("({0}) median_rel_err: {1:2.6}".format(
                        self.sampling_task_name, median_rel_err))
                    if median_rel_err < self.hybrid_inference_params.log_emission_sampling_median_rel_error:
                        _logger.info('{0} converged after {1} rounds with final '
                                     'median relative error {2:.3}.'.format(self.sampling_task_name, i_round + 1,
                                                                            median_rel_err))
                        raise StopIteration

            except StopIteration:
                progress_bar.set_description("({0}) [final] median_rel_err: {1:2.6}".format(
                    self.sampling_task_name, median_rel_err))
                progress_bar.refresh()
                converged = True

            except KeyboardInterrupt:
                progress_bar.close()
                raise KeyboardInterrupt

            finally:
                if not converged:
                    _logger.warning('{0} did not converge (median relative error '
                                    '= {1:.3}). Increase sampling rounds ({2}). Proceeding...'
                                    .format(self.sampling_task_name, median_rel_err,
                                            self.hybrid_inference_params.log_emission_sampling_rounds))

        return converged

    def _update_discrete_posteriors(self, i_epoch):
        self._log_start(self.calling_task_name, i_epoch)
        converged = False
        caller_summary = "N/A"
        with tqdm.trange(self.hybrid_inference_params.max_calling_iters,
                         desc="({0})".format(self.calling_task_name)) as progress_bar:
            try:
                for _ in progress_bar:
                    progress_bar.set_description("({0}) ...".format(self.calling_task_name))
                    caller_summary = self.caller.call()
                    progress_bar.set_description("({0}) {1}".format(self.calling_task_name, repr(caller_summary)))
                    caller_update_size_scalar = caller_summary.reduce_to_scalar()
                    if caller_update_size_scalar < self.hybrid_inference_params.caller_update_convergence_threshold:
                        converged = True
                        raise StopIteration

            except StopIteration:
                progress_bar.set_description("({0}) [final] {1}".format(self.calling_task_name, repr(caller_summary)))
                progress_bar.refresh()
                progress_bar.close()
                self._log_stop(self.calling_task_name, i_epoch)

            except KeyboardInterrupt:
                progress_bar.close()
                self._log_interrupt(self.calling_task_name, i_epoch)
                raise KeyboardInterrupt

            finally:
                if not converged:
                    _logger.warning('{0} did not converge. Increase maximum calling rounds ({1})'.format(
                        self.calling_task_name, self.hybrid_inference_params.max_calling_iters))

        return converged


class HybridInferenceParameters:
    """ Hybrid ADVI (for continuous RVs) + external calling (for discrete RVs) inference parameters """
    def __init__(self,
                 learning_rate: float = 0.2,
                 obj_n_mc: int = 1,
                 total_grad_norm_constraint: Optional[float] = None,
                 log_emission_samples_per_round: int = 50,
                 log_emission_sampling_median_rel_error: float = 5e-3,
                 log_emission_sampling_rounds: int = 10,
                 max_advi_iter_first_epoch: int = 500,
                 max_advi_iter_subsequent_epochs: int = 300,
                 max_training_epochs: int = 50,
                 track_model_params: bool = False,
                 track_model_params_every: int = 10,
                 param_tracker_config: Optional['ParamTrackerConfig'] = None,
                 convergence_snr_averaging_window: int = 100,
                 convergence_snr_trigger_threshold: float = 0.1,
                 convergence_snr_countdown_window: int = 10,
                 max_calling_iters: int = 10,
                 caller_update_convergence_threshold: float = 1e-6,
                 caller_admixing_rate: float = 0.75,
                 caller_summary_statistics_reducer: Callable[[np.ndarray], float] = np.mean):
        self.learning_rate = learning_rate
        self.obj_n_mc = obj_n_mc
        self.total_grad_norm_constraint = total_grad_norm_constraint
        self.log_emission_samples_per_round = log_emission_samples_per_round
        self.log_emission_sampling_median_rel_error = log_emission_sampling_median_rel_error
        self.log_emission_sampling_rounds = log_emission_sampling_rounds
        self.max_advi_iter_first_epoch = max_advi_iter_first_epoch
        self.max_advi_iter_subsequent_epochs = max_advi_iter_subsequent_epochs
        self.max_training_epochs = max_training_epochs
        self.track_model_params = track_model_params
        self.track_model_params_every = track_model_params_every
        self.param_tracker_config = param_tracker_config
        self.convergence_snr_averaging_window = convergence_snr_averaging_window
        self.convergence_snr_trigger_threshold = convergence_snr_trigger_threshold
        self.convergence_snr_countdown_window = convergence_snr_countdown_window
        self.max_calling_iters = max_calling_iters
        self.caller_update_convergence_threshold = caller_update_convergence_threshold
        self.caller_admixing_rate = caller_admixing_rate
        self.caller_summary_statistics_reducer = caller_summary_statistics_reducer

        self._assert_params()

    # todo (complete this)
    def _assert_params(self):
        assert self.learning_rate >= 0
        assert self.obj_n_mc >= 0
        assert self.log_emission_samples_per_round >= 1
        assert self.log_emission_sampling_rounds >= 1
        assert 0.0 < self.log_emission_sampling_median_rel_error < 1.0

        if self.track_model_params:
            assert self.param_tracker_config is not None
