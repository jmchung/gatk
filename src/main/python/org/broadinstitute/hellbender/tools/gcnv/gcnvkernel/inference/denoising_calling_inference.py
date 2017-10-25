import numpy as np
import pymc3 as pm
import logging
import time
import tqdm
from pymc3.variational.callbacks import Callback
from typing import List, Tuple, Callable, Optional
from abc import abstractmethod

from .. import config, types
from ..utils.rls import NonStationaryLinearRegression
from ..models.denoising_calling_model import DenoisingModel, DenoisingModelConfig,\
    LogCopyNumberEmissionPosteriorSampler, InitialModelParametersSupplier,\
    DenoisingCallingWorkspace, CopyNumberCallingConfig, HHMMClassAndCopyNumberCaller

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

_log_copy_number_emission_sampling_task_name = "log copy number emission sampling task"
_log_ploidy_emission_sampling_task_name = "log ploidy emission sampling task"


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
                 track_model_params: bool = True,
                 track_model_params_every: int = 10,
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
        self.convergence_snr_averaging_window = convergence_snr_averaging_window
        self.convergence_snr_trigger_threshold = convergence_snr_trigger_threshold
        self.convergence_snr_countdown_window = convergence_snr_countdown_window
        self.max_calling_iters = max_calling_iters
        self.caller_update_convergence_threshold = caller_update_convergence_threshold
        self.caller_admixing_rate = caller_admixing_rate
        self.caller_summary_statistics_reducer = caller_summary_statistics_reducer

        self._assert_params()

    def _assert_params(self):
        assert self.learning_rate >= 0
        assert self.obj_n_mc >= 0
        assert self.log_emission_samples_per_round >= 1
        assert self.log_emission_sampling_rounds >= 1
        assert 0.0 < self.log_emission_sampling_median_rel_error < 1.0
        # todo (rest)


class NoisyELBOConvergenceTracker(Callback):
    """ Convergence stopping criterion based on the linear trend of the noisy ELBO observations """
    MIN_WINDOW_SIZE = 10

    def __init__(self,
                 window: int = 50,
                 snr_stop_trigger_threshold: float = 0.1,
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
        self.eipi: float = None  # effective loss per iteration
        self.snr: float = None  # signal-to-noise ratio
        self.variance: float = None  # variance of elbo in the window
        self.drift: float = None  # absolute elbo change in the window

    def __call__(self, approx, loss, i):
        self._lin_reg.add_observation(loss)
        self._n_obs += 1
        self.eipi = self._lin_reg.get_slope()
        self.variance = self._lin_reg.get_variance()
        if self.eipi is not None and self.variance is not None:
            self.eipi *= -1
            self.drift = np.abs(self.eipi) * self.window
            self.snr = self.drift / np.sqrt(2 * self.variance)
            if self.snr < self.snr_stop_trigger_threshold:
                self._n_obs_snr_under_threshold += 1
            else:  # reset countdown
                self._n_obs_snr_under_threshold = 0
            if self._n_obs_snr_under_threshold == self.stop_countdown_window:
                raise StopIteration("Convergence criterion satisfied: SNR remained below {0} for "
                                    "{1} iterations.".format(self.snr_stop_trigger_threshold, self.stop_countdown_window))

    def _assert_params(self):
        assert self.window > self.MIN_WINDOW_SIZE, \
            "ELBO linear regression window size is too small (minimum is {0})".format(self.MIN_WINDOW_SIZE)
        assert self.snr_stop_trigger_threshold > 0, "bad SNR stop trigger threshold (must be positive)"
        assert self.stop_countdown_window >= 1, "bad SNR-under-threshold countdown window (must be >= 1)"


class ParamTracker:
    def __init__(self,
                 param_list: List[str],
                 trans_list: List[Callable[[np.ndarray], np.ndarray]],
                 trans_param_list: List[str]):
        """ Constructor.
        :param param_list: list of parameter names (must match pymc3 var names)
        :param trans_list: list of (inverse) transformations to apply on the parameters
        :param trans_param_list: list of transformed parameter names (arbitrary)
        """
        self.param_list = param_list
        self.trans_list = trans_list
        self.trans_param_list = trans_param_list
        self.tracked_param_values_dict = {}
        for key in trans_param_list:
            self.tracked_param_values_dict[key] = []

    def _extract_param_mean(self, approx: pm.approximations.MeanField):
        all_means = approx.mean.eval()
        out = dict()
        for param_name, trans, trans_param_name in zip(self.param_list, self.trans_list, self.trans_param_list):
            _, slc, _, dtype = approx._global_view[param_name]
            bare_param_mean = all_means[..., slc].astype(dtype)
            if trans is None:
                out[trans_param_name] = bare_param_mean
            else:
                out[trans_param_name] = trans(bare_param_mean)
        return out

    def record(self, approx, _loss, _i):
        out = self._extract_param_mean(approx)
        for key in self.trans_param_list:
            self.tracked_param_values_dict[key].append(out[key])

    __call__ = record

    def clear(self):
        for key in self.trans_param_list:
            self.tracked_param_values_dict[key] = []

    def __getitem__(self, key):
        return self.tracked_param_values_dict[key]


class InferenceTask:
    # Lay in a course for starbase one two warp nine point five...
    @abstractmethod
    def engage(self):
        raise NotImplementedError("Core breach imminent!")


class CohortLearnAndCall(InferenceTask):
    """
    todo
    """

    sampling_task_name = _log_copy_number_emission_sampling_task_name
    advi_task_name = "denoising"
    calling_task_name = "calling"

    def __init__(self,
                 denoising_model_config: DenoisingModelConfig,
                 calling_config: CopyNumberCallingConfig,
                 hybrid_inference_params: HybridInferenceParameters,
                 shared_workspace: DenoisingCallingWorkspace,
                 initial_param_supplier: InitialModelParametersSupplier):
        self.denoising_model_config = denoising_model_config
        self.calling_config = calling_config
        self.hybrid_inference_params = hybrid_inference_params
        self.shared_workspace = shared_workspace

        _logger.info("Instantiating the denoising model...")
        self.denoising_model = DenoisingModel(denoising_model_config, calling_config, shared_workspace,
                                              initial_param_supplier)

        _logger.info("Instantiating the copy number caller...")
        self.copy_number_caller = HHMMClassAndCopyNumberCaller(calling_config, hybrid_inference_params, shared_workspace)

        _logger.info("Instantiating the sampler...")
        self.log_emission_posterior_sampler = LogCopyNumberEmissionPosteriorSampler(
            denoising_model_config, calling_config, hybrid_inference_params, shared_workspace, self.denoising_model)
        self.log_emission_posterior_sampling_task = LogCopyNumberEmissionPosteriorSamplingTask(
            hybrid_inference_params, calling_config, shared_workspace, self.log_emission_posterior_sampler)

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
        with self.denoising_model:
            self.denoising_model_advi = pm.ADVI()
            self.denoising_model_opt = pm.adamax(learning_rate=self.hybrid_inference_params.learning_rate)
            self.denoising_model_step_func = self.denoising_model_advi.objective.step_function(
                score=True,
                obj_optimizer=self.denoising_model_opt,
                total_grad_norm_constraint=self.hybrid_inference_params.total_grad_norm_constraint,
                obj_n_mc=self.hybrid_inference_params.obj_n_mc)

        self.elbo_normalization_factor = shared_workspace.num_targets * shared_workspace.num_samples

        self.converged = False
        self._t0 = None
        self._t1 = None
        self.elbo_hist: List[float] = []
        self.snr_hist: List[float] = []

    def engage(self):
        try:
            for i_epoch in range(self.hybrid_inference_params.max_training_epochs):
                _logger.info("Starting epoch {0}...".format(i_epoch))
                self._update_denoising_parameters(i_epoch)
                self._update_log_copy_number_emission_posterior(i_epoch)
                self._update_copy_number_posterior(i_epoch)

                if self.converged:
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

    @staticmethod
    def _create_param_tracker():
        # todo perhaps we'd want to expose these?
        param_list = ['alpha_u_log__', 'psi_t_log__', 'log_mean_bias_t', 'depth_s_log__', 'W_tu']
        trans_list = [np.exp, np.exp, None, np.exp, None]
        trans_param_list = ['alpha_u', 'psi_t', 'log_mean_bias_t', 'depth_s', 'W_tu']
        return ParamTracker(param_list, trans_list, trans_param_list)

    def _update_denoising_parameters(self, i_epoch):
        self._log_start(self.advi_task_name, i_epoch)
        max_advi_iters = self.hybrid_inference_params.max_advi_iter_subsequent_epochs if i_epoch > 0 \
            else self.hybrid_inference_params.max_advi_iter_first_epoch
        with tqdm.trange(max_advi_iters, desc="(denoising) starting...") as progress_bar:
            try:
                for i in progress_bar:
                    loss = self.denoising_model_step_func() / self.elbo_normalization_factor
                    self.convergence_tracker(self.denoising_model_advi.approx, loss, i)
                    snr = self.convergence_tracker.snr
                    eipi = self.convergence_tracker.eipi
                    if snr is not None:
                        self.snr_hist.append(snr)
                    self.elbo_hist.append(-loss)
                    progress_bar.set_description("(denoising) ELBO: {0:2.6}, SNR: {1}, EIPI: {2}".format(
                        -loss,
                        "{0:2.2}".format(snr) if snr is not None else "N/A",
                        "{0:2.2}".format(eipi) if eipi is not None else "N/A"))
                    if self.param_tracker is not None \
                            and i % self.hybrid_inference_params.track_model_params_every == 0:
                        self.param_tracker(self.denoising_model_advi.approx, loss, i)

            except StopIteration as ex:
                progress_bar.close()
                _logger.info(ex)
                self.converged = True
                self._log_stop("denoising task", i_epoch)

            except KeyboardInterrupt:
                progress_bar.close()
                self._log_interrupt("denoising_task", i_epoch)
                raise KeyboardInterrupt

    def _update_log_copy_number_emission_posterior(self, i_round):
        self._log_start(self.sampling_task_name, i_round)
        try:
            self.log_emission_posterior_sampling_task.engage(self.denoising_model_advi.approx)
        except StopIteration:
            self._log_stop(self.sampling_task_name, i_round)
        except KeyboardInterrupt:
            self._log_interrupt(self.sampling_task_name, i_round)
            raise KeyboardInterrupt

    def _update_copy_number_posterior(self, i_epoch):
        self._log_start(self.calling_task_name, i_epoch)
        converged = False
        copy_number_update_summary = np.nan
        class_update_summary = np.nan
        with tqdm.trange(self.hybrid_inference_params.max_calling_iters,
                         desc="({0})".format(self.calling_task_name)) as progress_bar:
            try:
                for _ in progress_bar:
                    progress_bar.set_description("({0}) ...".format(self.calling_task_name))
                    (copy_number_update_s, copy_number_log_likelihoods_s,
                     class_update_summary, class_log_likelihood) = self.copy_number_caller.call(
                        copy_number_update_summary_statistic_reducer=
                            self.hybrid_inference_params.caller_summary_statistics_reducer,
                        class_update_summary_statistic_reducer=
                            self.hybrid_inference_params.caller_summary_statistics_reducer)
                    copy_number_update_summary = self.hybrid_inference_params\
                        .caller_summary_statistics_reducer(copy_number_update_s)
                    progress_bar.set_description("({0}) q_c update: {1:2.6}, q_tau update: "
                                                 "{2:2.6}".format(self.calling_task_name,
                                                                  copy_number_update_summary,
                                                                  class_update_summary))
                    if (copy_number_update_summary < self.hybrid_inference_params.caller_update_convergence_threshold
                        and class_update_summary < self.hybrid_inference_params.caller_update_convergence_threshold):
                        converged = True
                        raise StopIteration

            except StopIteration:
                progress_bar.set_description("({0}) [final] q_c update: {1:2.6}, q_tau update: "
                                             "{2:2.6}".format(self.calling_task_name,
                                                              copy_number_update_summary,
                                                              class_update_summary))
                progress_bar.refresh()
                progress_bar.close()
                self._log_stop(self.calling_task_name, i_epoch)

            except KeyboardInterrupt:
                progress_bar.close()
                self._log_interrupt(self.calling_task_name, i_epoch)
                raise KeyboardInterrupt

            finally:
                if not converged:
                    _logger.warning('Copy number calling did not converge. Increase maximum rounds ({0})'
                                    .format(self.hybrid_inference_params.max_calling_iters))


class ErrorControlledSamplingTask:
    """
    todo
    """
    def __init__(self,
                 task_name: str,
                 output_tensor: types.TensorSharedVariable,
                 output_shape: Tuple[int],
                 sampling_rounds: int,
                 target_median_rel_err: float):
        """
        todo
        :param task_name:
        :param output_tensor:
        :param output_shape:
        :param sampling_rounds:
        :param target_median_rel_err:
        """
        self.task_name = task_name
        self.output_tensor = output_tensor
        self.output_shape = output_shape
        self.sampling_rounds = sampling_rounds
        self.target_median_rel_err = target_median_rel_err

    @abstractmethod
    def _prepare(self, approx: pm.approximations.MeanField):
        raise NotImplementedError

    @abstractmethod
    def _draw(self) -> np.ndarray:
        raise NotImplementedError

    def engage(self, approx: pm.approximations.MeanField):
        self._prepare(approx)
        self.output_tensor.set_value(np.zeros(self.output_shape, dtype=types.floatX), borrow=config.borrow_numpy)
        converged = False
        median_rel_err = np.nan
        with tqdm.trange(self.sampling_rounds, desc="({0})".format(self.task_name)) as progress_bar:
            try:
                for i_round in progress_bar:
                    mean_new_samples = np.mean(self._draw(), axis=0)
                    output_update = (mean_new_samples - self.output_tensor.get_value(borrow=True)) / (i_round + 1)
                    self.output_tensor.set_value(self.output_tensor.get_value(borrow=True) + output_update,
                                                 borrow=config.borrow_numpy)
                    median_rel_err = np.median(
                        np.abs(output_update / self.output_tensor.get_value(borrow=True)).flatten())
                    progress_bar.set_description("({0}) median_rel_err: {1:2.6}".format(
                        self.task_name, median_rel_err))
                    if median_rel_err < self.target_median_rel_err:
                        converged = True
                        _logger.info('{0} converged after {1} rounds with final '
                                     'median relative error {2:.3}.'.format(self.task_name, i_round + 1,
                                                                            median_rel_err))
                        raise StopIteration

            except StopIteration:
                progress_bar.set_description("({0}) [final] median_rel_err: {1:2.6}".format(
                    self.task_name, median_rel_err))
                progress_bar.refresh()
                raise StopIteration

            except KeyboardInterrupt:
                progress_bar.close()
                raise KeyboardInterrupt

            finally:
                if not converged:
                    _logger.warning('{0} did not converge (median relative error '
                                    '= {1:.3}). Increase sampling rounds ({2}). Proceeding...'
                                    .format(self.task_name, median_rel_err, self.sampling_rounds))


class LogCopyNumberEmissionPosteriorSamplingTask(ErrorControlledSamplingTask):
    """
    todo
    """
    def __init__(self,
                 hybrid_inference_params: HybridInferenceParameters,
                 calling_config: CopyNumberCallingConfig,
                 shared_workspace: DenoisingCallingWorkspace,
                 log_copy_number_emission_sampler: LogCopyNumberEmissionPosteriorSampler):
        shape: Tuple[int] = (shared_workspace.num_samples, shared_workspace.num_targets,
                             calling_config.num_copy_number_states)
        super().__init__(_log_copy_number_emission_sampling_task_name,
                         shared_workspace.log_copy_number_emission_stc,
                         shape,
                         hybrid_inference_params.log_emission_sampling_rounds,
                         hybrid_inference_params.log_emission_sampling_median_rel_error)
        self.log_copy_number_emission_sampler = log_copy_number_emission_sampler

    def _prepare(self, approx):
        self.log_copy_number_emission_sampler.update_approximation(approx)

    def _draw(self):
        return self.log_copy_number_emission_sampler.draw()

