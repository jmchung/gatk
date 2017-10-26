import numpy as np
import pymc3 as pm
import logging
from typing import Callable

from .inference_task_base import Sampler, Caller, CallerUpdateSummary, HybridInferenceTask, HybridInferenceParameters
from .. import config, types
from ..models.model_denoising_calling import DenoisingModel, DenoisingModelConfig,\
    CopyNumberEmissionBasicSampler, InitialModelParametersSupplier,\
    DenoisingCallingWorkspace, CopyNumberCallingConfig, HHMMClassAndCopyNumberBasicCaller

_logger = logging.getLogger(__name__)
_logger.setLevel(config.log_level)


class HHMMClassAndCopyNumberCaller(Caller):
    """ This is a wrapper around HHMMClassAndCopyNumberBasicCaller to be used in a HybridInferenceTask """
    def __init__(self,
                 calling_config: CopyNumberCallingConfig,
                 hybrid_inference_params: HybridInferenceParameters,
                 shared_workspace: DenoisingCallingWorkspace):
        self.hybrid_inference_params = hybrid_inference_params
        self.copy_number_basic_caller = HHMMClassAndCopyNumberBasicCaller(
            calling_config, hybrid_inference_params, shared_workspace)

    def call(self) -> 'HHMMClassAndCopyNumberCallerUpdateSummary':
        (copy_number_update_s, copy_number_log_likelihoods_s,
         class_update, class_log_likelihood) = self.copy_number_basic_caller.call(
            self.hybrid_inference_params.caller_summary_statistics_reducer,
            self.hybrid_inference_params.caller_summary_statistics_reducer)
        return HHMMClassAndCopyNumberCallerUpdateSummary(
            copy_number_update_s, copy_number_log_likelihoods_s,
            class_update, class_log_likelihood,
            self.hybrid_inference_params.caller_summary_statistics_reducer)


class HHMMClassAndCopyNumberCallerUpdateSummary(CallerUpdateSummary):
    def __init__(self,
                 copy_number_update_s: np.ndarray,
                 copy_number_log_likelihoods_s: np.ndarray,
                 class_update: float,
                 class_log_likelihood: float,
                 reducer: Callable[[np.ndarray], float]):
        self.copy_number_update_s = copy_number_update_s
        self.copy_number_log_likelihoods_s = copy_number_log_likelihoods_s
        self.class_update = class_update
        self.class_log_likelihood = class_log_likelihood
        self.copy_number_update_reduced = reducer(copy_number_update_s)

    def __repr__(self):
        return "class update size: {0:2.6}, CN update size: {1:2.6}".format(
            self.class_update, self.copy_number_update_reduced)

    def reduce_to_scalar(self) -> float:
        return max(self.class_update, self.copy_number_update_reduced)


class CopyNumberEmissionSampler(Sampler):
    """ This is a wrapper around CopyNumberEmissionBasicSampler to be used in a HybridInferenceTask """
    def __init__(self,
                 hybrid_inference_params: HybridInferenceParameters,
                 denoising_config: DenoisingModelConfig,
                 calling_config: CopyNumberCallingConfig,
                 shared_workspace: DenoisingCallingWorkspace,
                 denoising_model: DenoisingModel):
        super().__init__(hybrid_inference_params)
        self.shared_workspace = shared_workspace
        self.calling_config = calling_config
        self.copy_number_emission_basic_sampler = CopyNumberEmissionBasicSampler(
            denoising_config, calling_config, hybrid_inference_params, shared_workspace, denoising_model)

    def update_approximation(self, approx: pm.approximations.MeanField):
        self.copy_number_emission_basic_sampler.update_approximation(approx)

    def draw(self) -> np.ndarray:
        return self.copy_number_emission_basic_sampler.draw()

    def erase(self):
        self.shared_workspace.log_copy_number_emission_stc.set_value(
            np.zeros((self.shared_workspace.num_samples,
                      self.shared_workspace.num_targets,
                      self.calling_config.num_copy_number_states),
                     dtype=types.floatX), borrow=config.borrow_numpy)

    def increment(self, update):
        self.shared_workspace.log_copy_number_emission_stc.set_value(
            self.shared_workspace.log_copy_number_emission_stc.get_value(borrow=True) + update)

    def get_latest_log_emission_expectation_estimator(self) -> np.ndarray:
        return self.shared_workspace.log_copy_number_emission_stc.get_value(borrow=True)


class CohortDenoisingAndCallingTask(HybridInferenceTask):
    def __init__(self,
                 denoising_config: DenoisingModelConfig,
                 calling_config: CopyNumberCallingConfig,
                 hybrid_inference_params: HybridInferenceParameters,
                 shared_workspace: DenoisingCallingWorkspace,
                 initial_param_supplier: InitialModelParametersSupplier):
        _logger.info("Instantiating the denoising model...")
        denoising_model = DenoisingModel(
            denoising_config, calling_config, shared_workspace, initial_param_supplier)

        _logger.info("Instantiating the sampler...")
        copy_number_emission_sampler = CopyNumberEmissionSampler(
            hybrid_inference_params, denoising_config, calling_config, shared_workspace, denoising_model)

        _logger.info("Instantiating the copy number caller...")
        copy_number_caller = HHMMClassAndCopyNumberCaller(
            calling_config, hybrid_inference_params, shared_workspace)

        elbo_normalization_factor = shared_workspace.num_samples * shared_workspace.num_targets
        super().__init__(hybrid_inference_params, denoising_model, copy_number_emission_sampler, copy_number_caller,
                         elbo_normalization_factor=elbo_normalization_factor,
                         advi_task_name="denoising",
                         calling_task_name="calling CNVs")
