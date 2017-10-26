import numpy as np
import pymc3 as pm
import logging
from typing import List, Tuple, Callable, Optional

from .hybrid_inference_base import Sampler, Caller, CallerUpdateSummary, HybridInferenceTask, HybridInferenceParameters
from .. import config, types
from ..models.ploidy_model import PloidyModelConfig, PloidyModel, PloidyWorkspace, PloidyEmissionBasicSampler, PloidyBasicCaller

_logger = logging.getLogger(__name__)
_logger.setLevel(config.log_level)


class PloidyCaller(Caller):
    def __init__(self,
                 hybrid_inference_params: HybridInferenceParameters,
                 ploidy_workspace: PloidyWorkspace):
        self.hybrid_inference_params = hybrid_inference_params
        self.ploidy_basic_caller = PloidyBasicCaller(ploidy_workspace)

    def call(self) -> 'PloidyCallerUpdateSummary':
        update_norm_sj = self.ploidy_basic_caller.call()
        return PloidyCallerUpdateSummary(update_norm_sj,
                                         self.hybrid_inference_params.caller_summary_statistics_reducer)


class PloidyCallerUpdateSummary(CallerUpdateSummary):
    def __init__(self,
                 update_norm_sj: np.ndarray,
                 reducer: Callable[[np.ndarray], float]):
        self.scalar_update = reducer(update_norm_sj)

    def to_string(self):
        return "update size summary: {0:2.6}".format(self.scalar_update)

    def reduce_to_scalar(self) -> float:
        return self.scalar_update

    __repr__ = to_string
    __str__ = to_string


class PloidyEmissionSampler(Sampler):
    def __init__(self,
                 hybrid_inference_params: HybridInferenceParameters,
                 ploidy_model: PloidyModel,
                 ploidy_workspace: PloidyWorkspace):
        super().__init__(hybrid_inference_params)
        self.ploidy_workspace = ploidy_workspace
        self.ploidy_emission_basic_sampler = PloidyEmissionBasicSampler(
            ploidy_model, self.hybrid_inference_params.log_emission_samples_per_round)

    def update_approximation(self, approx: pm.approximations.MeanField):
        assert self.ploidy_emission_basic_sampler is not None, "model is not attached yet"
        self.ploidy_emission_basic_sampler.update_approximation(approx)

    def draw(self) -> np.ndarray:
        return self.ploidy_emission_basic_sampler.draw()

    def erase(self):
        self.ploidy_workspace.log_ploidy_emission_sjk.set_value(
            np.zeros((self.ploidy_workspace.num_samples,
                      self.ploidy_workspace.num_contigs,
                      self.ploidy_workspace.num_ploidy_states),
                     dtype=types.floatX), borrow=config.borrow_numpy)

    def increment(self, update):
        self.ploidy_workspace.log_ploidy_emission_sjk.set_value(
            self.ploidy_workspace.log_ploidy_emission_sjk.get_value(borrow=True) + update)

    def get_latest_log_emission_expectation_estimator(self) -> np.ndarray:
        return self.ploidy_workspace.log_ploidy_emission_sjk.get_value(borrow=True)


class PloidyInferenceTask(HybridInferenceTask):
    def __init__(self,
                 hybrid_inference_params: HybridInferenceParameters,
                 ploidy_config: PloidyModelConfig,
                 ploidy_workspace: PloidyWorkspace):
        ploidy_model = PloidyModel(ploidy_config, ploidy_workspace)
        ploidy_caller = PloidyCaller(hybrid_inference_params, ploidy_workspace)
        ploidy_emission_sampler = PloidyEmissionSampler(hybrid_inference_params, ploidy_model, ploidy_workspace)
        super().__init__(hybrid_inference_params, ploidy_model, ploidy_emission_sampler, ploidy_caller)
