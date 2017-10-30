import logging
from typing import List, Dict, Set

import numpy as np
import pymc3 as pm
import theano as th
import theano.tensor as tt
from pymc3 import Model, Normal, HalfFlat, Deterministic, DensityDist, Bound
from typing import Optional

from ..structs.interval import Interval
from ..structs.metadata import TargetsIntervalListMetadata, SampleCoverageMetadataCollection
from . import commons
from .. import config, types

_logger = logging.getLogger(__name__)


class PloidyModelConfig:
    """
    todo
    mention that good priors are extremely important
    """
    def __init__(self,
                 contig_prior_ploidy_map: Dict[str, np.ndarray],
                 mean_bias_mu: float = 1.0,
                 mean_bias_sd: float = 1e-2,
                 mapping_error_rate: float = 1e-2):
        self.mean_bias_mu = mean_bias_mu
        self.mean_bias_sd = mean_bias_sd
        self.mapping_error_rate = mapping_error_rate
        self.contig_prior_ploidy_map, self.num_ploidy_states = self._get_validated_contig_prior_ploidy_map(
            contig_prior_ploidy_map)
        self.contig_set = set(contig_prior_ploidy_map.keys())

    @staticmethod
    def _get_validated_contig_prior_ploidy_map(given_contig_prior_ploidy_map: Dict[str, np.ndarray],
                                               min_prob: float = 0):
        given_contigs = set(given_contig_prior_ploidy_map.keys())
        num_ploidy_states: int = 0
        for contig in given_contigs:
            num_ploidy_states = max(num_ploidy_states, given_contig_prior_ploidy_map[contig].size)
        validated_contig_prior_ploidy_map: Dict[str, np.ndarray] = dict()
        for contig in given_contigs:
            padded_validated_prior = np.zeros((num_ploidy_states,), dtype=types.floatX) + min_prob
            given_prior = given_contig_prior_ploidy_map[contig].flatten()
            padded_validated_prior[:given_prior.size] = padded_validated_prior[:given_prior.size] + given_prior
            padded_validated_prior = commons.get_normalized_prob_vector(padded_validated_prior, config.prob_sum_tol)
            validated_contig_prior_ploidy_map[contig] = padded_validated_prior
        return validated_contig_prior_ploidy_map, num_ploidy_states


class PloidyWorkspace:
    def __init__(self,
                 ploidy_config: PloidyModelConfig,
                 targets_metadata: TargetsIntervalListMetadata,
                 sample_metadata_collection: SampleCoverageMetadataCollection):
        self.targets_metadata = targets_metadata
        self.sample_metadata_collection = sample_metadata_collection
        self.ploidy_config = ploidy_config
        self.num_contigs = targets_metadata.num_contigs
        self.num_samples: int = sample_metadata_collection.num_samples
        self.num_ploidy_states = ploidy_config.num_ploidy_states
        assert all([contig in ploidy_config.contig_set for contig in targets_metadata.contig_set]), \
            "Some contigs do not have ploidy priors; cannot continue."

        # number of targets per contig as a shared theano tensor
        self.t_j: types.TensorSharedVariable = th.shared(targets_metadata.t_j,
                                                         name='t_j', borrow=config.borrow_numpy)

        # count per contig and total count as shared theano tensors
        n_sj = np.zeros((self.num_samples, self.num_contigs), dtype=types.big_uint)
        n_s = np.zeros((self.num_samples,), dtype=types.big_uint)
        for si in range(self.num_samples):
            sample_metadata = sample_metadata_collection.get_sample_coverage_metadata_by_index(si)
            n_sj[si, :] = sample_metadata.n_j[:]
            n_s[si] = sample_metadata.n_total
        self.n_sj: types.TensorSharedVariable = th.shared(n_sj, name='n_sj', borrow=config.borrow_numpy)
        self.n_s: types.TensorSharedVariable = th.shared(n_s, name='n_s', borrow=config.borrow_numpy)

        # integer ploidy values
        int_ploidy_values_k = np.arange(0, ploidy_config.num_ploidy_states, dtype=np.int)
        self.int_ploidy_values_k = th.shared(int_ploidy_values_k, name='int_ploidy_values_k',
                                             borrow=config.borrow_numpy)

        # ploidy priors
        p_ploidy_jk = np.zeros((self.num_contigs, self.ploidy_config.num_ploidy_states), dtype=types.floatX)
        for j, contig in enumerate(targets_metadata.contig_list):
            p_ploidy_jk[j, :] = ploidy_config.contig_prior_ploidy_map[contig][:]
        log_p_ploidy_jk = np.log(p_ploidy_jk)
        self.log_p_ploidy_jk: types.TensorSharedVariable = th.shared(log_p_ploidy_jk, name='log_p_ploidy_jk',
                                                                    borrow=config.borrow_numpy)

        # ploidy log posteriors (placeholder)
        #
        log_q_ploidy_sjk = np.tile(log_p_ploidy_jk, (self.num_samples, 1, 1))
        self.log_q_ploidy_sjk: types.TensorSharedVariable = th.shared(
            log_q_ploidy_sjk, name='log_q_ploidy_sjk', borrow=config.borrow_numpy)

        # ploidy log emission (placeholder)
        log_ploidy_emission_sjk = np.zeros(
            (self.num_samples, self.num_contigs, ploidy_config.num_ploidy_states), dtype=types.floatX)
        self.log_ploidy_emission_sjk: types.TensorSharedVariable = th.shared(
            log_ploidy_emission_sjk, name="log_ploidy_emission_sjk", borrow=config.borrow_numpy)

        # exclusion mask; mask(j, k) = 1 - delta(j, k)
        contig_exclusion_mask_jj = (np.ones((self.num_contigs, self.num_contigs), dtype=np.int)
                                    - np.eye(self.num_contigs, dtype=np.int))
        self.contig_exclusion_mask_jj = th.shared(contig_exclusion_mask_jj, name='contig_exclusion_mask_jj')

        # post-processed results; will be set by process_ploidy_posteriors once
        self.most_likely_ploidy_sj: Optional[np.ndarray] = None
        self.ploidy_genotyping_quality_sj: Optional[np.ndarray] = None

    @staticmethod
    def _get_contig_set_from_interval_list(targets_interval_list: List[Interval]) -> Set[str]:
        return {target.contig for target in targets_interval_list}

    # todo warn if ploidy genotyping quality is low
    # todo warn if ploidy genotyping is incompatible with a given list of sex genotypes
    def post_process(self, update_sample_metadata=True):
        self.most_likely_ploidy_sj = np.zeros((self.num_samples, self.num_contigs), dtype=types.small_uint)
        self.ploidy_genotyping_quality_sj = np.zeros((self.num_samples, self.num_contigs), dtype=types.floatX)
        log_q_ploidy_sjk = self.log_q_ploidy_sjk.get_value(borrow=True)
        for si in range(self.num_samples):
            for j in range(self.num_contigs):
                (self.most_likely_ploidy_sj[si, j],
                 self.ploidy_genotyping_quality_sj[si, j]) = commons.perform_genotyping(log_q_ploidy_sjk[si, j, :])

        if update_sample_metadata:
            for si in range(self.num_samples):
                self.sample_metadata_collection.get_sample_coverage_metadata_by_index(si).set_ploidy_and_read_depth(
                    self.most_likely_ploidy_sj[si, :])


class PloidyModel(Model):
    PositiveNormal = Bound(Normal, lower=0)  # how cool is this?

    def __init__(self,
                 ploidy_config: PloidyModelConfig,
                 ploidy_workspace: PloidyWorkspace):
        super().__init__()

        # shorthands
        t_j = ploidy_workspace.t_j
        contig_exclusion_mask_jj = ploidy_workspace.contig_exclusion_mask_jj
        n_s = ploidy_workspace.n_s
        n_sj = ploidy_workspace.n_sj
        ploidy_k = ploidy_workspace.int_ploidy_values_k
        q_ploidy_sjk = tt.exp(ploidy_workspace.log_q_ploidy_sjk)
        eps = ploidy_config.mapping_error_rate

        # mean per-contig bias
        mean_bias_j = self.PositiveNormal('mean_bias_j',
                                          mu=ploidy_config.mean_bias_mu,
                                          sd=ploidy_config.mean_bias_sd,
                                          shape=ploidy_workspace.num_contigs)

        # todo informative prior?
        # per-contig NB over-dispersion parameter
        alpha_j = HalfFlat('alpha_j', shape=ploidy_workspace.num_contigs)

        # mean ploidy per contig per sample
        mean_ploidy_sj = tt.sum(tt.exp(ploidy_workspace.log_q_ploidy_sjk)
                               * ploidy_workspace.int_ploidy_values_k.dimshuffle('x', 'x', 0), axis=2)

        # mean-field amplification coefficient per contig
        gamma_sj = mean_ploidy_sj * t_j.dimshuffle('x', 0) * mean_bias_j.dimshuffle('x', 0)

        # gamma_rest_sj \equiv sum_{j' \neq j} gamma_sj
        gamma_rest_sj = tt.dot(gamma_sj, contig_exclusion_mask_jj)

        # NB per-contig counts
        mu_num_sjk = (t_j.dimshuffle('x', 0, 'x') * mean_bias_j.dimshuffle('x', 0, 'x')
                      * ploidy_k.dimshuffle('x', 'x', 0))
        mu_den_sjk = gamma_rest_sj.dimshuffle(0, 1, 'x') + mu_num_sjk
        eps_j = eps * t_j / tt.sum(t_j)  # proportion of fragments erroneously mapped to contig j
        mu_sjk = ((1.0 - eps) * (mu_num_sjk / mu_den_sjk)
                  + eps_j.dimshuffle('x', 0, 'x')) * n_s.dimshuffle(0, 'x', 'x')
        alpha_adj_sjk = tt.sqrt(1 + tt.inv(alpha_j)).dimshuffle('x', 0, 'x')

        def _get_logp_sjk(_n_sj):
            _logp_sjk = commons.negative_binomial_logp(alpha_adj_sjk * mu_sjk,  # mean
                                                       alpha_j.dimshuffle('x', 0, 'x'),  # over-dispersion
                                                       _n_sj.dimshuffle(0, 1, 'x'))  # contig counts
            return _logp_sjk

        DensityDist(name='n_sj_obs',
                    logp=lambda _n_sj: tt.sum(q_ploidy_sjk * _get_logp_sjk(_n_sj)),
                    observed=n_sj)

        # for log ploidy emission sampling
        Deterministic(name='logp_sjk', var=_get_logp_sjk(n_sj))


class PloidyEmissionBasicSampler:
    """ Draws posterior samples from the ploidy log emission probability for a given variational approximation to
    the ploidy determination model posterior """
    def __init__(self, ploidy_model: PloidyModel, samples_per_round: int):
        self.ploidy_model = ploidy_model
        self.samples_per_round = samples_per_round
        self._simultaneous_log_ploidy_emission_sampler = None

    def update_approximation(self, approx: pm.approximations.MeanField):
        self._simultaneous_log_ploidy_emission_sampler = \
            self._get_compiled_simultaneous_log_ploidy_emission_sampler(approx)

    def is_sampler_initialized(self):
        return self._simultaneous_log_ploidy_emission_sampler is not None

    def draw(self):
        return self._simultaneous_log_ploidy_emission_sampler()

    def _get_compiled_simultaneous_log_ploidy_emission_sampler(self, approx: pm.approximations.MeanField):
        """ For a given variational approximation, returns a compiled theano function that draws posterior samples
        from the log ploidy emission """
        log_ploidy_emission_sjk_sampler = approx.sample_node(self.ploidy_model['logp_sjk'], size=self.samples_per_round)
        return th.function(inputs=[], outputs=log_ploidy_emission_sjk_sampler)


class PloidyBasicCaller:
    """ Simple Bayesian update of contig ploidy log posteriors """
    def __init__(self,
                 ploidy_workspace: PloidyWorkspace):
        self.ploidy_workspace = ploidy_workspace
        self._update_log_q_ploidy_sjk_theano_func = self._get_update_log_q_ploidy_sjk_theano_func()

    @th.configparser.change_flags(compute_test_value="ignore")
    def _get_update_log_q_ploidy_sjk_theano_func(self):
        new_log_q_ploidy_sjk = (self.ploidy_workspace.log_p_ploidy_jk.dimshuffle('x', 0, 1)
                               + self.ploidy_workspace.log_ploidy_emission_sjk)
        new_log_q_ploidy_sjk -= pm.logsumexp(new_log_q_ploidy_sjk, axis=2)
        update_norm_sj = commons.get_hellinger_distance(new_log_q_ploidy_sjk,
                                                        self.ploidy_workspace.log_q_ploidy_sjk)
        return th.function(inputs=[],
                           outputs=[update_norm_sj],
                           updates=[(self.ploidy_workspace.log_q_ploidy_sjk, new_log_q_ploidy_sjk)])

    def call(self) -> np.ndarray:
        return self._update_log_q_ploidy_sjk_theano_func()

