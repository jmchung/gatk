import numpy as np
import theano as th
import theano.tensor as tt
import theano.sparse as tst
import scipy.sparse as sp
import pymc3 as pm
import logging

from pymc3 import Model, Normal, Exponential, HalfFlat, Deterministic, Lognormal, DensityDist, Bound
from . import commons

from typing import List, Dict, Set
from ..utils.interval import Interval
from .. import config, types

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


class ContigPloidyDeterminationConfig:
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
                                               min_prob: float = 1e-12):
        given_contigs = set(given_contig_prior_ploidy_map.keys())
        num_ploidy_states = 0
        for contig in given_contigs:
            num_ploidy_states = max(num_ploidy_states, given_contig_prior_ploidy_map[contig].size)
        validated_contig_prior_ploidy_map: Dict[str, np.ndarray] = dict()
        for contig in given_contigs:
            validated_prior = commons.get_normalized_prob_vector(
                given_contig_prior_ploidy_map[contig].flatten() + min_prob, config.prob_sum_tol)
            padded_validated_prior = np.zeros((num_ploidy_states,), dtype=types.floatX)
            padded_validated_prior[:validated_prior.size] = validated_prior[:]
            validated_contig_prior_ploidy_map[contig] = padded_validated_prior
        return validated_contig_prior_ploidy_map, num_ploidy_states


class ContigPloidyDeterminationWorkspace:
    def __init__(self,
                 ploidy_config: ContigPloidyDeterminationConfig,
                 n_st: np.ndarray,
                 targets_interval_list: List[Interval]):
        self.targets_interval_list = targets_interval_list
        self.ploidy_config = ploidy_config
        self.contig_set = self._get_contig_set_from_interval_list(targets_interval_list)
        self.contig_list = sorted(list(self.contig_set))
        self.num_contigs = len(self.contig_list)
        self.num_samples: int = n_st.shape[0]
        assert all([contig in ploidy_config.contig_set for contig in self.contig_set]), \
            "Some contigs do not have ploidy priors; cannot continue."

        self.contig_target_indices: Dict[str, List[int]] = \
            {contig: [ti for ti in range(len(targets_interval_list))
                      if targets_interval_list[ti].contig == contig]
             for contig in self.contig_set}

        # number of targets per contig
        t_j = np.asarray([len(self.contig_target_indices[self.contig_list[j]])
                          for j in range(self.num_contigs)], dtype=np.int)
        self.t_j: types.TensorSharedVariable = th.shared(t_j, name='t_j', borrow=config.borrow_numpy)

        # total count per contig
        n_sj = np.zeros((self.num_samples, self.num_contigs), dtype=np.int)
        for j in range(self.num_contigs):
            n_sj[:, j] = np.sum(n_st[:, self.contig_target_indices[self.contig_list[j]]], axis=1)
        self.n_sj: types.TensorSharedVariable = th.shared(n_sj, name='n_sj', borrow=config.borrow_numpy)

        # total count per sample
        n_s = np.sum(n_sj, axis=1)
        self.n_s: types.TensorSharedVariable = th.shared(n_s, name='n_s', borrow=config.borrow_numpy)

        # integer ploidy values
        int_ploidy_values_k = np.arange(0, ploidy_config.num_ploidy_states, dtype=np.int)
        self.int_ploidy_values_k = th.shared(int_ploidy_values_k, name='int_ploidy_values_k',
                                             borrow=config.borrow_numpy)

        # ploidy priors
        p_kappa_jk = np.zeros((self.num_contigs, self.ploidy_config.num_ploidy_states), dtype=types.floatX)
        for j in range(self.num_contigs):
            p_kappa_jk[j, :] = ploidy_config.contig_prior_ploidy_map[self.contig_list[j]][:]
        log_p_kappa_jk = np.log(p_kappa_jk)
        self.log_p_kappa_jk: types.TensorSharedVariable = th.shared(log_p_kappa_jk, name='log_p_kappa_jk',
                                                                    borrow=config.borrow_numpy)

        # ploidy log posteriors (placeholder)
        log_q_kappa_sjk = np.tile(log_p_kappa_jk, (self.num_samples, 1, 1))
        self.log_q_kappa_sjk: types.TensorSharedVariable = th.shared(
            log_q_kappa_sjk, name='log_q_kappa_sjk', borrow=config.borrow_numpy)

        # mean ploidy (placeholder)
        mean_ploidy_j = np.dot(p_kappa_jk, int_ploidy_values_k)
        mean_ploidy_sj = np.tile(mean_ploidy_j, (self.num_samples, 1))
        self.mean_ploidy_sj: types.TensorSharedVariable = th.shared(
            mean_ploidy_sj, name='mean_ploidy_sj', borrow=config.borrow_numpy)

        # exclusion mask
        contig_exclusion_mask_jj = np.ones((self.num_contigs, self.num_contigs), dtype=np.int) \
                                   - np.eye(self.num_contigs, dtype=np.int)
        self.contig_exclusion_mask_jj = th.shared(contig_exclusion_mask_jj, name='contig_exclusion_mask_jj')

    @staticmethod
    def _get_contig_set_from_interval_list(targets_interval_list: List[Interval]) -> Set[str]:
        return {target.contig for target in targets_interval_list}


class PloidyDeterminationBiasModel(Model):
    PositiveNormal = Bound(Normal, lower=0)  # how cool is this?

    def __init__(self,
                 ploidy_config: ContigPloidyDeterminationConfig,
                 ploidy_workspace: ContigPloidyDeterminationWorkspace):
        super().__init__()
        print("test")

        # shorthands
        mean_ploidy_sj = ploidy_workspace.mean_ploidy_sj
        t_j = ploidy_workspace.t_j
        contig_exclusion_mask_jj = ploidy_workspace.contig_exclusion_mask_jj
        n_s = ploidy_workspace.n_s
        n_sj = ploidy_workspace.n_sj
        kappa_k = ploidy_workspace.int_ploidy_values_k
        q_kappa_sjk = tt.exp(ploidy_workspace.log_q_kappa_sjk)
        eps = ploidy_config.mapping_error_rate

        # mean per-contig bias
        mean_bias_j = self.PositiveNormal('mean_bias_j',
                                          mu=ploidy_config.mean_bias_mu,
                                          sd=ploidy_config.mean_bias_sd,
                                          shape=ploidy_workspace.num_contigs)

        # per-contig NB over-dispersion parameter
        alpha_j = HalfFlat('alpha_j', shape=ploidy_workspace.num_contigs)

        # mean-field amplification coefficient per contig
        gamma_sj = mean_ploidy_sj * t_j.dimshuffle('x', 0) * mean_bias_j.dimshuffle('x', 0)

        # gamma_rest_sj \equiv sum_{j' \neq j} gamma_sj
        gamma_rest_sj = tt.dot(gamma_sj, contig_exclusion_mask_jj)

        # NB per-contig counts
        mu_num_sjk = (t_j.dimshuffle('x', 0, 'x') * mean_bias_j.dimshuffle('x', 0, 'x')
                      * kappa_k.dimshuffle('x', 'x', 0))
        mu_den_sjk = gamma_rest_sj.dimshuffle(0, 1, 'x') + mu_num_sjk
        eps_j = eps * t_j / tt.sum(t_j)  # proportion of fragments erroneously mapped to contig j
        mu_sjk = ((1.0 - eps) * (mu_num_sjk / mu_den_sjk)
                  + eps_j.dimshuffle('x', 0, 'x')) * n_s.dimshuffle(0, 'x', 'x')
        alpha_adj_sjk = tt.sqrt(1 + tt.inv(alpha_j)).dimshuffle('x', 0, 'x')

        def _get_contig_counts_logp(n_sj):
            _logp_sjk = commons.negative_binomial_logp(alpha_adj_sjk * mu_sjk,  # mean
                                                alpha_j.dimshuffle('x', 0, 'x'),  # over-dispersion
                                                n_sj.dimshuffle(0, 1, 'x'))  # contig counts
            return tt.sum(q_kappa_sjk * _logp_sjk)

        DensityDist(name='n_sj_obs', logp=_get_contig_counts_logp, observed=n_sj)


class LogPloidyEmissionPosteriorSampler:
    pass


class PloidyCaller:
    pass
