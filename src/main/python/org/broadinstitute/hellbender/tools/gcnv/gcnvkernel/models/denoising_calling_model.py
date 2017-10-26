import numpy as np
import theano as th
import theano.tensor as tt
import theano.sparse as tst
import scipy.sparse as sp
import pymc3 as pm
import logging

from pymc3 import Model, Normal, Exponential, HalfFlat, Deterministic, Lognormal, DensityDist
from . import commons

from typing import List, Tuple, Callable
from abc import abstractmethod
from ..utils.interval import Interval, GCContentAnnotation
from .hmm import TheanoForwardBackward
from .. import config, types
from ..inference.hybrid_inference_base import HybridInferenceParameters

_logger = logging.getLogger(__name__)
_logger.setLevel(config.log_level)


class DenoisingModelConfig:
    # approximation schemes for calculating posterior expectations with respect to copy number posteriors
    _q_c_expectation_modes = ['map', 'exact']

    """ Configuration of the denoising model """
    def __init__(self,
                 max_bias_factors: int = 5,
                 mapping_error_rate: float = 1e-2,
                 mean_psi: float = 0.1,
                 tau_depth: float = 1e+1,
                 sigma_log_bias: float = 5e-1,
                 init_alpha_rel_psi: float = 0.1,
                 num_gc_bins: int = 20,
                 gc_curve_sd: float = 1.0,
                 q_c_expectation_mode: str = 'map',
                 enable_bias_factors: bool = True,
                 enable_explicit_gc_bias_modeling: bool = True):
        """ Constructor
        :param max_bias_factors: maximum number of bias factors
        :param mapping_error_rate: typical mapping error rate
        :param mean_psi: mean expected unexplained variance
        :param tau_depth: precision for pinning the read depth to the raw coverage median
        :param sigma_log_bias: standard deviation of the log mean bias
        :param init_alpha_rel_psi: initial ARD precision relative to mean_psi
        :param num_gc_bins: number of GC bins (if enable_explicit_gc_bias_modeling is True)
        :param gc_curve_sd: standard deviation of each knob in the GC curve
        :param q_c_expectation_mode: approximation scheme to use for calculating posterior expectations
                                     with respect to the copy number posteriors
        :param enable_bias_factors: enable bias factor discovery
        :param enable_explicit_gc_bias_modeling: enable explicit GC bias modeling
        """

        self.max_bias_factors = max_bias_factors
        self.mapping_error_rate = mapping_error_rate
        self.mean_psi = mean_psi
        self.tau_depth = tau_depth
        self.sigma_log_bias = sigma_log_bias
        self.init_alpha_rel_psi = init_alpha_rel_psi
        self.num_gc_bins = num_gc_bins
        self.gc_curve_sd = gc_curve_sd
        self.q_c_expectation_mode = q_c_expectation_mode

        self.enable_bias_factors = enable_bias_factors
        self.enable_explicit_gc_bias_modeling = enable_explicit_gc_bias_modeling

        # assert validity of parameters
        self._assert_params()

    def _assert_params(self):
        assert self.q_c_expectation_mode in self._q_c_expectation_modes,\
            "wrong q_c expectation calculation mode; available modes: {0}".format(self._q_c_expectation_modes)


class CopyNumberCallingConfig:
    """ Configuration of the copy number caller """
    def __init__(self,
                 pi_kc: np.ndarray,
                 class_probs_k: np.ndarray,
                 class_coherence_k: np.ndarray,
                 copy_number_coherence_c: np.ndarray):
        """
        :param pi_kc: each row represents a copy number class
        :param class_probs_k: prior probabilities for choosing a class
        :param class_coherence_k: coherence length of each class
        :param copy_number_coherence_c: coherence length of each copy number state
        """
        # TODO consider doing all of this in the log space
        _pi_kc = np.asarray(pi_kc, dtype=types.floatX)
        assert pi_kc.ndim == 2, "bad copy number prior matrix"
        num_copy_number_classes: int = _pi_kc.shape[0]
        num_copy_number_states: int = _pi_kc.shape[1]

        _class_probs_k = commons.get_normalized_prob_vector(
            np.asarray(class_probs_k, dtype=types.floatX).reshape((num_copy_number_classes,)),
            config.prob_sum_tol)

        _class_coherence_k: np.ndarray = np.asarray(class_coherence_k, dtype=types.floatX).reshape(
            (num_copy_number_classes,))

        _copy_number_coherence_c: np.ndarray = np.asarray(copy_number_coherence_c, dtype=types.floatX).reshape(
            (num_copy_number_states,))

        for k in range(num_copy_number_classes):
            _pi_kc[k, :] = commons.get_normalized_prob_vector(_pi_kc[k, :], config.prob_sum_tol)

        # calculate copy number prior mean
        _pi_c = np.dot(_pi_kc.T, _class_probs_k).reshape((num_copy_number_states,))

        # set member variables
        self.num_copy_number_states = num_copy_number_states
        self.num_copy_number_classes = num_copy_number_classes
        self.pi_kc: types.TensorSharedVariable = th.shared(_pi_kc, name="pi_kc", borrow=config.borrow_numpy)
        self.pi_c: types.TensorSharedVariable = th.shared(_pi_c, name="pi_c", borrow=config.borrow_numpy)
        self.prior_mean_c: float = np.dot(_pi_c, np.arange(0, num_copy_number_states, dtype=types.floatX))
        self.class_probs_k: types.TensorSharedVariable = th.shared(_class_probs_k, name="class_probs_k",
                                                                   borrow=config.borrow_numpy)
        self.class_coherence_k: types.TensorSharedVariable = th.shared(_class_coherence_k, name="class_coherence_k",
                                                                       borrow=config.borrow_numpy)
        self.copy_number_coherence_c: types.TensorSharedVariable = th.shared(_copy_number_coherence_c,
                                                                             name="copy_number_coherence_c",
                                                                             borrow=config.borrow_numpy)


class PosteriorInitializer:
    """ Base class for posterior initializers """
    @staticmethod
    @abstractmethod
    def initialize_posterior(denoising_config: DenoisingModelConfig,
                             calling_config: CopyNumberCallingConfig,
                             shared_workspace: 'DenoisingCallingWorkspace') -> None:
        pass


class InitializeToPrior(PosteriorInitializer):
    """ Initialize posteriors to priors """
    @staticmethod
    def initialize_posterior(denoising_config: DenoisingModelConfig,
                             calling_config: CopyNumberCallingConfig,
                             shared_workspace: 'DenoisingCallingWorkspace'):
        # class log posterior probs
        log_q_tau_tk = np.tile(np.log(calling_config.class_probs_k.get_value(borrow=True)),
                               (shared_workspace.num_targets, 1))
        shared_workspace.log_q_tau_tk = th.shared(log_q_tau_tk, name="log_q_tau_tk", borrow=config.borrow_numpy)

        # copy number log posterior probs
        log_q_c_stc = np.tile(np.log(calling_config.pi_c.get_value(borrow=True)),
                              (shared_workspace.num_samples, shared_workspace.num_targets, 1))
        shared_workspace.log_q_c_stc = th.shared(log_q_c_stc, name="log_q_c_stc", borrow=config.borrow_numpy)


class DenoisingCallingWorkspace:
    """ This class contains objects (numpy arrays, theano tensors, etc) shared between the denoising model
    and the copy number caller """
    def __init__(self,
                 denoising_config: DenoisingModelConfig,
                 calling_config: CopyNumberCallingConfig,
                 targets_interval_list: List[Interval],
                 n_st: np.ndarray,
                 initializer: PosteriorInitializer = InitializeToPrior):
        assert n_st.ndim == 2, "read counts matrix must be a dim=2 ndarray with shape (num_samples, num_targets)"
        assert n_st.shape[1] == len(targets_interval_list), "the length of the targets interval list does not match" \
                                                            " the shape of the read counts matrix"
        self.targets_interval_list: List[Interval] = targets_interval_list
        self.n_st: types.TensorSharedVariable = th.shared(n_st.astype(types.floatX), name="n_st",
                                                          borrow=config.borrow_numpy)
        self.num_targets: int = len(targets_interval_list)
        self.num_samples: int = n_st.shape[0]

        # distances between subsequent targets
        self.dist_t: types.TensorSharedVariable = th.shared(
            np.asarray([targets_interval_list[ti + 1].distance(targets_interval_list[ti])
                        for ti in range(self.num_targets - 1)], dtype=types.floatX),
            borrow=config.borrow_numpy)

        # copy number log posterior
        #   initialized by PosteriorInitializer.initialize(),
        #   subsequently updated by HHMMClassAndCopyNumberCaller.update_copy_number_log_posterior()
        self.log_q_c_stc: types.TensorSharedVariable = None

        # copy number emission log posterior
        #   updated by LogEmissionPosteriorSampler.update_log_copy_number_emission_posterior()
        log_copy_number_emission_stc = np.zeros(
            (self.num_samples, self.num_targets, calling_config.num_copy_number_states), dtype=types.floatX)
        self.log_copy_number_emission_stc: types.TensorSharedVariable = th.shared(
            log_copy_number_emission_stc, name="log_copy_number_emission_stc", borrow=config.borrow_numpy)

        # copy number Markov chain log prior
        #   updated by HHMMClassAndCopyNumberCaller.update_copy_number_hmm_specs()
        log_prior_c = np.zeros((calling_config.num_copy_number_states,), dtype=types.floatX)
        self.log_prior_c: types.TensorSharedVariable = th.shared(log_prior_c, name="log_prior_c",
                                                                 borrow=config.borrow_numpy)

        # copy number Markov chain log transition,
        #   updated by HHMMClassAndCopyNumberCaller.update_copy_number_hmm_specs()
        log_trans_tcc = np.zeros(
            (self.num_targets, self.num_targets, calling_config.num_copy_number_states), dtype=types.floatX)
        self.log_trans_tcc: types.TensorSharedVariable = th.shared(log_trans_tcc, name="log_trans_tcc",
                                                                   borrow=config.borrow_numpy)

        # class log posterior
        #   initialized by PosteriorInitializer.initialize()
        #   subsequently updated by HHMMClassAndCopyNumberCaller.update_class_log_posterior()
        self.log_q_tau_tk: types.TensorSharedVariable = None

        # class emission log posterior
        #   updated by HHMMClassAndCopyNumberCaller.update_log_class_emission_tk()
        log_class_emission_tk = np.zeros(
            (self.num_targets, calling_config.num_copy_number_classes), dtype=types.floatX)
        self.log_class_emission_tk: types.TensorSharedVariable = th.shared(
            log_class_emission_tk, name="log_class_emission_tk", borrow=True)

        # class Markov chain log prior
        #   initialized here and remains constant throughout
        self.log_prior_k: np.ndarray = np.log(calling_config.class_probs_k.get_value(borrow=True))

        # class Markov chain log transition
        #   initialized here and remains constant throughout
        self.log_trans_tkk: np.ndarray = self._get_log_trans_tkk(
            self.dist_t.get_value(borrow=True),
            calling_config.class_coherence_k.get_value(borrow=True),
            calling_config.num_copy_number_classes,
            calling_config.class_probs_k.get_value(borrow=True))

        # GC bias factors
        self.W_gc_tg: tst.SparseConstant = None
        if denoising_config.enable_explicit_gc_bias_modeling:
            self.W_gc_tg = self._create_sparse_gc_bin_tensor_tg(
                self.targets_interval_list, denoising_config.num_gc_bins)

        # read depth prior
        self.depth_prior_s = (np.median(n_st, axis=1) / calling_config.prior_mean_c).astype(types.floatX)

        # initialize posterior
        initializer.initialize_posterior(denoising_config, calling_config, self)

    @staticmethod
    def _get_log_trans_tkk(dist_t: np.ndarray,
                           class_coherence_k: np.ndarray,
                           num_copy_number_classes: int,
                           class_probs_k: np.ndarray) -> np.ndarray:
        """ Calculates the log transition probability between copy number classes """
        stay_tk = np.exp(-dist_t[:, None] / class_coherence_k[None, :])
        not_stay_tk = np.ones_like(stay_tk) - stay_tk
        delta_kl = np.eye(num_copy_number_classes, dtype=types.floatX)
        trans_tkl = not_stay_tk[:, :, None] * class_probs_k[None, None, :] + stay_tk[:, :, None] * delta_kl[None, :, :]
        return np.log(trans_tkl)

    @staticmethod
    def _create_sparse_gc_bin_tensor_tg(targets_interval_list: List[Interval], num_gc_bins: int) -> tst.SparseConstant:
        """ Creates a sparse 2d theano tensor with shape (num_targets, gc_bin). The sparse tensor represents a
        1-hot mapping of each target to its GC bin index. The range [0, 1] is uniformly divided into num_gc_bins.
        """
        assert all([GCContentAnnotation.get_key() in interval.annotations.keys()
                    for interval in targets_interval_list]), "explicit GC bias modeling is enabled, however, " \
                                                             "some or all targets lack the GC_CONTENT annotation."

        def get_gc_bin_idx(gc_content):
            return min(int(gc_content * num_gc_bins), num_gc_bins - 1)

        num_targets = len(targets_interval_list)
        data = np.ones((num_targets,))
        indices = [get_gc_bin_idx(interval.get_annotation(GCContentAnnotation.get_key()))
                   for interval in targets_interval_list]
        indptr = np.arange(0, num_targets + 1)
        scipy_gc_matrix = sp.csr_matrix((data, indices, indptr), shape=(num_targets, num_gc_bins), dtype=np.int8)
        theano_gc_matrix: tst.SparseConstant = tst.as_sparse(scipy_gc_matrix)
        return theano_gc_matrix


class InitialModelParametersSupplier:
    def __init__(self,
                 denoising_model_config: DenoisingModelConfig,
                 calling_config: CopyNumberCallingConfig,
                 shared_workspace: DenoisingCallingWorkspace):
        self.denoising_model_config = denoising_model_config
        self.calling_config = calling_config
        self.shared_workspace = shared_workspace

    @abstractmethod
    def get_init_psi_t(self) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_init_log_mean_bias_t(self) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_init_alpha_u(self) -> np.ndarray:
        raise NotImplementedError


class DefaultInitialModelParametersSupplier(InitialModelParametersSupplier):
    """ TODO """
    def __init__(self,
                 denoising_model_config: DenoisingModelConfig,
                 calling_config: CopyNumberCallingConfig,
                 shared_workspace: DenoisingCallingWorkspace):
        super().__init__(denoising_model_config, calling_config, shared_workspace)

    def get_init_psi_t(self) -> np.ndarray:
        return self.denoising_model_config.mean_psi * np.ones((self.shared_workspace.num_targets,), dtype=types.floatX)

    def get_init_log_mean_bias_t(self) -> np.ndarray:
        depth_normalized_n_st: np.ndarray = self.shared_workspace.n_st.get_value(borrow=True) / (
            self.calling_config.prior_mean_c * self.shared_workspace.depth_prior_s[:, np.newaxis])
        bias_t = np.mean(depth_normalized_n_st, axis=0)
        return np.log(bias_t + config.log_eps)

    def get_init_alpha_u(self) -> np.ndarray:
        fact = self.denoising_model_config.mean_psi * self.denoising_model_config.init_alpha_rel_psi
        return fact * np.ones((self.denoising_model_config.max_bias_factors,), dtype=types.floatX)


class DenoisingModel(Model):
    """ The gCNV coverage denoising model """
    def __init__(self,
                 denoising_model_config: DenoisingModelConfig,
                 calling_config: CopyNumberCallingConfig,
                 shared_workspace: DenoisingCallingWorkspace,
                 test_value_supplier: InitialModelParametersSupplier):
        super().__init__()

        # target-specific unexplained variance
        psi_t = Exponential(name='psi_t', lam=1.0 / denoising_model_config.mean_psi,
                            shape=shared_workspace.num_targets,
                            testval=test_value_supplier.get_init_psi_t())

        # target-specific mean log bias
        log_mean_bias_t = Normal(name='log_mean_bias_t', mu=0.0, sd=denoising_model_config.sigma_log_bias,
                                 shape=shared_workspace.num_targets,
                                 testval=test_value_supplier.get_init_log_mean_bias_t())

        # sample-specific depth
        depth_s = Lognormal(name='depth_s', mu=np.log(shared_workspace.depth_prior_s),
                            tau=denoising_model_config.tau_depth,
                            shape=shared_workspace.num_samples,
                            testval=shared_workspace.depth_prior_s)
        # bias modeling
        log_bias_st = tt.tile(log_mean_bias_t, (shared_workspace.num_samples, 1))

        if denoising_model_config.enable_bias_factors:
            # ARD prior precisions
            alpha_u = HalfFlat(name='alpha_u', shape=denoising_model_config.max_bias_factors,
                               testval=test_value_supplier.get_init_alpha_u())

            # bias factors
            W_tu = Normal(name='W_tu', mu=0.0, tau=alpha_u.dimshuffle('x', 0),
                          shape=(shared_workspace.num_targets, denoising_model_config.max_bias_factors))

            # sample-specific bias factor latent variables
            z_su = Normal(name='z_su', mu=0.0, sd=1.0,
                          shape=(shared_workspace.num_samples, denoising_model_config.max_bias_factors))

            # total bias = mean bias + sample-specific bias factors
            log_bias_st += tt.dot(W_tu, z_su.T).T

        # gc bias
        if denoising_model_config.enable_explicit_gc_bias_modeling:
            z_sg = Normal(name='z_sg', mu=0.0, sd=denoising_model_config.gc_curve_sd,
                          shape=(shared_workspace.num_samples, denoising_model_config.num_gc_bins))

            log_bias_st += tst.dot(shared_workspace.W_gc_tg, z_sg.T).T

        bias_st = Deterministic(name='bias_st', var=tt.exp(log_bias_st))

        # n_st (observed)
        if denoising_model_config.q_c_expectation_mode == 'map':
            DensityDist(name='n_st_obs',
                        logp=self._logp_negative_binomial_q_c_expectation_map(
                            bias_st, depth_s, psi_t,shared_workspace.log_q_c_stc,
                            denoising_model_config.mapping_error_rate),
                        observed=shared_workspace.n_st)
        elif denoising_model_config.q_c_expectation_mode == 'exact':
            DensityDist(name='n_st_obs',
                        logp=self._logp_negative_binomial_q_c_expectation_exact(
                            bias_st, depth_s, psi_t, shared_workspace.log_q_c_stc,
                            denoising_model_config.mapping_error_rate,
                            calling_config.num_copy_number_states),
                        observed=shared_workspace.n_st)
        else:
            raise Exception("Unknown q_c expectation mode")

    @staticmethod
    def _logp_negative_binomial_q_c_expectation_exact(bias_st: types.TheanoMatrix,
                                                      depth_s: types.TheanoVector,
                                                      psi_t: types.TheanoVector,
                                                      log_q_c_stc: types.TheanoTensor3,
                                                      mapping_error_rate: types.floatX,
                                                      num_copy_number_states: int)\
            -> Callable[[types.TheanoMatrix], types.TheanoScalar]:
        """ Calculates the variational lower bound to the emission probability for given parameters of the
        coverage denoising model.
        :return: A function from n_st (2d theano tensor) to a scalar theano tensor; evaluates to the log
        probability density.
        """
        def _eval_logp(n_st):
            depth_bias_st = (1.0 - mapping_error_rate) * depth_s.dimshuffle(0, 'x') * bias_st
            eps_st = (mapping_error_rate * depth_s).dimshuffle(0, 'x')
            mu_adj_fact_st = tt.exp(0.5 * psi_t).dimshuffle('x', 0)
            alpha_st = tt.inv((tt.exp(psi_t) - 1.0)).dimshuffle('x', 0)
            log_copy_number_emission_stc = tt.stack([commons.negative_binomial_logp(
                c * mu_adj_fact_st * depth_bias_st + eps_st, alpha_st, n_st)
                for c in range(num_copy_number_states)]).dimshuffle(1, 2, 0)
            q_c_stc = tt.exp(log_q_c_stc)
            return tt.sum(q_c_stc * (log_copy_number_emission_stc - log_q_c_stc))

        return _eval_logp

    @staticmethod
    def _logp_negative_binomial_q_c_expectation_map(bias_st: types.TheanoMatrix,
                                                    depth_s: types.TheanoVector,
                                                    psi_t: types.TheanoVector,
                                                    log_q_c_stc: types.TheanoTensor3,
                                                    mapping_error_rate: types.floatX)\
            -> Callable[[types.TheanoMatrix], types.TheanoScalar]:
        """ Calculates the variational lower bound to the emission probability for given parameters of the
        coverage denoising model. Instead of exact q_c posterior expectation, it uses MAP(c).
        :return: A function from n_st (2d theano tensor) to a scalar theano tensor; evaluates to the log
        probability density.
        """
        def _eval_logp(n_st):
            depth_bias_st = (1.0 - mapping_error_rate) * depth_s.dimshuffle(0, 'x') * bias_st
            eps_st = (mapping_error_rate * depth_s).dimshuffle(0, 'x')
            mu_adj_fact_st = tt.exp(0.5 * psi_t).dimshuffle('x', 0)
            alpha_st = tt.inv((tt.exp(psi_t) - 1.0)).dimshuffle('x', 0)
            c_argmax_st = tt.argmax(log_q_c_stc, axis=2)
            log_copy_number_emission_st = commons.negative_binomial_logp(
                c_argmax_st * mu_adj_fact_st * depth_bias_st + eps_st, alpha_st, n_st)
            return tt.sum(log_copy_number_emission_st)

        return _eval_logp


class LogCopyNumberEmissionPosteriorSampler:
    """ Draws posterior samples from the log emission probability for a given variational approximation to
    the denoising model posterior """
    def __init__(self,
                 denoising_model_config: DenoisingModelConfig,
                 calling_config: CopyNumberCallingConfig,
                 inference_params: HybridInferenceParameters,
                 shared_workspace: DenoisingCallingWorkspace,
                 denoising_model: DenoisingModel):
        self.model_config = denoising_model_config
        self.calling_config = calling_config
        self.inference_params = inference_params
        self.shared_workspace = shared_workspace
        self.denoising_model = denoising_model
        self._simultaneous_log_copy_number_emission_sampler = None

    def update_approximation(self, approx: pm.approximations.MeanField):
        self._simultaneous_log_copy_number_emission_sampler =\
            self._get_compiled_simultaneous_log_copy_number_emission_sampler(approx)

    def is_sampler_initialized(self):
        return self._simultaneous_log_copy_number_emission_sampler is not None

    def draw(self):
        return self._simultaneous_log_copy_number_emission_sampler()

    def _get_compiled_simultaneous_log_copy_number_emission_sampler(self, approx: pm.approximations.MeanField):
        """ For a given variational approximation, returns a compiled theano function that draws posterior samples
        from the log copy number emission """
        depth_bias_st = ((1.0 - self.model_config.mapping_error_rate)
                         * self.denoising_model['depth_s'].dimshuffle(0, 'x')
                         * self.denoising_model['bias_st'])
        eps_st = self.model_config.mapping_error_rate * self.denoising_model['depth_s'].dimshuffle(0, 'x')
        mu_adj_fact_st = tt.exp(0.5 * self.denoising_model['psi_t']).dimshuffle('x', 0)
        alpha_st = tt.inv((tt.exp(self.denoising_model['psi_t']) - 1.0)).dimshuffle('x', 0)
        log_copy_number_emission_stc = tt.stack([commons.negative_binomial_logp(
            c * mu_adj_fact_st * depth_bias_st + eps_st,
            alpha_st, self.shared_workspace.n_st)
            for c in range(self.calling_config.num_copy_number_states)]).dimshuffle(1, 2, 0)
        log_copy_number_emission_stc_sampler = approx.sample_node(
            log_copy_number_emission_stc, size=self.training_params.log_copy_number_emission_samples_per_round)
        return th.function(inputs=[], outputs=log_copy_number_emission_stc_sampler)


class HHMMClassAndCopyNumberCaller:
    """ This class updates copy number and class posteriors.

        class_prior_k --► (tau_1) --► (tau_2) --► (tau_3) --► ...
                             |           |           |
                             |           |           |
                             ▼           ▼           ▼
                           (c_s1) --►  (c_s2) --►  (c_s3) --► ...
                             |           |           |
                             |           |           |
                             ▼           ▼           ▼
                            n_s1        n_s2        n_s3

        We assume the variational ansatz \prod_s p(tau, c_s | n) ~ q(tau) \prod_s q(c_s)
        Accordingly, q(tau) and q(c_s) are determined by minimizing the KL divergence w.r.t. the true
        posterior. This yields the following iterative scheme:

        - Given q(tau), the (variational) copy number prior for the first state and the copy number
          transition probabilities are determined (see _get_update_copy_number_hmm_specs_compiled_function).
          Along with the given emission probabilities to sample read counts, q(c_s) is updated using the
          forward-backward algorithm for each sample (see _update_copy_number_log_posterior)

        - Given q(c_s), the emission probability of each copy number class (tau) is determined
          (see _get_update_log_class_emission_tk_theano_func). The class prior and transition probabilities
          are fixed hyperparameters. Therefore, q(tau) can be updated immediately using a single run
          of forward-backward algorithm (see _update_class_log_posterior).
    """
    def __init__(self,
                 calling_config: CopyNumberCallingConfig,
                 inference_params: HybridInferenceParameters,
                 shared_workspace: DenoisingCallingWorkspace):
        self.calling_config = calling_config
        self.inference_params = inference_params
        self.shared_workspace = shared_workspace

        # compiled functions for forward-backward updates of copy number and class posteriors
        self._hmm_q_copy_number = TheanoForwardBackward(shared_workspace.log_q_c_stc, 'slice',
                                                        self.inference_params.caller_admixing_rate,
                                                        True)
        self._hmm_q_class = TheanoForwardBackward(shared_workspace.log_q_tau_tk, 'whole',
                                                  self.inference_params.caller_admixing_rate,
                                                  True)

        # compiled function for variational update of copy number HMM specs
        self._update_copy_number_hmm_specs_theano_func = self._get_update_copy_number_hmm_specs_theano_func()

        # compiled function for update of class log emission
        self._update_log_class_emission_tk_theano_func = self._get_update_log_class_emission_tk_theano_func()

    def call(self,
             copy_number_update_summary_statistic_reducer,
             class_update_summary_statistic_reducer) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """
        todo
        :param copy_number_update_summary_statistic_reducer:
        :param class_update_summary_statistic_reducer:
        :return:
        """
        # copy number posterior update
        self._update_copy_number_hmm_specs()
        copy_number_update_s, copy_number_log_likelihoods_s = self._update_copy_number_log_posterior(
            copy_number_update_summary_statistic_reducer)

        # class posterior update
        self._update_log_class_emission_tk()
        class_update, class_log_likelihood = self._update_class_log_posterior(class_update_summary_statistic_reducer)

        return copy_number_update_s, copy_number_log_likelihoods_s, class_update, class_log_likelihood

    def _update_copy_number_hmm_specs(self):
        self._update_copy_number_hmm_specs_theano_func()

    def _update_copy_number_log_posterior(self, copy_number_update_summary_statistic_reducer)\
            -> Tuple[np.ndarray, np.ndarray]:
        """
        todo
        :param copy_number_update_summary_statistic_reducer:
        :return:
        """
        copy_number_update_s = np.zeros((self.shared_workspace.num_samples,), dtype=types.floatX)
        copy_number_log_likelihoods_s = np.zeros((self.shared_workspace.num_samples,), dtype=types.floatX)
        for sample_index in range(self.shared_workspace.num_samples):
            output = self._hmm_q_copy_number.update_log_posterior_slice(
                self.calling_config.num_copy_number_states, sample_index,
                self.shared_workspace.log_prior_c.get_value(borrow=True),
                self.shared_workspace.log_trans_tcc.get_value(borrow=True),
                self.shared_workspace.log_copy_number_emission_stc.get_value(borrow=True)[sample_index, ...])
            copy_number_update_s[sample_index] = copy_number_update_summary_statistic_reducer(output[0])
            copy_number_log_likelihoods_s[sample_index] = float(output[1])
        return copy_number_update_s, copy_number_log_likelihoods_s

    def _update_log_class_emission_tk(self):
        self._update_log_class_emission_tk_theano_func()

    def _update_class_log_posterior(self, class_update_summary_statistic_reducer) -> Tuple[float, float]:
        """
        todo
        :param class_update_summary_statistic_reducer:
        :return:
        """
        output = self._hmm_q_class.update_log_posterior_whole(
            self.calling_config.num_copy_number_classes,
            self.shared_workspace.log_prior_k,
            self.shared_workspace.log_trans_tkk,
            self.shared_workspace.log_class_emission_tk.get_value(borrow=True))
        return class_update_summary_statistic_reducer(output[0]), float(output[1])

    @th.configparser.change_flags(compute_test_value="ignore")
    def _get_update_copy_number_hmm_specs_theano_func(self):
        """ Returns a compiled function that calculates the class-averaged and probability-sum-normalized log copy
        number transition matrix and log copy number prior for the first state, and updates the corresponding
        placeholders in the shared workspace

        Note:
            In the following, we use "a" and "b" subscripts in the variable names to refer to the departure
            and destination states, respectively. Like before, "t" and "k" denote target and class.
        """
        # shorthands
        dist_t = self.shared_workspace.dist_t
        log_q_tau_tk = self.shared_workspace.log_q_tau_tk
        pi_kc = self.calling_config.pi_kc
        copy_number_coherence_c = self.calling_config.copy_number_coherence_c
        num_copy_number_states = self.calling_config.num_copy_number_states

        # log prior probability for the first target
        log_prior_c_first_state = tt.dot(tt.log(pi_kc.T), tt.exp(log_q_tau_tk[0, :]))
        log_prior_c_first_state -= pm.logsumexp(log_prior_c_first_state)

        # log transition matrix
        stay_ta = tt.exp(-dist_t.dimshuffle(0, 'x') / copy_number_coherence_c.dimshuffle('x', 0))
        not_stay_ta = tt.ones_like(stay_ta) - stay_ta
        delta_ab = tt.eye(num_copy_number_states)
        trans_ktab = not_stay_ta.dimshuffle('x', 0, 1, 'x') * pi_kc.dimshuffle(0, 'x', 'x', 1)\
                     + stay_ta.dimshuffle('x', 0, 1, 'x') * delta_ab.dimshuffle('x', 'x', 0, 1)
        q_tau_ktab = tt.exp(log_q_tau_tk[:-1, :]).dimshuffle(1, 0, 'x', 'x')
        log_trans_tab = tt.sum(q_tau_ktab * tt.log(trans_ktab), axis=0)
        log_trans_tab -= pm.logsumexp(log_trans_tab, axis=2)

        return th.function(inputs=[], outputs=[], updates=[
            (self.shared_workspace.log_prior_c, log_prior_c_first_state),
            (self.shared_workspace.log_trans_tcc, log_trans_tab)])

    @th.configparser.change_flags(compute_test_value="ignore")
    def _get_update_log_class_emission_tk_theano_func(self):
        """ Returns a compiled function that calculates the log class emission and updates the
        corresponding placeholder in the shared workspace
        """
        # shorthands
        dist_t = self.shared_workspace.dist_t
        q_c_stc = tt.exp(self.shared_workspace.log_q_c_stc)
        pi_kc = self.calling_config.pi_kc
        copy_number_coherence_c = self.calling_config.copy_number_coherence_c
        num_copy_number_states = self.calling_config.num_copy_number_states

        # log copy number transition matrix for each class
        stay_ta = tt.exp(-dist_t.dimshuffle(0, 'x') / copy_number_coherence_c.dimshuffle('x', 0))
        not_stay_ta = tt.ones_like(stay_ta) - stay_ta
        delta_ab = tt.eye(num_copy_number_states)
        log_trans_tkab = tt.log(
            not_stay_ta.dimshuffle(0, 'x', 1, 'x') * pi_kc.dimshuffle('x', 0, 'x', 1)
            + stay_ta.dimshuffle(0, 'x', 1, 'x') * delta_ab.dimshuffle('x', 'x', 0, 1))

        # posterior copy number probability of two subsequent targets
        # Note:
        #   correlations are ignored, i.e. we assume:
        #       xi(c_t, c_{t+1}) \equiv q(c_t, c_{t+1}) \approx q(c_t) q(c_{t+1})
        #   if needed, xi can be calculated exactly from the forward-backward tables
        xi_tab = tt.sum(q_c_stc[:, :-1, :].dimshuffle(0, 1, 2, 'x')
                        * q_c_stc[:, 1:, :].dimshuffle(0, 1, 'x', 2), axis=0)

        # TODO can this be optimized, or does theano cast this to BLAS ops?
        log_class_emission_tk_except_for_last = tt.sum(tt.sum(
            xi_tab.dimshuffle(0, 'x', 1, 2) * log_trans_tkab, axis=-1), axis=-1)

        log_class_emission_k_last = tt.dot(tt.log(pi_kc), tt.sum(q_c_stc[:, -1, :], axis=0))
        log_class_emission_tk = tt.concatenate((log_class_emission_tk_except_for_last,
                                                log_class_emission_k_last.dimshuffle('x', 0)))

        return th.function(inputs=[], outputs=[], updates=[
            (self.shared_workspace.log_class_emission_tk, log_class_emission_tk)])

