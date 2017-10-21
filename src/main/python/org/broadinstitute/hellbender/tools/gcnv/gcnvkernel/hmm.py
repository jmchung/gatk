import numpy as np
import theano as th
import theano.tensor as tt
import pymc3 as pm
from typing import Optional, Tuple
from . import types


class TheanoForwardBackward:
    """ Implementation of the forward-backward algorithm using th.scan """
    def __init__(self,
                 log_posterior_output: Optional[types.TensorSharedVariable],
                 update_type: str,
                 admixing_rate: float,
                 extended_output: bool = False):
        """
        :param log_posterior_output: shared theano tensor to update
        :param update_type: 'slice' or 'whole'. If 'slice', a slice of the output tensor (along the first axis)
        will be updated. In this case, it is expected that log_posterior_output is a 3d tensor. If 'whole', the
        whole tensor will be update, in which case, log_posterior_output is expected to be a 2d tensor.
        :param admixing_rate: a float in range [0, 1] denoting the amount of the new posterior to admix with the
        old posterior
        :param extended_output: if True, an extended output will be provided (see below)
        """
        assert update_type in ['slice', 'whole'], "Unrecognized updated type. Available types: 'slice', 'whole'"
        self.update_type = update_type
        self.admixing_rate = admixing_rate
        assert 0.0 < admixing_rate <= 1.0, "Admixing rate must be in range (0, 1]"
        self.output_tensor = log_posterior_output
        if update_type == 'slice':
            self._update_log_posterior_theano_func = self._get_compiled_log_posterior_calculator_slice(extended_output)
        else:  # 'whole'
            self._update_log_posterior_theano_func = self._get_compiled_log_posterior_calculator_whole(extended_output)

    def update_log_posterior_slice(self, num_states: int, slice_index: int, log_prior_c: np.ndarray,
                                   log_trans_tcc: np.ndarray, log_emission_tc: np.ndarray):
        assert self.update_type == 'slice', "This instance of the class does not update by slice"
        return self._update_log_posterior_theano_func(
            num_states, slice_index, log_prior_c, log_trans_tcc, log_emission_tc)

    def update_log_posterior_whole(self, num_states: int, log_prior_c: np.ndarray,
                                   log_trans_tcc: np.ndarray, log_emission_tc: np.ndarray):
        assert self.update_type == 'whole', "This instance of the class does not update as a whole"
        return self._update_log_posterior_theano_func(num_states, log_prior_c, log_trans_tcc, log_emission_tc)

    @th.configparser.change_flags(compute_test_value="ignore")
    def _get_compiled_log_posterior_calculator_slice(self, extended_output: bool):
        """ Returns a compiled theano function that updates log posterior probabilities.

        The theano function takes 5 inputs:
            num_states (integer scalar), slice_index (integer scalar), log_prior_c (float vector),
            log_trans_tcc (float tensor3), log_emission_tc (float matrix)
        The output is either an empty list (if extended_output is False) or otherwise a tuple of:
            update norm_inf (float scalar), data_log_likelihood (float vector)
        As a side effect, the log posterior will be written to self.output_tensor[slice_idx, ...].

        :param extended_output: whether or not provided extended output
        :return: a theano function
        """
        num_states = tt.iscalar('num_states')
        slice_index = tt.iscalar('slice_index')
        log_prior_c = tt.vector('log_prior_c')
        log_trans_tcc = tt.tensor3('log_trans_tcc')
        log_emission_tc = tt.matrix('log_emission_tc')
        new_log_posterior_tc, log_data_likelihood_t = self._get_symbolic_log_posterior(
            num_states, log_prior_c, log_trans_tcc, log_emission_tc)
        old_log_posterior_tc = self.output_tensor[slice_index, ...]
        admixed_log_posterior_tc = pm.logaddexp(new_log_posterior_tc + np.log(self.admixing_rate),
                                                old_log_posterior_tc + np.log(1.0 - self.admixing_rate))

        if extended_output:
            log_data_likelihood = log_data_likelihood_t[-1]  # in theory, they are all the same
            update_norm_t = self._get_jensen_shannon_divergence(admixed_log_posterior_tc, old_log_posterior_tc)
            outputs = [update_norm_t, log_data_likelihood]
        else:
            outputs = None
        update_log_posterior_output = tt.set_subtensor(old_log_posterior_tc, admixed_log_posterior_tc)
        return th.function(inputs=[num_states, slice_index, log_prior_c, log_trans_tcc, log_emission_tc],
                           outputs=outputs,
                           updates=[(self.output_tensor, update_log_posterior_output)])

    @th.configparser.change_flags(compute_test_value="ignore")
    def _get_compiled_log_posterior_calculator_whole(self, extended_output: bool):
        """ Returns a compiled theano function that updates log posterior probabilities.

        The theano function takes 4 inputs:
            num_states (integer scalar), log_prior_c (float vector), log_trans_tcc (float tensor3),
            log_emission_tc (float matrix)
        The output is either an empty list (if extended_output is False) or otherwise a tuple of:
            update norm_inf (float scalar), data_log_likelihood (float vector)
        As a side effect, the log posterior will be written to self.output_tensor.

        :param extended_output: whether or not provided extended output
        :return: a theano function
        """
        num_states = tt.iscalar('num_states')
        log_prior_c = tt.vector('log_prior_c')
        log_trans_tcc = tt.tensor3('log_trans_tcc')
        log_emission_tc = tt.matrix('log_emission_tc')
        new_log_posterior_tc, log_data_likelihood_t = self._get_symbolic_log_posterior(
            num_states, log_prior_c, log_trans_tcc, log_emission_tc)
        old_log_posterior_tc = self.output_tensor
        admixed_log_posterior_tc = pm.logaddexp(new_log_posterior_tc + np.log(self.admixing_rate),
                                                old_log_posterior_tc + np.log(1.0 - self.admixing_rate))
        if extended_output:
            log_data_likelihood = log_data_likelihood_t[-1]  # in theory, they are all the same
            update_norm_t = self._get_jensen_shannon_divergence(admixed_log_posterior_tc, old_log_posterior_tc)
            outputs = [update_norm_t, log_data_likelihood]
        else:
            outputs = None
        return th.function(inputs=[num_states, log_prior_c, log_trans_tcc, log_emission_tc],
                           outputs=outputs,
                           updates=[(old_log_posterior_tc, admixed_log_posterior_tc)])

    @staticmethod
    def _get_jensen_shannon_divergence(log_p_1, log_p_2):
        p_1 = tt.exp(log_p_1)
        p_2 = tt.exp(log_p_2)
        return 0.5 * tt.sum((p_1 * (log_p_1 - log_p_2) + p_2 * (log_p_2 - log_p_1)), axis=-1)

    @staticmethod
    def _get_symbolic_log_posterior(num_states: tt.iscalar,
                                    log_prior_c: types.TheanoVector,
                                    log_trans_tcc: types.TheanoTensor3,
                                    log_emission_tc: types.TheanoMatrix) -> Tuple[types.TheanoMatrix, types.TheanoVector]:
        """ Returns a symbolic tensor for log posterior and log data likelihood
        :return: tuple of (log_posterior_probs, log_data_likelihood)
        """

        def calculate_next_alpha(c_log_trans_mat: types.TheanoMatrix, c_log_emission_vec: types.TheanoVector,
                                 p_alpha_vec: types.TheanoVector):
            """ Calculates the next entry on the forward table, alpha(t), from alpha(t-1)
            :param c_log_trans_mat: a 2d tensor with rows and columns corresponding to log transition probability
                                    from the previous state at position t-1 and to the next state at position t,
                                    respectively
            :param c_log_emission_vec: a 1d tensor representing the emission probability to each state at position t
            :param p_alpha_vec: a 1D tensor representing alpha(t-1)
            :return: a 1d tensor representing alpha(t)
            """
            mu = tt.tile(p_alpha_vec, (num_states, 1)) + c_log_trans_mat.T
            return c_log_emission_vec + pm.math.logsumexp(mu, axis=1).dimshuffle(0)

        def calculate_prev_beta(n_log_trans_mat: types.TheanoMatrix, n_log_emission_vec: types.TheanoVector,
                                n_beta_vec: types.TheanoVector):
            """ Calculates the previous entry on the backward table, beta(t-1), from beta(t)
            :param n_log_trans_mat: a 2d tensor with rows and columns corresponding to log transition probability
                                    from the previous state at position t-1 and to the next state at position t,
                                    respectively
            :param n_log_emission_vec: a 1d tensor representing the emission probability to each state at position t
            :param n_beta_vec: a 1d tensor representing beta(t)
            :return: a 1d tensor representing beta(t-1)
            """
            nu = tt.tile(n_beta_vec + n_log_emission_vec, (num_states, 1)) + n_log_trans_mat
            return pm.math.logsumexp(nu, axis=1).dimshuffle(0)

        # first entry of the forward table
        alpha_first = log_prior_c + log_emission_tc[0, :]

        # the rest of the forward table
        alpha_outputs, alpha_updates = th.scan(
            fn=calculate_next_alpha,
            sequences=[log_trans_tcc, log_emission_tc[1:, :]],
            outputs_info=[alpha_first])

        # concatenate with the first alpha
        alpha_full_outputs = tt.concatenate((alpha_first.dimshuffle('x', 0), alpha_outputs))

        # last entry of the backward table (zero for all states)
        beta_last = tt.zeros_like(log_prior_c)

        # the rest of the backward table
        beta_outputs, beta_updates = th.scan(
            fn=calculate_prev_beta,
            sequences=[log_trans_tcc, log_emission_tc[1:, :]],
            go_backwards=True,
            outputs_info=[beta_last])

        # concatenate with the last beta and reverse
        beta_full_outputs = tt.concatenate((beta_last.dimshuffle('x', 0), beta_outputs))[::-1, :]
        log_unnormalized_posterior_probs = alpha_full_outputs + beta_full_outputs
        log_data_likelihood = pm.math.logsumexp(log_unnormalized_posterior_probs, axis=1)
        log_posterior_probs = log_unnormalized_posterior_probs - log_data_likelihood

        return log_posterior_probs, log_data_likelihood.dimshuffle(0)
