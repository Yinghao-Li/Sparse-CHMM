import logging
import numpy as np
from typing import Optional

import torch
import torch.nn as nn
from torch.distributions.dirichlet import Dirichlet

from seqlbtoolkit.data import label_to_span
from seqlbtoolkit.training.eval import Metric

from .math import (
    log_matmul,
    log_maxmul,
    validate_prob,
    logsumexp,
    prob_scaling,
    entity_emiss_diag,
    entity_emiss_o,
    entity_emiss_nondiag
)
from .args import Config

logger = logging.getLogger(__name__)


class SparseCHMMMetric(Metric):
    def __init__(self, conc_o2o=None, conc_l2l=None, conc_l2l_batch=None, swxor=None):
        super(SparseCHMMMetric, self).__init__()
        self.conc_o2o = conc_o2o
        self.conc_l2l = conc_l2l
        self.swxor = swxor
        self.conc_l2l_batch = conc_l2l_batch


class TransitionModule(nn.Module):
    def __init__(self, config: Config):
        super(TransitionModule, self).__init__()
        self._d_hidden = config.d_hidden

        # transition network layers
        self.neural_transition = nn.Linear(config.d_emb, config.d_hidden * config.d_hidden)

        nn.init.xavier_uniform_(self.neural_transition.weight.data)

    def forward(self, embs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        embs: BERT sequence embeddings

        Returns
        -------
        Transition probabilities
        """
        batch_size, max_seq_length, _ = embs.size()
        trans_logits = self.neural_transition(embs).view(
            batch_size, max_seq_length, self._d_hidden, self._d_hidden
        )
        trans_probs = torch.softmax(trans_logits, dim=-1)
        return trans_probs


class DirParamBaseModule(nn.Module):
    def __init__(self, config: Config):
        super(DirParamBaseModule, self).__init__()

        self._n_src = config.n_src
        self._d_obs = config.d_obs
        self._device = config.device

        self._ent_exp1 = config.diag_exp_t1
        self._ent_exp2 = config.diag_exp_t2
        self._nondiag_exp = config.nondiag_exp
        self._nondiag_r = config.nondiag_split_ratio
        self._nondiag_r_decay = config.nondiag_split_decay

        self._reliab_level = config.reliability_level

        # network layers
        self.emission_2o = nn.Linear(config.d_emb, self._n_src)
        if self._reliab_level == 'label':
            self.emission_2e = nn.Linear(config.d_emb, self._n_src * (config.n_lbs - 1))  # excludes "O"
        elif self._reliab_level == 'entity':
            self.emission_2e = nn.Linear(config.d_emb, self._n_src * config.n_ent)

        self._source_softmax = nn.Softmax(dim=-2)
        self._sigmoid = nn.Sigmoid()

        # define emission prior indices
        self._l2l_diag_idx = torch.eye(self._d_obs, dtype=torch.bool)
        self._l2l_diag_idx[0, 0] = False
        self._l2l_nondiag_idx = ~torch.eye(self._d_obs, dtype=torch.bool)
        self._l2l_nondiag_idx[0, :] = False
        self._l2l_nondiag_idx[:, 0] = False

        # parameter weight initialization
        nn.init.xavier_uniform_(self.emission_2o.weight.data)
        nn.init.xavier_uniform_(self.emission_2e.weight.data)

    def forward(self, embs: torch.Tensor, apply_ratio_decay: Optional[bool] = False):
        batch_size, max_seq_length, _ = embs.size()

        # initialize the dirichlet parameter matrix; shape: batch_size X n_src X d_obs X d_obs
        diric_params = torch.zeros(
            [batch_size, self._n_src, self._d_obs, self._d_obs], device=self._device
        ) + 1E-9  # Add a small value to guarantee positivity. Should not be necessary but nice to have

        # get Dirichlet parameters
        # assume the reliability of a source remains constant across the whole sequence
        # TODO: incorporate token-dependency
        # predict the concentration parameter for o2o from sequence embedding; batch_size X n_src
        conc_o2o = self._sigmoid(self.emission_2o(embs[:, 0, :]))

        # assign the first row of the dirichlet parameter matrix
        diric_params[:, :, 0, 0] = conc_o2o
        diric_params[:, :, 0, 1:] = ((1 - conc_o2o) / (self._d_obs - 1)).unsqueeze(-1)

        # predict the concentration parameter for e2e; batch_size X n_src X (n_lbs or n_ent)
        conc_e2e = self.emission_2e(embs[:, 0, :]).view([batch_size, self._n_src, -1])
        conc_e2e_norm = self._source_softmax(conc_e2e)
        # scale up the result
        conc_e2e_scale = prob_scaling(conc_e2e_norm, 1 / self._n_src, self._ent_exp1, self._ent_exp2)

        if self._reliab_level == 'label':
            conc_l2l = conc_e2e_scale
        elif self._reliab_level == 'entity':
            conc_l2l = conc_e2e_scale.repeat_interleave(2, dim=-1)
        else:
            raise ValueError(f"Unknown reliability score level: {self._reliab_level}")

        # assign the l2l emission probabilities
        diric_params[:, :, self._l2l_diag_idx] = entity_emiss_diag(conc_l2l)

        # get non-diagonal l2l emissions (heuristic)
        # Rayleigh distribution is hard to design. Use piecewise polynomial instead
        if not apply_ratio_decay:
            split_point = self._nondiag_r / self._d_obs
        else:
            split_point = self._nondiag_r * self._nondiag_r_decay / self._d_obs

        conc_l2l_nondiag = entity_emiss_nondiag(
            x=conc_l2l, n_lbs=self._d_obs, tp=split_point, exp_term=self._nondiag_exp
        ).detach()

        # assign the non-diagonal e2e emission probabilities
        diric_params[:, :, self._l2l_nondiag_idx] = conc_l2l_nondiag.repeat_interleave(self._d_obs - 2, dim=-1)

        # the first column other than [0,0]
        l2o_values = entity_emiss_o(
            x=conc_l2l, n_lbs=self._d_obs, tp=split_point, exp_term=self._nondiag_exp
        )

        diric_params[:, :, 1:, 0] = l2o_values

        return diric_params, (conc_o2o, conc_l2l)


class DirParamExpanModule(nn.Module):
    def __init__(self, config: Config):
        super(DirParamExpanModule, self).__init__()

        self._n_src = config.n_src

        # network layers
        self.emission_expand = nn.Linear(config.d_emb, config.n_src * (config.n_lbs - 1))
        self._sigmoid = nn.Sigmoid()

        # initialize parameters
        nn.init.xavier_uniform_(self.emission_expand.weight.data)

    def forward(self,
                embs: torch.Tensor,
                wxor_lut: torch.Tensor):
        """
        Parameters
        ----------
        embs: Bert sequence embeddings, batch_size X max_len_seq X d_emb
        wxor_lut: weighted xor loop-up table, n_src X n_lbs X n_lbs
        """
        batch_size = embs.shape[0]

        # predict the addon scaling factor for emission parameters; batch_size X n_src X (n_lbs-1)
        # batch_size X n_src X n_lbs (l^query)
        wxor_scale_factor = self._sigmoid(self.emission_expand(embs[:, 0, :])).view([batch_size, self._n_src, -1])
        wxor_potential = wxor_scale_factor.unsqueeze(-1) * wxor_lut.unsqueeze(0)

        return wxor_potential.transpose(-1, -2)


class NeuralModule(nn.Module):
    def __init__(self,
                 config: Config):
        super(NeuralModule, self).__init__()

        self.transition_module = TransitionModule(config)
        self.dir_param_base_module = DirParamBaseModule(config)
        self.dir_param_expan_module = DirParamExpanModule(config)

        self._disable_dirichlet = config.disable_dirichlet
        self._rel_level = config.reliability_level

        # Dirichlet sampling concentration parameter base and range values
        self._concentration_base = config.dirichlet_conc_base
        self._concentration_range = config.dirichlet_conc_max - config.dirichlet_conc_base

        # weighted xor look-up table
        self._wxor_lut = None

    @property
    def wxor_lut(self):
        return self._wxor_lut

    @wxor_lut.setter
    def wxor_lut(self, wxor: torch.Tensor):
        self._wxor_lut = wxor

    def forward(self,
                embs: torch.Tensor,
                add_wxor_lut: Optional[bool] = False,
                sample_emiss: Optional[bool] = True,
                apply_ratio_decay: Optional[bool] = False):

        trans_probs = self.transition_module(embs)
        diric_params, (conc_o2o, conc_l2l) = self.dir_param_base_module(embs, apply_ratio_decay=apply_ratio_decay)

        if self._disable_dirichlet:
            return trans_probs, diric_params, (conc_o2o, conc_l2l)

        if add_wxor_lut:
            wxor_mat = torch.zeros_like(diric_params)
            scaled_wxor = self.dir_param_expan_module(embs, self._wxor_lut)
            wxor_mat[:, :, 1:, 1:] = scaled_wxor
            diric_params = diric_params + wxor_mat
        else:
            scaled_wxor = None

        ranged_diric_params = diric_params * self._concentration_range + self._concentration_base
        if sample_emiss:
            # construct dirichlet distribution
            dirichlet_distr = Dirichlet(ranged_diric_params)
            # sample from distribution
            emiss_probs = dirichlet_distr.rsample()
        else:
            emiss_probs = ranged_diric_params / ranged_diric_params.sum(dim=-1, keepdim=True)

        return trans_probs, emiss_probs, (conc_o2o, conc_l2l, scaled_wxor)


class SparseCHMM(nn.Module):

    def __init__(self, config: Config, state_prior=None):
        super(SparseCHMM, self).__init__()

        self._n_src = config.n_src
        self._d_obs = config.d_obs  # number of possible obs_set
        self._d_hidden = config.d_hidden  # number of states

        self._device = config.device

        self.neural_module = NeuralModule(config)

        # initialize unnormalized state-prior
        self._initialize_model(state_prior=state_prior)
        self.to(self._device)

        self._inter_results = SparseCHMMMetric()

    @property
    def log_trans(self):
        try:
            return self._log_trans
        except NameError:
            logger.error('DirCHMM.log_trans is not defined!')
            return None

    @property
    def log_emiss(self):
        try:
            return self._log_emiss
        except NameError:
            logger.error('DirCHMM.log_emiss is not defined!')
            return None

    @property
    def state_priors(self):
        return self._state_priors

    @property
    def inter_results(self) -> "SparseCHMMMetric":
        return self._inter_results

    def pop_inter_results(self) -> "SparseCHMMMetric":
        result = self._inter_results
        self._inter_results = SparseCHMMMetric()
        return result

    def _initialize_model(self, state_prior: torch.Tensor):
        """
        Initialize model parameters

        Parameters
        ----------
        state_prior: state prior (pi)

        Returns
        -------
        self
        """

        logger.info('Initializing Dirichlet CHMM...')

        if state_prior is None:
            priors = torch.zeros(self._d_hidden, device=self._device) + 1E-3
            priors[0] = 1
            self._state_priors = nn.Parameter(torch.log(priors))
        else:
            state_prior.to(self._device)
            priors = validate_prob(state_prior, dim=0)
            self._state_priors = nn.Parameter(torch.log(priors))

        logger.info("Dirichlet CHMM initialized!")

        return self

    def _initialize_states(self,
                           embs: torch.Tensor,
                           obs: torch.Tensor,
                           src_usg_ids: Optional[list[int]] = None,
                           add_wxor_lut: Optional[bool] = False,
                           apply_ratio_decay: Optional[bool] = False,
                           sample_emiss: Optional[bool] = True,
                           track_conc_params: Optional[bool] = True):
        """
        Initialize inference states. Should be called before forward inference.

        Parameters
        ----------
        embs: token embeddings
        obs: observations
        src_usg_ids: the indices of the sources being used in inference. Leave None if use all sources
        add_wxor_lut: whether use weighted xor look-up table in the process
        apply_ratio_decay: whether apply split ratio decay to the dirichlet base neural module
        sample_emiss: whether sample emission matrix from the Dirichlet distribution

        Returns
        -------
        self
        """
        # normalize and put the probabilities into the log domain
        batch_size, max_seq_length, n_src, _ = obs.size()
        self._log_state_priors = torch.log_softmax(self._state_priors, dim=-1)

        # get neural transition and emission matrices
        # TODO: we can add layer-norm later to see what happens
        nn_trans, nn_emiss, (conc_o2o, conc_l2l, swxor) = self.neural_module(
            embs=embs,
            sample_emiss=sample_emiss,
            add_wxor_lut=add_wxor_lut,
            apply_ratio_decay=apply_ratio_decay
        )

        self._log_trans = torch.log(nn_trans)
        self._log_emiss = torch.log(nn_emiss)

        if track_conc_params:
            # save the record
            self._inter_results.conc_l2l_batch = conc_l2l
            if self._inter_results.conc_o2o is None:
                self._inter_results.conc_o2o = conc_o2o.detach().cpu().numpy()
                self._inter_results.conc_l2l = conc_l2l.detach().cpu().numpy()
            else:
                self._inter_results.conc_o2o = np.r_[self._inter_results.conc_o2o, conc_o2o.detach().cpu().numpy()]
                self._inter_results.conc_l2l = np.r_[self._inter_results.conc_l2l, conc_l2l.detach().cpu().numpy()]
                
            if self._inter_results.swxor is None:
                if swxor is not None:
                    self._inter_results.swxor = swxor.detach().cpu().numpy()
            else:
                if swxor is not None:
                    self._inter_results.swxor = np.r_[self._inter_results.swxor, swxor.detach().cpu().numpy()]

        # only keep observations of desired sources during inference
        if src_usg_ids:
            log_emiss_ = self._log_emiss[:, src_usg_ids, :, :]
            obs_ = obs[:, :, src_usg_ids, :]
        else:
            log_emiss_ = self._log_emiss
            obs_ = obs
        # Calculate the emission probabilities in one time, so that we don't have to compute this repeatedly
        # log-domain subtract is regular-domain divide
        self._log_emiss_evidence = log_matmul(
            log_emiss_.unsqueeze(1), torch.log(obs_).unsqueeze(-1)
        ).squeeze(-1).sum(dim=-2)

        self._log_alpha = torch.zeros([batch_size, max_seq_length, self._d_hidden], device=self._device)
        self._log_beta = torch.zeros([batch_size, max_seq_length, self._d_hidden], device=self._device)
        # Gamma can be readily computed and need no initialization
        self._log_gamma = None
        # only values in 1:max_seq_length are valid. The first state is a dummy
        self._log_xi = torch.zeros([batch_size, max_seq_length, self._d_hidden, self._d_hidden], device=self._device)
        return self

    def _forward_step(self, t):
        # initial alpha state
        if t == 0:
            log_alpha_t = self._log_state_priors + self._log_emiss_evidence[:, t, :]
        # do the forward step
        else:
            log_alpha_t = self._log_emiss_evidence[:, t, :] + \
                          log_matmul(self._log_alpha[:, t - 1, :].unsqueeze(1), self._log_trans[:, t, :, :]).squeeze(1)

        # normalize the result
        normalized_log_alpha_t = log_alpha_t - log_alpha_t.logsumexp(dim=-1, keepdim=True)
        return normalized_log_alpha_t

    def _backward_step(self, t):
        # do the backward step
        # beta is not a distribution, so we do not need to normalize it
        log_beta_t = log_matmul(
            self._log_trans[:, t, :, :],
            (self._log_emiss_evidence[:, t, :] + self._log_beta[:, t + 1, :]).unsqueeze(-1)
        ).squeeze(-1)
        return log_beta_t

    def _forward_backward(self, seq_lengths):
        max_seq_length = seq_lengths.max().item()
        # calculate log alpha
        for t in range(0, max_seq_length):
            self._log_alpha[:, t, :] = self._forward_step(t)

        # calculate log beta
        # The last beta state beta[:, -1, :] = log1 = 0, so no need to re-assign the value
        for t in range(max_seq_length - 2, -1, -1):
            self._log_beta[:, t, :] = self._backward_step(t)
        # shift the output (since beta is calculated in backward direction,
        # we need to shift each instance in the batch according to its length)
        shift_distances = seq_lengths - max_seq_length
        self._log_beta = torch.stack(
            [torch.roll(beta, s.item(), 0) for beta, s in zip(self._log_beta, shift_distances)]
        )
        return None

    def _compute_xi(self, t):
        temp_1 = self._log_emiss_evidence[:, t, :] + self._log_beta[:, t, :]
        temp_2 = log_matmul(self._log_alpha[:, t - 1, :].unsqueeze(-1), temp_1.unsqueeze(1))
        log_xi_t = self._log_trans[:, t, :, :] + temp_2
        return log_xi_t

    def _expected_complete_log_likelihood(self, seq_lengths):
        batch_size = len(seq_lengths)
        max_seq_length = seq_lengths.max().item()

        # calculate expected sufficient statistics: gamma_t(j) = P(z_t = j|x_{1:T})
        self._log_gamma = self._log_alpha + self._log_beta
        # normalize as gamma is a distribution
        log_gamma = self._log_gamma - self._log_gamma.logsumexp(dim=-1, keepdim=True)

        # calculate expected sufficient statistics: psi_t(i, j) = P(z_{t-1}=i, z_t=j|x_{1:T})
        for t in range(1, max_seq_length):
            self._log_xi[:, t, :, :] = self._compute_xi(t)
        stabled_norm_term = logsumexp(self._log_xi[:, 1:, :, :].view(batch_size, max_seq_length - 1, -1), dim=-1) \
            .view(batch_size, max_seq_length - 1, 1, 1)
        log_xi = self._log_xi[:, 1:, :, :] - stabled_norm_term

        # calculate the expected complete data log likelihood
        log_prior = torch.sum(torch.exp(log_gamma[:, 0, :]) * self._log_state_priors, dim=-1)
        log_prior = log_prior.mean()
        # sum over j, k
        log_tran = torch.sum(torch.exp(log_xi) * self._log_trans[:, 1:, :, :], dim=[-2, -1])
        # sum over valid time steps, and then average over batch. Note this starts from t=2
        log_tran = torch.mean(torch.stack([inst[:length].sum() for inst, length in zip(log_tran, seq_lengths - 1)]))
        # same as above
        log_emis = torch.sum(torch.exp(log_gamma) * self._log_emiss_evidence, dim=-1)
        log_emis = torch.mean(torch.stack([inst[:length].sum() for inst, length in zip(log_emis, seq_lengths)]))
        log_likelihood = log_prior + log_tran + log_emis

        return log_likelihood

    def forward(self,
                emb: torch.Tensor,
                obs: torch.Tensor,
                seq_lengths: torch.Tensor,
                src_usg_ids: Optional[list[int]] = None,
                add_wxor_lut: Optional[bool] = False,
                apply_ratio_decay: Optional[bool] = False,
                sample_emiss: Optional[bool] = True,
                track_conc_params: Optional[bool] = True):

        # the row of obs should be one-hot or at least sum to 1
        # assert (obs.sum(dim=-1) == 1).all()

        batch_size, max_seq_length, n_src, n_obs = obs.size()
        assert n_obs == self._d_obs
        assert n_src == self._n_src

        # Initialize alpha, beta and xi
        self._initialize_states(embs=emb,
                                obs=obs,
                                src_usg_ids=src_usg_ids,
                                add_wxor_lut=add_wxor_lut,
                                apply_ratio_decay=apply_ratio_decay,
                                sample_emiss=sample_emiss,
                                track_conc_params=track_conc_params)
        self._forward_backward(seq_lengths=seq_lengths)
        log_likelihood = self._expected_complete_log_likelihood(seq_lengths=seq_lengths)
        return log_likelihood, (self.log_trans, self.log_emiss)

    def viterbi(self,
                emb: torch.Tensor,
                obs: torch.Tensor,
                seq_lengths: torch.Tensor,
                src_usg_ids: Optional[list[int]] = None,
                add_wxor_lut: Optional[bool] = False,
                apply_ratio_decay: Optional[bool] = False,
                sample_emiss: Optional[bool] = False):
        """
        Find argmax_z log p(z|obs) for each (obs) in the batch.
        """
        batch_size = len(seq_lengths)
        max_seq_length = seq_lengths.max().item()

        # initialize states
        self._initialize_states(embs=emb,
                                obs=obs,
                                src_usg_ids=src_usg_ids,
                                add_wxor_lut=add_wxor_lut,
                                apply_ratio_decay=apply_ratio_decay,
                                sample_emiss=sample_emiss)
        # maximum probabilities
        log_delta = torch.zeros([batch_size, max_seq_length, self._d_hidden], device=self._device)
        # most likely previous state on the most probable path to z_t = j. a[0] is undefined.
        pre_states = torch.zeros([batch_size, max_seq_length, self._d_hidden], dtype=torch.long, device=self._device)

        # the initial delta state
        log_delta[:, 0, :] = self._log_state_priors + self._log_emiss_evidence[:, 0, :]
        for t in range(1, max_seq_length):
            # udpate delta and a. The location of the emission probabilities does not matter
            max_log_prob, argmax_val = log_maxmul(
                log_delta[:, t - 1, :].unsqueeze(1),
                self._log_trans[:, t, :, :] + self._log_emiss_evidence[:, t, :].unsqueeze(1)
            )
            log_delta[:, t, :] = max_log_prob.squeeze(1)
            pre_states[:, t, :] = argmax_val.squeeze(1)

        # The terminal state
        batch_max_log_prob = list()
        batch_z_t_star = list()

        for l_delta, length in zip(log_delta, seq_lengths):
            max_log_prob, z_t_star = l_delta[length - 1, :].max(dim=-1)
            batch_max_log_prob.append(max_log_prob)
            batch_z_t_star.append(z_t_star)

        # Trace back
        batch_z_star = [[z_t_star.item()] for z_t_star in batch_z_t_star]
        for p_states, z_star, length in zip(pre_states, batch_z_star, seq_lengths):
            for t in range(length - 2, -1, -1):
                z_t = p_states[t + 1, z_star[0]].item()
                z_star.insert(0, z_t)

        # compute the smoothed marginal p(z_t = j | obs_{1:T})
        self._forward_backward(seq_lengths)
        log_marginals = self._log_alpha + self._log_beta
        norm_marginals = torch.exp(log_marginals - logsumexp(log_marginals, dim=-1, keepdim=True))
        batch_marginals = list()
        for marginal, length in zip(norm_marginals, seq_lengths):
            mgn_list = marginal[:length].detach().cpu().numpy()
            batch_marginals.append(mgn_list)

        return batch_z_star, batch_marginals

    def annotate(self, emb, obs, seq_lengths, label_types):
        batch_label_indices, batch_probs = self.viterbi(emb, obs, seq_lengths)
        batch_labels = [[label_types[lb_index] for lb_index in label_indices]
                        for label_indices in batch_label_indices]

        # For batch_spans, we are going to compare them with the true spans,
        # and the true spans is already shifted, so we do not need to shift predicted spans back
        batch_spans = list()
        batch_scored_spans = list()
        for labels, probs, indices in zip(batch_labels, batch_probs, batch_label_indices):
            spans = label_to_span(labels)
            batch_spans.append(spans)

            ps = [p[s] for p, s in zip(probs, indices[1:])]
            scored_spans = dict()
            for k, v in spans.items():
                if k == (0, 1):
                    continue
                start = k[0] - 1 if k[0] > 0 else 0
                end = k[1] - 1
                score = np.mean(ps[start:end])
                scored_spans[(start, end)] = [(v, score)]
            batch_scored_spans.append(scored_spans)

        return batch_spans, (batch_scored_spans, batch_probs)
