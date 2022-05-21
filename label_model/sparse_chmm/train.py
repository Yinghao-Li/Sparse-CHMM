import sys
sys.path.append('../..')

import os
import gc
import logging
import numpy as np
from tqdm.auto import tqdm
from typing import Optional

import torch
from torch.nn import functional as F

from seqlbtoolkit.base_model.eval import Metric, get_ner_metrics
from seqlbtoolkit.chmm.train import CHMMBaseTrainer

from .math import get_dataset_wxor
from .args import SparseCHMMConfig
from .model import SparseCHMM, SparseCHMMMetric
from .dataset import CHMMDataset
from .macro import *

logger = logging.getLogger(__name__)


class SparseCHMMTrainer(CHMMBaseTrainer):
    def __init__(self,
                 config: SparseCHMMConfig,
                 collate_fn=None,
                 training_dataset=None,
                 valid_dataset=None,
                 test_dataset=None,
                 pretrain_optimizer=None,
                 optimizer=None):

        super().__init__(
            config, collate_fn, training_dataset, valid_dataset, test_dataset, pretrain_optimizer, optimizer
        )

    @property
    def config(self) -> SparseCHMMConfig:
        return self._config

    @property
    def neural_module(self):
        return self._model.neural_module

    def initialize_trainer(self):
        """
        Initialize necessary components for training, returns self
        """
        CHMMBaseTrainer.initialize_trainer(self)
        return self

    def initialize_model(self):
        self._model = SparseCHMM(
            config=self.config,
            state_prior=self._init_state_prior
        )
        return self

    def pretrain_step(self,
                      data_loader,
                      optimizer,
                      trans_: Optional[torch.Tensor] = None,
                      emiss_: Optional[torch.Tensor] = None,
                      add_wxor_lut: Optional[bool] = False,
                      apply_ratio_decay: Optional[bool] = False):

        self.neural_module.train()
        if trans_ is not None:
            trans_ = trans_.to(self.config.device)
        if emiss_ is not None:
            emiss_ = emiss_.to(self.config.device)

        num_iters = len(data_loader)
        use_part_data = False
        train_iter_ids = np.arange(num_iters)
        # select a sub-dataset for training
        if self.config.training_ratio_per_epoch is not None and self.config.training_ratio_per_epoch < 1:
            use_part_data = True
            train_iter_ids = np.random.choice(
                num_iters, size=round(self.config.training_ratio_per_epoch * num_iters), replace=False
            )

        train_loss = 0
        num_samples = 0

        for i, batch in enumerate(tqdm(data_loader)):
            # skip unselected batches
            if use_part_data and (i not in train_iter_ids):
                continue

            emb_batch, obs_batch, seq_lens = map(lambda x: x.to(self.config.device), batch[:3])
            batch_size = len(obs_batch)
            num_samples += batch_size

            optimizer.zero_grad()
            nn_trans, nn_emiss, _ = self.neural_module(
                embs=emb_batch, add_wxor_lut=add_wxor_lut, apply_ratio_decay=apply_ratio_decay
            )
            batch_size, max_seq_len, n_hidden, _ = nn_trans.size()

            if trans_ is not None and nn_trans is not None:
                loss_mask = torch.zeros([batch_size, max_seq_len], device=self.config.device)
                for n in range(batch_size):
                    loss_mask[n, :seq_lens[n]] = 1
                trans_mask = loss_mask.view(batch_size, max_seq_len, 1, 1)
                trans_pred = trans_mask * nn_trans
                trans_true = trans_mask * trans_.view(1, 1, n_hidden, n_hidden).repeat(batch_size, max_seq_len, 1, 1)

                l1 = F.mse_loss(trans_pred, trans_true)
            else:
                l1 = 0

            if emiss_ is not None and nn_emiss is not None:
                emiss_pred = nn_emiss
                emiss_true = emiss_.view(
                    1, self.config.n_src, self.config.d_hidden, self.config.d_obs
                ).repeat(batch_size, 1, 1, 1)

                l2 = F.mse_loss(emiss_pred, emiss_true)
            else:
                l2 = 0

            loss = l1 + l2
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_size
        train_loss /= num_samples
        return train_loss

    def training_step(self,
                      data_loader,
                      optimizer,
                      src_usg_ids: Optional[list[int]] = None,
                      add_wxor_lut: Optional[bool] = False,
                      apply_ratio_decay: Optional[bool] = False,
                      track_conc_params: Optional[bool] = True):

        self._model.train()

        num_iters = len(data_loader)
        use_part_data = False
        train_iter_ids = np.arange(num_iters)
        # select a sub-dataset for training
        if self.config.training_ratio_per_epoch is not None and self.config.training_ratio_per_epoch < 1:
            use_part_data = True
            train_iter_ids = np.random.choice(
                num_iters, size=round(self.config.training_ratio_per_epoch * num_iters), replace=False
            )

        train_loss = 0
        num_samples = 0

        for i, batch in enumerate(tqdm(data_loader)):
            # skip unselected batches
            if use_part_data and (i not in train_iter_ids):
                continue

            # get data
            emb_batch, obs_batch, seq_lens = map(lambda x: x.to(self.config.device), batch[:3])
            batch_size = len(obs_batch)
            num_samples += batch_size

            # training step
            optimizer.zero_grad()
            log_probs, _ = self._model(
                emb=emb_batch,
                obs=obs_batch,
                seq_lengths=seq_lens,
                src_usg_ids=src_usg_ids,
                normalize_observation=self.config.obs_normalization,
                add_wxor_lut=add_wxor_lut,
                track_conc_params=track_conc_params,
                apply_ratio_decay=apply_ratio_decay
            )
            loss = -log_probs.mean()

            if self.config.apply_entity_emiss_loss:
                l2l_emiss = self._model.inter_results.conc_l2l_batch
                loss = loss + torch.sum((l2l_emiss[:, :, ::2] - l2l_emiss[:, :, 1::2]) ** 2)

            loss.backward()
            optimizer.step()

            # track loss
            train_loss += loss.item() * batch_size
        train_loss /= num_samples

        return train_loss

    def train(self):

        logger.info("---Stage 1---")
        if self.config.load_s1_model:
            try:
                logger.info("loading stage-1 model")
                self.load(model_name='chmm-stage1')
            except Exception as err:
                logger.exception(f"Failed to load stage-1 model: {err}; will train from the beginning")
                self.initialize_model()
                self.initialize_optimizers()

                stage1_valid_results = self.stage1()
                self.save_results(output_dir=self.config.output_dir,
                                  valid_results=stage1_valid_results,
                                  file_name="dir-chmm-stage1")
        else:
            stage1_valid_results = self.stage1()
            self.save_results(output_dir=self.config.output_dir,
                              valid_results=stage1_valid_results,
                              file_name="dir-chmm-stage1")

        gc.collect()
        torch.cuda.empty_cache()

        if self.config.include_s2:
            logger.info("---Stage 2---")
            s2_loaded = False
            if self.config.load_s2_model:
                try:
                    logger.info("loading stage-2 model")
                    self.load(model_name='chmm-stage2', load_wxor=True)
                    s2_loaded = True
                except Exception as err:
                    logger.exception(f"Failed to load stage-2 model: {err}; will train from stage-1")
                    self.load(model_name='chmm-stage1')

            if not s2_loaded:
                stage2_valid_results = self.stage2()
                self.save_results(output_dir=self.config.output_dir,
                                  valid_results=stage2_valid_results,
                                  file_name="dir-chmm-stage2",
                                  add_wxor_lut=True,
                                  apply_ratio_decay=True)

        gc.collect()
        torch.cuda.empty_cache()

        if self.config.include_s3:
            logger.info("---Stage 3---")
            s3_loaded = False
            if self.config.load_s3_model:
                try:
                    logger.info("loading stage-3 model")
                    self.load(model_name='chmm-stage3')
                    s3_loaded = True
                except Exception as err:
                    logger.exception(f"Failed to load stage-3 model: {err}")
                    self.load(model_name='chmm-stage2')

            if not s3_loaded:
                stage3_valid_results = self.stage3()
                self.save_results(output_dir=self.config.output_dir,
                                  valid_results=stage3_valid_results,
                                  file_name="dir-chmm-stage3",
                                  add_wxor_lut=True,
                                  apply_ratio_decay=True)

        gc.collect()
        torch.cuda.empty_cache()

        return self

    def stage1(self):
        """
        Train the Transition matrix and the Dirichlet base model
        """
        self.unfreeze_trans().unfreeze_dir_param_base()

        if self.config.transduction:
            training_dataloader = self.get_dataloader(self._test_dataset, shuffle=True)
        else:
            training_dataloader = self.get_dataloader(self._training_dataset, shuffle=True)

        # ----- pre-train neural module -----
        if self.config.num_lm_nn_pretrain_epochs > 0:
            logger.info(" ----- ")
            logger.info("Pre-training neural module...")
            for epoch_i in range(self.config.num_lm_nn_pretrain_epochs):
                train_loss = self.pretrain_step(
                    training_dataloader, self._pretrain_optimizer, self._init_trans_mat, self._init_emiss_mat
                )
                logger.info(f"Epoch: {epoch_i}, Loss: {train_loss}")
            logger.info("Neural module pretrained!")

        valid_results = SparseCHMMMetric()
        best_metric = 0
        metric_buffer = list()
        tolerance_epoch = 0

        # ----- start training process -----
        logger.info(" ----- ")
        logger.info("Training Dir-CHMM...")
        for epoch_i in range(self.config.num_lm_train_epochs):
            logger.info("------")
            logger.info(f"Epoch {epoch_i + 1} of {self.config.num_lm_train_epochs}")

            train_loss = self.training_step(training_dataloader, self._optimizer)
            logger.info(f"Training loss: {train_loss:.4f}")

            self._model.pop_inter_results()  # remove training inter results
            valid_metrics = self.valid()  # get and log validation results

            # ----- save model -----
            metric_buffer.append(valid_metrics['f1'])
            if len(metric_buffer) == self.config.num_lm_valid_smoothing + 1:
                metric_buffer.pop(0)
                curr_metric = np.mean(metric_buffer)
            else:
                curr_metric = -np.inf

            if curr_metric >= best_metric:
                self.save(model_name='chmm-stage1')
                logger.info("Checkpoint Saved!")
                best_metric = curr_metric
                tolerance_epoch = 0
            else:
                tolerance_epoch += 1

            # ----- log history -----
            inter_results = self._model.pop_inter_results()  # get validation inter results
            inter_results.conc_l2l = inter_results.conc_l2l.mean(axis=0)
            inter_results.conc_o2o = inter_results.conc_o2o.mean(axis=0)
            inter_results.conc_l2l_batch = None
            inter_results.swxor = None

            valid_results.append(valid_metrics).append(inter_results)
            if tolerance_epoch > self.config.num_lm_valid_tolerance:
                logger.info("Training stopped because of exceeding tolerance")
                break

        # retrieve the best state dict
        self.load(model_name='chmm-stage1')

        return valid_results

    def stage2(self):
        """
        Freeze the transition matrix, train the emission addon model;
        Freezing the dirichlet emission base model is optional.
        """
        logger.info("Constructing initial emission matrix from the previous stage...")
        xor_cal_dataset = self._valid_dataset if self.config.calculate_wxor_on_valid else self._training_dataset
        trans_mat_pred, emiss_mat_pred = self.get_trans_and_emiss(xor_cal_dataset)
        del trans_mat_pred  # delete this variable as it is never used
        emiss_mat_pred = emiss_mat_pred.mean(dim=0)
        init_emiss_mat = 0.8 * emiss_mat_pred + 0.2 * self._init_emiss_mat

        # calculate wxor_lut and assign it to the neural module
        if self.neural_module.wxor_lut is None:
            logger.info("Calculating weighted xor lookup table...")
            self.neural_module.wxor_lut = self.get_weighted_xor_lut(xor_cal_dataset)

        logger.info("Initializing parameter status")
        # freeze transition and base emission parameters
        # Should only pre-train the addon matrix.
        self.freeze_trans().freeze_dir_param_base()

        # remove any inter results stored in the model
        self._model.pop_inter_results()

        self.initialize_optimizers()  # re-initialize optimizers
        for g in self._optimizer.param_groups:
            g['lr'] *= self.config.s2_lr_decay  # decrease the learning rate for more stable training

        if self.config.transduction:
            training_dataloader = self.get_dataloader(self._test_dataset, shuffle=True)
        else:
            training_dataloader = self.get_dataloader(self._training_dataset, shuffle=True)

        # ----- pre-train neural module -----
        if self.config.num_lm_nn_pretrain_epochs > 0:
            logger.info(" ----- ")
            logger.info("Pre-training neural module...")
            for epoch_i in range(self.config.num_lm_nn_pretrain_epochs):
                train_loss = self.pretrain_step(
                    data_loader=training_dataloader,
                    optimizer=self._pretrain_optimizer,
                    emiss_=init_emiss_mat,
                    add_wxor_lut=True,
                    apply_ratio_decay=False
                )
                logger.info(f"Epoch: {epoch_i}, Loss: {train_loss}")
            logger.info("Neural module pretrained!")

        # The dir base parameters should only be trained during the EM optimization process
        if not self.config.freeze_s2_base_emiss:
            self.unfreeze_dir_param_base()

        # collect and dump garbage
        gc.collect()
        torch.cuda.empty_cache()

        valid_results = SparseCHMMMetric()
        best_metric = 0
        tolerance_epoch = 0

        # ----- start training process -----
        logger.info(" ----- ")
        logger.info("Training Dir-CHMM...")
        for epoch_i in range(self.config.num_lm_train_epochs):
            logger.info("------")
            logger.info(f"Epoch {epoch_i + 1} of {self.config.num_lm_train_epochs}")

            train_loss = self.training_step(
                data_loader=training_dataloader,
                optimizer=self._optimizer,
                add_wxor_lut=True,
                apply_ratio_decay=True
            )
            logger.info(f"Training loss: {train_loss:.4f}")
            self._model.pop_inter_results()  # remove training inter results

            valid_metrics = self.valid(
                add_wxor_lut=True,
                apply_ratio_decay=True
            )  # get and log validation results

            # ----- save model -----
            curr_metric = valid_metrics['f1']

            if curr_metric >= best_metric:
                self.save(model_name='chmm-stage2')
                logger.info("Checkpoint Saved!")
                best_metric = curr_metric
                tolerance_epoch = 0
            else:
                tolerance_epoch += 1

            # ----- log history -----
            inter_results = self._model.pop_inter_results()  # get validation inter results
            inter_results.conc_l2l = None
            inter_results.conc_o2o = None
            inter_results.conc_l2l_batch = None
            inter_results.swxor = inter_results.swxor.mean(axis=0)

            valid_results.append(valid_metrics).append(inter_results)
            if tolerance_epoch > self.config.num_lm_s2_valid_tolerance:
                logger.info("Training stopped because of exceeding tolerance")
                break

        # retrieve the best state dict
        self.load(model_name='chmm-stage2', load_wxor=True)

        return valid_results

    def stage3(self):
        """
        Freeze the emission model, refine the transition matrix
        """

        # calculate wxor_lut and assign it to the neural module
        assert self.neural_module.wxor_lut is not None, ValueError('`wxor_lut` is not defined!')

        logger.info("Initializing parameter status")
        # freeze transition and base emission parameters
        # Should only pre-train the addon matrix.
        self.freeze_emiss()

        self.initialize_optimizers()  # re-initialize optimizers
        for g in self._optimizer.param_groups:
            g['lr'] *= self.config.s3_lr_decay  # decrease the learning rate for more stable training

        if self.config.transduction:
            training_dataloader = self.get_dataloader(self._test_dataset, shuffle=True)
        else:
            training_dataloader = self.get_dataloader(self._training_dataset, shuffle=True)

        # ----- no pre-training is needed in this stage (if we do not re-initialize the transition matrix) -----
        # ----- pre-train neural module -----
        if self.config.num_lm_nn_pretrain_epochs > 0 and self.config.reinit_s3_trans:
            logger.info(" ----- ")
            logger.info("Pre-training neural module...")
            for epoch_i in range(self.config.num_lm_nn_pretrain_epochs):
                train_loss = self.pretrain_step(
                    data_loader=training_dataloader,
                    optimizer=self._pretrain_optimizer,
                    trans_=self._init_trans_mat,
                    add_wxor_lut=True,
                    apply_ratio_decay=False
                )
                logger.info(f"Epoch: {epoch_i}, Loss: {train_loss}")
            logger.info("Neural module pretrained!")

        # remove any inter results stored in the model
        self._model.pop_inter_results()

        valid_results = SparseCHMMMetric()
        best_metric = 0
        tolerance_epoch = 0

        # remove majority voting lf during inference
        if self.config.keep_s3_mv or (MV_LF_NAME not in self.config.sources):
            src_usg_ids = None
        else:
            src_idx = self.config.sources.index(MV_LF_NAME)
            src_usg_ids = list(range(self.config.n_src))
            src_usg_ids.remove(src_idx)

        # ----- start training process -----
        logger.info(" ----- ")
        logger.info("Training Dir-CHMM...")
        for epoch_i in range(self.config.num_lm_train_epochs):
            logger.info("------")
            logger.info(f"Epoch {epoch_i + 1} of {self.config.num_lm_train_epochs}")

            train_loss = self.training_step(
                data_loader=training_dataloader,
                optimizer=self._optimizer,
                src_usg_ids=src_usg_ids,
                add_wxor_lut=True,
                apply_ratio_decay=True,
                track_conc_params=False
            )
            logger.info(f"Training loss: {train_loss:.4f}")
            self._model.pop_inter_results()  # remove training inter results

            valid_metrics = self.valid(
                add_wxor_lut=True,
                apply_ratio_decay=True
            )  # get and log validation results

            # ----- save model -----
            curr_metric = valid_metrics['f1']

            if curr_metric >= best_metric:
                self.save(model_name='chmm-stage3')
                logger.info("Checkpoint Saved!")
                best_metric = curr_metric
                tolerance_epoch = 0
            else:
                tolerance_epoch += 1

            # ----- log history -----
            valid_results.append(valid_metrics)
            if tolerance_epoch > self.config.num_lm_s3_valid_tolerance:
                logger.info("Training stopped because of exceeding tolerance")
                break

        # retrieve the best state dict
        self.load(model_name='chmm-stage3', load_wxor=True)

        return valid_results

    def valid(self,
              add_wxor_lut: Optional[bool] = False,
              apply_ratio_decay: Optional[bool] = False) -> Metric:
        self._model.to(self.config.device)
        valid_metrics = self.evaluate(
            dataset=self._valid_dataset,
            add_wxor_lut=add_wxor_lut,
            apply_ratio_decay=apply_ratio_decay
        )

        logger.info("Validation results:")
        for k, v in valid_metrics.items():
            logger.info(f"  {k}: {v:.4f}")
        return valid_metrics

    def test(self,
             add_wxor_lut: Optional[bool] = False,
             apply_ratio_decay: Optional[bool] = False) -> Metric:
        self._model.to(self.config.device)
        test_metrics = self.evaluate(
            dataset=self._test_dataset,
            add_wxor_lut=add_wxor_lut,
            apply_ratio_decay=apply_ratio_decay
        )

        logger.info("Test results:")
        for k, v in test_metrics.items():
            logger.info(f"  {k}: {v:.4f}")
        return test_metrics

    def evaluate(self,
                 dataset: CHMMDataset,
                 add_wxor_lut: Optional[bool] = False,
                 apply_ratio_decay: Optional[bool] = False) -> Metric:

        data_loader = self.get_dataloader(dataset)
        self._model.eval()

        # remove majority voting lf during inference
        if self.config.keep_inference_mv or (MV_LF_NAME not in self.config.sources):
            src_usg_ids = None
        else:
            src_idx = self.config.sources.index(MV_LF_NAME)
            src_usg_ids = list(range(self.config.n_src))
            src_usg_ids.remove(src_idx)

        pred_lbs = list()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(data_loader)):
                # get data
                emb_batch, obs_batch, seq_lens = map(lambda x: x.to(self.config.device), batch[:3])

                # get prediction
                pred_lb_indices, _ = self._model.viterbi(
                    emb=emb_batch,
                    obs=obs_batch,
                    seq_lengths=seq_lens,
                    src_usg_ids=src_usg_ids,
                    add_wxor_lut=add_wxor_lut,
                    apply_ratio_decay=apply_ratio_decay,
                    normalize_observation=self.config.obs_normalization,
                    sample_emiss=True if self.config.enable_inference_sampling else False
                )
                pred_lb_batch = [[self.config.bio_label_types[lb_index] for lb_index in label_indices]
                                 for label_indices in pred_lb_indices]
                pred_lbs += pred_lb_batch

        true_lbs = dataset.lbs
        metric_values = get_ner_metrics(true_lbs, pred_lbs)

        return metric_values

    def predict(self,
                dataset: CHMMDataset,
                add_wxor_lut: Optional[bool] = False,
                apply_ratio_decay: Optional[bool] = False):

        data_loader = self.get_dataloader(dataset)
        self._model.eval()

        # remove majority voting lf during inference
        if self.config.keep_inference_mv or (MV_LF_NAME not in self.config.sources):
            src_usg_ids = None
        else:
            src_idx = self.config.sources.index(MV_LF_NAME)
            src_usg_ids = list(range(self.config.n_src))
            src_usg_ids.remove(src_idx)

        pred_lbs = list()
        pred_probs = list()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(data_loader)):
                # get data
                emb_batch, obs_batch, seq_lens = map(lambda x: x.to(self.config.device), batch[:3])

                # get prediction
                pred_lb_indices, pred_prob_batch = self._model.viterbi(
                    emb=emb_batch,
                    obs=obs_batch,
                    seq_lengths=seq_lens,
                    src_usg_ids=src_usg_ids,
                    add_wxor_lut=add_wxor_lut,
                    apply_ratio_decay=apply_ratio_decay,
                    normalize_observation=self.config.obs_normalization,
                    sample_emiss=True if self.config.enable_inference_sampling else False
                )
                pred_lb_batch = [[self.config.bio_label_types[lb_index] for lb_index in label_indices]
                                 for label_indices in pred_lb_indices]

                pred_probs += pred_prob_batch
                pred_lbs += pred_lb_batch

        return pred_lbs, pred_probs

    def get_trans_and_emiss(self, dataset: CHMMDataset) -> tuple[list[torch.Tensor], torch.Tensor]:
        data_loader = self.get_dataloader(dataset)
        self.neural_module.eval()

        transitions = None
        emissions = None
        with torch.no_grad():
            for i, batch in enumerate(tqdm(data_loader)):
                emb_batch = batch[0].to(self.config.device)
                seq_lens = batch[2].to(self.config.device)

                # predict reliability scores
                trans_probs, emiss_probs, _ = self.neural_module(embs=emb_batch, sample_emiss=False)

                if transitions is None:
                    transitions = [trans[:seq_len] for trans, seq_len in zip(trans_probs.detach().cpu(), seq_lens)]
                    emissions = emiss_probs.detach().cpu()
                else:
                    transitions += [trans[:seq_len] for trans, seq_len in zip(trans_probs.detach().cpu(), seq_lens)]
                    emissions = torch.cat([emissions, emiss_probs.detach().cpu()], 0)
        return transitions, emissions

    def get_src_relibs(self, dataset: CHMMDataset):
        """
        Calculate the reliability scores of each label/entity of each source
        """
        data_loader = self.get_dataloader(dataset)
        self.neural_module.eval()

        src_relibs = None
        with torch.no_grad():
            for i, batch in enumerate(tqdm(data_loader)):
                emb_batch = batch[0].to(self.config.device)

                # predict reliability scores
                _, (_, conc_e2e) = self.neural_module.dir_param_base_module(embs=emb_batch)

                if src_relibs is None:
                    src_relibs = conc_e2e.detach()
                else:
                    src_relibs = torch.cat([src_relibs, conc_e2e.detach()], 0)
        return src_relibs

    def get_weighted_xor_lut(self, dataset: CHMMDataset):
        """
        Get the weighted xor lookup table using source reliability scores predicted from the nn module
        and the labels observed by the sources
        """
        # exclude observations of `O`
        # data_size X len_seq X n_src X (n_lbs-1)
        obs_insts = [o[:, :, 1:].to(self.config.device) for o in dataset.obs]

        relib_insts = self.get_src_relibs(dataset=dataset)

        dataset_wxor = get_dataset_wxor(obs_insts=obs_insts, relib_insts=relib_insts, device=self.config.device)
        wxor = dataset_wxor ** self.config.wxor_temperature

        # Make every column of wxor sum up to 1. Facilitates the following scaling method
        wxor = wxor / (wxor.sum(axis=-1, keepdims=True) + 1E-12)

        # de-emphasize the less probable emission addon factors
        ids_0 = list()
        ids_1 = list()
        for i in range(self.config.d_obs - 1):
            for j in range(self.config.d_obs - 1):
                if abs(i - j) % 2 == 0 or ((j - i == 1) and i % 2 == 0) or ((i - j == 1) and i % 2 == 1):
                    pass
                else:
                    ids_0.append(i)
                    ids_1.append(j)

        wxor[:, ids_0, ids_1] *= 0.5

        return wxor

    def get_pretrain_optimizer(self):
        pretrain_optimizer = torch.optim.Adam(
            self.neural_module.parameters(),
            lr=self.config.pretrain_lr,
            weight_decay=1e-5
        )
        return pretrain_optimizer

    def get_optimizer(self):
        # ----- initialize optimizer -----
        hmm_params = [self._model.state_priors]
        optimizer = torch.optim.Adam(
            [{'params': self.neural_module.parameters(), 'lr': self.config.nn_lr},
             {'params': hmm_params}],
            lr=0.01,
            weight_decay=1E-5
        )
        return optimizer

    def freeze_dir_param_base(self):
        """
        stop updating base dirichlet parameters matrix
        """
        for param in self.neural_module.dir_param_base_module.parameters():
            param.requires_grad = False

        return self

    def unfreeze_dir_param_base(self):
        """
        start updating base dirichlet parameters matrix
        """
        for param in self.neural_module.dir_param_base_module.parameters():
            param.requires_grad = True

        return self

    def freeze_dir_param_expan(self):
        """
         stop updating expand dirichlet parameters matrix
         """
        for param in self.neural_module.dir_param_expan_module.parameters():
            param.requires_grad = False

        return self

    def unfreeze_dir_param_expan(self):
        """
         start updating expand dirichlet parameters matrix
         """
        for param in self.neural_module.dir_param_expan_module.parameters():
            param.requires_grad = True

        return self

    def freeze_emiss(self):
        """
        stop updating emission matrix in training
        """
        return self.freeze_dir_param_base().freeze_dir_param_expan()

    def unfreeze_emiss(self):
        """
        start updating emission matrix in training
        """
        return self.unfreeze_dir_param_base().unfreeze_dir_param_expan()

    def freeze_trans(self):
        """
        stop updating transition matrix in training
        """
        for param in self.neural_module.transition_module.parameters():
            param.requires_grad = False

        return self

    def unfreeze_trans(self):
        """
        start updating transition matrix in training
        """

        for param in self.neural_module.transition_module.parameters():
            param.requires_grad = True

        return self

    def save(self,
             output_dir: Optional[str] = None,
             save_optimizer: Optional[bool] = False,
             model_name: Optional[str] = 'chmm',
             optimizer_name: Optional[str] = 'chmm-optimizer',
             pretrain_optimizer_name: Optional[str] = 'chmm-pretrain-optimizer'):
        super().save(output_dir, save_optimizer, model_name, optimizer_name, pretrain_optimizer_name)

        output_dir = output_dir if output_dir is not None else self.config.output_dir
        if self.neural_module.wxor_lut is not None:
            torch.save(self.neural_module.wxor_lut, os.path.join(output_dir, 'wxor_lut.pt'))

        return self

    def load(self,
             input_dir: Optional[str] = None,
             load_optimizer: Optional[bool] = False,
             load_wxor: Optional[bool] = False,
             model_name: Optional[str] = 'chmm',
             optimizer_name: Optional[str] = 'chmm-optimizer',
             pretrain_optimizer_name: Optional[str] = 'chmm-pretrain-optimizer'):
        super().load(input_dir, load_optimizer, model_name, optimizer_name, pretrain_optimizer_name)

        input_dir = input_dir if input_dir is not None else self.config.output_dir
        if load_wxor:
            try:
                self.neural_module.wxor_lut = torch.load(os.path.join(input_dir, 'wxor_lut.pt'))
            except Exception as err:
                logger.exception(f"Failed to load `wxor_lut`: {err}")
                self.neural_module.wxor_lut = None

        return self

    def save_results(self,
                     output_dir: str,
                     valid_results: Optional[Metric] = None,
                     file_name: Optional[str] = 'results',
                     add_wxor_lut: Optional[bool] = False,
                     apply_ratio_decay: Optional[bool] = False,
                     disable_final_valid: Optional[bool] = False,
                     disable_test: Optional[bool] = False,
                     disable_inter_results: Optional[bool] = False) -> None:
        """
        Save training (validation) results

        Parameters
        ----------
        output_dir: output directory, should be a folder
        valid_results: validation results during the training process
        file_name: file name
        add_wxor_lut: use wxor lookup table in inference
        apply_ratio_decay: apply emission non-diagonal ratio decay
        disable_final_valid: disable final validation process (getting validation results of the trained model)
        disable_test: disable test process
        disable_inter_results: do not save inter-results

        Returns
        -------
        None
        """
        if not disable_final_valid:
            logger.info("Getting final validation metrics")
            valid_metrics = self.valid(
                add_wxor_lut=add_wxor_lut,
                apply_ratio_decay=apply_ratio_decay
            )
        else:
            valid_metrics = None

        if not disable_test:
            logger.info("Getting test metrics.")
            test_metrics = self.test(
                add_wxor_lut=add_wxor_lut,
                apply_ratio_decay=apply_ratio_decay
            )
        else:
            test_metrics = None

        # write validation and test results
        result_file = os.path.join(output_dir, f'{file_name}.txt')
        logger.info(f"Writing results to {result_file}")
        self.write_result(file_path=result_file,
                          valid_results=valid_results,
                          final_valid_metrics=valid_metrics,
                          test_metrics=test_metrics)

        if not disable_inter_results:
            # save validation inter results
            logger.info(f"Saving inter results")
            inter_result_file = os.path.join(output_dir, f'{file_name}-inter.pt')
            torch.save(valid_results.__dict__, inter_result_file)
        return None
