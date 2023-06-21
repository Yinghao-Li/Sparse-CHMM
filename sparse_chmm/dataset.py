import os
import json
import regex
import copy
import torch
import numpy as np
import logging
from string import printable
from typing import Optional, Union

from seqlbtoolkit.data import (
    probs_to_lbs,
    label_to_span,
    span_to_label,
    rand_argmax,
    span_list_to_dict,
    entity_to_bio_labels,
    one_hot
)
from seqlbtoolkit.embs import build_bert_token_embeddings

from .args import Config
from .macro import *


logger = logging.getLogger(__name__)


class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 text: Optional[list[list[str]]] = None,
                 embs: Optional[list[torch.Tensor]] = None,
                 obs: Optional[list[torch.Tensor]] = None,  # batch, src, token
                 lbs: Optional[list[list[str]]] = None,
                 src: Optional[list[str]] = None,
                 ents: Optional[list[str]] = None
                 ):
        super().__init__()
        self._embs = embs
        self._obs = obs
        self._text = text
        self._lbs = lbs
        self._src = src
        self._ents = ents

    @property
    def n_insts(self):
        return len(self.obs)

    @property
    def embs(self):
        return self._embs if self._embs else list()

    @property
    def text(self):
        return self._text if self._text else list()

    @property
    def lbs(self):
        return self._lbs if self._lbs else list()

    @property
    def obs(self):
        return self._obs if self._obs else list()

    @property
    def src(self):
        return self._src

    @property
    def ents(self):
        return self._ents

    @text.setter
    def text(self, value):
        logger.warning(f'{type(self)}: text has been changed')
        self._text = value

    @obs.setter
    def obs(self, value):
        logger.warning(f'{type(self)}: observations have been changed')
        self._obs = value

    @lbs.setter
    def lbs(self, value):
        logger.warning(f'{type(self)}: labels have been changed')
        self._lbs = value

    @embs.setter
    def embs(self, value):
        logger.warning(f'{type(self)}: embeddings have been changed')
        self._embs = value

    @src.setter
    def src(self, value):
        logger.warning(f'{type(self)}: sources have been changed')
        self._src = value

    @ents.setter
    def ents(self, value):
        logger.warning(f'{type(self)}: entity types have been changed')
        self._ents = value

    def __len__(self):
        return self.n_insts

    def __getitem__(self, idx):
        if self._lbs is not None and len(self._lbs) > 0:
            return self._text[idx], self._embs[idx], self._obs[idx], self._lbs[idx]
        else:
            return self._text[idx], self._embs[idx], self._obs[idx]

    def __add__(self, other: "Dataset") -> "Dataset":
        assert self.src and other.src and self.src == other.src, ValueError("Sources not matched!")
        assert self.ents and other.ents and self.ents == other.ents, ValueError("Entity types not matched!")

        return Dataset(
            text=copy.deepcopy(self.text + other.text),
            embs=copy.deepcopy(self.embs + other.embs),
            obs=copy.deepcopy(self.obs + other.obs),
            lbs=copy.deepcopy(self.lbs + other.lbs),
            ents=copy.deepcopy(self.ents),
            src=copy.deepcopy(self.src)
        )

    def __iadd__(self, other: "Dataset") -> "Dataset":
        if self.src:
            assert other.src and self.src == other.src, ValueError("Sources do not match!")
        else:
            assert other.src, ValueError("Attribute `src` not found!")

        if self.ents:
            assert other.ents and self.ents == other.ents, ValueError("Entity types do not match!")
        else:
            assert other.ents, ValueError("Attribute `ents` not found!")

        self.text = copy.deepcopy(self.text + other.text)
        self.embs = copy.deepcopy(self.embs + other.embs)
        self.obs = copy.deepcopy(self.obs + other.obs)
        self.lbs = copy.deepcopy(self.lbs + other.lbs)
        self.ents = copy.deepcopy(other.ents)
        self.src = copy.deepcopy(other.src)
        return self

    def save(self, file_dir: str, dataset_type: str, config, force_save: Optional[bool] = False):
        """
        Save dataset for future usage

        Parameters
        ----------
        file_dir: the folder which the dataset will be stored in.
        dataset_type: decides if the dataset is training, validation or test set
        config: configuration file
        force_save: force to save the file even if a file of the same path exists.

        Returns
        -------
        None
        """
        assert dataset_type in ['train', 'valid', 'test']
        output_path = os.path.join(file_dir, f"{dataset_type}.chmmdp")
        if os.path.exists(output_path) and not force_save:
            return None

        chmm_data_dict = {
            'text': self.text,
            'lbs': self.lbs,
            'obs': self.obs,
            'src': self.src,
            'ents': self.ents,
            'embs': self.embs,
            'src_priors': config.src_priors
        }
        torch.save(chmm_data_dict, output_path)
        return None

    def load(self, file_dir: str, dataset_type: str, config):
        """
        Load saved datasets and configurations

        Parameters
        ----------
        file_dir: the folder which the dataset is stored in.
        dataset_type: decides if the dataset is training, validation or test set
        config: configuration file

        Returns
        -------
        self
        """
        assert dataset_type in ['train', 'valid', 'test']
        if os.path.isdir(file_dir):
            file_path = os.path.join(file_dir, f'{dataset_type}.chmmdp')
            assert os.path.isfile(file_path), FileNotFoundError(f"{file_path} does not exist!")
        else:
            raise FileNotFoundError(f"{file_dir} does not exist!")

        chmm_data_dict = torch.load(file_path)
        for attr, value in chmm_data_dict.items():
            if attr == 'src_priors':
                continue
            try:
                setattr(self, f'_{attr}', value)
            except AttributeError as err:
                logger.exception(f"Failed to set attribute {attr}: {err}")
                raise err

        config.sources = copy.deepcopy(self.src)
        config.entity_types = copy.deepcopy(self.ents)
        config.bio_label_types = entity_to_bio_labels(self.ents)
        config.src_priors = chmm_data_dict['src_priors']
        config.d_emb = self._embs[0].shape[-1]
        return self

    def load_file(self, file_path: str, config) -> Union["Dataset", tuple["Dataset", "Config"]]:
        """
        Load data from disk

        Parameters
        ----------
        file_path: the directory of the file. In JSON or PT
        config: chmm configuration; Optional to make function testing easier.

        Returns
        -------
        self (MultiSrcNERDataset)
        """
        bert_model = getattr(config, "bert_model_name_or_path", 'bert-base-uncased')
        device = getattr(config, "device", torch.device('cpu'))

        file_path = os.path.normpath(file_path)
        logger.info(f'Loading data from {file_path}')

        file_dir, file_name = os.path.split(file_path)
        if file_path.endswith('.json'):
            sentence_list, label_list, weak_label_list = load_data_from_json(file_path, config)
            # get embedding directory
            emb_name = f"{'.'.join(file_name.split('.')[:-1])}-emb.pt"
            emb_dir = os.path.join(file_dir, emb_name)
        else:
            logger.error(f"Unsupported data type: {file_path}")
            raise TypeError(f"Unsupported data type: {file_path}")

        self._text = sentence_list
        self._lbs = label_list
        self._obs = weak_label_list
        logger.info(f'Data loaded from {file_path}.')

        logger.info(f'Searching for corresponding BERT embeddings...')
        if os.path.isfile(emb_dir):
            logger.info(f"Found embedding file: {emb_dir}. Loading to memory...")
            embs = torch.load(emb_dir)
            if isinstance(embs[0], torch.Tensor):
                self._embs = embs
            elif isinstance(embs[0], np.ndarray):
                self._embs = [torch.from_numpy(emb).to(torch.float) for emb in embs]
            else:
                logger.error(f"Unknown embedding type: {type(embs[0])}")
                raise RuntimeError
        else:
            logger.info(f"{emb_dir} does not exist. Building embeddings instead...")
            self.build_embs(bert_model, device, emb_dir)

        self._src = copy.deepcopy(config.sources)
        self._ents = config.entity_types
        config.d_emb = self._embs[0].shape[-1]
        if getattr(config, 'debug_mode', False):
            self._embs = self._embs[:100]

        # append dummy token/labels in front of the text/lbs/obs
        logger.info("Appending dummy token/labels in front of the text/lbs/obs for CHMM compatibility")
        self._text = [['[CLS]'] + txt for txt in self._text]
        self._lbs = [['O'] + lb for lb in self._lbs]
        prefix = torch.zeros([1, self._obs[0].shape[-2], self._obs[0].shape[-1]])  # shape: 1, n_src, d_obs
        prefix[:, :, 0] = 1
        self._obs = [torch.cat([prefix, inst]) for inst in self._obs]

        if config.add_majority_voting:
            # --- construct the majority voting source ---
            # shape: n_insts X seq_len X n_src
            obs_lbs = [probs_to_lbs(ob_probs, label_types=config.bio_label_types) for ob_probs in self.obs]
            mv_lbs = list()
            for obs_inst, sent_tks in zip(obs_lbs, self.text):
                mv_lb_inst = majority_voting(obs_inst, label_types=config.bio_label_types)
                mv_lb_inst = span_to_label(label_to_span(mv_lb_inst), sent_tks)  # remove possible invalid entity labels
                mv_lbs.append(mv_lb_inst)

            self.update_obs(mv_lbs, MV_LF_NAME, config)

        return self

    def build_embs(self,
                   bert_model,
                   device: Optional[torch.device] = torch.device('cpu'),
                   save_dir: Optional[str] = None):
        """
        build bert embeddings

        Parameters
        ----------
        bert_model: the location/name of the bert model to use
        device: device
        save_dir: location to update/store the BERT embeddings. Leave None if do not want to save

        Returns
        -------
        self (MultiSrcNERDataset)
        """
        assert bert_model is not None, AssertionError('Please specify BERT model to build embeddings')
        logger.info(f'Building BERT embeddings with {bert_model} on {device}')
        self._embs = build_bert_token_embeddings(
            tk_seq_list=self._text,
            model_or_name=bert_model,
            tokenizer_or_name=bert_model,
            device=device,
            prepend_cls_embs=True
        )
        if save_dir:
            save_dir = os.path.normpath(save_dir)
            logger.info(f'Saving embeddings to {save_dir}...')
            embs = [emb.numpy().astype(np.float32) for emb in self.embs]
            torch.save(embs, save_dir)
        return self

    def update_obs(self, obs: list[list[Union[int, str]]], src_name: str, config):
        """
        update weak labels (chmm observations)

        Parameters
        ----------
        obs: input observations (week annotations)
        src_name: source name
        config: configuration file

        Returns
        -------
        self (MultiSrcNERDataset)
        """
        if isinstance(obs[0][0], str):
            lb2ids = {lb: i for i, lb in enumerate(config.bio_label_types)}
            np_map = np.vectorize(lambda lb: lb2ids[lb])
            obs = [np_map(np.asarray(weak_lbs)).tolist() for weak_lbs in obs]

        if len(obs[0]) == len(self.text[0]):
            weak_lbs_one_hot = [one_hot(np.asarray(weak_lbs), n_class=config.n_lbs) for weak_lbs in obs]
        elif len(obs[0]) == len(self.text[0]) - 1:
            weak_lbs_one_hot = [one_hot(np.asarray([0] + weak_lbs), n_class=config.n_lbs) for weak_lbs in obs]
        else:
            logger.error("The length of the input observation does not match the dataset sentences!")
            raise ValueError("The length of the input observation does not match the dataset sentences!")

        if src_name in self._src:
            src_idx = self._src.index(src_name)
            for i in range(len(self._obs)):
                self._obs[i][:, src_idx, :] = torch.tensor(weak_lbs_one_hot[i])
        else:
            self._src.append(src_name)
            for i in range(len(self._obs)):
                self._obs[i] = torch.cat([self._obs[i], torch.tensor(weak_lbs_one_hot[i]).unsqueeze(1)], dim=1)
            # add the source into config and give a heuristic source prior
            if src_name not in config.sources:
                config.sources.append(src_name)
                config.src_priors[src_name] = {ent: (0.7, 0.7) for ent in config.entity_types}

        return self

    def remove_src(self,
                   src_name: str,
                   config):
        """
        remove a source and its observations from the dataset

        Parameters
        ----------
        src_name: source name
        config: configuration file

        Returns
        -------
        self (MultiSrcNERDataset)
        """

        if src_name in config.sources:
            config.sources.remove(src_name)
        if src_name in config.src_priors.keys():
            config.src_priors.pop(src_name, None)

        if src_name not in self._src:
            logger.warning(f"Labeling function {src_name} is not presented in dataset. Nothing is changed!")
            return self

        # remove source name
        src_idx = self._src.index(src_name)
        other_idx = list(range(len(self.src)))
        other_idx.remove(src_idx)

        # remove the corresponding observation
        self._src.remove(src_name)
        for i in range(len(self._obs)):
            self._obs[i] = self._obs[i][:, other_idx, :]

        # remove the cached property
        try:
            delattr(self, "src_metrics")
        except Exception:
            pass

        return self


def majority_voting(obs_inst: np.ndarray, label_types: list[str]):
    """

    Parameters
    ----------
    obs_inst: seq_len X n_src
    label_types: list of string
    """
    seq_len, n_src = obs_inst.shape
    lb_types = copy.deepcopy(label_types)
    lb_types.remove("O")

    mv_lbs = np.empty([seq_len], dtype=object)
    mv_lb_counts = np.zeros([seq_len, len(lb_types)])

    for i, lb_type in enumerate(lb_types):
        mv_lb_counts[:, i] = np.sum(obs_inst == lb_type, axis=-1)

    non_obs_ids = mv_lb_counts.sum(axis=-1) == 0
    mv_lb_ids = rand_argmax(mv_lb_counts, axis=-1)

    for i, lb_type in enumerate(lb_types):
        mv_lbs[mv_lb_ids == i] = lb_type

    mv_lbs[non_obs_ids] = "O"

    return mv_lbs


def load_data_from_json(file_dir: str, config: Optional = None):
    """
    Load data stored in the current data format.


    Parameters
    ----------
    file_dir: file directory
    config: configuration

    """
    with open(file_dir, 'r', encoding='utf-8') as f:
        data_dict = json.load(f)

    # Load meta if exist
    file_loc = os.path.split(file_dir)[0]
    meta_dir = os.path.join(file_loc, 'meta.json')

    if not os.path.isfile(meta_dir):
        raise FileNotFoundError('Meta file does not exist!')

    with open(meta_dir, 'r', encoding='utf-8') as f:
        meta_dict = json.load(f)

    bio_labels = entity_to_bio_labels(meta_dict['entity_types'])
    label_to_id = {lb: i for i, lb in enumerate(bio_labels)}
    np_map = np.vectorize(lambda lb: label_to_id[lb])

    load_all_sources = config is not None and getattr(config, "load_all_sources", False)
    if 'lf_rec' in meta_dict.keys() and not load_all_sources:
        lf_rec_ids = [meta_dict['lf'].index(lf) for lf in meta_dict['lf_rec']]
    else:
        lf_rec_ids = list(range(meta_dict['num_lf']))

    sentence_list = list()
    lbs_list = list()
    w_lbs_list = list()

    for i in range(len(data_dict)):
        data = data_dict[str(i)]
        # get tokens
        tks = [regex.sub("[^{}]+".format(printable), "", tk) for tk in data['data']['text']]
        sent_tks = ['[UNK]' if not tk else tk for tk in tks]
        sentence_list.append(sent_tks)
        # get true labels
        lbs = span_to_label(span_list_to_dict(data['label']), sent_tks)
        lbs_list.append(lbs)
        # get lf annotations (weak labels) in one-hot format tensor
        w_lbs = [span_to_label(span_list_to_dict(data['weak_labels'][lf_idx]), sent_tks) for lf_idx in lf_rec_ids]
        w_lbs = np_map(np.asarray(w_lbs).T)
        w_lbs_one_hot = one_hot(w_lbs, n_class=len(bio_labels))
        w_lbs_list.append(torch.from_numpy(w_lbs_one_hot).to(dtype=torch.float))

    # update config
    if config:
        config.sources = meta_dict['lf_rec'] \
            if 'lf_rec' in meta_dict.keys() and not load_all_sources else meta_dict['lf']
        config.entity_types = meta_dict['entity_types']
        config.bio_label_types = bio_labels
        if 'priors' in meta_dict.keys():
            config.src_priors = meta_dict['priors']
        else:
            config.src_priors = {src: {lb: (0.7, 0.7) for lb in config.entity_types} for src in config.sources}

    if config and getattr(config, 'debug_mode', False):
        return sentence_list[:100], lbs_list[:100], w_lbs_list[:100]
    return sentence_list, lbs_list, w_lbs_list
