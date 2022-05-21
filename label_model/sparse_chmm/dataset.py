import sys
sys.path.append('../..')

import copy
import torch
import numpy as np
from typing import List, Optional

from seqlbtoolkit.chmm.dataset import CHMMBaseDataset
from seqlbtoolkit.data import (
    probs_to_lbs,
    label_to_span,
    span_to_label,
    rand_argmax
)

from .args import SparseCHMMConfig
from .macro import *


class CHMMDataset(CHMMBaseDataset):
    def __init__(self,
                 text: Optional[List[List[str]]] = None,
                 embs: Optional[List[torch.Tensor]] = None,
                 obs: Optional[List[torch.Tensor]] = None,  # batch, src, token
                 lbs: Optional[List[List[str]]] = None,
                 src: Optional[List[str]] = None,
                 ents: Optional[List[str]] = None):
        super().__init__(text, embs, obs, lbs, src, ents)

    def load_file(self,
                  file_path: str,
                  config: Optional[SparseCHMMConfig] = None) -> "CHMMDataset":
        super().load_file(file_path, config)

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


# noinspection PyTypeChecker
def majority_voting(obs_inst: np.array, label_types: list[str]):
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
