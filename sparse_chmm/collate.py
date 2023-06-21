import logging
import torch
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)


def batch_prep(emb_list: list[torch.Tensor],
               obs_list: list[torch.Tensor],
               txt_list: Optional[list[list[str]]] = None,
               lbs_list: Optional[list[dict]] = None):
    """
    Pad the instance to the max seq max_seq_length in batch

    All input should already have the dummy element appended to the beginning of the sequence
    """
    for emb, obs, txt, lbs in zip(emb_list, obs_list, txt_list, lbs_list):
        assert len(obs) == len(emb) == len(txt) == len(lbs)
    d_emb = emb_list[0].shape[-1]
    _, n_src, n_obs = obs_list[0].size()
    seq_lens = [len(obs) for obs in obs_list]
    max_seq_len = np.max(seq_lens)

    emb_batch = torch.stack([
        torch.cat([inst, torch.zeros([max_seq_len-len(inst), d_emb])], dim=-2) for inst in emb_list
    ])

    prefix = torch.zeros([1, n_src, n_obs])
    prefix[:, :, 0] = 1
    obs_batch = torch.stack([
        torch.cat([inst, prefix.repeat([max_seq_len-len(inst), 1, 1])]) for inst in obs_list
    ])
    obs_batch /= obs_batch.sum(dim=-1, keepdim=True)

    seq_lens = torch.tensor(seq_lens, dtype=torch.long)

    # we don't need to append the length of txt_list and lbs_list
    return emb_batch, obs_batch, seq_lens, txt_list, lbs_list


def collate_fn(insts):
    """
    Principle used to construct dataloader

    :param insts: original instances
    :return: padded instances
    """
    all_insts = list(zip(*insts))
    if len(all_insts) == 4:
        txt, embs, obs, lbs = all_insts
        batch = batch_prep(emb_list=embs, obs_list=obs, txt_list=txt, lbs_list=lbs)
    elif len(all_insts) == 3:
        txt, embs, obs = all_insts
        batch = batch_prep(emb_list=embs, obs_list=obs, txt_list=txt)
    else:
        logger.error('Unsupported number of instances')
        raise ValueError('Unsupported number of instances')
    return batch
