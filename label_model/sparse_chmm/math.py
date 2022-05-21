import torch
import numpy as np
from typing import Optional, Union, List
from itertools import permutations


def log_matmul(a: torch.Tensor, b: torch.Tensor):
    """
    a : m \times n
    b : n \times p

    output : m \times p matrix

    Normally, a matrix multiplication
    computes out_{i,j} = sum_k A_{i,k} \times B_{k,j}

    A log domain matrix multiplication
    computes out_{i,j} = logsumexp_k log_A_{i,k} + log_B_{k,j}

    This is needed for numerical stability when A and B are probability matrices.
    """
    a1 = a.unsqueeze(-1)
    b1 = b.unsqueeze(-3)
    return (a1 + b1).logsumexp(-2)


def log_maxmul(a, b):
    a1 = a.unsqueeze(-1)
    b1 = b.unsqueeze(-3)
    return (a1 + b1).max(-2)


# noinspection PyTypeChecker
def logsumexp(x, dim=None, keepdim=False):
    if dim is None:
        x, dim = x.view(-1), 0
    xm, _ = torch.max(x, dim=dim, keepdim=True)
    x = torch.where(
        (xm == np.inf) | (xm == -np.inf),
        xm,
        xm + torch.logsumexp(x - xm, dim=dim, keepdim=True)
    )
    return x if keepdim else x.squeeze(dim)


def validate_prob(x, dim=-1):
    if (x <= 0).any():
        prob = normalize(x, dim=dim)
    elif (x.sum(dim=dim) != 1).any():
        prob = x / x.sum(dim=dim, keepdim=True)
    else:
        prob = x
    return prob


def normalize(x, dim=-1, epsilon=1e-6):
    result = x - x.min(dim=dim, keepdim=True)[0] + epsilon
    result = result / result.sum(dim=dim, keepdim=True)
    return result


def entropy(p: torch.Tensor, dim: Optional[int] = -1):
    """
    Calculate entropy

    Parameters
    ----------
    p: probabilities
    dim: dimension

    Returns
    -------
    entropy
    """
    h = torch.sum(-p * torch.log(p), dim=dim)
    return h


def prob_scaling(p: Union[float, torch.Tensor, np.ndarray],
                 r: Optional[float] = 0.5,
                 e: Optional[float] = 2,
                 n: Optional[float] = 2):
    """
    scale the probabilities: pushing the probability values to extreme

    Parameters
    ----------
    p: input probabilities
    r: split point: the point that separates the "pushing up" and "pushing down" operations
    e: tier 1 inverse exponential term
    n: tier 2 exponential term

    Returns
    -------
    type(p)
    """
    p_ = p ** (1 / e)

    pu = p_ > r
    pd = p_ <= r

    ru = pu * (-(1 / (1 - r)) ** (n - 1) * (1 - p_) ** n + 1)
    rd = pd * ((1 / r) ** (n - 1) * p_ ** n)

    return ru + rd


def entity_emiss_diag(x):
    """
    emission prior of entity to itself

    Parameters
    ----------
    x

    Returns
    -------
    x
    """
    return x


def entity_emiss_o(x, n_lbs, tp, exp_term=2):
    """
    The function that calculates the emission prior of entity labels to the non-entity label 'O'
    according to the diagonal values of the emission prior

    Parameters
    ----------
    x: diagonal values
    n_lbs: number of entity labels (2e+1)
    tp: turning point
    exp_term: the exponential term that controls the slope of the function

    Returns
    -------
    non-diagonal emission priors
    """
    # separating piecewise function
    low = x < tp
    high = x >= tp

    # parameters for the first piece
    a = (2 - n_lbs) / ((exp_term - 1) * tp ** exp_term - exp_term * tp ** (exp_term - 1))
    b = 1 - n_lbs
    # parameter for the second piece
    f_tp = a * tp ** exp_term + b * tp + 1
    c = f_tp / (tp - 1)

    # piecewise result
    y = low * (a * x ** exp_term + b * x + 1) + high * (c * x - c)
    return y


def entity_emiss_nondiag(x, n_lbs, tp, exp_term=2):
    """
    emission prior of entity to other entities

    Parameters
    ----------
    x: diagonal values
    n_lbs: number of entity labels (2e+1)
    tp: turning point
    exp_term: the exponential term that controls the slope of the function

    Returns
    -------

    """
    return (1 - entity_emiss_diag(x) - entity_emiss_o(x, n_lbs, tp, exp_term)) / (n_lbs - 2)


def calc_weighted_xor(obs: torch.Tensor,
                      relibs: torch.Tensor,
                      l_query: int,
                      l_tgt: int) -> torch.Tensor:
    """
    Calculate weighted xor scores of each source labels

    Parameters
    ----------
    obs: the values observed by the sources. Generally represented by x
    relibs: reliability scores, usually represented by `a` or `conc_e2e_scale`
    l_query: query line idx
    l_tgt: target line idx

    Returns
    -------
    weighted xor score for one sentence

    """
    prod_obs_a = obs * relibs
    prod_obs_1ma = obs * (1 - relibs)

    tgt_norm = prod_obs_a[:, :, l_tgt].sum(dim=1) / (obs[:, :, l_tgt].sum(dim=1) + 1e-12)  # prevent dividing by 0
    score = (prod_obs_1ma[:, :, l_query] * tgt_norm.unsqueeze(-1)).sum(dim=0)

    return score


def get_dataset_wxor(obs_insts: List[torch.Tensor],
                     relib_insts: torch.Tensor,
                     device: Optional[torch.device] = torch.device('cpu')) -> torch.Tensor:
    """
    Calculate weighted xor scores of each source labels for the entire dataset

    Parameters
    ----------
    obs_insts: the values observed by the sources. Generally represented by x;
               shape: data_size X len_seq X n_src X (n_lbs-1)
    relib_insts: reliability scores, usually represented by `a` or `conc_e2e_scale`
               shape: data_size X n_src X (n_lbs-1)
    device: device

    Returns
    -------
    weighted xor score for the dataset

    """

    _, n_src, n_lbs = obs_insts[0].size()
    perm = torch.tensor(list(permutations(list(range(n_lbs)), 2)))

    wxor_lut_sum = torch.zeros([n_src, n_lbs, n_lbs], device=device)

    for obs, relibs in zip(obs_insts, relib_insts):

        # shape of `relibs`: n_src X (n_lbs-1)
        wxor_lut = torch.zeros_like(wxor_lut_sum)  # weighted xor lookup table
        for (l_query, l_tgt) in perm:
            weighted_xor = calc_weighted_xor(obs, relibs, l_query, l_tgt)
            wxor_lut[:, l_query, l_tgt] = weighted_xor

        wxor_lut_sum += wxor_lut

    obs_sum = torch.zeros([n_src, n_lbs], device=device)
    for obs in obs_insts:
        obs_sum += obs.sum(dim=0)
    obs_sum[obs_sum < 1] = 0

    wxor_lut_avg = wxor_lut_sum / (obs_sum.unsqueeze(-1) + 1E-12)

    return wxor_lut_avg
