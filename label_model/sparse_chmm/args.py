import torch
import logging
from typing import Optional
from dataclasses import dataclass, field
from transformers.file_utils import cached_property, torch_required
from seqlbtoolkit.chmm.config import CHMMBaseConfig

logger = logging.getLogger(__name__)


@dataclass
class SparseCHMMArguments:
    """
    Arguments regarding the training of Neural hidden Markov Model
    """

    # --- manage directories and IO ---
    train_path: Optional[str] = field(
        default='', metadata={'help': 'Training data file path'}
    )
    valid_path: Optional[str] = field(
        default='', metadata={'help': 'Validation data file path'}
    )
    test_path: Optional[str] = field(
        default='', metadata={'help': 'Test data file path'}
    )
    output_dir: Optional[str] = field(
        default='.',
        metadata={"help": "The folder where the models and outputs will be written."},
    )
    bert_model_name_or_path: Optional[str] = field(
        default='', metadata={"help": "Path to pretrained BERT model or model identifier from huggingface.co/models; "
                                      "Used to construct BERT embeddings if not exist"}
    )
    save_dataset: Optional[bool] = field(
        default=False, metadata={"help": "Whether save the datasets used for training & validation & test"}
    )
    save_dataset_to_data_dir: Optional[bool] = field(
        default=False, metadata={"help": "Whether save the datasets to the original dataset folder. "
                                         "If not, the dataset would be saved to the result folder."}
    )
    load_preprocessed_dataset: Optional[bool] = field(
        default=False, metadata={"help": "Whether load the pre-processed datasets from disk"}
    )
    load_s1_model: Optional[bool] = field(
        default=False, metadata={'help': 'Whether load the trained stage-1 model parameters'}
    )
    load_s2_model: Optional[bool] = field(
        default=False, metadata={'help': 'Whether load the trained stage-2 model parameters.'
                                         'Usually used for testing model.'}
    )
    load_s3_model: Optional[bool] = field(
        default=False, metadata={'help': 'Whether load the trained stage-3 model parameters.'
                                         'Usually used for testing model.'}
    )
    training_ratio_per_epoch: Optional[float] = field(
        default=None, metadata={'help': 'How much data in the training set is used for one epoch.'
                                        'Leave None if use the whole training set'}
    )
    load_init_mat: Optional[bool] = field(
        default=False, metadata={'help': 'Whether to load initial transition and emission matrix from disk'}
    )
    save_init_mat: Optional[bool] = field(
        default=False, metadata={'help': 'Whether to save initial transition and emission matrix from disk'}
    )
    log_dir: Optional[str] = field(
        default=None,
        metadata={"help": "the directory of the log file. Set to '' to disable logging"}
    )

    # --- training and data arguments ---
    nn_lr: Optional[float] = field(
        default=0.001, metadata={'help': 'learning rate of the neural networks in CHMM'}
    )
    pretrain_lr: Optional[float] = field(
        default=5E-4, metadata={'help': 'learning rate of the pretraining stage'}
    )
    s2_lr_decay: Optional[float] = field(
        default=0.5, metadata={'help': 'Stage 2 learning rate decay rate'}
    )
    s3_lr_decay: Optional[float] = field(
        default=1, metadata={'help': 'Stage 3 learning rate decay rate'}
    )
    lm_batch_size: Optional[int] = field(
        default=128, metadata={'help': 'denoising model training batch size'}
    )
    num_lm_train_epochs: Optional[int] = field(
        default=15, metadata={'help': 'number of denoising model training epochs'}
    )
    num_lm_nn_pretrain_epochs: Optional[int] = field(
        default=5, metadata={'help': 'number of denoising model pre-training epochs'}
    )
    num_lm_valid_tolerance: Optional[int] = field(
        default=10, metadata={"help": "How many tolerance epochs before quiting training"}
    )
    num_lm_s2_valid_tolerance: Optional[int] = field(
        default=5, metadata={"help": "How many tolerance epochs before quiting training"}
    )
    num_lm_s3_valid_tolerance: Optional[int] = field(
        default=5, metadata={"help": "How many tolerance epochs before quiting training"}
    )
    num_lm_valid_smoothing: Optional[int] = field(
        default=1, metadata={'help': "Number of validation performance smoothing epochs"}
    )
    freeze_s2_base_emiss: Optional[bool] = field(
        default=False, metadata={'help': "Whether freeze the base emission model in stage 2"}
    )
    include_s2: Optional[bool] = field(
        default=False, metadata={'help': 'Whether include stage 2 training'}
    )
    include_s3: Optional[bool] = field(
        default=False, metadata={'help': 'Whether include stage 3 training'}
    )
    reinit_s3_trans: Optional[bool] = field(
        default=False, metadata={'help': "Whether to re-initialize the transition matrix in stage 3"}
    )
    transduction: Optional[bool] = field(
        default=False, metadata={'help': "Use transductive learning instead of inductive learning"}
    )
    add_majority_voting: Optional[bool] = field(
        default=False, metadata={'help': "Add an additional majority voting labeling function"}
    )
    keep_inference_mv: Optional[bool] = field(
        default=False, metadata={'help': "Keep the majority voting LF during inference"}
    )
    keep_s3_mv: Optional[bool] = field(
        default=False, metadata={'help': "Keep the majority voting LF in stage 3"}
    )
    no_cuda: Optional[bool] = field(
        default=False, metadata={"help": "Disable CUDA even when it is available"}
    )
    seed: Optional[int] = field(
        default=42, metadata={"help": "Random seed that will be set at the beginning of training."}
    )
    debug_mode: Optional[bool] = field(
        default=False, metadata={"help": "Debugging mode with fewer training data"}
    )

    # --- (Dirichlet) model parameters ---
    dirichlet_conc_base: Optional[float] = field(
        default=1.0, metadata={'help': 'the basic concentration parameter (lower-bound)'}
    )
    dirichlet_conc_max: Optional[float] = field(
        default=100.0, metadata={'help': 'the maximum concentration parameter value'}
    )
    reliability_level: Optional[str] = field(
        default='entity', metadata={'help': 'entity-level emission score or token-level score'}
    )
    disable_dirichlet: Optional[bool] = field(
        default=False, metadata={'help': 'whether completely disable the dirichlet sampling process'}
    )
    enable_inference_sampling: Optional[bool] = field(
        default=False, metadata={'help': 'whether sample from dirichlet while inference'}
    )
    apply_entity_emiss_loss: Optional[bool] = field(
        default=False, metadata={'help': 'whether add loss for different entity label emission probabilities'}
    )
    diag_exp_t1: Optional[float] = field(
        default=2.0, metadata={'help': 'Tier 1 emission term for scaling up the emission diagonal values, '
                                       'should be >= 1.'}
    )
    diag_exp_t2: Optional[float] = field(
        default=3.0, metadata={'help': 'Tier 2 emission term for scaling up the emission diagonal values, '
                                       'should be >= 1.'}
    )
    nondiag_exp: Optional[float] = field(
        default=3.0, metadata={'help': 'Exponential term that controls how quick the e2o emission prob descents.'}
    )
    nondiag_split_ratio: Optional[float] = field(
        default=0.5, metadata={'help': 'Decides how large the non-diagonal values can be. Should <=1.'}
    )
    nondiag_split_decay: Optional[float] = field(
        default=1, metadata={'help': 'The decay ratio of the nondiag split point. Should <=1.'}
    )
    wxor_temperature: Optional[float] = field(
        default=1, metadata={'help': 'The temperature of the weighted xor loopup table. Should be less than 1.'}
    )
    calculate_wxor_on_valid: Optional[bool] = field(
        default=False, metadata={'help': 'Whether calculate the weighted xor score on the validation set to save time'}
    )

    def __post_init__(self):
        assert self.reliability_level in ('entity', 'label'),\
            ValueError("`reliability_level` parameter must be in ('entity', 'label')")

    # The following three functions are copied from transformers.training_args
    @cached_property
    @torch_required
    def _setup_devices(self) -> "torch.device":
        if self.no_cuda:
            device = torch.device("cpu")
            self._n_gpu = 0
        else:
            device = torch.device("cuda")
            self._n_gpu = 1

        return device

    @property
    @torch_required
    def device(self) -> "torch.device":
        """
        The device used by this process.
        """
        return self._setup_devices

    @property
    @torch_required
    def n_gpu(self) -> "int":
        """
        The number of GPUs used by this process.

        Note:
            This will only be greater than one when you have multiple GPUs available but are not using distributed
            training. For distributed training, it will always be 1.
        """
        # Make sure `self._n_gpu` is properly setup.
        _ = self._setup_devices
        return self._n_gpu


@dataclass
class SparseCHMMConfig(SparseCHMMArguments, CHMMBaseConfig):
    pass
