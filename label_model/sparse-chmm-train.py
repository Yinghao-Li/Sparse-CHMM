# coding=utf-8
""" Train the conditional hidden Markov model """

import sys
sys.path.append('..')

import logging
import os
import sys
import gc
import torch
from datetime import datetime

from transformers import (
    HfArgumentParser,
    set_seed,
)

from seqlbtoolkit.io import set_logging, logging_args
from seqlbtoolkit.chmm.dataset import collate_fn

from label_model.sparse_chmm.train import SparseCHMMTrainer
from label_model.sparse_chmm.args import SparseCHMMArguments, SparseCHMMConfig
from label_model.sparse_chmm.dataset import CHMMDataset
from label_model.sparse_chmm.macro import *

logger = logging.getLogger(__name__)


def chmm_train(args: SparseCHMMArguments):
    set_seed(args.seed)
    config = SparseCHMMConfig().from_args(args)

    # create output dir if it does not exist
    os.makedirs(args.output_dir, exist_ok=True)

    # load dataset
    training_dataset = valid_dataset = test_dataset = None
    if args.load_preprocessed_dataset:
        logger.info('Loading pre-processed datasets...')
        file_dir = os.path.split(args.train_path)[0]
        try:
            training_dataset = CHMMDataset().load(
                file_dir=file_dir,
                dataset_type='train',
                config=config
            )
            valid_dataset = CHMMDataset().load(
                file_dir=file_dir,
                dataset_type='valid',
                config=config
            )
            test_dataset = CHMMDataset().load(
                file_dir=file_dir,
                dataset_type='test',
                config=config
            )
        except Exception as err:
            logger.exception(f"Encountered error {err} while loading the pre-processed datasets")
            training_dataset = valid_dataset = test_dataset = None

    if training_dataset is None:
        if args.train_path:
            logger.info('Loading training dataset...')
            training_dataset = CHMMDataset().load_file(
                file_path=args.train_path,
                config=config
            )
        if args.valid_path:
            logger.info('Loading validation dataset...')
            valid_dataset = CHMMDataset().load_file(
                file_path=args.valid_path,
                config=config
            )
        if args.test_path:
            logger.info('Loading test dataset...')
            test_dataset = CHMMDataset().load_file(
                file_path=args.test_path,
                config=config
            )

        if config.save_dataset:
            logger.info(f"Saving datasets")
            output_dir = os.path.split(config.train_path)[0] if config.save_dataset_to_data_dir else args.output_dir

            training_dataset.save(output_dir, 'train', config)
            valid_dataset.save(output_dir, 'valid', config)
            test_dataset.save(output_dir, 'test', config)

    if not config.add_majority_voting:
        training_dataset.remove_src(MV_LF_NAME, config)
        valid_dataset.remove_src(MV_LF_NAME, config)
        test_dataset.remove_src(MV_LF_NAME, config)

    chmm_trainer = SparseCHMMTrainer(
        config=config,
        collate_fn=collate_fn,
        training_dataset=training_dataset,
        valid_dataset=valid_dataset,
        test_dataset=test_dataset,
    ).initialize_trainer()

    logger.info("Start training CHMM")
    chmm_trainer.train()

    logger.info("Collecting garbage")
    gc.collect()
    torch.cuda.empty_cache()

    logger.info("Process finished!")


if __name__ == '__main__':

    _time = datetime.now().strftime("%m.%d.%y-%H.%M")
    _current_file_name = os.path.basename(__file__)
    if _current_file_name.endswith('.py'):
        _current_file_name = _current_file_name[:-3]

    # --- set up arguments ---
    parser = HfArgumentParser(SparseCHMMArguments)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        chmm_args, = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        chmm_args, = parser.parse_args_into_dataclasses()

    # Setup logging
    if chmm_args.log_dir is None:
        chmm_args.log_dir = os.path.join('logs', f'{_current_file_name}', f'{_time}.log')

    set_logging(log_dir=chmm_args.log_dir)
    logging_args(chmm_args)

    try:
        chmm_train(args=chmm_args)
    except Exception as e:
        logger.exception(e)
        raise e
