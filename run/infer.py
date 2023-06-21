# coding=utf-8
""" Train the conditional hidden Markov model """

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

from sparse_chmm.train import Trainer
from sparse_chmm.args import Arguments, Config
from sparse_chmm.dataset import Dataset

logger = logging.getLogger(__name__)


def chmm_infer(args: Arguments):
    set_seed(args.seed)
    config = Config().from_args(args)

    # load dataset
    test_dataset = None
    if args.load_preprocessed_dataset:
        logger.info('Loading pre-processed datasets...')
        try:
            test_dataset = Dataset().load(
                file_dir=os.path.split(args.test_path)[0],
                dataset_type='test',
                config=config
            )
        except Exception as err:
            logger.exception(f"Encountered error {err} while loading the pre-processed datasets")

    if test_dataset is None:
        if args.test_path:
            logger.info('Loading test dataset...')
            test_dataset = Dataset().load_file(
                file_path=args.test_path,
                config=config
            )

        if config.save_dataset:
            logger.info(f"Saving datasets")
            output_dir = os.path.split(config.test_path)[0] if config.save_dataset_to_data_dir else args.output_dir

            test_dataset.save(output_dir, 'test', config)

    chmm_trainer = Trainer(
        config=config,
        test_dataset=test_dataset,
    ).initialize_model()

    logger.info("Inference started!")
    chmm_trainer.inference()

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
    parser = HfArgumentParser(Arguments)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        chmm_args, = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        chmm_args, = parser.parse_args_into_dataclasses()

    # Setup logging
    if chmm_args.log_path is None:
        chmm_args.log_path = os.path.join('logs', f'{_current_file_name}', f'{_time}.log')

    set_logging(log_dir=chmm_args.log_path)
    logging_args(chmm_args)

    try:
        chmm_infer(args=chmm_args)
    except Exception as e:
        logger.exception(e)
        raise e
