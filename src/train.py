import argparse
import logging
import os
import typing as tp
from runpy import run_path

import wandb
from clearml import Task
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import RichModelSummary
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger

from configs.base import Config
from data.dataset import OCRBarcodeDataModule
from model import OCRModel
from utils import get_config_dict


def parse() -> tp.Any:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', default='src/configs/config.py', type=str, help='Path to experiment config file (*.py)',
    )
    parser.add_argument(
        '--log', action=argparse.BooleanOptionalAction, help='Log experiment or not (for debug)',
    )
    parser.add_argument(
        '--logger', type=str, choices=['clearml', 'wandb'], help='Logger',
    )

    return parser.parse_args()


def main(args: tp.Any, config: Config):
    logging.basicConfig(level=logging.INFO)

    model = OCRModel(
        model=config.model,
        optimizer=config.train.optimizer,
        lr_scheduler=config.train.lr_scheduler,
        criterion=config.train.criterion,
    )

    datamodule = OCRBarcodeDataModule(
        images_dir=config.dataset.images_dir,
        annot_file=config.dataset.annot_file,
        img_height=config.dataset.img_height,
        img_width=config.dataset.img_width,
        batch_size=config.dataset.batch_size,
        test_size=config.dataset.test_size,
        num_workers=config.dataset.num_workers,
        add_border=config.dataset.add_border,
    )

    if args.log:
        config_dict = get_config_dict(model=model, datamodule=datamodule, config=config)
        if args.logger == 'clearml':
            task = Task.init(project_name=config.project.project_name, task_name=config.project.task_name)
            task.connect(config_dict)
            task.upload_artifact('exp_config', artifact_object=args.config, delete_after_upload=False)
            logger=True

        elif args.logger == 'wandb':
            logger = WandbLogger(
                project=config.project.project_name,
                config=config_dict,
                log_model='all',
            )
            logger.watch(
                model=model,
                log='all',
                log_freq=5,  # log gradients and parameters every log_freq batches
            )
            base_path = os.path.split(args.config)[0]
            wandb.save(args.config, base_path=base_path, policy='now')

    # trainer
    trainer_params = config.train.trainer_params
    callbacks = list(config.train.callbacks.__dict__.values())
    callbacks = filter(lambda callback: callback is not None, callbacks)
    trainer = Trainer(
        logger=logger,
        callbacks=[
            TQDMProgressBar(refresh_rate=1),
            RichModelSummary(),
            *callbacks,
        ],
        **trainer_params,
    )

    if trainer_params['auto_scale_batch_size'] is not None or trainer_params['auto_lr_find'] is not None:
        trainer.tune(model=model, datamodule=datamodule)

    trainer.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=config.train.ckpt_path,
    )

    trainer.test(
        model=model,
        datamodule=datamodule,
    )
    if args.log and args.logger == 'clearml':
        task.close()


if __name__ == '__main__':
    args = parse()
    config_module = run_path(args.config)
    exp_config = config_module['CONFIG']
    seed_everything(exp_config.common.seed, workers=True)
    main(args, exp_config)

