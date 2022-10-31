"""Use this file to change experiment settings."""

import pytorch_lightning as pl
from torch import nn

from configs.base import (Callbacks, Common, Config, Criterion, Dataset,
                   LRScheduler, Model, Optimizer, Project, Train)

NUM_CLASSES = 17
IMG_SIZE = 256

CONFIG = Config(
    project=Project(
        project_name='cvr-hw2-ocr-modelling',
        task_name='ocr_resnet34',
    ),

    common=Common(seed=8),

    dataset=Dataset(
        images_dir='data/barcodes-annotated-gorai/images',
        annot_file='data/barcodes-annotated-gorai/full_annotation.tsv',
        batch_size=48,
        test_size=0.1,
        num_workers=6,
        img_height=128,
        img_width=256,
    ),

    model=Model(
        number_class_symbols=11,  # 10 (0-9) + 1 (blank)
        time_feature_count=324,
        lstm_hidden=256,
        lstm_len=3,
    ),

    train=Train(
        trainer_params={
            'devices': 1,
            'accelerator': 'auto',
            'accumulate_grad_batches': 2,
            'auto_scale_batch_size': None,
            'gradient_clip_val': 0,
            'benchmark': True,
            'precision': 32,
            'profiler': None,
            'max_epochs': 1,
            'auto_lr_find': None,
        },

        callbacks=Callbacks(
            model_checkpoint=pl.callbacks.ModelCheckpoint(
                dirpath='/root/cvr-hw2-ocr/checkpoints/resnet_34/',
                save_top_k=2,
                monitor='val_loss_epoch',
                mode='min',
            ),

            lr_monitor=pl.callbacks.LearningRateMonitor(logging_interval='step'),
        ),

        optimizer=Optimizer(
            name='Adam',
            opt_params={
                'lr': 0.001,
                'weight_decay': 0.0001,
            },
        ),

        lr_scheduler=LRScheduler(
            name='CosineAnnealingLR',
            lr_sched_params={
                'T_max': 50,
                'eta_min': 0.00001,
            },
        ),

        criterion=Criterion(
            loss=nn.CTCLoss(blank=0, reduction='mean', zero_infinity=False)
        ),
        ckpt_path=None,
    ),
)
