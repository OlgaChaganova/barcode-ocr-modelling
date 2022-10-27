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
        batch_size=8,
        test_size=0.1,
        num_workers=6,
    ),

    model=Model(
        number_class_symbols=11,  # 10 (0-9) + 1 (blank)
        time_feature_count=512,
        lstm_hidden=256,
        lstm_len=3,
    ),

    train=Train(
        trainer_params={
            'devices': 1,
            'accelerator': 'auto',
            'accumulate_grad_batches': 4,
            'auto_scale_batch_size': None,
            'gradient_clip_val': 0,
            'benchmark': True,
            'precision': 32,
            'profiler': 'simple',
            'max_epochs': 10,
            'auto_lr_find': None,
        },

        callbacks=Callbacks(
            model_checkpoint=pl.callbacks.ModelCheckpoint(
                dirpath='/root/cvr-hw2-ocr/checkpoints/resnet_34/',
                save_top_k=3,
                monitor='val_loss_epocg',
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
            name='ReduceLROnPlateau',
            lr_sched_params={
                'patience': 3,
                'factor': 0.1,
                'mode': 'min',
                'min_lr': 0.00001,
            },
        ),

        criterion=Criterion(
            loss=nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
        ),
        ckpt_path=None,
    ),
)
