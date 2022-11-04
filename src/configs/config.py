"""Use this file to change experiment settings."""

import pytorch_lightning as pl
from torch import nn

from configs.base import (Callbacks, Common, Config, Criterion, Dataset,
                   LRScheduler, Model, Optimizer, Project, Train)


CONFIG = Config(
    project=Project(
        project_name='cvr-hw2-ocr-modelling',
        task_name='ocr_resnet34_20-22',
    ),

    common=Common(seed=8),

    dataset=Dataset(
        images_dir='data/barcodes-annotated-gorai/images',
        annot_file='data/barcodes-annotated-gorai/full_annotation.tsv',
        batch_size=128,
        test_size=0.1,
        num_workers=6,
        img_height=128,
        img_width=512,
        add_border=5,
    ),

    model=Model(
        number_class_symbols=11,  # 10 (0-9) + 1 (blank)
        time_feature_count=64,
        lstm_hidden=128,
        lstm_len=1,
    ),

    train=Train(
        trainer_params={
            'devices': 1,
            'accelerator': 'auto',
            'accumulate_grad_batches': 1,
            'auto_scale_batch_size': None,
            'gradient_clip_val': 0,
            'benchmark': True,
            'precision': 32,
            'profiler': None,
            'max_epochs': 100,
            'auto_lr_find': None,
        },

        callbacks=Callbacks(
            model_checkpoint=pl.callbacks.ModelCheckpoint(
                dirpath='checkpoints/resnet34_20-22/',
                save_top_k=2,
                monitor='val_loss_epoch',
                mode='min',
            ),

            lr_monitor=pl.callbacks.LearningRateMonitor(logging_interval='step'),
        ),

        optimizer=Optimizer(
            name='Adam',
            opt_params={
                'lr': 1e-3,
                'weight_decay': 0.0001,
            },
        ),

        lr_scheduler=LRScheduler(
            name='CosineAnnealingLR',
            lr_sched_params={
                'T_max': 500,
                'eta_min': 1e-8,
            },
        ),

        criterion=Criterion(
            loss=nn.CTCLoss(blank=10, reduction='mean', zero_infinity=True)
        ),
        ckpt_path=None,
    ),
)
