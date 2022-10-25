import typing as tp
from dataclasses import dataclass

import pytorch_lightning as pl
from torch import nn


@dataclass
class Project:
    project_name: str
    task_name: str


@dataclass
class Common:
    seed: int = 8


@dataclass
class Dataset:
    root: str  # path to root directory with images
    num_workers: int


@dataclass
class Model:
    number_class_symbols: int
    time_feature_count: int = 256
    lstm_hidden: int = 256
    lstm_len: int = 3
    pretrained_backbone: bool = True


@dataclass
class Callbacks:
    model_checkpoint: pl.callbacks.ModelCheckpoint
    early_stopping: tp.Optional[pl.callbacks.EarlyStopping] = None
    lr_monitor: tp.Optional[pl.callbacks.LearningRateMonitor] = None
    model_summary: tp.Optional[tp.Union[pl.callbacks.ModelSummary, pl.callbacks.RichModelSummary]] = None
    timer: tp.Optional[pl.callbacks.Timer] = None


@dataclass
class Optimizer:
    name: str
    opt_params: dict


@dataclass
class LRScheduler:
    name: str
    lr_sched_params: dict


@dataclass
class Criterion:
    loss: nn.Module


@dataclass
class Train:
    trainer_params: dict
    callbacks: Callbacks
    optimizer: Optimizer
    lr_scheduler: LRScheduler
    criterion: Criterion
    ckpt_path: tp.Optional[str] = None


@dataclass
class Config:
    project: Project
    common: Common
    dataset: Dataset
    model: Model
    train: Train
