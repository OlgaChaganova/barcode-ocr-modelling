import pytorch_lightning as pl


from configs.config import Config
from configs.config import LRScheduler
from configs.config import Optimizer


def _get_dict_for_optimizer(config_optimizer: Optimizer) -> dict:
    _dict = dict()
    _dict['_'.join(['optimizer', 'name'])] = config_optimizer.name
    for key, value in config_optimizer.opt_params.items():
        _dict['_'.join(['optimizer', key])] = value
    return _dict


def _get_dict_for_lr_scheduler(config_lr_scheduler: LRScheduler) -> dict:
    _dict = dict()
    _dict['_'.join(['lr_scheduler', 'name'])] = config_lr_scheduler.name
    for key, value in config_lr_scheduler.lr_sched_params.items():
        _dict['_'.join(['lr_scheduler', key])] = value
    return _dict


def get_config_dict(model: pl.LightningModule, datamodule: pl.LightningDataModule, config: Config) -> dict:
    config_dict = dict()
    model_dict = dict(model.hparams)
    datamodule_dict = dict(datamodule.hparams)
    trainer_dict = config.train.trainer_params
    optimizer_dict = _get_dict_for_optimizer(config.train.optimizer)
    lr_sched_dict = _get_dict_for_lr_scheduler(config.train.lr_scheduler)

    # using update to avoid doubling keys
    config_dict.update(model_dict)
    config_dict.update(datamodule_dict)
    config_dict.update(trainer_dict)
    config_dict.update(optimizer_dict)
    config_dict.update(lr_sched_dict)
    return config_dict
