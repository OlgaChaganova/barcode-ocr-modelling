import typing as tp
from collections import namedtuple

import pytorch_lightning as pl


from configs.config import Config
from configs.config import LRScheduler
from configs.config import Optimizer


def convert_dict_to_tuple(dictionary: dict) -> tp.NamedTuple:
    for key, value in dictionary.items():
        if isinstance(value, dict):
            dictionary[key] = convert_dict_to_tuple(value)
    return namedtuple('GenericDict', dictionary.keys())(**dictionary)


def flatten_dict(dictionary: dict, parent_key: str = '', sep: str = '_') -> dict:
    items = []
    for k, v in dictionary.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def convert_namedtuple_to_dict(named_tuple: tp.NamedTuple) -> dict:
    params_dict = dict(named_tuple._asdict())
    params_dict.pop('name', None)
    return params_dict


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

    # using update to avoid doubling keys for img_size, num_channels, num_classes
    config_dict.update(model_dict)
    config_dict.update(datamodule_dict)
    config_dict.update(trainer_dict)
    config_dict.update(optimizer_dict)
    config_dict.update(lr_sched_dict)
    return config_dict
