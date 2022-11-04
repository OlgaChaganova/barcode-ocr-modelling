import argparse
import logging
import os
import typing as tp
from runpy import run_path

import numpy as np
import onnxruntime
import torch

from configs.base import Config
from model import OCRModel


def parse() -> tp.Any:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', default='src/configs/config.py', type=str, help='Path to experiment config file (*.py)',
    )
    parser.add_argument(
        '--ckpt_path', required=True, type=str, help='Path to experiment checkpoint (*.ckpt)',
    )
    parser.add_argument(
        '--dir_to_save', type=str, default='weights/', help='Path to directory where .pt model will be saved',
    )
    parser.add_argument(
        '--check', action='store_true', help='Check correctness of converting by shape of output',
    )
    return parser.parse_args()


def convert_from_checkpoint(args: tp.Any, config: Config) -> tp.Tuple[str, torch.tensor, torch.tensor]:
    model = OCRModel.load_from_checkpoint(
        args.ckpt_path,
        model=config.model,
        optimizer=config.train.optimizer,
        lr_scheduler=config.train.lr_scheduler,
        criterion=config.train.criterion,
    ).eval()

    model_name = os.path.split(args.ckpt_path)[-1].replace('ckpt', 'onnx')
    model_path = os.path.join(args.dir_to_save, model_name)

    input_sample = torch.randn((1, 3, config.dataset.img_height, config.dataset.img_width))
    output_sample = model(input_sample)

    model.to_onnx(
        model_path,
        input_sample,
        export_params=True,
        input_names = ['input'],  # the model's input names
        output_names = ['output'],  # the model's output names
    )

    if os.path.isfile(model_path):
        logging.info(f'Model was successfully saved. File name: {model_path}')
    else:
        raise ValueError('An error was occurred. Check paths and try again.')
    return model_path, input_sample, output_sample.detach()


def check(model_path: str, input_sample: torch.tensor, output_sample: torch.tensor):
    logging.info(input_sample.mean)
    ort_session = onnxruntime.InferenceSession(model_path)
    ort_inputs = {ort_session.get_inputs()[0].name: input_sample.numpy()}
    output_sample_onnx = ort_session.run(None, ort_inputs)[0]

    if np.allclose(output_sample_onnx, output_sample.numpy(), atol=1e-2):
        logging.info('Model can be loaded and outputs look good!')
    else:
        logging.info(f'ONNX: {output_sample_onnx.shape}, torch: {output_sample.numpy().shape}')
        logging.info(f'ONNX: {output_sample_onnx[0]}, torch: {output_sample.numpy()[0]}')
        logging.error('Outputs of the converted model don\'t match output of the original torch model.')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = parse()
    config_module = run_path(args.config)
    exp_config = config_module['CONFIG']
    pt_model_path, input_sample, output_sample = convert_from_checkpoint(args, exp_config)
    if args.check:
        check(pt_model_path, input_sample, output_sample)
