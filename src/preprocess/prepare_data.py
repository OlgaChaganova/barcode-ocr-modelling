"""Prepare dataset: crop images by the bounding boxes."""

import argparse
import logging
import os
import shutil
import typing as tp
from pathlib import Path

import cv2
from detector import BarcodeDetector
from omegaconf import OmegaConf
from tqdm import tqdm


def parse():
    parser = argparse.ArgumentParser(
        'Argument parser for dataset preparing for OCR model training. '
        'Detect barcode, crop image and save it.',
    )
    parser.add_argument(
        'root', type=str, help='Path to raw dataset root directory (e.g. data/dataset_ocr/images/)',
    )
    parser.add_argument(
        'new_root', type=str, help='Path to preprocessed dataset root directory (e.g. data/dataset_ocr_detected/)',
    )
    parser.add_argument(
        '--detector_cfg', type=str, default='src/configs/detector.yml', help='Path to detector config',
    )
    parser.add_argument(
        '--img_ext', type=str, default='jpg', help='Image extention',
    )
    return parser.parse_args()


def _ignore_files(directory: str, files: str):
    return [file for file in files if os.path.isfile(os.path.join(directory, file))]


def copy_folders_structure(root: str, new_root: str) -> None:
    shutil.copytree(
        src=root,
        dst=new_root,
        ignore=_ignore_files,
        dirs_exist_ok=True,
    )


def crop_images(root: str, new_root: str, detector_cfg: str, img_ext: tp.Literal['jpg', 'jpeg', 'png']):
    detector_cfg = OmegaConf.load(detector_cfg)
    detector = BarcodeDetector(config=detector_cfg)
    files = Path(root).rglob(f'*.{img_ext}')
    files = list(map(str, files))

    for file in tqdm(files):
        image = cv2.imread(file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        try:  # noqa: WPS229
            detected_image = detector.detect(image)
            save_path = file.replace(root, new_root)
            cv2.imwrite(save_path, cv2.cvtColor(detected_image, cv2.COLOR_RGB2BGR))
        except IndexError:
            logging.error(f'Could not detect barcode for image: {file}')


def main(args: tp.Any):
    copy_folders_structure(args.root, args.new_root)
    crop_images(args.root, args.new_root, args.detector_cfg, args.img_ext)


if __name__ == '__main__':
    args = parse()
    logging.basicConfig(level=logging.INFO)
    main(args)
