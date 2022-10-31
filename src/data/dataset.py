import logging
import os
import random
import typing as tp

import albumentations as alb
import albumentations.pytorch as alb_pt
import cv2
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as tvf

from .encoder import Encoder


class OCRBarcodeDataset(Dataset):
    """Barcode OCR dataset."""

    def __init__(
            self,
            images_dir: str,
            annot_df: pd.DataFrame,
            add_border,
            img_height: int,
            img_width: int,
    ):
        """
        Init OCR Barcode dataset.

        Parameters
        ----------
        images_dir : str
            Path to folder with images.

        annot_df : pd.DataFrame
            Annotations dataframe (filename, code, p1, p2).

        add_border : int
            Number of pixel to be added to the true borders.

        img_height : int
            Height of the image after preprocessing and augmentations.

        img_width : int
            Width of the image after preprocessing and augmentations.
        """
        self.images_dir = images_dir
        self.annot_df = annot_df
        self.add_border = add_border
        self.img_height = img_height
        self.img_width = img_width

        self.transform = alb.Compose(
            [
                # alb.Perspective(scale=0.05, keep_size=True, pad_mode=0, pad_val=(0, 0, 0), always_apply=True),
                alb.SmallestMaxSize(max_size=img_height, interpolation=0, always_apply=True),
                alb.PadIfNeeded(min_height=img_height, min_width=img_width, border_mode=0, value=(0, 0, 0), always_apply=True),
                # alb.Normalize(),
                alb_pt.ToTensorV2(),
            ],
        )

    def __len__(self):
        return len(self.annot_df)

    def load_image(self, filename: str) -> np.array:
        """
        Load image.

        Parameters
        ----------
        filename : str
            Name of the file with image.

        Returns
        -------
            Image (np.array)
        """
        filename = os.path.join(self.images_dir, filename)
        image = Image.open(filename)
        image = image.convert('RGB')
        image.load()
        return np.array(image)

    def __getitem__(self, ind):
        add = self.add_border
        filename, text, p1, p2 = self.annot_df.iloc[ind, :]
        image = self.load_image(filename)
        y_min, x_min = list(map(int, p1.replace('(', '').replace(')', '').split(',')))
        y_max, x_max = list(map(int, p2.replace('(', '').replace(')', '').split(',')))

        image = image[(y_min - add):(y_max + add), (x_min - add):(x_max + add), :]  # crop by bbox
        image = image[(image.shape[0] // 2):, :, :]  # crop by half of the image

        if image.shape[0] > image.shape[1]:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

        image = self.transform(image=image)['image'] / 255.

        if image.shape != (3, self.img_height, self.img_width):
            image = tvf.resize(image, size=[self.img_height, self.img_width])

        encoded_text = Encoder.encode(text)
        return image, text, encoded_text, len(text)


class OCRBarcodeDataModule(pl.LightningDataModule):
    def __init__(
        self,
        images_dir: str,
        annot_file: str,
        img_height: int,
        img_width: int,
        batch_size: int,
        test_size: float,
        num_workers: int,
        add_border: int = 0,
    ):
        """Create Data Module for Barcodes OCR.

        Parameters
        ----------
        images_dir : str
            Path to folder with images.

        annot_file : str
            Path to annotations file (full_annotation.tsv).

        img_height : int
            Height of the image after preprocessing and augmentations.

        img_width : int
            Width of the image after preprocessing and augmentations.

        batch_size : int
            Batch size for dataloaders.

        test_size : float
            Size (share) of valid and test datasets.

        num_workers : int
            Number of workers in dataloaders.

        add_border : int
            Number of pixel to be added to the true borders.

        """
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size

        self.images_dir = images_dir
        self.num_workers = num_workers
        self.add_border = add_border
        self.img_height = img_height
        self.img_width = img_width

        self._train_val_test_split(annot_file, test_size)

    def setup(self, stage: tp.Optional[str] = None):
        if stage == 'fit' or stage is None:
            self.train_dataset = OCRBarcodeDataset(
                images_dir=self.images_dir,
                annot_df=self.train_annot_df,
                add_border=self.add_border,
                img_height=self.img_height,
                img_width=self.img_width,
            )
            num_train_files = len(self.train_dataset)
            logging.info(f'Mode: train, number of files: {num_train_files}')

            self.val_dataset = OCRBarcodeDataset(
                images_dir=self.images_dir,
                annot_df=self.val_annot_df,
                add_border=self.add_border,
                img_height=self.img_height,
                img_width=self.img_width,
            )
            num_val_files = len(self.val_dataset)
            logging.info(f'Mode: val, number of files: {num_val_files}')

        elif stage == 'test':
            self.test_dataset = OCRBarcodeDataset(
                images_dir=self.images_dir,
                annot_df=self.test_annot_df,
                add_border=self.add_border,
                img_height=self.img_height,
                img_width=self.img_width,
            )
            num_test_files = len(self.test_dataset)
            logging.info(f'Mode: test, number of files: {num_test_files}')

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, drop_last=False,
        )

    def _train_val_test_split(self, annot_file: str, test_size):
        annot_df = pd.read_table(annot_file, sep='\t', dtype={'filename': str, 'code': str})
        indexes = list(annot_df.index)
        random.shuffle(indexes)
        train_cnt = int(len(annot_df) * (1 - test_size))
        test_val_cnt = (len(annot_df) - train_cnt) // 2
        train_inds = indexes[:train_cnt]
        val_inds = indexes[train_cnt:train_cnt + test_val_cnt]
        test_inds = indexes[train_cnt + test_val_cnt:]
        self.train_annot_df = annot_df.iloc[train_inds]
        self.val_annot_df = annot_df.iloc[val_inds]
        self.test_annot_df = annot_df.iloc[test_inds]
