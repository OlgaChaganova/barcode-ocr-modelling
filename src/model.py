import logging

import pytorch_lightning as pl
import torch
import torchvision
from torch import nn  # noqa: WPS458

from configs.base import Criterion, LRScheduler, Model, Optimizer


def _get_resnet34_backbone():
    model = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1)
    input_conv = nn.Conv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=7,
        stride=1,
        padding=3,
    )
    blocks = [
        input_conv,
        model.bn1,
        model.relu,
        model.maxpool,
        model.layer1,
        model.layer2,
        model.layer3,
    ]
    return nn.Sequential(*blocks)


class BiLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,  # The number of expected features in the input x
            hidden_size,  # The number of features in the hidden state h
            num_layers,
            dropout=dropout,
            bidirectional=True,
            batch_first=True,
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return out  # output features (h_t) from the last layer of the LSTM, for each t


class CRNN(nn.Module):
    def __init__(
        self,
        number_class_symbols: int,
        time_feature_count: int = 256,
        lstm_hidden: int = 256,
        lstm_len: int = 3,
    ):
        super().__init__()
        self.feature_extractor = _get_resnet34_backbone()
        self.bi_lstm = BiLSTM(
            input_size=time_feature_count,
            hidden_size=lstm_hidden,
            num_layers=lstm_len,
        )
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, time_feature_count),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(time_feature_count, number_class_symbols),
        )
        self.time_feature_count = time_feature_count

    def forward(self, x):
        x = self.feature_extractor(x)
        b, c, h, w = x.size()
        x = x.view(b, c * h, w)
        x = nn.functional.adaptive_avg_pool2d(x, (self.time_feature_count, w))
        x = x.transpose(1, 2)
        x = self.bi_lstm(x)
        x = self.classifier(x)  # [batch_size, time_steps, alphabet_size]
        return nn.functional.log_softmax(x, dim=2).permute(1, 0, 2)  # [time_steps, batch_size, alphabet_size]


class OCRModel(pl.LightningModule):
    def __init__(
        self,
        model: Model,
        optimizer: Optimizer,
        lr_scheduler: LRScheduler,
        criterion: Criterion,
    ):
        super().__init__()

        self.crnn = CRNN(
            number_class_symbols=model.number_class_symbols,
            time_feature_count=model.time_feature_count,
            lstm_hidden=model.lstm_hidden,
            lstm_len=model.lstm_len,
        )

        self.criterion = criterion.loss
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def forward(self, batch: torch.tensor):
        return self.crnn(batch)

    def training_step(self, batch, batch_idx):
        images, _, encoded_text, target_length = batch

        preds = self.forward(images)  # [time_steps, batch_size, alphabet_size (10 цифр + 1 для blank символа)]

        input_length = torch.full(size=(preds.size(1),), fill_value=preds.size(0), dtype=torch.long)

        loss = self.criterion(preds, encoded_text, input_length, target_length)
        self.log('train_loss_batch', loss, on_epoch=False, on_step=True, batch_size=images.shape[0])
        self.log('train_loss_epoch', loss, on_epoch=True, on_step=False, batch_size=images.shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        images, _, encoded_text, target_length = batch

        preds = self.forward(images)

        input_length = torch.full(size=(preds.size(1),), fill_value=preds.size(0), dtype=torch.long)

        loss = self.criterion(preds, encoded_text, input_length, target_length)
        logging.info(f'VAl loss: {loss.item()}')
        self.log('val_loss_batch', loss, on_epoch=False, on_step=True, batch_size=images.shape[0])
        self.log('val_loss_epoch', loss, on_epoch=True, on_step=False, batch_size=images.shape[0])
        return loss

    def test_step(self, batch, batch_idx):
        images, _, encoded_text, target_length = batch

        preds = self.forward(images)

        input_length = torch.full(size=(preds.size(1),), fill_value=preds.size(0), dtype=torch.long)

        loss = self.criterion(preds, encoded_text, input_length, target_length)
        self.log('test_loss_batch', loss, on_epoch=False, on_step=True, batch_size=images.shape[0])
        self.log('test_loss_epoch', loss, on_epoch=True, on_step=False, batch_size=images.shape[0])
        return loss

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.optimizer.name)(
            self.parameters(),
            **self.optimizer.opt_params,
        )
        optim_dict = {
            'optimizer': optimizer,
            'monitor': 'val_loss_epoch',
        }

        if self.lr_scheduler is not None:
            lr_scheduler = getattr(torch.optim.lr_scheduler, self.lr_scheduler.name)(
                optimizer,
                **self.lr_scheduler.lr_sched_params,
            )
            optim_dict.update({'lr_scheduler': lr_scheduler})
        return optim_dict
