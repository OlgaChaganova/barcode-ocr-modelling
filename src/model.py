import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable

from src.configs.base import Model, Optimizer, LRScheduler, Criterion


def _get_resnet34_backbone(pretrained: bool = True):
    model = torchvision.models.resnet34(pretrained=pretrained)
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
            input_size,
            hidden_size,
            num_layers,
            dropout=dropout,
            bidirectional=True,
            batch_first=True,
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return out


class CRNN(nn.Module):
    def __init__(
        self,
        number_class_symbols: int,
        time_feature_count: int = 256,
        lstm_hidden: int = 256,
        lstm_len: int = 3,
        pretrained_backbone: bool = True,
    ):
        super().__init__()
        self.feature_extractor = _get_resnet34_backbone(pretrained=pretrained_backbone)
        self.avg_pool = nn.AdaptiveAvgPool2d((time_feature_count, time_feature_count))
        self.bi_lstm = BiLSTM(time_feature_count, lstm_hidden, lstm_len)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, time_feature_count),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(time_feature_count, number_class_symbols)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        b, c, h, w = x.size()
        x = x.view(b, c * h, w)
        x = self.avg_pool(x)
        x = x.transpose(1, 2)
        x = self.bi_lstm(x)
        x = self.classifier(x)
        return x


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
            pretrained_backbone=model.pretrained_backbone,
        )

        self.criterion = criterion.loss
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def forward(self, batch: torch.tensor):
        return self.crnn(batch).log_softmax(2)

    def training_step(self, batch, batch_idx):
        images, text_labels = batch
        batch_size = images.shape[0]

        preds = self.forward(images)

        input_length = Variable(torch.IntTensor([preds.size(0)] * batch_size))
        target_length = ...

        loss = self.criterion(preds, text_labels, input_length, target_length)
        self.log('train_loss_batch', loss, on_epoch=False, on_step=True)
        self.log('train_loss_epoch', loss, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        images, text_labels = batch
        batch_size = images.shape[0]

        preds = self.forward(images)

        input_length = Variable(torch.IntTensor([preds.size(0)] * batch_size))
        target_length = ...

        loss = self.criterion(preds, text_labels, input_length, target_length)
        self.log('val_loss_batch', loss, on_epoch=False, on_step=True)
        self.log('val_loss_epoch', loss, on_epoch=True, on_step=False)
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


