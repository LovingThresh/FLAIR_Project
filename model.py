# -*- coding: utf-8 -*-
# @Time    : 2022/12/21 21:43
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : model.py
# @Software: PyCharm
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModel
import segmentation_models_pytorch as smp


class mtd_MLP(nn.Module):
    """
    Light MLP to encode metadata

    """

    def __init__(self):
        super(mtd_MLP, self).__init__()

        self.enc_mlp = nn.Sequential(
            nn.Linear(45, 64),
            nn.Dropout(0.4),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.Dropout(0.4),
            nn.ReLU(inplace=True),
            nn.Linear(32, 16),
            nn.Dropout(0.4),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.enc_mlp(x)
        return x


class SMP_Unet_meta(BaseModel):
    """
    Pytorch segmentation U-Net with ResNet34 (default)
    with added metadata information at encoder output

    """

    def __init__(self,
                 n_channels,
                 n_classes,
                 use_metadata=True
                 ):
        super(SMP_Unet_meta, self).__init__()

        self.seg_model = smp.create_model(arch="FPN", encoder_name="timm-regnety_160", classes=n_classes,
                                          in_channels=n_channels)
        self.use_metadata = use_metadata
        if use_metadata:
            self.enc = mtd_MLP()

    def forward(self, inputs, target, mode):
        if isinstance(inputs, list):
            input = inputs[0]
            met = inputs[1]
        else:
            input = inputs
            met = None
        if self.use_metadata:
            feats = self.seg_model.encoder(input)
            x_enc = self.enc(met)
            x_enc = x_enc.unsqueeze(1).unsqueeze(-1).repeat(1, 3024, 1, 16)
            feats[-1] = torch.add(feats[-1], x_enc)
            output = self.seg_model.decoder(*feats)
            output = self.seg_model.segmentation_head(output)
        else:
            output = self.seg_model(input)

        if mode == 'loss':
            return {'loss': F.cross_entropy(output, target)}
        elif mode == 'predict':
            return [inputs, output, target]
