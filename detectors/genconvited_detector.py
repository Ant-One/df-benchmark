'''
# author: Antoine Moix, adapted from Deressa Wodajo, Hannes Mareen, Peter Lambert, Solomon Atnafu, Zahid Akhtar, Glenn Van Wallendael
# email: antoine.moix@epfl.ch
# date: 2026-01
# description: Class for the GenConviTED detector

Reference:
@misc{wodajo2023deepfake,
      title={Deepfake Video Detection Using Generative Convolutional Vision Transformer}, 
      author={Deressa Wodajo and Hannes Mareen and Peter Lambert and Solomon Atnafu and Zahid Akhtar and Glenn Van Wallendael},
      year={2023},
      eprint={2307.07036},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
'''

import logging

from .base_detector import AbstractDetector
from detectors import DETECTOR
from networks import HybridEmbed

import torch
import torch.nn as nn
from torchvision import transforms

import timm
from timm import create_model

logger = logging.getLogger(__name__)

class Encoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(    
            nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),

            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),

            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),

            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)
        )

    def forward(self, x):
        return self.features(x)
    

class Decoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=(2, 2), stride=(2, 2)),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=(2, 2), stride=(2, 2)),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=(2, 2), stride=(2, 2)),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 16, kernel_size=(2, 2), stride=(2, 2)),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(16, 3, kernel_size=(2, 2), stride=(2, 2)),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.features(x)

@DETECTOR.register_module(module_name='genconvited')
class GenConVitEDDetector(AbstractDetector):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.encoder = Encoder()
        self.decoder = Decoder()
        self.backbone = timm.create_model(config['backbone_name'], pretrained=config['is_pretrained'])
        self.embedder = timm.create_model(config['embedder_name'], pretrained=config['is_pretrained'])
        self.backbone.patch_embed = HybridEmbed(self.embedder, img_size=config['img_size'], embed_dim=768)

        self.num_features = self.backbone.head.fc.out_features * 2
        self.fc = nn.Linear(self.num_features, self.num_features//4)
        self.fc2 = nn.Linear(self.num_features//4, 2)
        self.relu = nn.GELU()

    def features(self, data_dict: dict):
        raise NotImplementedError

    def forward(self, data_dict: dict, inference=False) -> dict:
        encimg = self.encoder(data_dict)
        decimg = self.decoder(encimg)

        x1 = self.backbone(decimg)
        x2 = self.backbone(data_dict)

        x = torch.cat((x1,x2), dim=1)

        x = self.fc2(self.relu(self.fc(self.relu(x))))

        return x

    def classifier(self, features):
        raise NotImplementedError

    def build_backbone(self, config):
        raise NotImplementedError

    def build_loss(self, config):
        raise NotImplementedError

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        raise NotImplementedError

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        raise NotImplementedError
