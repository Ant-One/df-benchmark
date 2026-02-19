'''
# author: Antoine Moix, adapted from Deressa Wodajo, Hannes Mareen, Peter Lambert, Solomon Atnafu, Zahid Akhtar, Glenn Van Wallendael
# email: antoine.moix@epfl.ch
# date: 2026-01
# description: Class for the GenConvit detector

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

from detectors import GenConVitEDDetector
from detectors import GenConVitVAEDetector

logger = logging.getLogger(__name__)

@DETECTOR.register_module(module_name='genconvit')
class GenConVitDetector(AbstractDetector):
    def __init__(self, config):
        super().__init__()

        self.net = net
        self.fp16 = fp16

        self.model_ed = GenConVitEDDetector(config)
        self.model_vae = GenConVitVAEDetector(config)