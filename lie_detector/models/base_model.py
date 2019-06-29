import numpy as np

from lie_detector.models.model import Model
from lie_detector.datasets.trial_dataset import TrialDataset
from lie_detector.networks.lstm_network import LSTM
from typing import Callable, Dict, Tuple


class BaseModel(Model):
    def __init__(self,
                 dataset_cls: type = TrialDataset,
                 network_fn: Callable = LSTM,
                 dataset_args: Dict = None,
                 network_args: Dict = None,
                 input_shape=None):
        """Define the default dataset and network values for this model."""
        super().__init__(network_fn=network_fn, network_args=network_args, input_shape=input_shape)
