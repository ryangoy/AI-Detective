from tensorflow.keras import backend as K
import numpy as np
from typing import Callable, Dict, Tuple

from lie_detector.models.model import Model
from lie_detector.datasets.trial_dataset import TrialDataset
from lie_detector.datasets.dataset_sequence import DatasetSequence
from lie_detector.networks.feature_network import RESNET50


class CNNModel(Model):
    def __init__(self,
                 dataset_cls: type = TrialDataset,
                 network_fn: Callable = RESNET50,
                 dataset_args: Dict = None,
                 network_args: Dict = None):
        """Define the default dataset and network values for this model."""
        super().__init__(network_fn=network_fn, network_args=network_args, input_shape=dataset_cls.get_input_shape())


    def generate_features(self, X, y=None, batch_size=32):
        s = X.shape
        X_inp = X.reshape((s[0]*s[1], s[2], s[3], s[4]))  
        X_inp = preprocess_input(X_inp.astype(np.float64)).astype(np.float32)

        print('Generating CNN features for each frame...')
        sequence = DatasetSequence(X_inp, batch_size=batch_size)
        preds_raw = self.network.predict_generator(sequence, steps=len(sequence))
        feats = preds_raw.reshape((s[0], s[1], -1))

        return feats, y


def preprocess_input(x, data_format=None):
    x_temp = np.copy(x)
    if data_format is None:
        data_format = K.image_data_format()
    assert data_format in {'channels_last', 'channels_first'}

    if data_format == 'channels_first':
        x_temp = x_temp[:, ::-1, ...]
        x_temp[:, 0, :, :] -= 93.5940
        x_temp[:, 1, :, :] -= 104.7624
        x_temp[:, 2, :, :] -= 129.1863
    else:
        x_temp = x_temp[..., ::-1]
        x_temp[..., 0] -= 93.5940
        x_temp[..., 1] -= 104.7624
        x_temp[..., 2] -= 129.1863

    return x_temp