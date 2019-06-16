import numpy as np

from lie_detector.models.base import Model
from lie_detector.datasets.trial_dataset import TrialDataset
from lie_detector.datasets.dataset_sequence import DatasetSequenceXOnly
from lie_detector.networks.face_net import RESNET50
from typing import Callable, Dict, Tuple
from keras import backend as K

class CNNModel(Model):
    def __init__(self,
                 dataset_cls: type = TrialDataset,
                 network_fn: Callable = RESNET50,
                 dataset_args: Dict = None,
                 network_args: Dict = None):
        """Define the default dataset and network values for this model."""
        super().__init__(dataset_cls, network_fn, dataset_args, network_args)


    def generate_features(self, X, y=None, batch_size=32):
        s = X.shape
        X_inp = X.reshape((s[0]*s[1], s[2], s[3], s[4]))
        # X_inp = X_inp[...,::-1]
        # X_inp = np.true_divide(X_inp, 255.0)
        # X_inp = X_inp.astype(np.float32)
        
        X_inp = preprocess_input(X_inp.astype(np.float64)).astype(np.float32)

        print('Generating CNN features for each frame...')
        sequence = DatasetSequenceXOnly(X_inp, batch_size=batch_size)
        
        
        preds_raw = self.network.predict_generator(sequence, steps=len(sequence))

        feats = preds_raw.reshape((s[0], s[1], -1))

        return feats

    def predict_on_image(self, image: np.ndarray) -> Tuple[str, float]:
        if image.dtype == np.uint8:
            image = (image / 255).astype(np.float32)
        pred_raw = self.network.predict(np.expand_dims(image, 0), batch_size=1).flatten()

        return pred_raw


def preprocess_input(x, data_format=None, version=1):
    x_temp = np.copy(x)
    if data_format is None:
        data_format = K.image_data_format()
    assert data_format in {'channels_last', 'channels_first'}

    if version == 1:
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

    elif version == 2:
        if data_format == 'channels_first':
            x_temp = x_temp[:, ::-1, ...]
            x_temp[:, 0, :, :] -= 91.4953
            x_temp[:, 1, :, :] -= 103.8827
            x_temp[:, 2, :, :] -= 131.0912
        else:
            x_temp = x_temp[..., ::-1]
            x_temp[..., 0] -= 91.4953
            x_temp[..., 1] -= 103.8827
            x_temp[..., 2] -= 131.0912
    else:
        raise NotImplementedError

    return x_temp