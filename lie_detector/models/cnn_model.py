import numpy as np

from lie_detector.models.base import Model
from lie_detector.datasets.emnist_dataset import EmnistDataset
from lie_detector.networks.face_net import senet50


class CNNModel(Model):
    def __init__(self,
                 dataset_cls: type = TrialDataset,
                 network_fn: Callable = senet50,
                 dataset_args: Dict = None,
                 network_args: Dict = None):
        """Define the default dataset and network values for this model."""
        super().__init__(dataset_cls, network_fn, dataset_args, network_args)

    def predict_on_image(self, image: np.ndarray) -> Tuple[str, float]:
        if image.dtype == np.uint8:
            image = (image / 255).astype(np.float32)
        pred_raw = self.network.predict(np.expand_dims(image, 0), batch_size=1).flatten()

        return pred_raw