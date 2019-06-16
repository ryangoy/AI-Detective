"""Model class, to be extended by specific types of models."""
from pathlib import Path
from typing import Callable, Dict, Optional

from keras.optimizers import RMSprop, Adam
import numpy as np

from lie_detector.datasets.dataset_sequence import DatasetSequence


DIRNAME = Path(__file__).parents[1].resolve() / 'weights' / 'cache'


class Model:
    """Base class, to be subclassed by predictors for specific type of data."""
    def __init__(self, dataset_cls: type, network_fn: Callable, dataset_args: Dict = None, network_args: Dict = None, input_shape=None):
        self.name = '{}_{}_{}'.format(self.__class__.__name__, dataset_cls.__name__ , network_fn.__name__)

        if dataset_args is None:
            dataset_args = {}
        

        if network_args is None:
            network_args = {}

        if input_shape is not None:
            self.network = network_fn(input_shape=input_shape, **network_args)
        else:
            self.data = dataset_cls(**dataset_args)
            self.network = network_fn(input_shape=self.data.input_shape, **network_args)
        # self.network.summary()

        self.batch_augment_fn = None
        self.batch_format_fn = None


    # @property
    # def image_shape(self):
    #     return self.data.input_shape

    @property
    def weights_filename(self) -> str:
        DIRNAME.mkdir(parents=True, exist_ok=True)
        return str(DIRNAME / '{}_weights.h5'.format(self.name))

    def fit(self, dataset, batch_size: int = 32, epochs: int = 10, augment_val: bool = True, callbacks: list = None):
        if callbacks is None:
            callbacks = []

        self.network.compile(loss=self.loss(), optimizer=self.optimizer(), metrics=self.metrics())

        print(dataset.y_trn)
        train_sequence = DatasetSequence(
            dataset.X_trn,
            dataset.y_trn,
            batch_size,
            augment_fn=self.batch_augment_fn,
            format_fn=self.batch_format_fn
        )
        test_sequence = DatasetSequence(
            dataset.X_val,
            dataset.y_val,
            batch_size,
            augment_fn=self.batch_augment_fn if augment_val else None,
            format_fn=self.batch_format_fn
        )

        self.network.fit_generator(
            generator=train_sequence,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=test_sequence,
            use_multiprocessing=False,
            workers=1,
            shuffle=True
        )

    def evaluate(self, x, y, batch_size=16, verbose=False):  # pylint: disable=unused-argument
        sequence = DatasetSequence(x, y, batch_size=batch_size)  # Use a small batch size to use less memory
        preds = self.network.predict_generator(sequence)
        
        length = len(preds)
        y = np.array(y).reshape((length,))
        preds = np.array(preds).reshape((length,))
        print(preds)
        print(y)
        return np.sqrt(np.sum(np.square(preds - y)))

    def loss(self):
        return 'binary_crossentropy'

    def optimizer(self):
        return Adam()

    def metrics(self):
        return ['accuracy']

    def load_weights(self):
        self.network.load_weights(self.weights_filename)

    def save_weights(self):
        self.network.save_weights(self.weights_filename) 