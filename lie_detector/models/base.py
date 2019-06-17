"""Model class, to be extended by specific types of models."""
from pathlib import Path
from typing import Callable, Dict, Optional
from keras.optimizers import RMSprop, Adam
import numpy as np
from keras.callbacks import EarlyStopping
from time import time
from typing import Dict, Optional
from sklearn.metrics import mean_squared_error

from lie_detector.datasets.dataset_sequence import DatasetSequence
from lie_detector.datasets.dataset import Dataset

WEIGHTS_DIRNAME = Path(__file__).parents[1].resolve() / 'weights' / 'cache'


class Model:
    """Base class, to be subclassed by predictors for specific type of data."""
    def __init__(self, dataset_cls: type, network_fn: Callable, dataset_args: Dict = None, network_args: Dict = None, input_shape=None):
        self.name = '{}_{}_{}'.format(self.__class__.__name__, dataset_cls.__name__ , network_fn.__name__)

        if dataset_args is None:
            self.dataset_args = {}
        else:
            self.dataset_args = dataset_args
        
        if network_args is None:
            self.network_args = {}
        else:
            self.network_args = network_args

        if input_shape is not None:
            self.network = network_fn(input_shape=input_shape, **self.network_args)
        else:
            self.data = dataset_cls(**self.dataset_args)
            self.network = network_fn(input_shape=self.data.input_shape, **self.network_args)

        self.batch_augment_fn = None
        self.batch_format_fn = None


    @property
    def weights_filename(self) -> str:
        WEIGHTS_DIRNAME.mkdir(parents=True, exist_ok=True)
        return str(WEIGHTS_DIRNAME / '{}_weights.h5'.format(self.name))


    def fit(self, dataset, batch_size: int = 32, epochs: int = 10, augment_val: bool = True, callbacks: list = None):
        if callbacks is None:
            callbacks = []

        self.network.compile(loss=self.loss(), optimizer=self.optimizer(), metrics=self.metrics())

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

    def train_model(self, dataset: Dataset, epochs: int, batch_size: int, gpu_ind: Optional[int] = None,
                    use_wandb: bool = False, early_stopping: bool = False):
        """Train model."""
        callbacks = []

        if early_stopping:
            early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=3, verbose=1, mode='auto')
            callbacks.append(early_stopping)

        # Hide lines below until Lab 4
        if use_wandb:
            wandb.init()
            wandb_callback = WandbCallback()
            callbacks.append(wandb_callback)

        # model.network.summary()

        t = time()
        history = self.fit(dataset=dataset, batch_size=batch_size, epochs=epochs, callbacks=callbacks)
        print('Training took {:2f} s'.format(time() - t))

        return history

    def evaluate(self, x, y, batch_size=16, verbose=False):  # pylint: disable=unused-argument
        sequence = DatasetSequence(x, y, batch_size=batch_size)  # Use a small batch size to use less memory
        preds = self.network.predict_generator(sequence)
        length = len(preds)
        y = np.array(y).reshape((length,))
        preds = np.array(preds).reshape((length,))

        return mean_squared_error(preds, y)

    def loss(self):
        if 'loss' in self.network_args:
            return self.network_args['loss']
        return 'binary_crossentropy'

    def optimizer(self):
        if 'optimizer' in self.network_args:
            return self.network_args['optimizer']
        return Adam()

    def metrics(self):
        if 'metrics' in self.network_args:
            return self.network_args['metrics']
        return ['accuracy']

    def load_weights(self):
        self.network.load_weights(self.weights_filename)

    def save_weights(self):
        self.network.save_weights(self.weights_filename) 