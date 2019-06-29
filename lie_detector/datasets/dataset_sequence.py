"""DatasetSequence class."""
import numpy as np
from tensorflow.keras.utils import Sequence


class DatasetSequence(Sequence):
    """
    Minimal implementation of https://keras.io/utils/#sequence.
    Allows easy use of fit_generator in training.
    """
    def __init__(self, x, y=None, batch_size=32, augment_fn=None, format_fn=None):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.augment_fn = augment_fn
        self.format_fn = format_fn

    def __len__(self):
        """Return length of the dataset."""
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        """Return a single batch."""
        # idx = 0  # If you want to intentionally overfit to just one batch
        begin = idx * self.batch_size
        end = (idx + 1) * self.batch_size
        batch_x = self.x[begin:end]
        if self.y is not None:
            batch_y = self.y[begin:end]
        else:
            batch_y = None

        if self.augment_fn:
            batch_x, batch_y = self.augment_fn(batch_x, batch_y)

        if self.format_fn:
            batch_x, batch_y = self.format_fn(batch_x, batch_y)

        if self.y is not None:
            return batch_x, batch_y
        else:
            return batch_x

    def on_epoch_end(self) -> None:
        """Shuffle data."""
        self.x, self.y = _shuffle(self.x, self.y)


def _shuffle(x, y=None):
    """Shuffle x and y maintaining their association."""
    shuffled_indices = np.random.permutation(x.shape[0])
    if y is not None:
        return x[shuffled_indices], y[shuffled_indices]
    else:
        return x[shuffled_indices], None