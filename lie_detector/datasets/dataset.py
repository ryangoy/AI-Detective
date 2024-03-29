"""Dataset class to be extended by dataset-specific classes."""
from pathlib import Path
import argparse
import os


def download_url(url, filename):
    """Download a file from url to filename, with a progress bar."""
    with TqdmUpTo(unit='B', unit_scale=True, unit_divisor=1024, miniters=1) as t:
        urlretrieve(url, filename, reporthook=t.update_to, data=None)


class Dataset:
    """Simple abstract class for datasets."""
    @classmethod
    def data_dirname(cls):
        return Path(__file__).resolve().parents[0]

    def load_or_generate_data(self):
        pass


def _download_raw_dataset(metadata):
    if os.path.exists(metadata['filename']):
        return
    print('Downloading raw dataset...')
    download_url(metadata['url'], metadata['filename'])
  

def _sample_to_balance(x, y):
    """Because the dataset is not balanced, we take at most the mean number of instances per class."""
    np.random.seed(0)
    num_to_sample = int(np.bincount(y.flatten()).mean())
    all_sampled_inds = []
    for label in np.unique(y.flatten()):
        inds = np.where(y == label)[0]
        sampled_inds = np.unique(np.random.choice(inds, num_to_sample))
        all_sampled_inds.append(sampled_inds)
    ind = np.concatenate(all_sampled_inds)
    x_sampled = x[ind]
    y_sampled = y[ind]
    return x_sampled, y_sampled


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsample_fraction",
                        type=float,
                        default=None,
                        help="If given, is used as the fraction of data to expose.")
    return parser.parse_args()