"""
Real life dataset. Downloads from UMichigan website and saves as .npz file if not already present.
"""

# to do: move settings to metadata file

import json
import os
from pathlib import Path
import shutil
import zipfile
import h5py
import numpy as np
import toml
import pandas as pd
from sklearn.model_selection import GroupKFold

from lie_detector.datasets.dataset import _download_raw_dataset, Dataset, _parse_args
from lie_detector.video_face_detector import generate_cropped_face_video

SAMPLE_TO_BALANCE = False  # If true, take at most the mean number of instances per class.

RAW_DATA_DIRNAME = Dataset.data_dirname() / 'raw'
METADATA_FILENAME = Dataset.data_dirname() / 'metadata.toml'

PROCESSED_DATA_DIRNAME = Dataset.data_dirname() / 'processed' / 'TrialData'
PROCESSED_DATA_FILENAME = PROCESSED_DATA_DIRNAME / 'X_faces.npy'
PROCESSED_LABELS_FILENAME = PROCESSED_DATA_DIRNAME / 'y.npy'

ANNOTATION_CSV_FILENAME = 'TrialData/Annotation/All_Gestures_Deceptive and Truthful.csv'


class TrialDataset(Dataset):


    def __init__(self, subsample_fraction: float = None, num_folds: int = 3, frames_per_sample = 64):
        if not os.path.exists(str(PROCESSED_DATA_FILENAME)):
            _download_and_process_trial()

        self.output_shape = 1
        self.subsample_fraction = subsample_fraction
        self.X = None
        self.y = None
        self.groups = None
        self.num_folds = num_folds 
        self.trn_folds = []
        self.val_folds = []
        self.input_shape = [224, 224, 3]
        self.frames_per_sample = frames_per_sample


    def load_or_generate_data(self):
        self.metadata = toml.load(METADATA_FILENAME)
        if not os.path.exists(str(PROCESSED_DATA_FILENAME)):
            _download_and_process_trial()
        
        print("\nLoading Trial Dataset into memory...")
        self.X = np.load(PROCESSED_DATA_FILENAME, allow_pickle=True)
        self.y = np.load(PROCESSED_LABELS_FILENAME)
        if SAMPLE_TO_BALANCE:
            print('Balancing classes to reduce amount of data')
            X, y = _sample_to_balance(x_train, y_train)
        self._fix_data_length()
        self._initialize_fold()
        # self._subsample()


    def _fix_data_length(self):
        X_new = []
        y_new = []
        groups = []
        metadata_groups = self.metadata['trial']['groups']['Deceptive'] + self.metadata['trial']['groups']['Truthful']
        for i, x in enumerate(self.X):
            if metadata_groups[i] == -1:
                continue
            n_samps = len(x) // self.frames_per_sample
            for j in range(n_samps):
                X_new.append(x[j*self.frames_per_sample: (j+1)*self.frames_per_sample])
                y_new.append(self.y[i])
                groups.append(metadata_groups[i])
        self.X = np.array(X_new)
        self.y = np.array(y_new)
        self.groups = np.array(groups)


    def _initialize_fold(self):
        kf = GroupKFold(n_splits=self.num_folds)
        for trn_index, val_index in kf.split(self.X, groups=self.groups):
            self.trn_folds.append(trn_index)
            self.val_folds.append(val_index)


    def set_fold(self, index):
        self.X_trn = self.X[self.trn_folds[index]]
        self.y_trn = self.y[self.trn_folds[index]]
        self.X_val = self.X[self.val_folds[index]]
        self.y_val = self.y[self.val_folds[index]]


    def _subsample(self):
        """Only this fraction of data will be loaded."""
        if self.subsample_fraction is None:
            return
        num_train = int(self.x_train.shape[0] * self.subsample_fraction)
        num_test = int(self.x_test.shape[0] * self.subsample_fraction)
        self.x_train = self.x_train[:num_train]
        self.y_train_int = self.y_train_int[:num_train]
        self.x_test = self.x_test[:num_test]
        self.y_test_int = self.y_test_int[:num_test]

    def preprocess(self, fn):
        self.X, self.y = fn(self.X, self.y)

        self.input_shape = [self.X.shape[-1]]

def _download_and_process_trial():
    curdir = os.getcwd()
    os.chdir(str(RAW_DATA_DIRNAME))
    _download_raw_dataset(self.metadata['trial'])
    _process_raw_dataset(self.metadata['trial']['filename'])
    os.chdir(curdir)


def _process_raw_dataset(filename: str):
    if not os.path.isdir('TrialData'):
        print('Unzipping trial_data.zip...')
        zip_file = zipfile.ZipFile(filename, 'r')
        for file in zip_file.namelist():
            if file.startswith('Real-life_Deception_Detection_2016/'):
                zip_file.extract(file, 'temp')
        zip_file.close()
        os.rename('temp/Real-life_Deception_Detection_2016/', 'TrialData/')
        os.rmdir('temp')  

    print('\nLoading training data from folder')

    X_fnames = []
    y = []
    microexpressions = []
    annotation_path = os.path.join(str(RAW_DATA_DIRNAME), ANNOTATION_CSV_FILENAME)
    annotation_csv = pd.read_csv(annotation_path)

    for f in os.listdir('TrialData/Clips/Deceptive'):
        X_fnames.append(os.path.join(str(RAW_DATA_DIRNAME), 'TrialData', 'Clips', 'Deceptive', f))
        microexpressions.append(list(annotation_csv[annotation_csv.id==f]))
        y.append(1)
    for f in os.listdir('TrialData/Clips/Truthful'):
        X_fnames.append(os.path.join(str(RAW_DATA_DIRNAME), 'TrialData', 'Clips', 'Truthful', f))
        microexpressions.append(list(annotation_csv[annotation_csv.id==f]))
        y.append(0)

    X = []
    print('\nDetecting face in videos...')
    for counter, f in enumerate(X_fnames):
        X.append(generate_cropped_face_video(f, fps=10))
        if (counter+1) % 1 == 0:
            print('Successfully detected faces in video {}/{} with shape {}'.format(counter+1, len(X_fnames), np.array(X[counter]).shape))
    X = np.array(X)
    y = np.array(y)
    np.save(os.path.join(PROCESSED_DATA_DIRNAME, 'X_faces.npy'), X)
    np.save(os.path.join(PROCESSED_DATA_DIRNAME, 'y.npy'), y)


def main():
    """Load trial dataset and print info."""
    args = _parse_args()
    dataset = TrialDataset(subsample_fraction=args.subsample_fraction)
    dataset.load_or_generate_data()

    print(dataset)
    print(dataset.x_train.shape, dataset.y_train.shape)  # pylint: disable=E1101
    print(dataset.x_test.shape, dataset.y_test.shape)  # pylint: disable=E1101


if __name__ == '__main__':
    main()