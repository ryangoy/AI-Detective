#!/usr/bin/env python
"""Script to run an experiment."""
import argparse
import importlib
import json
import keras.backend as K
import numpy as np
import os
import sys
import tensorflow as tf
from time import time
from typing import Dict, Optional
import wandb
from wandb.keras import WandbCallback

sys.path.insert(0, './')
from lie_detector.datasets.dataset import Dataset
from lie_detector.models.base import Model

K.set_image_dim_ordering('tf')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
# dir_path = os.path.dirname(os.path.realpath(__file__))
DEFAULT_TRAIN_ARGS = {
    'batch_size': 4,
    'epochs': 5
}


def run_experiment(experiment_config: Dict, save_weights: bool, gpu_ind: int, use_wandb: bool = True):
    """
    Run a training experiment.
    Parameters
    ----------
    experiment_config (dict)
        Of the form
        {
            "dataset": "EmnistLinesDataset",
            "dataset_args": {
                "max_overlap": 0.4,
                "subsample_fraction": 0.2
            },
            "model": "LineModel",
            "network": "line_cnn_all_conv",
            "network_args": {
                "window_width": 14,
                "window_stride": 7
            },
            "train_args": {
                "batch_size": 128,
                "epochs": 10
            }
        }
    save_weights (bool)
        If True, will save the final model weights to a canonical location (see Model in models/base.py)
    gpu_ind (int)
        specifies which gpu to use (or -1 for first available)
    use_wandb (bool)
        sync training run to wandb
    """
    print('\nRunning experiment with config {} on GPU {}'.format(experiment_config, gpu_ind))

    datasets_module = importlib.import_module('lie_detector.datasets')
    dataset_class_ = getattr(datasets_module, experiment_config['dataset'])
    dataset_args = experiment_config.get('dataset_args', {})
    dataset = dataset_class_(**dataset_args)
    dataset.load_or_generate_data()

    models_module = importlib.import_module('lie_detector.models')
    model_class_ = getattr(models_module, experiment_config['model'])

    networks_module = importlib.import_module('lie_detector.networks')
    base_network_fn_ = getattr(networks_module, experiment_config['base_network'])
    head_network_fn_ = getattr(networks_module, experiment_config['head_network'])
    network_args = experiment_config.get('network_args', {})

    experiment_config['train_args'] = {**DEFAULT_TRAIN_ARGS, **experiment_config.get('train_args', {})}
    experiment_config['experiment_group'] = experiment_config.get('experiment_group', None)
    experiment_config['gpu_ind'] = gpu_ind
    
    if experiment_config['end2end'] == "False":
        print('Initializing feature model...')
        feature_model_class_ = getattr(models_module, experiment_config['feature_model'])
        feature_model = feature_model_class_(network_fn=head_network_fn_)
        dataset.preprocess(feature_model.generate_features)

    for k in range(dataset.num_folds):
        print('\nRunning fold {}'.format(k))
        dataset.set_fold(k)

        print('Initializing model...')
        model = model_class_(
            dataset_cls=dataset_class_,
            network_fn=base_network_fn_,
            dataset_args=dataset_args,
            network_args=network_args,
            input_shape=dataset.input_shape
        )

        if use_wandb:
            wandb.init()
            wandb.config.update(experiment_config)
        
        print('Beginning training...')
        history = model.train_model(
            dataset,
            epochs=experiment_config['train_args']['epochs'],
            batch_size=experiment_config['train_args']['batch_size'],
            gpu_ind=gpu_ind,
            use_wandb=use_wandb
        )

        score = model.evaluate(dataset.X_val, dataset.y_val)
        print('Test evaluation: {}'.format(score))

        # Hide lines below until Lab 4
        if use_wandb:
            wandb.log({'test_metric': score})
        # Hide lines above until Lab 4

        if save_weights:
            model.save_weights()




def _parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="Provide index of GPU to use."
    )
    parser.add_argument(
        "--save",
        default=False,
        dest='save',
        action='store_true',
        help="If true, then final weights will be saved to canonical, version-controlled location"
    )
    parser.add_argument(
        "--experiment_config",
        type=str,
        default="{\"dataset\": \"TrialDataset\", \"model\": \"LSTMModel\", \"network\": \"LSTM\", \"end2end\": \"False\"}"
    )
    parser.add_argument(
        "--nowandb",
        default=True,
        action='store_false',
        help='If true, do not use wandb for this run'
    )
    args = parser.parse_args()
    return args


def main():
    """Run experiment."""
    args = _parse_args()
    experiment_config = json.loads(args.experiment_config)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    run_experiment(experiment_config, args.save, args.gpu, use_wandb=not args.nowandb)


if __name__ == '__main__':
    main()