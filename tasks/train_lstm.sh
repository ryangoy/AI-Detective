#!/bin/bash
python lie_detector/training/run_experiment.py --save --experiment_config '{"dataset": "TrialDataset", "feature_model":"CNNModel", "model": "LSTMModel", "base_network": "LSTM", "head_network": "RESNET50", "train_args": {"batch_size": 8}, "network_args": {"frames":64}, "end2end": "False"}'
