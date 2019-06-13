#!/bin/bash
pipenv run python lie_detector/training/run_experiment.py '{"dataset": "TrialDataset", "model": "E2EModel", "base_network": "lstm", "head_network": "senet50", train_args": {"batch_size": 256}}'
