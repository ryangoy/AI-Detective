import json
import importlib
import os
import numpy as np
import sys
sys.path.insert(0, './')
from lie_detector.video_face_detector import generate_cropped_face_video
# to do: change magic numbers
# to do: automatically import correct models
# to do: rename head_network feature_network, rename model_class to base_model_class

def predict_example(vpath, experiment_config_path):
    with open(experiment_config_path) as f:

        experiment_config = json.load(f)

    datasets_module = importlib.import_module('lie_detector.datasets')
    dataset_class_ = getattr(datasets_module, experiment_config['dataset'])
    # dataset_args = experiment_config.get('dataset_args', {})
    # dataset = dataset_class_(**dataset_args)
    # dataset.load_or_generate_data()

    models_module = importlib.import_module('lie_detector.models')
    model_class_ = getattr(models_module, experiment_config['model'])
    feature_model_class_ = getattr(models_module, experiment_config['feature_model'])

    networks_module = importlib.import_module('lie_detector.networks')
    base_network_fn_ = getattr(networks_module, experiment_config['base_network'])
    head_network_fn_ = getattr(networks_module, experiment_config['head_network'])
    network_args = experiment_config.get('network_args', {})

    face_cropped = generate_cropped_face_video(vpath)
    X = fix_data_length(face_cropped)
    
    
    feature_model = feature_model_class_(network_fn=head_network_fn_)

    X = feature_model.generate_features(X)

    model = model_class_(dataset_cls=dataset_class_, network_fn=base_network_fn_, network_args=network_args, input_shape=[2048])

    preds = model.network.predict(X)

    return np.mean(preds)

    return(preds)


def fix_data_length(x):
        
    X_new = []

    n_samps = len(x) // 64
    for j in range(n_samps):
        X_new.append(x[j*64: (j+1)*64])

    return np.array(X_new)

if __name__ == '__main__':
    
    p = predict_example('/home/ryan/cs/fs-lie-detector/lie_detector/datasets/raw/TrialData/Clips/Deceptive/trial_lie_058.mp4', '/home/ryan/cs/fs-lie-detector/lie_detector/training/experiments/base_LSTM.json')
    print(p)