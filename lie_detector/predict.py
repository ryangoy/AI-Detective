import json
import importlib
import os
import numpy as np
import sys
sys.path.insert(0, './')
from lie_detector.video_face_detector import generate_cropped_face_video
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import time
import tensorflow as tf
# to do: change magic numbers
# to do: automatically import correct models

def predict_example(vpath, experiment_config_path=None, table=None, fname=None):


    if experiment_config_path is not None:
        with open(experiment_config_path) as f:

            experiment_config = json.load(f)
    else:
        experiment_config = {
            "dataset": "TrialDataset", 
            "feature_model":"CNNModel", 
            "model": "BaseModel", 
            "base_network": "LSTM", 
            "head_network": "RESNET50", 
            "train_args": {
                "batch_size": 8, 
                "end2end": "False"
            },
            "network_args": {
                "frames": 64
            }
        }

    if table:
        table.update_item(
            Key={
                'id': fname
            },
            UpdateExpression='SET stage= :val1',
            ExpressionAttributeValues={
                ':val1': 'initialization'
            }
        )
        
    datasets_module = importlib.import_module('lie_detector.datasets')
    dataset_class_ = getattr(datasets_module, experiment_config['dataset'])

    models_module = importlib.import_module('lie_detector.models')
    model_class_ = getattr(models_module, experiment_config['model'])
    feature_model_class_ = getattr(models_module, experiment_config['feature_model'])

    networks_module = importlib.import_module('lie_detector.networks')
    base_network_fn_ = getattr(networks_module, experiment_config['base_network'])
    head_network_fn_ = getattr(networks_module, experiment_config['head_network'])
    network_args = experiment_config.get('network_args', {})

    if table:
        table.update_item(
            Key={
                'id': fname
            },
            UpdateExpression='SET stage= :val1',
            ExpressionAttributeValues={
                ':val1': 'face detection'
            }
        )
    face_cropped = generate_cropped_face_video(vpath)

    if type(face_cropped) is float or type(face_cropped) is int:
        return face_cropped

    X = fix_data_length(face_cropped)
    
    graph = tf.get_default_graph()
    if table:
        table.update_item(
            Key={
                'id': fname
            },
            UpdateExpression='SET stage= :val1',
            ExpressionAttributeValues={
                ':val1': 'feature generation'
            }
        )
    with graph.as_default():
        feature_model = feature_model_class_(network_fn=head_network_fn_)
        X, _ = feature_model.generate_features(X)
    if table:
        table.update_item(
            Key={
                'id': fname
            },
            UpdateExpression='SET stage= :val1',
            ExpressionAttributeValues={
                ':val1': 'prediction'
            }
        )
    model = model_class_(network_fn=base_network_fn_, network_args=network_args, input_shape=[X.shape[-1]])
    preds = model.network.predict(X)
    print("Example predicted with average score as {}".format(np.mean(preds)))

    return np.mean(preds)*100


def fix_data_length(x):
    X_new = []

    n_samps = len(x) // 64
    for j in range(n_samps):
        X_new.append(x[j*64: (j+1)*64])

    return np.array(X_new)
    

if __name__ == '__main__':
    p = predict_example('/home/ryan/cs/fs-lie-detector/lie_detector/datasets/raw/TrialData/Clips/Deceptive/trial_lie_058.mp4', '/home/ryan/cs/fs-lie-detector/lie_detector/training/experiments/base_LSTM.json')
    print(p)