import os

import sagemaker
from sagemaker.tensorflow import TensorFlow
import tensorflow as tf

from deep_attribution.train.utilities import get_X_sample

# !wget -q https://raw.githubusercontent.com/aws-samples/amazon-sagemaker-script-mode/master/local_mode_setup.sh
# !wget -q https://raw.githubusercontent.com/aws-samples/amazon-sagemaker-script-mode/master/daemon.json    
# !/bin/bash ./local_mode_setup.sh


sess = sagemaker.Session()


model_dir = 'model'
train_instance_type = 'local'
hyperparameters = {
        "n_hidden_units_embedding":40,
        "n_hidden_units_lstm":100,
        "dropout_lstm":0.2,
        "recurrent_dropout_lstm":0.1,
        "learning_rate":0.01,
        "epochs":5
    }

local_estimator = TensorFlow(
    source_dir='deep_attribution/train',
    entry_point='train.py',
    model_dir=model_dir,
    instance_type=train_instance_type,
    instance_count=1,
    hyperparameters=hyperparameters,
    role=sagemaker.get_execution_role(),
    base_job_name='deep-attribution-local-training',
    framework_version='2.2',
    py_version='py37',
    script_mode=True
    )


local_estimator.fit()


X_sample = get_X_sample()

local_predictor = local_estimator.deploy(initial_instance_count=1, instance_type='local')
local_results = local_predictor.predict(X_sample)['predictions']

local_predictor.delete_endpoint()
