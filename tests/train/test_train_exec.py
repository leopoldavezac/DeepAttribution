import pytest

import sagemaker
from sagemaker.tensorflow import TensorFlow
from sagemaker.local import LocalSession

from deep_attribution.train.utilities import get_X_sample, get_nb_campaigns_from_s3


role = "arn:aws:iam::689575273089:role/service-role/AmazonSageMaker-ExecutionRole-20210728T121043"
sagemaker_session = LocalSession()
sagemaker_session.config = {'local': {'local_code': True}}

config = {
    "journey_max_len":10,
    "bucket_nm":"deep-attribution"
}

def test_execute():

    sagemaker.Session()

    model_dir = '/opt/ml/model'
    train_instance_type = 'local'
    hyperparameters = {
            "n_hidden_units_embedding":40,
            "n_hidden_units_lstm":100,
            "dropout_lstm":0.2,
            "recurrent_dropout_lstm":0.1,
            "learning_rate":0.01,
            "epochs":1
        }

    local_estimator = TensorFlow(
        dependencies=["deep_attribution", "config"],
        entry_point='deep_attribution/train/train.py',
        model_dir=model_dir,
        instance_type=train_instance_type,
        instance_count=1,
        hyperparameters=hyperparameters,
        role=role,
        base_job_name='deep-attribution-local-training',
        framework_version='2.2',
        py_version='py37',
        script_mode=True,
        )

    local_estimator.fit({
        "sets_parent_dir_path":"file://tests/train/test_data/",
        })

    nb_campaigns = get_nb_campaigns_from_s3(config["bucket_nm"])

    X_sample = get_X_sample(config["journey_max_len"], nb_campaigns, config["bucket_nm"])

    local_predictor = local_estimator.deploy(initial_instance_count=1, instance_type=train_instance_type)
    print(local_predictor.predict(X_sample))

    local_predictor.delete_endpoint()

    assert True
