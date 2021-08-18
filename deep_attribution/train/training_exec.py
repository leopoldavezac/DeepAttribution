from typing import Dict

from time import gmtime, strftime 

from sagemaker.tuner import IntegerParameter, ContinuousParameter, HyperparameterTuner
from sagemaker.tensorflow import TensorFlow
from sagemaker.estimator import EstimatorBase
import sagemaker


def main(config: Dict) -> EstimatorBase:

    sagemaker.Session()
        
    hyperparameters = {
            "n_hidden_units_embedding":40,
            "n_hidden_units_lstm":100,
            "dropout_lstm":0.2,
            "recurrent_dropout_lstm":0.1,
            "learning_rate":0.01,
            "epochs":5
        }

    estimator = TensorFlow(
        dependencies=['deep_attribution', 'smote'],
        entry_point='train.py',
        model_dir="model",
        instance_type=config["training"]["instance_type"],
        instance_count=config["preprocessing"]["instance_count"],
        hyperparameters=hyperparameters,
        role=sagemaker.get_execution_role(),
        base_job_name='deep-attribution-training',
        framework_version='2.2',
        py_version='py37',
        script_mode=True
        )


    hyperparameter_ranges = {
        'learning_rate': ContinuousParameter(0.001, 0.2, scaling_type="Logarithmic"),
        'epochs': IntegerParameter(10, 50),
        'n_hidden_units_embedding': IntegerParameter(20, 100),
        "n_hidden_units_lstm":IntegerParameter(64, 256),
        "dropout_lstm":ContinuousParameter(0.001, 0.3, scaling_type="Logarithmic"),
        "recurrent_dropout_lstm":ContinuousParameter(0.001, 0.3, scaling_type="Logarithmic")
    }

    metric_definitions = [{
        'Name': 'test_auc',
        'Regex': ' auc - test: ([0-9\\.]+)'}
        ]

    objective_metric_name = 'test_auc'
    objective_type = 'Maximize'


    tuner = HyperparameterTuner(estimator,
                                objective_metric_name,
                                hyperparameter_ranges,
                                metric_definitions,
                                max_jobs=5,
                                max_parallel_jobs=2,
                                objective_type=objective_type)

    tuning_job_name = "deep-attribution-training-{}".format(strftime("%d-%H-%M-%S", gmtime()))
    
    tuner.fit(job_name=tuning_job_name, inputs=config)
    tuner.wait()

    estimator = tuner.best_estimator()

    return estimator