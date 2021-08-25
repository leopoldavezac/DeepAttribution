import os

from typing import Dict

import sagemaker
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput

from deep_attribution.utilities import format_config_as_job_args

INPUT_PREFIX = "feature_store"
OUTPUT_PREFIX = "feature_store_preprocessed"


def main(config: Dict) -> None:

    job_args = format_config_as_job_args(config)

    sklearn_job = SKLearnProcessor(
        framework_version="0.23-1",
        role=sagemaker.get_execution_role(),
        instance_type=config["preprocessing"]["instance_type"],
        instance_count=config["preprocessing"]["instance_count"],
        base_job_name="deep-attribution-preprocessing"
    )

    sklearn_job.run(
        code="preprocess/preprocessing.py",
        logs=False,
        arguments=job_args
        )