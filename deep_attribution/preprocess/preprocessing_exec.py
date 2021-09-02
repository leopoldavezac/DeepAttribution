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
        code="deep_attribution/preprocess/preprocessing.py",
        logs=False,
        arguments=job_args,
        inputs=[
            ProcessingInput(
                source='s3://' + os.path.join(config["bucket_nm"], "feature_store", 'train.parquet'),
                destination='/opt/ml/processing/input/train'),
            ProcessingInput(
                source='s3://' + os.path.join(config["bucket_nm"], "feature_store", 'test.parquet'),
                destination='/opt/ml/processing/input/test'),
            ProcessingInput(
                source='s3://' + os.path.join(config["bucket_nm"], "feature_store", 'val.parquet'),
                destination='/opt/ml/processing/input/val')
            ],
        outputs=[
            ProcessingOutput(
                output_name='train_preprocessed',
                source='/opt/ml/processing/output/train',
                destination='s3://' + os.path.join(config["bucket_nm"], "feature_store_preprocessed", 'train.parquet')),
            ProcessingOutput(
                output_name='test_preprocessed',
                source='/opt/ml/processing/output/test',
                destination='s3://' + os.path.join(config["bucket_nm"], "feature_store_preprocessed", 'test.parquet')),
            ProcessingOutput(
                output_name='val_preprocessed',
                source='/opt/ml/processing/output/val',
                destination='s3://' + os.path.join(config["bucket_nm"], "feature_store_preprocessed", 'val.parquet'))
            ]
        )