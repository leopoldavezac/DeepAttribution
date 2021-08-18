import os

from typing import Dict

import sagemaker
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput

INPUT_PREFIX = "feature_store"
OUTPUT_PREFIX = "feature_store_preprocessed"


def main(config: Dict) -> None:

    job_args = []
    for arg_nm, arg_val in config.items():
        job_args += ["--"+arg_nm, arg_val]

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
        arguments=job_args,
        inputs=[
            ProcessingInput(
                input_name="train_features",
                source=os.path.join("s3://", config["bucket_nm"], INPUT_PREFIX, "train.parquet"),
                destination="/opt/ml/processing/train_input"),
            ProcessingInput(
                input_name="test_features",
                source=os.path.join("s3://", config["bucket_nm"], INPUT_PREFIX, "test.parquet"),
                destination="/opt/ml/processing/test_input"),
            ProcessingInput(
                input_name="val_features",
                source=os.path.join("s3://", config["bucket_nm"], INPUT_PREFIX, "val.parquet"),
                destination="/opt/ml/processing/val_input"),
                ],
        outputs=[
            ProcessingOutput(
                output_name="train_preprocessed",
                source="/opt/ml/processing/output/train",
                destination=os.path.join("s3://", config["bucket_nm"], OUTPUT_PREFIX, "train.parquet")),
            ProcessingOutput(
                output_name="test_preprocessed",
                source="/opt/ml/processing/output/test",
                destination=os.path.join("s3://", config["bucket_nm"], OUTPUT_PREFIX, "test.parquet")),
            ProcessingOutput(
                output_name="val_preprocessed",
                source="/opt/ml/processing/output/val",
                destination=os.path.join("s3://", config["bucket_nm"], OUTPUT_PREFIX, "val.parquet"))
                ]
        )