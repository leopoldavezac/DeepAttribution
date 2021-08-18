import os

from typing import Dict

import sagemaker
from sagemaker.spark.processing import PySparkProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput


def main(config: Dict) -> None:

    job_args = []
    for arg_nm, arg_val in config.items():
        job_args += ["--"+arg_nm, str(arg_val)]

    role = sagemaker.get_execution_role()

    spark_processor = PySparkProcessor(
        base_job_name="deep-attribution-feature-engineering",
        framework_version="2.4",
        role=role,
        instance_count=config["feature_engineering"]["instance_count"],
        instance_type=config["feature_engineering"]["instance_type"],
        max_runtime_in_seconds=1200,
    )

    spark_processor.run(
        submit_app="feature_engineering/feature_engineering.py",
        spark_event_logs_s3_uri=os.path.join("s3://", config["bucket_nm"], "feature_store", "spark_event_logs"),
        logs=False,
        arguments=job_args,
        inputs=[
            ProcessingInput(
                input_name="raw",
                source= os.path.join("s3://", config["bucket_nm"], "raw", "impressions.parquet"),
                destination="/opt/ml/processing/raw")
        ],
        outputs=[
            ProcessingOutput(
                output_name="train_features",
                source="/opt/ml/processing/output/train",
                destination= os.path.join("s3://", config["bucket_nm"], "feature_store", "train.parquet")
            ),
            ProcessingOutput(
                output_name="test_features",
                source="/opt/ml/processing/output/test",
                destination= os.path.join("s3://", config["bucket_nm"], "feature_store", "test.parquet")
            ),
            ProcessingOutput(
                output_name="val_features",
                source="/opt/ml/processing/output/val",
                destination= os.path.join("s3://", config["bucket_nm"], "feature_store", "val.parquet")
            )
        ]
    )