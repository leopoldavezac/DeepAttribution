import os

from typing import Dict

import sagemaker
from sagemaker.spark.processing import PySparkProcessor

from deep_attribution.utilities import format_config_as_job_args


def main(config: Dict) -> None:

    job_args = format_config_as_job_args(config)

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
        spark_event_logs_s3_uri="s3://"+os.path.join(config["bucket_nm"], "feature_store", "spark_event_logs"),
        logs=False,
        arguments=job_args
    )