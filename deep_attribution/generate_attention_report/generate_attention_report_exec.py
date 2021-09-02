import os

from typing import Dict

import sagemaker
from sagemaker.spark.processing import PySparkProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput

from deep_attribution.utilities import format_config_as_job_args


def main(config: Dict) -> None:

    job_args = format_config_as_job_args(config)

    role = sagemaker.get_execution_role()

    spark_processor = PySparkProcessor(
        base_job_name="deep-attribution-generate-attention-report",
        framework_version="2.4",
        role=role,
        instance_count=config["attention_report_generation"]["instance_count"],
        instance_type=config["attention_report_generation"]["instance_type"],
        max_runtime_in_seconds=1200,
    )

    spark_processor.run(
        submit_app="deep_attribution/generate_attention_report/generate_attention_report.py",
        spark_event_logs_s3_uri=os.path.join("s3://", config["bucket_nm"], "attention_report", "spark_event_logs"),
        logs=False,
        arguments=job_args
    )