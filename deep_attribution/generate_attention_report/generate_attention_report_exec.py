import os

from typing import Dict

import sagemaker
from sagemaker.spark.processing import PySparkProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput


def main(config: Dict) -> None:

    job_args = []
    for arg_nm, arg_val in config.items():
        job_args += ["--"+arg_nm, arg_val]

    role = sagemaker.get_execution_role()

    spark_processor = PySparkProcessor(
        base_job_name="deep-attribution-generate-attention_report",
        framework_version="2.4",
        role=role,
        instance_count=config["attention_report_generation"]["instance_count"],
        instance_type=config["attention_report_generation"]["instance_type"],
        max_runtime_in_seconds=1200,
    )

    spark_processor.run(
        submit_app="root/DeepAttribution/deep_attribution/generate_attention_report/generate_attention_report.py",
        spark_event_logs_s3_uri="s3://deep-attribution/attention_report/spark_event_logs",
        logs=False,
        arguments=job_args,
        inputs=[
            ProcessingInput(
                input_name='campaigns_at_journey_level',
                source='s3://' + os.path.join(config["bucket_nm"], "feature_store", 'train.parquet'),
                destination='/opt/ml/processing/feature_store'),
            ProcessingInput(
                input_name='attentions_at_journey_level',
                source='s3://' + os.path.join(config["bucket_nm"], "attention_report", 'attention_score.parquet'),
                destination='/opt/ml/processing/attention_report')
        ],
        outputs=[
            ProcessingOutput(
                output_name="attention_report",
                source="opt/ml/processing/output/attention_report",
                destination='s3://' + os.path.join(config["bucket_nm"], "attention_report", 'campaign_attention.parquet')
            )
        ]
    )