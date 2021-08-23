from sagemaker.spark.processing import PySparkProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.local import LocalSession

from deep_attribution.utilities import format_config_as_job_args

role = "arn:aws:iam::689575273089:role/service-role/AmazonSageMaker-ExecutionRole-20210728T121043"
sagemaker_session = LocalSession()
sagemaker_session.config = {'local': {'local_code': True}}

def test_main() -> None:

    config = {"bucket_nm":"deep-attribution", "journey_max_len":3}

    job_args = format_config_as_job_args(config)

    spark_processor = PySparkProcessor(
        base_job_name="deep-attribution-feature-engineering",
        framework_version="2.4",
        role=role,
        instance_type="local",
        instance_count=1,
        max_runtime_in_seconds=1200,
    )

    spark_processor.run(
        submit_app="../deep_attribution/feature_engineering/feature_engineering.py",
        arguments=job_args,
        inputs=[
            ProcessingInput(
                input_name="raw",
                source= "./test_data/raw/",
                destination="/opt/ml/processing/raw")
        ],
        outputs=[
            ProcessingOutput(
                output_name="train_features",
                source="/opt/ml/processing/output/train"
                ),
            ProcessingOutput(
                output_name="test_features",
                source="/opt/ml/processing/output/test"
            ),
            ProcessingOutput(
                output_name="val_features",
                source="/opt/ml/processing/output/val"
            )
        ]
    )
