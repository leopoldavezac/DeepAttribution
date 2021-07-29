import os

import sagemaker
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput

bucket_nm = 'deep-attribution'
code_prefix = 'deep_attribution/preprocess'
input_prefix = 'feature_store'
output_prefix = 'feature_store_preprocessed'


sklearn_job = SKLearnProcessor(
    framework_version='0.23-1',
    role=sagemaker.get_execution_role(),
    instance_type='ml.t2.medium',
    instance_count=1, # single machine computing
    base_job_name='deep-attribution-preprocessing'
)

sklearn_job.run(
    code='/root/DeepAttribution/deep_attribution/preprocess/preprocessing.py',
    inputs=[
        ProcessingInput(
            input_name='train_features',
            source='s3://' + os.path.join(bucket_nm, input_prefix, 'train.parquet'),
            destination='/opt/ml/processing/train_input'),
        ProcessingInput(
            input_name='test_features',
            source='s3://' + os.path.join(bucket_nm, input_prefix, 'test.parquet'),
            destination='/opt/ml/processing/test_input'),
        ProcessingInput(
            input_name='val_features',
            source='s3://' + os.path.join(bucket_nm, input_prefix, 'val.parquet'),
            destination='/opt/ml/processing/val_input'),
            ],
    outputs=[
        ProcessingOutput(
            output_name='train_preprocessed',
            source='/opt/ml/processing/output/train',
            destination='s3://' + os.path.join(bucket_nm, output_prefix, 'train.parquet')),
        ProcessingOutput(
            output_name='test_preprocessed',
            source='/opt/ml/processing/output/test',
            destination='s3://' + os.path.join(bucket_nm, output_prefix, 'test.parquet')),
        ProcessingOutput(
            output_name='val_preprocessed',
            source='/opt/ml/processing/output/val',
            destination='s3://' + os.path.join(bucket_nm, output_prefix, 'val.parquet'))
            ]
    )