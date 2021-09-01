
from typing import Dict

from pandas import DataFrame, read_parquet
from numpy import zeros, ndarray

from sagemaker.estimator import EstimatorBase

from deep_attribution.train.utilities import get_nb_campaigns_from_s3, reshape_X_with_one_hot_along_z


def main(estimator: EstimatorBase, config: Dict) -> None:

    nb_campaigns = get_nb_campaigns_from_s3(config["bucket_nm"])

    df_train = get_df_train(config["bucket_nm"])

    array_journey_id = df_train.journey_id.values
    # drop journey id
    X_tensor = reshape_X_with_one_hot_along_z(df_train.drop(columns="conversion"), config["journey_max"], nb_campaigns)
    del df_train

    predictor = estimator.deploy(
        initial_instance_count=config["prediction"]["instance_type"],
        instance_type=config["prediction"]["instance_type"]
    )

    array_attention = zeros((X_tensor.shape[0], config["journey_max_len"]+1))
    array_attention[:,0] = array_journey_id

    array_attention[:,1:] = predictor.predict(X_tensor)

    save_attention_as_parquet(array_attention, config["bucket_nm"])

    predictor.delete_endpoint()



def get_df_train(bucket_nm: str) -> DataFrame:

    df = read_parquet("s3://{}/feature_store_preprocessed/train.parquet".format(bucket_nm))

    return df.drop(columns="conversion")



def save_attention_as_parquet(array_attention: ndarray, journey_max_len: int, bucket_nm: str) -> None:

    attention_col_nms = ["attention_at_index_%d_in_journey" % i for i in range(1,journey_max_len+1)]

    df_attention = DataFrame(array_attention, columns=["journey_id"]+attention_col_nms)

    df_attention.to_parquet("s3://%s/attention_report/attention_score.parquet" % bucket_nm, index=False)
