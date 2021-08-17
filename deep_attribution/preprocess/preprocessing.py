import argparse
from typing import List, Dict

from json import loads

from numpy import concatenate, ndarray, zeros
from pandas import read_parquet, DataFrame

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import boto3



def main():

    args = parse_args()
        
    campaign_nm_to_index = load_json_from_s3(args.bucket_nm, "campaign_nm_to_one_hot_index.json")
    one_hot_categories = create_categories_for_one_hot_encoding(campaign_nm_to_index)

    pipeline = create_pipeline(one_hot_categories)

    set_nms = ["train", "test", "val"]
    for set_nm in set_nms:

        df_set_obs = load_set(set_nm)
        X = pipeline.transform(
            df_set_obs.drop(columns=["journey_id", "conversion_status"]).values
            )
        df_set_obs = format_preprocessed_obs(
            X, df_set_obs["journey_id"].values, df_set_obs["conversion_status"].values,
            args.journey_max_len
            )
        save_as_parquet(df_set_obs, set_nm)


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser()

    parser.add_argument('--bucket_nm', type=str)

    parser.add_argument('--journey_max_len', type=int)

    return parser.parse_args()


def load_json_from_s3(bucket_nm: str, path: str) -> Dict:

    s3 = boto3.resource('s3')

    file_conn = s3.Object(bucket_nm, path)
    file_content = file_conn.get()['Body'].read().decode('utf-8')

    return loads(file_content)

def create_categories_for_one_hot_encoding(campaign_nm_to_index: Dict) -> List[str]:

    nb_categories = len(campaign_nm_to_index.keys())
    arr_category_nms = zeros(nb_categories)

    for k, v in campaign_nm_to_index.items():
        arr_category_nms[v] = k
    
    return arr_category_nms.tolist()


def load_set(set_nm: str) -> DataFrame:

    path = '/opt/ml/processing/%s_input/%s.parquet' %(set_nm, set_nm)
    df_set_obs = read_parquet(path)

    return df_set_obs


def get_categorical_col_nms(df_set_obs: DataFrame) -> List[str]:

    return [col for col in df_set_obs.columns if col not in ["journey_id", "conversion"]]


def create_pipeline(one_hot_category_nms: List[str]) -> Pipeline:

    pipeline = Pipeline(steps=[
        ('one_hot_encoder', OneHotEncoder(categories=one_hot_category_nms))
        ])


    return pipeline
    

def format_preprocessed_obs(
    X: ndarray,
    arr_journey_id: ndarray,
    arr_conversion_status: ndarray,
    ohe_category_nms: List[str],
    journey_max_len: int
    ) -> DataFrame:

    feature_nms = []
    for i in range(1, journey_max_len+1):
        for ohe_category_nm in ohe_category_nms:
            feature_nms.append(
                "campaign_nm_at_index_%d_in_journey_is_%s" % (i, ohe_category_nm)
                )

    arr_obs = concatenate([
        arr_journey_id.reshape((-1,1)),
        X,
        arr_conversion_status.reshape((-1,1))
        ], axis=1)

    df_set_obs_obs = DataFrame(arr_obs, columns=["journey_id"]+feature_nms+["conversion_status"])
    
    df_set_obs_obs["journey_id"] = df_set_obs_obs["journey_id"].astype("uint32")
    for col_nm in feature_nms+["conversion_status"]:
        df_set_obs_obs[col_nm] = df_set_obs_obs[col_nm].astype("bool")

    return df_set_obs_obs


def save_as_parquet(df_set_obs: DataFrame, set_nm: str) -> None:

    df_set_obs.to_parquet("/opt/ml/processing/output/%s.parquet"%set_nm, index=False)


if __name__ == '__main__':
    main()
