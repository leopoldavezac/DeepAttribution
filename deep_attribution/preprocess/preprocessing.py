import os

import argparse
from typing import List, Dict

from json import loads

from numpy import (
    concatenate,
    ndarray,
    array,
    zeros,
    bool as np_bool,
    object as np_object
)
from pandas import read_parquet, DataFrame

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import boto3



def main() -> None:

    args = parse_args()
        
    campaign_nm_to_index = load_json_from_s3(args.bucket_nm, "campaign_nm_to_one_hot_index.json")
    category_nms = get_category_nms(campaign_nm_to_index)
    one_hot_categories = create_categories_for_one_hot_encoding(
        category_nms, args.journey_max_len)

    ohe = create_one_hot_encoder(one_hot_categories)

    set_nms = ["train", "test", "val"]
    for set_nm in set_nms:

        df_set_obs = load_set(set_nm)

        X = df_set_obs.drop(columns=["journey_id", "conversion_status"]).values
        X_encoded = ohe.fit_transform(X)
        df_set_obs = format_preprocessed_obs(
            X_encoded,
            df_set_obs["journey_id"].values,
            df_set_obs["conversion_status"].values,
            category_nms,
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

def get_category_nms(campaign_nm_to_index: Dict) -> List[str]:

    nb_categories = len(campaign_nm_to_index.keys())
    category_nms = list(zeros(nb_categories))

    for k, v in campaign_nm_to_index.items():
        category_nms[v] = k

    return category_nms

def create_categories_for_one_hot_encoding(
    category_nms: List[str],
    journey_max_len:int
    ) -> ndarray:

    nb_categories = len(category_nms)
    
    categories_ohe = zeros((journey_max_len, nb_categories), dtype=np_object)
    for i in range(journey_max_len):
        categories_ohe[i,:] = array(category_nms)

    return categories_ohe


def load_set(set_nm: str) -> DataFrame:

    path = "/opt/ml/processing/input/%s"% set_nm
    df_set_obs = read_parquet(path, engine="pyarrow")

    return df_set_obs


def get_categorical_col_nms(df_set_obs: DataFrame) -> List[str]:

    return [col for col in df_set_obs.columns if col not in ["journey_id", "conversion"]]


def create_one_hot_encoder(one_hot_category_nms: ndarray) -> Pipeline:

    ohe = OneHotEncoder(
        categories=one_hot_category_nms.tolist(),
        handle_unknown="ignore",
        sparse=False,
        dtype=np_bool
        )

    return ohe
    

def format_preprocessed_obs(
    X: ndarray,
    arr_journey_id: ndarray,
    arr_conversion_status: ndarray,
    category_nms: List[str],
    journey_max_len: int
    ) -> DataFrame:

    feature_nms = []
    for i in range(1, journey_max_len+1):
        for ohe_category_nm in category_nms:
            feature_nms.append(
                "campaign_nm_at_index_%d_in_journey_is_%s" % (
                    i, ohe_category_nm
                    )
                )

    print(X)

    arr_obs = concatenate([
        arr_journey_id.reshape((-1,1)),
        X,
        arr_conversion_status.reshape((-1,1))
        ], axis=1)

    df_set_obs_obs = DataFrame(
        arr_obs,
        columns=["journey_id"]+feature_nms+["conversion_status"])
    
    df_set_obs_obs["journey_id"] = df_set_obs_obs["journey_id"].astype("uint32")
    for col_nm in feature_nms+["conversion_status"]:
        df_set_obs_obs[col_nm] = df_set_obs_obs[col_nm].astype("bool")

    return df_set_obs_obs


def save_as_parquet(
    df_set_obs: DataFrame, set_nm: str) -> None:

    path = os.path.join("/opt/ml/processing/output/%s"% set_nm, "%s.parquet"% set_nm)
    df_set_obs.to_parquet(path, index=False)

if __name__ == '__main__':
    main()
