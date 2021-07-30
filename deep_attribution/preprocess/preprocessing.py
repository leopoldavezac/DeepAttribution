from typing import List, Dict

import os

from json import loads

from numpy import ndarray, zeros
from pandas import read_parquet, DataFrame

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import boto3


BUCKET_NM = "deep-attribution"
CAMPAIGN_NM_TO_INDEX_PATH = "feature_store/campaign_nm_to_index.json"

JOURNEY_MAX_LENGTH = 10

PREDICTOR_NMS = ["campaign_at_index_%d"%index for index in range(1, JOURNEY_MAX_LENGTH)]
TARGET_NM = "conversion"

def main():
        
    campaign_nm_to_index = load_json_from_s3(BUCKET_NM, CAMPAIGN_NM_TO_INDEX_PATH)
    one_hot_categories = create_categories_for_one_hot_encoding(campaign_nm_to_index)

    pipeline = create_pipeline(one_hot_categories)

    set_nms = ["train", "test", "val"]
    for set_nm in set_nms:

        X, y = load_set(set_nm)
        X = pipeline.transform(X)
        df = convert_to_dataframe(X, y)
        save_as_parquet(df, set_nm)


def load_json_from_s3(bucket_nm: str, path: str) -> Dict:

    s3 = boto3.resource('s3')

    file_conn = s3.Object(bucket_nm, path)
    file_content = file_conn.get()['Body'].read().decode('utf-8')

    return loads(file_content)

def create_categories_for_one_hot_encoding(campaign_nm_to_index: Dict) -> List:

    nb_categories = len(campaign_nm_to_index.keys())
    categories = zeros(nb_categories)

    for k, v in campaign_nm_to_index.items():
        categories[v] = k
    
    return categories


def load_set(set_nm: str) -> List[ndarray]:

    path = '/opt/ml/processing/%s_input/%s.parquet' %(set_nm, set_nm)
    df = read_parquet(path, usecols=PREDICTOR_NMS+[TARGET_NM])

    X = df.loc[:, PREDICTOR_NMS].values
    y = df.loc[:,TARGET_NM].values

    return [X, y]

def convert_to_dataframe(X:ndarray, y:ndarray) -> DataFrame:

    df = DataFrame(X)
    df.insert(0, "conversion", y)

    return df

def save_as_parquet(df: DataFrame, set_nm: str) -> None:

    df.to_parquet("/opt/ml/processing/output/%s.parquet"%set_nm, index=False)


def create_pipeline(one_hot_categories: List[str]) -> Pipeline:

    return Pipeline(steps=[
        ('one_hot_encoder', OneHotEncoder(categories=one_hot_categories))
        ])


if __name__ == '__main__':
    main()
