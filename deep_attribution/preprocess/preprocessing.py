from typing import List

import os

from numpy import ndarray
from pandas import read_parquet, DataFrame

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from deep_attribution.s3_utilities import load_json_from_s3
from utilities import create_categories_for_one_hot_encoding

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
        save_set(X, y, set_nm)

def load_set(set_nm: str) -> List[ndarray]:

    path = os.path.join('/opt/ml/processing/%s_input', '%s.parquet'%(set_nm, set_nm))
    df = read_parquet(path, usecols=PREDICTOR_NMS+TARGET_NM)

    X = df.loc[:, PREDICTOR_NMS].values
    y = df.loc[:,TARGET_NM].values

    return [X, y]

def save_set(X: ndarray, y: ndarray, set_nm: str) -> None:

    df = DataFrame(X)
    df.insert(0, "conversion", y)

    df.to_parquet("/opt/ml/processing/output/%s.parquet"%set_nm, index=False)


def create_pipeline(one_hot_categories: List[str]) -> Pipeline:

    return Pipeline(steps=[
        ('one_hot_encoder', OneHotEncoder(categories=one_hot_categories))
        ])


if __name__ == '__main__':
    main()
