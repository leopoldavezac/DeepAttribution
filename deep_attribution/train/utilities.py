from typing import Dict

from json import dumps, loads

from numpy import ndarray

import boto3


def get_nb_campaigns_from_s3(bucket_nm: str) -> int:

    campaign_nm_to_ohe_index = load_json_from_s3(
        bucket_nm, "campaign_nm_to_one_hot_index.json"
        )
    
    return len(campaign_nm_to_ohe_index.keys())



def get_X_sample(journey_max_len: int, nb_campaigns: int, bucket_nm: str) -> ndarray:

    from pandas import read_parquet
    from numpy import zeros, bool8

    SET_PARENT_DIR_PATH = "feature_store_preprocessed/train.parquet"

    df = read_parquet(
        "s3://%s/%s/part_0.parquet" % (bucket_nm, SET_PARENT_DIR_PATH),
        engine="pyarrow")

    X = df.drop(columns=["conversion_status", "journey_id"]).values
    del df

    nb_obs = X.shape[0]
    X_tensor = zeros((nb_obs, journey_max_len, nb_campaigns), dtype=bool8)

    for index in range(journey_max_len):
        
        X_tensor[:,index,:] = X[:,nb_campaigns*index:nb_campaigns*(index+1)]
    
    return X_tensor



def write_json_to_s3(json_object:Dict, bucket_nm: str, path: str) -> None:

    s3 = boto3.resource("s3")
    obj = s3.Object(bucket_nm, path)

    obj.put(Body=bytes(dumps(json_object).encode("UTF-8")))



def load_json_from_s3(bucket_nm: str, path: str) -> Dict:

    s3 = boto3.resource('s3')

    file_conn = s3.Object(bucket_nm, path)
    file_content = file_conn.get()['Body'].read().decode('utf-8')

    return loads(file_content)


def write_as_txt_to_s3(text: str, bucket_nm: str, path:str) -> None:

    s3 = boto3.resource("s3")

    file_conn = s3.Object(bucket_nm, path)
    file_conn.put(Body=text)
