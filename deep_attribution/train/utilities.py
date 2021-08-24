from typing import List, Dict

from json import dumps, loads

from numpy import ndarray, zeros

import boto3


def get_nb_campaigns_from_s3(bucket_nm: str) -> int:

    campaign_nm_to_ohe_index = load_json_from_s3(
        bucket_nm, "campaign_nm_to_one_hot_index.json"
        )
    
    return len(campaign_nm_to_ohe_index.keys())



def get_X_sample(journey_max_len: int, nb_campaigns: int, bucket_nm: str) -> ndarray:

    from pandas import read_parquet
        
    BATCH_FILES_PATH = "feature_store_preprocessed/train.parquet"

    sample_file_nm = get_file_nms_in_s3(bucket_nm, BATCH_FILES_PATH)[0]

    df = read_parquet(sample_file_nm)

    X = df.drop(columns=["conversion", "journey_id"]).values
    del df

    X_tensor = reshape_X_with_one_hot_along_z(X, journey_max_len, nb_campaigns)

    return X_tensor

def reshape_X_with_one_hot_along_z(X: ndarray, journey_max_len: int, nb_campaigns: int) -> ndarray:

    nb_obs = X.shape[0]
    X_tensor = zeros((nb_obs, journey_max_len, nb_campaigns))

    for index in range(journey_max_len):
        X_tensor[:,index,:] = X[:,nb_campaigns*index, nb_campaigns*(index+1)]

    return X_tensor



def write_json_to_s3(json_object:Dict, bucket_nm: str, path: str) -> None:

    s3 = boto3.resource("s3")
    obj = s3.Object(bucket_nm, path)

    obj.put(Body=bytes(dumps(json_object).encode("UTF-8")))


def get_file_nms_in_s3(
    bucket_nm: str, dir_path: str, file_ext: str = None
    ) -> List[str]:

    file_nms = []

    conn = boto3.resource("s3")
    bucket = conn.get_bucket(bucket_nm)

    for file_nm in bucket.list(prefix=dir_path):

        if file_ext is not None:
            if file_nm[-len(file_ext):] == file_ext:
                file_nms.append(file_nm)
        else:
            file_nms.append(file_nm)

    return file_nms


def load_json_from_s3(bucket_nm: str, path: str) -> Dict:

    s3 = boto3.resource('s3')

    file_conn = s3.Object(bucket_nm, path)
    file_content = file_conn.get()['Body'].read().decode('utf-8')

    return loads(file_content)


def write_as_txt_to_s3(text: str, bucket_nm: str, path:str) -> None:

    s3 = boto3.resource("s3")

    file_conn = s3.Object(bucket_nm, path)
    file_conn.put(Body=text)
