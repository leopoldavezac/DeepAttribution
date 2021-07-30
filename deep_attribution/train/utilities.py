from typing import List, Dict

from json import dumps

from numpy import ndarray

from imblearn.over_sampling import SMOTE #will fail

import boto3


def over_sample(X:ndarray, y:ndarray) -> List[ndarray]:

    resampler = SMOTE(random_state=42)

    m, n, p = X.shape

    X_resampled, y_resampled = resampler.fit_resample(X.reshape((m, n*p)), y)
    X_resampled = X_resampled.reshape((-1, n, p))

    return [X_resampled, y_resampled]



def get_X_sample() -> ndarray:

    from pandas import read_parquet
        
    BUCKET_NM = "deep-attribution"
    BATCH_FILES_PATH = "feature_store_preprocessed"

    sample_file_nm = get_file_nms_in_s3(BUCKET_NM, BATCH_FILES_PATH)[0]

    df = read_parquet(sample_file_nm)

    return df.drop(columns="conversion").values



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