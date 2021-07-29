from typing import List, Dict

from json import loads, dumps

import boto3

def load_json_from_s3(bucket_nm: str, path: str) -> Dict:

    s3 = boto3.resource('s3')

    file_conn = s3.Object(bucket_nm, path)
    file_content = file_conn.get()['Body'].read().decode('utf-8')

    return loads(file_content)

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
