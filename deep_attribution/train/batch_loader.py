from typing import List

from numpy import ndarray
from pandas import read_parquet

from tensorflow.keras.utils import Sequence


from utilities import over_sample
from deep_attribution.s3_utilities import get_file_nms_in_s3

BUCKET_NM = "deep-attribution"
BATCH_FILES_PATH = "feature_store_preprocessed"

class BatchLoader(Sequence):

    def __init__(
        self,
        target_nm: str,
        set_nm: str
        ) -> None:

        self.__target_nm = target_nm
        self.__set_nm = set_nm

        self.__batch_file_nms = get_file_nms_in_s3(
            BUCKET_NM,
            "%s/%s.parquet" % (BATCH_FILES_PATH, set_nm)
        )
    
    def __len__(self) -> int:

        return len(self.__batch_file_nms)

    
    def __getitem__(self, index: int) -> List[ndarray]:

        train_file_nm = self.__batch_file_nms[index]

        X, y = self.__load_batch(train_file_nm)

        if self.__set_nm == "train":
            X, y = over_sample(X, y)

        return [X, y]

    
    def __load_batch(self, file_name: str) -> List[ndarray]:

        df = read_parquet(
            "s3://%s/feature_store_preprocessed/%s.parquet/%s" % (BUCKET_NM, self.__set_nm, file_name)
            )
        y = df.loc[:,self.__target_nm].values
        X = df.drop(columns=self.__target_nm).values

        return [X, y]


