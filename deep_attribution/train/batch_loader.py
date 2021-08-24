from typing import List

from numpy import uint8, ndarray, zeros
from pandas import read_parquet

from tensorflow.keras.utils import Sequence

from deep_attribution.train.utilities import get_file_nms_in_s3
from deep_attribution.train.oversampling import oversample

class BatchLoader(Sequence):

    def __init__(
        self,
        target_nm: str,
        nb_campaigns:int,
        journey_max_len:int,
        data_path: str,
        oversample: bool = False
        ) -> None:

        self.__target_nm = target_nm
        self.__nb_campaigns = nb_campaigns
        self.__journey_max_len = journey_max_len
        self.__data_path = data_path
        self.__oversample = oversample

        self.__batch_file_nms = get_file_nms_in_s3(
            self.__data_path.split("/")[2],
            "/".join(self.__data_path.split("/")[3:])
        )
    
    def __len__(self) -> int:

        return len(self.__batch_file_nms)

    
    def __getitem__(self, index: int) -> ndarray:

        file_nm = self.__batch_file_nms[index]

        X, y = self.__load_batch(file_nm)

        if self.__oversample:
            X, y = oversample(X, y)

        X = self.__reshape_as_tensor_with_one_hot_along_z(X)

        return X, y

    
    def __reshape_as_tensor_with_one_hot_along_z(self, X:ndarray) -> ndarray:

        nb_obs = X.shape[0]
        X_tensor = zeros((nb_obs, self.__journey_max_len, self.__nb_campaigns), dtype=uint8)

        for index in range(self.__journey_max_len):
            
            X_tensor[:,index,:] = X[:,self.__nb_campaigns*index:self.__nb_campaigns*(index+1)]

        return X_tensor


    
    def __load_batch(self, file_nm: str) -> List[ndarray]:

        df = read_parquet(
            self.__data_path+"/"+file_nm
            )

        y = df.loc[:,self.__target_nm].values
        X = df.drop(columns=[self.__target_nm, "journey_id"]).values

        return [X, y]


