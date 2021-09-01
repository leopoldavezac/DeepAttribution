from typing import List

from glob import iglob

from numpy import uint8, ndarray, zeros
from numpy.random import shuffle
from pandas import read_parquet

from tensorflow.keras.utils import Sequence

from deep_attribution.train.oversampling import oversample

class BatchLoader(Sequence):

    def __init__(
        self,
        target_nm: str,
        set_nm:str,
        nb_campaigns:int,
        journey_max_len:int,
        set_parent_dir_path: str,
        oversample: bool = False,
        ) -> None:

        self.__target_nm = target_nm
        self.__set_nm = set_nm
        self.__nb_campaigns = nb_campaigns
        self.__journey_max_len = journey_max_len
        self.__set_parent_dir_path = set_parent_dir_path
        self.__oversample = oversample

        self.__set_batch_file_paths()

        # self.on_epoch_end()

    def __set_batch_file_paths(self):

        set_dir_path = "%s/%s.parquet" % (
            self.__set_parent_dir_path, self.__set_nm)

        self.__batch_file_paths = []
        for file_path in iglob(set_dir_path+"/*.parquet"):
            self.__batch_file_paths.append(file_path)

    # def on_epoch_end(self):
        
    #     self.__batch_file_paths = shuffle(self.__batch_file_paths)
    
    def __len__(self) -> int:

        return len(self.__batch_file_paths)

    
    def __getitem__(self, index: int) -> ndarray:

        file_nm = self.__batch_file_paths[index]

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


    
    def __load_batch(self, file_path: str) -> List[ndarray]:

        df = read_parquet(file_path)
        df.iloc[:,1:] = df.iloc[:,1:].astype("bool")

        y = df.loc[:,self.__target_nm].values
        X = df.drop(columns=[self.__target_nm, "journey_id"]).values

        print("X.dtypes", X.dtype)

        return [X, y]


