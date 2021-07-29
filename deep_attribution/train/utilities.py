from deep_attribution.train.batch_loader import BUCKET_NM
from typing import List


from numpy import ndarray

from imblearn.over_sampling import SMOTE

from deep_attribution.utilities import get_file_nms_in_s3


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
