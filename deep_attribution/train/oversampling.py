
from typing import List

from numpy import ndarray, flatnonzero, concatenate
from numpy.random import choice


def oversample(X: ndarray, y:ndarray) -> List[ndarray]:
    
    filter_true = y == True

    if filter_true.sum() == 0:
        return X, y

    filter_false = ~filter_true

    X_false = X[filter_false]
    y_false = y[filter_false]

    indices_true = flatnonzero(y)

    oversampled_indices = choice(indices_true, size=len(y_false))

    X_true_resampled = X[oversampled_indices]
    y_true_resampled = y[oversampled_indices]

    X_resampled = concatenate([X_true_resampled, X_false], axis=0)
    y_resampled = concatenate([y_true_resampled, y_false], axis=0)

    return [X_resampled, y_resampled]