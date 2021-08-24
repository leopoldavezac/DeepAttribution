import pytest

from numpy import array

from deep_attribution.train.oversampling import oversample

def test_oversample():

    X = array([
        [1, 0, 1],
        [0, 0, 1],
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 1]
    ])

    Y = array([1, 0, 0, 0, 0])

    EXPECTED_Y = array([1, 1, 1, 1, 0, 0, 0, 0])

    EXPECTED_X = array([
        [1, 0, 1],
        [1, 0, 1],
        [1, 0, 1],
        [1, 0, 1],
        [0, 0, 1],
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 1]
    ])

    obtained_X, obtained_y = oversample(X, Y)

    assert (EXPECTED_X == obtained_X).all()

    assert (EXPECTED_Y == obtained_y).all()