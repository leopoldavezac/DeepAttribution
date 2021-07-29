from typing import List

import pytest
from numpy import loadtxt, ndarray

from deep_attribution.model.journey_based_deepnn import JourneyBasedDeepNN

JOURNEY_MAX_LENGTH = 10
NB_OF_CAMPAIGNS = 5


def test_nn_is_compilable():

    try:

        JourneyBasedDeepNN(
            JOURNEY_MAX_LENGTH,
            NB_OF_CAMPAIGNS
        )
        assert True

    except:

        assert False


def test_nn_is_learning():

    X, y = load_test_X_y()
    
    journey_based_deepnn = JourneyBasedDeepNN(
        JOURNEY_MAX_LENGTH,
        NB_OF_CAMPAIGNS
    )

    initial_loss = journey_based_deepnn.evaluate(X, y)[0]
    journey_based_deepnn.fit(X, y, epochs=1)
    post_training_loss = journey_based_deepnn.evaluate(X, y)[0]

    assert initial_loss > post_training_loss


def load_test_X_y() -> List[ndarray]:

    X = loadtxt("tests/model/data/X.csv").reshape(
        (-1, JOURNEY_MAX_LENGTH, NB_OF_CAMPAIGNS)
        )
    y = loadtxt("tests/model/data/y.csv")

    return [X, y]


def test_all_nn_weights_are_updated():

    X, y = load_test_X_y()

    journey_based_deepnn = JourneyBasedDeepNN(
        JOURNEY_MAX_LENGTH,
        NB_OF_CAMPAIGNS
    )

    initial_weights = journey_based_deepnn._JourneyBasedDeepNN__nn.get_weights()
    journey_based_deepnn.fit(X, y, epochs=1)
    post_training_weigths = journey_based_deepnn._JourneyBasedDeepNN__nn.get_weights()

    for inital, post_training in zip(initial_weights, post_training_weigths):

        assert (inital != post_training).all()
