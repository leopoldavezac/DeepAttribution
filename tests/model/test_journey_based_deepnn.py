import pytest

from numpy import array

from deep_attribution.model.journey_based_deepnn import JourneyBasedDeepNN

JOURNEY_MAX_LENGTH = 3
NB_OF_CAMPAIGNS = 3

X = array([
    [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
    [[1, 0, 0], [0, 0, 1], [0, 0, 0]],
    [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
    [[1, 0, 0], [0, 0, 1], [0, 0, 0]]
    ])
Y = array([True, False, True, False])


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

    
    journey_based_deepnn = JourneyBasedDeepNN(
        JOURNEY_MAX_LENGTH,
        NB_OF_CAMPAIGNS
    )

    initial_loss = journey_based_deepnn.evaluate(X, Y)[0]
    journey_based_deepnn.fit(X, Y)
    post_training_loss = journey_based_deepnn.evaluate(X, Y)[0]

    assert initial_loss > post_training_loss


def test_all_nn_weights_are_updated():

    journey_based_deepnn = JourneyBasedDeepNN(
        JOURNEY_MAX_LENGTH,
        NB_OF_CAMPAIGNS
    )

    initial_weights = journey_based_deepnn._JourneyBasedDeepNN__nn.get_weights()
    journey_based_deepnn.fit(X, Y)
    post_training_weigths = journey_based_deepnn._JourneyBasedDeepNN__nn.get_weights()

    for inital, post_training in zip(initial_weights, post_training_weigths):

        assert (inital != post_training).all()
