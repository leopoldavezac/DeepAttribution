from numpy import array
from pandas.core.frame import DataFrame

import pytest

from deep_attribution.preprocess.preprocessing import (
    create_categories_for_one_hot_encoding,
    create_pipeline,
    format_preprocessed_obs
)
import deep_attribution.preprocess.preprocessing
    

def test_create_categories_for_one_hot_encoding():

    INPUT = {"display":2, "facebook":1, "google":0}
    EXXPECTED = ["google",  "facebook", "display"]

    obtained = create_categories_for_one_hot_encoding(INPUT)

    assert EXXPECTED == obtained


def test_pipeline_transform():

    INPUT = array(
        [["display", "google", "facebook"],
        ["google", "display", None]]
        )
    OHE_CATEGORIES = ["google",  "facebook", "display"]

    EXPECTED = array([
        [0, 0, 1, 1, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 1, 0, 0, 0]
    ])

    one_hot_pipeline = create_pipeline(OHE_CATEGORIES)
    obtained = one_hot_pipeline.fit_transform(INPUT)

    print(obtained)

    assert (EXPECTED == obtained).all()


def test_format_preprocessed_obs(mocker):

    X = array([
        [0, 0, 1, 1, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 1, 0, 0, 0]
    ])
    ARR_CONVERSION_STATUS = array([True, False])
    ARR_JOURNEY_ID = array([10, 20])
    OHE_CATEGORY_NMS = ["google", "facebook", "display"]
    JOURNEY_MAX_LEN = 3

    FEATURE_NMS = [
        "campaign_nm_at_index_1_in_journey_is_google",
        "campaign_nm_at_index_1_in_journey_is_facebook",
        "campaign_nm_at_index_1_in_journey_is_display",
        "campaign_nm_at_index_2_in_journey_is_google",
        "campaign_nm_at_index_2_in_journey_is_facebook",
        "campaign_nm_at_index_2_in_journey_is_display",
        "campaign_nm_at_index_3_in_journey_is_google",
        "campaign_nm_at_index_3_in_journey_is_facebook",
        "campaign_nm_at_index_3_in_journey_is_display"
    ]
    EXPECTED = DataFrame(
        [
            [10, False, False, True, True, False, False, False, True, False, True],
            [20, True, False, False, False, False, True, False, False, False, False]
        ],
        columns=["journey_id"]+FEATURE_NMS+["conversion_status"]
    )

    obtained = format_preprocessed_obs(
        X, ARR_JOURNEY_ID, ARR_CONVERSION_STATUS, OHE_CATEGORY_NMS, JOURNEY_MAX_LEN
        )

    assert EXPECTED.columns.tolist() == obtained.columns.tolist()

    assert (EXPECTED.values == obtained.values).all()
