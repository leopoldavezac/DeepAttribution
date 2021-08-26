from json import load

from numpy import array
from pandas import DataFrame, read_parquet

import pytest

from deep_attribution.preprocess.preprocessing import (
    main,
    create_categories_for_one_hot_encoding,
    create_one_hot_encoder,
    format_preprocessed_obs
)
    
def test_main(mocker):

    BUCKET_NM = "deep-attribution"
    JOURNEY_MAX_LEN = 10
    PARSER = ParserMock(BUCKET_NM, JOURNEY_MAX_LEN)

    mocker.patch(
        "deep_attribution.preprocess.preprocessing.parse_args",
        return_value=PARSER
    )

    with open("tests/preprocess/campaign_nm_to_one_hot_index.json", "r") as f:
        CAMPAIGN_NM_TO_INDEX = load(f)

    mocker.patch(
        "deep_attribution.preprocess.preprocessing.load_json_from_s3",
        return_value=CAMPAIGN_NM_TO_INDEX
    )

    DF_SET_OBS = read_parquet("tests/preprocess/train_sample.parquet")

    mocker.patch(
        "deep_attribution.preprocess.preprocessing.load_set",
        return_value=DF_SET_OBS
    )

    mocker.patch(
        "deep_attribution.preprocess.preprocessing.save_as_parquet",
        return_value=None
    )

    main()

    assert True

class ParserMock:

    def __init__(self, bucket_nm, journey_max_len):

        self.bucket_nm = bucket_nm
        self.journey_max_len = journey_max_len



def test_create_categories_for_one_hot_encoding():

    INPUT = {"display":2, "facebook":1, "google":0}
    EXPECTED = array([
        ["google",  "facebook", "display"],
        ["google",  "facebook", "display"]
    ])
    JOURNEY_MAX_LEN = 2

    obtained = create_categories_for_one_hot_encoding(INPUT, JOURNEY_MAX_LEN)

    assert (EXPECTED == obtained).all()


def test_ohe_transform():

    INPUT = array(
        [["display", "google", "facebook"],
        ["google", "display", None]]
        )
    OHE_CATEGORIES = ["google",  "facebook", "display"]

    EXPECTED = array([
        [0, 0, 1, 1, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 1, 0, 0, 0]
    ])

    ohe = create_one_hot_encoder(OHE_CATEGORIES)
    obtained = ohe.fit_transform(INPUT)

    print(obtained)

    assert (EXPECTED == obtained).all()


def test_format_preprocessed_obs():

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
