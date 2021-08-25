import os
import sys

from datetime import datetime

import pytest

from pyspark.sql.session import SparkSession
from pyspark.sql.types import *

from deep_attribution.feature_engineering.feature_engineering import (
    main,
    create_conversion_id_field,
    create_campaign_index_in_journey_field,
    create_journey_id_field,
    get_campaign_nm_to_one_hot_index,
    get_campaigns_at_journey_level,
    get_conversion_status_at_journey_level,
    join_at_journey_level,
    pad_journey_length
)

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

SPARK = SparkSession.builder.getOrCreate()

def test_main(mocker):

    BUCKET_NM = "deep-attribution"
    JOURNEY_MAX_LEN = 3
    
    INPUT = SPARK.createDataFrame(
        SPARK.read.parquet("tests/feature_engineering/impressions_sample.parquet").rdd,
        StructType([
            StructField("timestamp", IntegerType(), False),
            StructField("uid", IntegerType(), False),
            StructField("campaign", StringType(), False),
            StructField("conversion", BooleanType(), False)
        ])
    )

    PARSER = ParserMock(BUCKET_NM, JOURNEY_MAX_LEN)

    mocker.patch(
        "deep_attribution.feature_engineering.feature_engineering.parse_args",
        return_value=PARSER
    )
    
    mocker.patch(
        "deep_attribution.feature_engineering.feature_engineering.load_impressions",
        return_value=INPUT
    )

    mocker.patch(
        "deep_attribution.feature_engineering.feature_engineering.save_campaign_nm_to_one_hot_index",
        return_value=None
    )

    mocker.patch(
        "deep_attribution.feature_engineering.feature_engineering.save",
        return_value=None
    )

    main()

    assert True


class ParserMock:

    def __init__(self, bucket_nm, journey_max_len):
        self.bucket_nm = bucket_nm
        self.journey_max_len = journey_max_len


def test_create_conversion_id_field():

    INPUT = SPARK.createDataFrame(
        [
            (1, False, 10, "google"),
            (1, True, 9, "display"),
            (1, False, 7, "facebook"),
            (2, False, 8, "google"),
            (2, False, 7, "facebook"),
            (2, False, 6, "display")
        ],
        StructType([
            StructField("uid", IntegerType(), False),
            StructField("conversion", BooleanType(), False),
            StructField("timestamp", IntegerType(), False),
            StructField("campaign", StringType(), False)
        ])
    )

    EXPECTED = SPARK.createDataFrame(
        [
            (1, False, 10, "google", 0),
            (1, True, 9, "display", 1),
            (1, False, 7, "facebook", 1),
            (2, False, 8, "google", 0),
            (2, False, 7, "facebook", 0),
            (2, False, 6, "display", 0)
        ],
        StructType([
            StructField("uid", IntegerType(), False),
            StructField("conversion", BooleanType(), False),
            StructField("timestamp", IntegerType(), False),
            StructField("campaign", StringType(), False),
            StructField("conversion_id", IntegerType(), True)
        ])
    )


    obtained = create_conversion_id_field(
        INPUT,
        SPARK
        )

    assert check_dataframes_schema_are_equal(EXPECTED, obtained)

    assert check_dataframes_data_are_equal(EXPECTED, obtained)


def test_create_journey_id_field():

    INPUT = SPARK.createDataFrame(
        [
            (1, False, 10, "google", 0),
            (1, True, 9, "display", 1),
            (1, False, 7, "facebook", 1),
            (2, False, 8, "google", 0),
            (2, False, 7, "facebook", 0),
            (2, False, 6, "display", 0)
        ],
        StructType([
            StructField("uid", IntegerType(), False),
            StructField("conversion", BooleanType(), False),
            StructField("timestamp", IntegerType(), False),
            StructField("campaign", StringType(), False),
            StructField("conversion_id", IntegerType(), True)
        ])
    )

    EXPECTED = SPARK.createDataFrame(
        [
            (False, 10, "google", 10),
            (True, 9, "display", 11),
            (False, 7, "facebook", 11),
            (False, 8, "google", 20),
            (False, 7, "facebook", 20),
            (False, 6, "display", 20)
        ],
        StructType([
            StructField("conversion", BooleanType(), False),
            StructField("timestamp", IntegerType(), False),
            StructField("campaign", StringType(), False),
            StructField("journey_id", IntegerType(), False)
        ])
    )

    obtained = create_journey_id_field(INPUT, SPARK)

    assert check_dataframes_schema_are_equal(EXPECTED, obtained)

    assert check_dataframes_data_are_equal(EXPECTED, obtained)


def test_create_campaign_index_in_journey_field():

    INPUT = SPARK.createDataFrame(
        [
            (False, 10, "google", 10),
            (True, 9, "display", 11),
            (False, 7, "facebook", 11),
            (False, 8, "google", 20),
            (False, 7, "facebook", 20),
            (False, 6, "display", 20)
        ],
        StructType([
            StructField("conversion", BooleanType(), False),
            StructField("timestamp", IntegerType(), False),
            StructField("campaign", StringType(), False),
            StructField("journey_id", IntegerType(), False)
        ])
    )

    EXPECTED = SPARK.createDataFrame(
        [
            (10, False, "google", 1),
            (11, True, "display", 2),
            (11, False, "facebook", 1),
            (20, False, "google", 3),
            (20, False, "facebook", 2),
            (20, False, "display", 1)
        ],
        StructType([
            StructField("journey_id", IntegerType(), False),
            StructField("conversion", BooleanType(), False),
            StructField("campaign", StringType(), False),
            StructField("campaign_index_in_journey", IntegerType(), False)
        ])
    )

    obtained = create_campaign_index_in_journey_field(INPUT, SPARK)

    assert check_dataframes_schema_are_equal(EXPECTED, obtained)

    assert check_dataframes_data_are_equal(EXPECTED, obtained)



def test_pad_journey_length():

    INPUT = SPARK.createDataFrame(
        [
            (10, False, "google", 1),
            (11, True, "display", 2),
            (11, False, "facebook", 1),
            (20, False, "google", 3),
            (20, False, "facebook", 2),
            (20, False, "display", 1)
        ],
        StructType([
            StructField("journey_id", IntegerType(), False),
            StructField("conversion", BooleanType(), False),
            StructField("campaign", StringType(), False),
            StructField("campaign_index_in_journey", IntegerType(), False)
        ])
    )

    JOURNEY_MAX_LEN = 2


    EXPECTED = SPARK.createDataFrame(
        [
            (False, "google", 10, 1),
            (True, "display", 11, 2),
            (False, "facebook", 11, 1),
            (False, "facebook", 20, 2),
            (False, "display", 20, 1)
        ],
        StructType([
            StructField("conversion", BooleanType(), False),
            StructField("campaign", StringType(), False),
            StructField("journey_id", IntegerType(), False),
            StructField("campaign_index_in_journey", IntegerType(), False)
        ])
    )

    obtained = pad_journey_length(INPUT, SPARK, JOURNEY_MAX_LEN)

    assert check_dataframes_schema_are_equal(EXPECTED, obtained)

    assert check_dataframes_data_are_equal(EXPECTED, obtained)


def test_get_campaign_nm_to_one_hot_index():

    INPUT = SPARK.createDataFrame(
        [
            (False, "google", 10, 1),
            (True, "display", 11, 2),
            (False, "facebook", 11, 1),
            (False, "facebook", 20, 2),
            (False, "display", 20, 1)
        ],
        StructType([
            StructField("conversion", BooleanType(), False),
            StructField("campaign", StringType(), False),
            StructField("journey_id", IntegerType(), False),
            StructField("campaign_index_in_journey", IntegerType(), False)
        ])
    )

    EXPECTED = {
        "display":0,
        "facebook":1,
        "google":2
    }

    obtained = get_campaign_nm_to_one_hot_index(INPUT)

    assert EXPECTED == obtained



def test_get_conversion_status_at_journey_level():

    INPUT = SPARK.createDataFrame(
        [
            (False, "google", 10, 1),
            (True, "display", 11, 2),
            (False, "facebook", 11, 1),
            (False, "facebook", 20, 2),
            (False, "display", 20, 1)
        ],
        StructType([
            StructField("conversion", BooleanType(), False),
            StructField("campaign", StringType(), False),
            StructField("journey_id", IntegerType(), False),
            StructField("campaign_index_in_journey", IntegerType(), False)
        ])
    )

    EXPECTED = SPARK.createDataFrame(
        [
            (10, False),
            (11, True),
            (20, False)
        ],
        StructType([
            StructField("journey_id", IntegerType(), False),
            StructField("conversion_status", BooleanType(), False)
        ])
    )

    obtained = get_conversion_status_at_journey_level(INPUT, SPARK)

    assert check_dataframes_schema_are_equal(EXPECTED, obtained)

    assert check_dataframes_data_are_equal(EXPECTED, obtained)


def test_get_campaigns_at_journey_level():
 
    INPUT = SPARK.createDataFrame(
        [
            (False, "google", 10, 1),
            (True, "display", 11, 2),
            (False, "facebook", 11, 1),
            (False, "facebook", 20, 2),
            (False, "display", 20, 1)
        ],
        StructType([
            StructField("conversion", BooleanType(), False),
            StructField("campaign", StringType(), False),
            StructField("journey_id", IntegerType(), False),
            StructField("campaign_index_in_journey", IntegerType(), False)
        ])
    )

    JOURNEY_MAX_LEN = 2
    
    EXPECTED = SPARK.createDataFrame(
        [
            ( 10, "google", None),
            (11,"facebook", "display"),
            (20, "display", "facebook")
        ],
        StructType([
            StructField("journey_id", IntegerType(), False),
            StructField("campaign_nm_at_index_1_in_journey", StringType(), False),
            StructField("campaign_nm_at_index_2_in_journey", StringType(), True),
        ])
    )

    obtained = get_campaigns_at_journey_level(INPUT, SPARK, JOURNEY_MAX_LEN)

    assert check_dataframes_schema_are_equal(EXPECTED, obtained)

    assert check_dataframes_data_are_equal(EXPECTED, obtained)


def test_join_at_journey_level():

    LEFT = SPARK.createDataFrame(
        [
            (10, "google", None),
            (11,"facebook", "display"),
            (20, "display", "facebook")
        ],
        StructType([
            StructField("journey_id", IntegerType(), False),
            StructField("campaign_nm_at_index_1_in_journey", StringType(), False),
            StructField("campaign_nm_at_index_2_in_journey", StringType(), True),
        ])
    )

    RIGHT = SPARK.createDataFrame(
        [
            (10, False),
            (11, True),
            (20, False)
        ],
        StructType([
            StructField("journey_id", IntegerType(), False),
            StructField("conversion_status", BooleanType(), False)
        ])
    )

    EXPECTED = SPARK.createDataFrame(
        [
            (10, "google", None, False),
            (11, "facebook", "display", True),
            (20, "display", "facebook", False)
        ],
        StructType([
            StructField("journey_id", IntegerType(), False),
            StructField("campaign_nm_at_index_1_in_journey", StringType(), False),
            StructField("campaign_nm_at_index_2_in_journey", StringType(), True),
            StructField("conversion_status", BooleanType(), False)
        ])
    )

    obtained = join_at_journey_level(LEFT, RIGHT)

    assert check_dataframes_schema_are_equal(EXPECTED, obtained)

    assert check_dataframes_data_are_equal(EXPECTED, obtained)


def to_datetime(string):
    return datetime.strptime(string, "%Y-%m-%d %H:%M:%S")

def check_dataframes_schema_are_equal(ref, test):

    field_to_list_transfo = lambda field: (field.name, field.dataType, field.nullable)

    ref_fields = [*map(field_to_list_transfo, ref.schema.fields)]
    test_fields = [*map(field_to_list_transfo, test.schema.fields)]

    print(ref_fields, test_fields)

    return set(ref_fields) == set(test_fields)


def check_dataframes_data_are_equal(ref, test):

    ref = ref.collect()
    test = test.collect()

    return set(ref) == set(test)