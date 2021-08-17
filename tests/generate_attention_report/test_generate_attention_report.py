import os, sys
from pyspark.sql.dataframe import DataFrame

import pytest

from numpy import allclose

from pyspark.sql.session import SparkSession
from pyspark.sql.types import *

from deep_attribution.generate_attention_report.generate_attention_report import (
    unpivot_on_journey_id,
    create_impression_id_field,
    join_on_impression_id,
    compute_total_attention_by_campaign,
    compute_average_attention_by_campaign
)

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

SPARK = SparkSession.builder.getOrCreate()
JOURNEY_MAX_LEN = 2

@pytest.mark.parametrize("input, expected", [
    (
        SPARK.createDataFrame(
            [
                (10, "google", None),
                (11, "facebook", "display"),
                (20, "display", "facebook")
            ],
            StructType([
                StructField("journey_id", IntegerType(), False),
                StructField("campaign_nm_at_index_1_in_journey", StringType(), False),
                StructField("campaign_nm_at_index_2_in_journey", StringType(), True)
            ])
        ),
        SPARK.createDataFrame(
            [
                (10, 1, "google"),
                (10, 2, None),
                (11, 1, "facebook"),
                (11, 2, "display"),
                (20, 1, "display"),
                (20, 2, "facebook")
            ],
            StructType([
                StructField("journey_id", IntegerType(), False),
                StructField("campaign_nm_index_in_journey", IntegerType(), False),
                StructField("campaign_nm", StringType(), True)
            ])
        )
    ),
    (
        SPARK.createDataFrame(
            [
                (10, 0.1, 0.2),
                (11, 0.3, 0.01),
                (20, 0.2, 0.2)
            ],
            StructType([
                StructField("journey_id", IntegerType(), False),
                StructField("attention_at_index_1_in_journey", FloatType(), False),
                StructField("attention_at_index_2_in_journey", FloatType(), False)
            ])
        ),
        SPARK.createDataFrame(
            [
                (10, 1, 0.1),
                (10, 2, 0.2),
                (11, 1, 0.3),
                (11, 2, 0.01),
                (20, 1, 0.2),
                (20, 2, 0.2)
            ],
            StructType([
                StructField("journey_id", IntegerType(), False),
                StructField("attention_index_in_journey", IntegerType(), False),
                StructField("attention", FloatType(), False)
            ])
        )
    )
])
def test_unpivot_on_journey_id(input, expected):

    obtained = unpivot_on_journey_id(SPARK, input, JOURNEY_MAX_LEN)

    assert check_dataframes_schema_are_equal(expected, obtained)

    assert check_dataframes_data_are_equal(expected, obtained)


@pytest.mark.parametrize("input, expected", [
    (
        SPARK.createDataFrame(
            [
                (10, 1, "google"),
                (10, 2, None),
                (11, 1, "facebook"),
                (11, 2, "display"),
                (20, 1, "display"),
                (20, 2, "facebook")
            ],
            StructType([
                StructField("journey_id", IntegerType(), False),
                StructField("campaign_nm_index_in_journey", IntegerType(), False),
                StructField("campaign_nm", StringType(), True)
            ])
        ),
        SPARK.createDataFrame(
            [
                (101, "google"),
                (102, None),
                (111, "facebook"),
                (112, "display"),
                (201, "display"),
                (202, "facebook")
            ],
            StructType([
                StructField("impression_id", IntegerType(), False),
                StructField("campaign_nm", StringType(), True)
            ])
        )
    ),
    (
        SPARK.createDataFrame(
            [
                (10, 1, 0.1),
                (10, 2, 0.2),
                (11, 1, 0.3),
                (11, 2, 0.01),
                (20, 1, 0.2),
                (20, 2, 0.2)
            ],
            StructType([
                StructField("journey_id", IntegerType(), False),
                StructField("attention_index_in_journey", IntegerType(), False),
                StructField("attention", FloatType(), False)
            ])
        ),
        SPARK.createDataFrame(
            [
                (101, 0.1),
                (102, 0.2),
                (111, 0.3),
                (112, 0.01),
                (201, 0.2),
                (202, 0.2)
            ],
            StructType([
                StructField("impression_id", IntegerType(), False),
                StructField("attention", FloatType(), False)
            ])
        )
    )
])
def test_create_impression_id_field(input, expected):
    
    obtained = create_impression_id_field(SPARK, input)

    assert check_dataframes_schema_are_equal(expected, obtained)

    assert check_dataframes_data_are_equal(expected, obtained)


def test_join_on_impression_id():

    RIGHT = SPARK.createDataFrame(
            [
                (101, 0.1),
                (102, 0.2),
                (111, 0.3),
                (112, 0.01),
                (201, 0.2),
                (202, 0.2)
            ],
            StructType([
                StructField("impression_id", IntegerType(), False),
                StructField("attention", FloatType(), False)
            ])
        )

    LEFT = SPARK.createDataFrame(
            [
                (101, "google"),
                (102, None),
                (111, "facebook"),
                (112, "display"),
                (201, "display"),
                (202, "facebook")
            ],
            StructType([
                StructField("impression_id", IntegerType(), False),
                StructField("campaign_nm", StringType(), True)
            ])
        )

    EXPECTED = SPARK.createDataFrame(
            [
                (101, "google", 0.1),
                (102, None, 0.2),
                (111, "facebook", 0.3),
                (112, "display", 0.01),
                (201, "display", 0.2),
                (202, "facebook", 0.2)
            ],
            StructType([
                StructField("impression_id", IntegerType(), False),
                StructField("campaign_nm", StringType(), True),
                StructField("attention", FloatType(), False)
            ])
        )

    obtained = join_on_impression_id(LEFT, RIGHT)

    assert check_dataframes_schema_are_equal(EXPECTED, obtained)

    assert check_dataframes_data_are_equal(EXPECTED, obtained)


def test_compute_total_attention_by_campaign():

    INPUT = SPARK.createDataFrame(
            [
                (101, "google", 0.1),
                (102, None, 0.2),
                (111, "facebook", 0.3),
                (112, "display", 0.01),
                (201, "display", 0.2),
                (202, "facebook", 0.2)
            ],
            StructType([
                StructField("impression_id", IntegerType(), False),
                StructField("campaign_nm", StringType(), True),
                StructField("attention", FloatType(), False)
            ])
        )

    EXPECTED = SPARK.createDataFrame(
            [
                ("google", 0.1),
                (None, 0.2),
                ("facebook", 0.5),
                ("display", 0.21),
            ],
            StructType([
                StructField("campaign_nm", StringType(), True),
                StructField("total_attention", FloatType(), False)
            ])
        )

    obtained = compute_total_attention_by_campaign(SPARK, INPUT)

    assert check_dataframes_schema_are_equal(EXPECTED, obtained)

    assert check_dataframes_are_approx_equal(EXPECTED, obtained)


def test_compute_average_attention_by_campaign():

    INPUT = SPARK.createDataFrame(
            [
                (101, "google", 0.1),
                (102, None, 0.2),
                (111, "facebook", 0.3),
                (112, "display", 0.01),
                (201, "display", 0.2),
                (202, "facebook", 0.2)
            ],
            StructType([
                StructField("impression_id", IntegerType(), False),
                StructField("campaign_nm", StringType(), True),
                StructField("attention", FloatType(), False)
            ])
        )

    EXPECTED = SPARK.createDataFrame(
            [
                ("google", 0.1),
                (None, 0.2),
                ("facebook", 0.25),
                ("display", 0.105),
            ],
            StructType([
                StructField("campaign_nm", StringType(), True),
                StructField("average_attention", FloatType(), False)
            ])
        )

    obtained = compute_average_attention_by_campaign(SPARK, INPUT)

    assert check_dataframes_schema_are_equal(EXPECTED, obtained)

    assert check_dataframes_are_approx_equal(EXPECTED, obtained)

    

def check_dataframes_schema_are_equal(ref, test):

    field_to_list_transfo = lambda field: (field.name, field.dataType, field.nullable)

    ref_fields = [*map(field_to_list_transfo, ref.schema.fields)]
    test_fields = [*map(field_to_list_transfo, test.schema.fields)]

    return set(ref_fields) == set(test_fields)


def check_dataframes_data_are_equal(ref, test):

    ref = ref.collect()
    test = test.collect()

    return set(ref) == set(test)


def check_dataframes_are_approx_equal(ref: DataFrame, test: DataFrame):

    ref = ref.toPandas().round(2).sort_values("campaign_nm").reset_index(drop=True)
    test = test.toPandas().round(2).sort_values("campaign_nm").reset_index(drop=True)

    return ref.equals(test)