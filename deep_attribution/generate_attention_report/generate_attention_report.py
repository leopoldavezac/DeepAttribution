from typing import Dict, List

import argparse

from pyspark.sql import DataFrame
from pyspark.sql.session import SparkSession
from pyspark.sql.types import *

def main() -> None:

    args = parse_args()

    spark = create_spark_session()

    df_campaigns_at_journey_level = load_campaigns_at_journey_level(spark, args.bucket_nm)
    df_attentions_at_journey_level = load_attention_at_journey_level(spark, args.bucket_nm)

    df_campaign = unpivot_on_journey_id(spark, df_campaigns_at_journey_level, args.journey_max_len)
    df_attention = unpivot_on_journey_id(spark, df_attentions_at_journey_level, args.journey_max_len)
    del df_attentions_at_journey_level, df_campaigns_at_journey_level

    df_campaign = create_impression_id_field(spark, df_campaign)
    df_attention = create_impression_id_field(spark, df_attention)

    df_campaign_attention = join_on_impression_id(df_campaign, df_attention)
    del df_campaign, df_attention

    df_campaign_total_attention = compute_total_attention_by_campaign(
        spark,
        df_campaign_attention
        )
    df_campaign_average_attention = compute_average_attention_by_campaign(
        spark,
        df_campaign_attention
    )
    del df_campaign_attention

    df_attention_at_campaign_level = join_on_campaign_nm(
        df_campaign_total_attention,
        df_campaign_average_attention
    )
    del df_campaign_total_attention, df_campaign_average_attention

    save_as_parquet(df_attention_at_campaign_level, args.bucket_nm)
    


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser()

    parser.add_argument('--bucket_nm', type=str)

    parser.add_argument('--journey_max_len', type=int)

    return parser.parse_args()

def create_spark_session() -> SparkSession:

    return SparkSession.builder.getOrCreate()

def load_campaigns_at_journey_level(spark: SparkSession, bucket_nm:str) -> DataFrame:
    
    return spark.read.parquet("s3://%s/feature_store/train.parquet" % bucket_nm)

def load_attention_at_journey_level(spark: SparkSession, bucket_nm:str) -> DataFrame:
    
    return spark.read.parquet("s3://%s/attention_report/attention_score.parquet"%bucket_nm)

def unpivot_on_journey_id(
    spark: SparkSession,
    df_journey: DataFrame,
    journey_max_len: int
    ) -> DataFrame:

    pivoted_col_nms = get_pivoted_col_nms(df_journey.columns)
    pivoted_values_nm = get_pivoted_values_nm(pivoted_col_nms)

    df_impression = stack(
        spark,
        df_journey,
        journey_max_len,
        pivoted_col_nms,
        pivoted_values_nm
        )

    schema = StructType([
            StructField("journey_id", LongType(), False),
            StructField(pivoted_values_nm+"_index_in_journey", IntegerType(), False),
            StructField(
                pivoted_values_nm,
                StringType() if pivoted_values_nm == "campaign_nm" else FloatType(),
                True if pivoted_values_nm == "campaign_nm" else False
                )
        ])

    return spark.createDataFrame(df_impression.rdd, schema=schema)
 

def get_pivoted_col_nms(col_nms: List[str]) -> List[str]:
    
    return [
        col_nm for col_nm in col_nms
        if col_nm != "journey_id"
        ]

def get_pivoted_values_nm(pivoted_col_nms: List[str]) -> str:
    
    if "campaign_nm" in pivoted_col_nms[0]:
        pivoted_values_nm = "campaign_nm"
    else:
        pivoted_values_nm = "attention"

    return pivoted_values_nm

def stack(
    spark: SparkSession,
    df_journey: DataFrame,
    journey_max_len: int,
    pivoted_col_nms: List[str],
    pivoted_values_nm: str
    ) -> DataFrame:

    pivoted_col_journey_indexes = [
        int(pivoted_col_nm.split("_")[-3])
        for pivoted_col_nm in pivoted_col_nms
    ]

    sql = """
    select journey_id, stack(%d, %s) as (%s, %s) from journey
    """ % (
        journey_max_len,
        ", ".join([
            "%d, %s" % (index_in_journey, pivoted_col_nm) 
            for index_in_journey, pivoted_col_nm
            in zip(pivoted_col_journey_indexes, pivoted_col_nms)
        ]),
        pivoted_values_nm+"_index_in_journey",
        pivoted_values_nm
    )

    df_journey.createOrReplaceTempView("journey")

    return spark.sql(sql)


def create_impression_id_field(spark: SparkSession, df_impression: DataFrame) -> DataFrame:

    values_nm = df_impression.columns[-1]

    sql = """
    select int(concat(string(journey_id), %s_index_in_journey)) as impression_id, %s
    from impression 
    """ % (values_nm, values_nm)

    df_impression.createOrReplaceTempView("impression")

    df_impression = spark.sql(sql)

    schema = StructType([
        StructField("impression_id", LongType(), False),
        StructField(
            values_nm,
            StringType() if values_nm == "campaign_nm" else FloatType(),
            True if values_nm == "campaign_nm" else False
            )
    ])

    return spark.createDataFrame(df_impression.rdd, schema=schema)

def join_on_impression_id(left: DataFrame, right: DataFrame) -> DataFrame:
    
    return left.join(right, on = "impression_id")

def compute_total_attention_by_campaign(spark: SparkSession, df_campaign_attention: DataFrame) -> DataFrame:

    df_total_attention_at_campaign_level = df_campaign_attention.groupBy("campaign_nm").sum("attention")

    schema = StructType([
        StructField("campaign_nm", StringType(), True),
        StructField("total_attention", FloatType(), False)
    ])

    return spark.createDataFrame(df_total_attention_at_campaign_level.rdd, schema=schema)

def compute_average_attention_by_campaign(spark: SparkSession, df_campaign_attention: DataFrame) -> DataFrame:

    df_average_attention_at_campaign_level = df_campaign_attention.groupBy("campaign_nm").mean("attention")

    schema = StructType([
        StructField("campaign_nm", StringType(), True),
        StructField("average_attention", FloatType(), False)
    ])

    return spark.createDataFrame(df_average_attention_at_campaign_level.rdd, schema=schema)

def join_on_campaign_nm(left: DataFrame, right: DataFrame) -> DataFrame:

    return left.join(right, on="campaign_nm")

def save_as_parquet(df: DataFrame, bucket_nm:str) -> None:

    df.write.parquet("s3://%s/attention_report/campaign_attention.parquet" % bucket_nm)

