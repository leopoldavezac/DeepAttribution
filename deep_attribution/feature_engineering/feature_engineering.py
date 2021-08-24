import argparse
from typing import List, Dict

import sys

from json import dumps

from pyspark.sql import DataFrame, Window
from pyspark.sql.session import SparkSession
from pyspark.sql.functions import col, last
from pyspark.sql.types import *


import boto3


def main() -> None:

    args = parse_args()

    spark = create_spark_session()

    df_impressions = load_impressions(spark, args.bucket_nm)
    df_impressions = create_conversion_id_field(df_impressions, spark)
    df_impressions = create_journey_id_field(df_impressions, spark)
    df_impressions = create_campaign_index_in_journey_field(df_impressions, spark)
    df_impressions = pad_journey_length(df_impressions, spark, args.journey_max_len)
    campaign_nm_to_one_hot_index = get_campaign_nm_to_one_hot_index(df_impressions)
    save_campaign_nm_to_one_hot_index(campaign_nm_to_one_hot_index, args.bucket_nm)

    df_journeys_conversion = get_conversion_status_at_journey_level(df_impressions, spark)
    df_journeys_campaigns = get_campaigns_at_journey_level(
        df_impressions, spark, args.journey_max_len)
    del df_impressions

    df_journeys = join_at_journey_level(df_journeys_campaigns, df_journeys_conversion)
    del df_journeys_campaigns, df_journeys_conversion

    df_sets = split_train_test_val(df_journeys)
    set_nms = ["train", "test", "val"]

    for set_nm, df_set in zip(set_nms, df_sets):
        save(df_set, set_nm, args.bucket_nm)



def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser()

    parser.add_argument('--bucket_nm', type=str)

    parser.add_argument('--journey_max_len', type=int)

    return parser.parse_args()

def create_spark_session() -> SparkSession:

    return SparkSession.builder.getOrCreate()

def load_impressions(spark: SparkSession, bucket_nm: str) -> DataFrame:

    df = spark.read.parquet("s3://%s/raw/impressions.parquet"%bucket_nm)

    schema = StructType([
            StructField("timestamp", IntegerType(), False),
            StructField("uid", IntegerType(), False),
            StructField("campaign", StringType(), False),
            StructField("conversion", BooleanType(), False)
        ])

    df = spark.createDataFrame(df.rdd, schema=schema)

    df = df.sort("uid", "timestamp")

    return df

def create_conversion_id_field(df: DataFrame, spark: SparkSession) -> DataFrame:

    df.createOrReplaceTempView("impressions")

    sql = """
    select uid, conversion, timestamp, campaign, 
    row_number() over(partition by uid order by timestamp) as conversion_id
    from impressions
    where conversion == 1
    union
    select uid, conversion, timestamp, campaign, 
    null as conversion_id
    from impressions
    where conversion == 0
    """
    df = spark.sql(sql)

    df = backward_fill_conversion_id_by_user(df)
    df = fillna_conversion_id(df, spark)

    return df


def backward_fill_conversion_id_by_user(df: DataFrame) -> DataFrame:

    window = Window.partitionBy("uid").orderBy("timestamp").rowsBetween(0, sys.maxsize)
    conversion_id = last(df["conversion_id"], ignorenulls=True).over(window)

    df = df.withColumn("conversion_id", conversion_id)

    return df


def fillna_conversion_id(df: DataFrame, spark: SparkSession) -> DataFrame:

    df.createOrReplaceTempView("impressions")

    sql = """
    select uid, conversion, timestamp, campaign, conversion_id
    from impressions
    where conversion_id is not null
    union
    select uid, conversion, timestamp, campaign,
    0 as conversion_id
    from impressions    
    where conversion_id is null
    """

    df = spark.sql(sql)

    return df


def create_journey_id_field(df: DataFrame, spark: SparkSession) -> DataFrame:

    df.createOrReplaceTempView("impressions")

    sql = """
    select conversion, timestamp, campaign, 
    int(concat(string(uid), conversion_id)) as journey_id
    from impressions
    order by journey_id, timestamp asc
    """
    df = spark.sql(sql)

    schema = StructType([
            StructField("conversion", BooleanType(), False),
            StructField("timestamp", IntegerType(), False),
            StructField("campaign", StringType(), False),
            StructField("journey_id", IntegerType(), False)
        ])

    df = spark.createDataFrame(df.rdd, schema)

    return df

def create_campaign_index_in_journey_field(
    df: DataFrame, spark: SparkSession
    ) -> DataFrame:

    df.createOrReplaceTempView("impressions")

    sql = """
    select journey_id, conversion, campaign, 
    row_number() over(partition by journey_id order by timestamp asc) as campaign_index_in_journey
    from impressions
    """

    df = spark.sql(sql)

    schema = StructType([
        StructField("journey_id", IntegerType(), False),
        StructField("conversion", BooleanType(), False),
        StructField("campaign", StringType(), False),
        StructField("campaign_index_in_journey", IntegerType(), False)

    ])

    df = spark.createDataFrame(df.rdd, schema)

    return df

def pad_journey_length(
    df: DataFrame, spark: SparkSession, journey_max_len: int
    ) -> DataFrame:

    df.createOrReplaceTempView("impressions")

    sql = """
    select conversion, campaign, journey_id, campaign_index_in_journey
    from impressions
    where campaign_index_in_journey <= %d
    """ % (journey_max_len)

    return spark.sql(sql)


def get_campaign_nm_to_one_hot_index(df: DataFrame) -> Dict:

    campaign_nms = [
        campaign_nm.campaign
        for campaign_nm in df.select("campaign").distinct().collect()
    ]
    campaign_nms.sort()

    campaign_nm_to_one_hot_index = {}
    for index, campaign_nm in enumerate(campaign_nms):
        campaign_nm_to_one_hot_index[campaign_nm] = index

    
    return campaign_nm_to_one_hot_index


def save_campaign_nm_to_one_hot_index(campaign_nm_to_one_hot_index: Dict, bucket_nm:str) -> None:

    s3 = boto3.resource("s3")
    obj = s3.Object(bucket_nm, "campaign_nm_to_one_hot_index.json")

    obj.put(Body=bytes(dumps(campaign_nm_to_one_hot_index).encode("UTF-8")))


def get_conversion_status_at_journey_level(df: DataFrame, spark: SparkSession) -> DataFrame:
    
    conversion = df.groupby("journey_id").agg({"conversion":"max"})
    conversion = conversion.withColumnRenamed("max(conversion)", "conversion_status")

    schema = StructType([
        StructField("journey_id", IntegerType(), False),
        StructField("conversion_status", BooleanType(), False),
    ])

    conversion = spark.createDataFrame(conversion.rdd, schema)
    
    return conversion


def get_campaigns_at_journey_level(
    df: DataFrame, spark: SparkSession, journey_max_len: int
    ) -> DataFrame:

    eng_indexes = [i for i in range(1, journey_max_len+1)]
    campaigns = df.groupby("journey_id").pivot(
        "campaign_index_in_journey", eng_indexes
        ).agg({"campaign":"max"})

    col_renaming = ["journey_id"] + [
        col(str(i)).alias("campaign_at_index_%d_in_journey" % i)
        for i in range(1, journey_max_len+1)
        ]
    campaigns = campaigns.select(*col_renaming)

    schema = [StructField("journey_id", IntegerType(), False)] + [
        StructField(
            "campaign_nm_at_index_%d_in_journey"%i,
            StringType(),
            False if i == 1 else True
            ) for i in range(1, journey_max_len+1)
            ]
    schema = StructType(schema)

    campaigns = spark.createDataFrame(campaigns.rdd, schema)

    return campaigns

def join_at_journey_level(
    left: DataFrame, right: DataFrame
    ) -> DataFrame:
    
    return left.join(right, on = "journey_id")

def split_train_test_val(journeys: DataFrame) -> List[DataFrame]:

    return journeys.randomSplit([0.8, 0.1, 0.1], 24)


def save(df: DataFrame, set_nm: str, bucket_nm:str) -> None:

    df.write.parquet("s3://%s/feature_store/%s.parquet" % (bucket_nm, set_nm))


if __name__ == "__main__":
    main()