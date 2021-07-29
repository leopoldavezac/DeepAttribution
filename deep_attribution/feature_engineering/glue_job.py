import sys

from json import dumps
from typing import List, Dict

from pyspark.sql import DataFrame
from pyspark.sql.session import SparkSession
from pyspark.context import SparkContext
from pyspark.sql.functions import col
from pyspark.ml.feature import OneHotEncoder, StringIndexer

from awsglue.utils import getResolvedOptions
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.dynamicframe import DynamicFrame

import boto3



JOURNEY_MAX_LEN = 10


def run():

    args = getResolvedOptions(sys.argv, ['JOB_NAME'])

    sc = SparkContext()
    glue = GlueContext(sc)
    spark = glue.spark_session
    job = Job(glue)
    job.init(args['JOB_NAME'], args)

    dfc_impressions = load_impressions(glue)
    df_impressions = convert_dynamic_frame_to_data_frame(dfc_impressions)

    df_journeys = create_feature_store(df_impressions, spark)

    set_vals = split_train_test_val(df_journeys)
    set_nms = ["train", "test", "val"]

    for set_nm, set_val in zip(set_nms, set_vals):

        set_val = convert_data_frame_to_dynamic_frame(glue, set_val)
        save_journeys(glue, set_val, set_nm)


def load_impressions(glue: GlueContext) -> DynamicFrame:

    return glue.create_dynamic_frame.from_options(
        connection_type = "s3",
        format = "parquet",
        connection_options = {"paths": ["s3://deep-attribution/raw/impressions.parquet"], "recurse":True},
        transformation_ctx = "raw_impressions"
        )

def convert_dynamic_frame_to_data_frame(dfc: DynamicFrame) -> DataFrame:
    
    return dfc.toDF()

def convert_data_frame_to_dynamic_frame(glue: GlueContext, df: DataFrame) -> DynamicFrame:
    
    dfc = DynamicFrame.fromDF(df, glue, "dfc")
    return dfc

def create_feature_store(impressions: DataFrame, spark: SparkSession) -> DataFrame:

    impressions = create_conversion_id_field(impressions, spark)
    impressions = create_journey_id_field(impressions, spark)
    impressions = create_engagement_index_in_journey_field(impressions, spark) 
    impressions = pad_journey_length(impressions, spark)

    cmpgn_nm_to_index = get_cmpgn_nm_to_index(impressions)
    save_cmpgn_nm_to_index(cmpgn_nm_to_index)

    conversion_at_journey_level = get_conversion_status_at_journey_level(
        impressions
        )
    campaigns_at_journey_level = get_campaigns_at_journey_level(
        impressions
    )
    del impressions

    journeys = join_at_journey_level(
        conversion_at_journey_level,
        campaigns_at_journey_level
    )

    return journeys
   


# testable
def create_conversion_id_field(df: DataFrame, spark: SparkSession) -> DataFrame:

    df.createOrReplaceTempView("impressions")

    sql_conversion_id = """
    select uid, conversion, timestamp, campaign, 
    row_number() over(partition by uid order by timestamp) as conversion_id
    from impressions
    where conversion != 0
    union
    select uid, conversion, timestamp, campaign, 
    -1 as conversion_id from impressions
    where conversion == 0
    """
    return spark.sql(sql_conversion_id)

# testable
def create_journey_id_field(df: DataFrame, spark: SparkSession) -> DataFrame:

    df.createOrReplaceTempView("impressions")

    sql_journey_id = """
    select conversion, timestamp, campaign, 
    concat(uid, conversion_id) as journey_id
    from impressions
    order by journey_id, timestamp asc
    """
    return spark.sql(sql_journey_id)

# testable
def create_engagement_index_in_journey_field(
    df: DataFrame, spark: SparkSession
    ) -> DataFrame:

    df.createOrReplaceTempView("impressions")

    sql_eng_index = """
    select journey_id, conversion, campaign, 
    row_number() over(partition by journey_id order by timestamp asc) as eng_index
    from impressions
    """

    return spark.sql(sql_eng_index)

# testable
def pad_journey_length(
    df: DataFrame, spark: SparkSession
    ) -> DataFrame:

    df.createOrReplaceTempView("impressions")

    sql_pad_journey_length = """
    select conversion, campaign, journey_id, 
    concat("eng_at_index_",cast(eng_index as string)) as eng_index
    from impressions
    where eng_index <= %d
    """ % (JOURNEY_MAX_LEN)

    return spark.sql(sql_pad_journey_length)


# testable
def get_cmpgn_nm_to_index(df: DataFrame) -> Dict:

    cmpgn_nm_to_index = df.groupby("campaign").count()
    cmpgn_nm_to_index = cmpgn_nm_to_index.sort(col("count").desc()).toPandas()
    cmpgn_nm_to_index.reset_index(inplace=True)
    cmpgn_nm_to_index.set_index("campaign", inplace=True)
    cmpgn_nm_to_index.drop(columns="count", inplace=True)
    return cmpgn_nm_to_index["index"].to_dict()


def save_cmpgn_nm_to_index(cmpgn_nm_to_index: Dict) -> None:

    s3 = boto3.resource("s3")
    obj = s3.Object("deep-attribution", "campaign_nm_to_index.json")

    obj.put(Body=bytes(dumps(cmpgn_nm_to_index).encode("UTF-8")))


def get_conversion_status_at_journey_level(df: DataFrame) -> DataFrame:
    
    conversion = df.groupby("journey_id").agg({"conversion":"max"})
    conversion = conversion.withColumnRenamed("max(conversion)", "conversion")
    
    return conversion


def get_campaigns_at_journey_level(
    df: DataFrame
    ) -> DataFrame:

    eng_indexes = [i for i in range(1, JOURNEY_MAX_LEN+1)]
    campaigns = df.groupby("journey_id").pivot(
        "eng_index", eng_indexes
        ).agg({"campaign":"max"})

    col_renaming = [col(str(i)).alias("campaign_at_index_"+str(i)) for i in range(1, JOURNEY_MAX_LEN)] + ["journey_id"]
    campaigns = campaigns.select(*col_renaming)

    return campaigns

def join_at_journey_level(
    left: DataFrame, right: DataFrame
    ) -> DataFrame:
    
    return left.join(right, on = "journey_id")

def split_train_test_val(journeys: DataFrame) -> List[DataFrame]:

    return journeys.randomSplit([0.8, 0.1, 0.1], 24)


def save_journeys(glue: GlueContext, dfc: DynamicFrame, set_nm: str):
    glue.write_dynamic_frame.from_options(
        frame = dfc,
        connection_type = "s3",
        format = "parquet",
        connection_options = {"path": "s3://deep-attribution/feature_store/%s.parquet/" % set_nm, "partitionKeys": []},
        transformation_ctx = "dfc"
        )


if __name__ == "__main__":
    run()