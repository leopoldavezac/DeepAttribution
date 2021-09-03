# Journey

## Definition

A journey is a succession of impressions that leads to a conversion or not.


### Case 1
If a client converted at some point then all the prior impressions are part of the same journey.

<img src="img/journey_case_1.png" width="375"/>

### Case 2
If a client converted multiple times then all the prior impressions are part of the same journey until the next past conversion.

<img src="img/journey_case_2.png"  width="750"/>

### Case 3

If a client never converted then all the impressions are part of the same journey.
<img src="img/journey_case_3.png"  width="375"/>

### Case 4
If a client received new impressions after a conversion and did not convert all impressions after the conversion are part of the same journey.
<img src="img/journey_case_4.png"  width="750"/>


## How to define the journey max length

In order to define the journey max length one might be interested in plotting the number of impressions per journey.

To do so you can use the following code:

The input to main should be the impressions dataset and an active spark session.

```python

from pyspark.sql import DataFrame, Window
from pyspark.sql.session import SparkSession
from pyspark.sql.functions import col, first
from pyspark.sql.types import *

import plotly.graph_objects as go

def main(df: DataFrame, spark: SparkSession):

    df = create_conversion_id_field(df, spark)
    df = create_journey_id_field(df, spark)
    df = compute_nb_impressions_per_journey(df)
    plot_nb_impressions_per_journey(df)

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
    conversion_id = first(df["conversion_id"], ignorenulls=True).over(window)

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
    bigint(concat(string(uid), string(conversion_id))) as journey_id
    from impressions
    order by journey_id, timestamp asc
    """
    df = spark.sql(sql)

    schema = StructType([
            StructField("conversion", BooleanType(), False),
            StructField("timestamp", IntegerType(), False),
            StructField("campaign", StringType(), False),
            StructField("journey_id", LongType(), False)
        ])

    df = spark.createDataFrame(df.rdd, schema=schema)

    return df

def compute_nb_impressions_per_journey(df: DataFrame) -> DataFrame:

    return df.groupby("journey_id").count()

def plot_nb_impressions_per_journey(df: DataFrame) -> None:

    df = df.toPandas()

    fig = go.Figure(
        go.Histogram(df.iloc[:,1])
    )

    fig.show()

```

Ideally you want find the journey_max_len that allows to keep >= 80% of the journeys uncut without having too many journey index mostly empty. A 80/20 rule might be useful here.

Once you have find the ideal values you can update appropriatly the [config.yaml](../config/config.yaml)

### Padding method

During the feature engineering job the journeys are padded using the journey_max_len value. **The first impressions of the padded journeys are dropped.**

