# Pipeline

The multi channel attribution pipeline consists of 4 jobs all running on sagemaker.

## Feature Engineering

The first job takes in raw data at impressions level from S3, clean it, compute features and conversion status at journey level and save the results to S3.

### Job summary

The feature engineering is done using pyspark via the PysparkProcessor object. It takes the **impressions dataset** (s3://**deep-attribution-bucket**/raw/impressions.parquet) as input and output three datasets correspond to the train, test and validation set (s3://**deep-attribution-bucket**/feature_store/set_nm.parquet).

### Scripts

There are two scripts associated with this part the:

- [feature_engineering.py](../deep_attribution/feature_engineering/feature_engineering.py): the job script
- [feature_engineering_exec.py](../deep_attribution/feature_engineering/feature_engineering_exec.py): the script to lauch the job

### Job description

0. parse command line args

1. load impressions data from S3

```bash
 uid | timestamp | campaign | conversion 
_______________________________________

 int |    int    |    str   |    bool 

pyspark.sql.DataFrame
```

2. create a conversion id field for each impressions, an identifier (unique at uid level) that specify which impressions happened in the same journey.

```bash
 uid | timestamp | campaign | conversion | conversion_id 
________________________________________________________

 int |    int    |    str   |    bool    |      int

pyspark.sql.DataFrame
```

3. create a journey id field for each impressions, an unique identifier that specify which impressions happened in the same journey.

```bash
 uid | timestamp | campaign | conversion | journey_id 
________________________________________________________

 int |    int    |    str   |    bool    |    bigint

pyspark.sql.DataFrame
```

4. create a campaign index field for each impressions, an index that specify the order in which impressions have happened at journey level.

```bash
 uid | campaign | conversion | journey_id | campaign_index_in_journey 
_____________________________________________________________________

 int |    str   |    bool    |    bigint  |            int

pyspark.sql.DataFrame
```

5. pad the journey based on the campaign index and the [journey_max_length](../config/config.yaml)

```bash
 uid | campaign | conversion | journey_id | campaign_index_in_journey 
_____________________________________________________________________

 int |    str   |    bool    |    bigint  |            int

pyspark.sql.DataFrame
```

6. get a campaign nm to one hot index mapper and save it to S3

```python

{ 
    "campaign_nm_1":0,
    "campaign_nm_*":5,
    "campaign_nm_n":10
 }

Dict
```

7. get the conversion status at journey level as a DataFrame

```bash
 conversion_status | journey_id
________________________________

        bool      |   bigint  

pyspark.sql.DataFrame
```

8. get the campaign at each journey index at journey level as a DataFrame

```bash
 campaign_nm_at_index_0 | campaign_nm_at_index_* | campaign_nm_at_index_n | journey_id
______________________________________________________________________________________

           str          |           str          |           str          |   bigint  

pyspark.sql.DataFrame
```

9. join the results from 8 and 9 together at journey level.

```bash
 conversion_status | campaign_nm_at_index_0 | campaign_nm_at_index_* | campaign_nm_at_index_n | journey_id
___________________________________________________________________________________________________________

        bool       |           str          |           str          |           str          |   bigint  

pyspark.sql.DataFrame
```

10. split the DataFrame into a train (0.8), a test (0.8) and a validation set (0.1).

10. save the results to S3.

### Minimal configuration

- intance_type: at least ml.c5.xlarge
- instance_count: at least 2

## Preprocessing

The second job takes in data from the feature_store (train, test, val) at journey level from S3, clean it, one hot encode the features and save the results to S3.

### Job summary

The processing is done using sklearn via the SklearnProcessor object. It takes the **feature store dataset** (s3://**deep-attribution-bucket**/feature_store/set_nm.parquet) as input and output three datasets correspond to the train, test and validation set (s3://**deep-attribution-bucket**/feature_store_preprocessed/set_nm.parquet).

### Scripts

There are two scripts associated with this part the:

- [preprocessing.py](../deep_attribution/preprocess/preprocessing.py): the job script
- [preprocessing_exec.py](../deep_attribution/preprocessing/preprocessing_exec.py): the script to lauch the job

### Job description

0. parse command line args

1. load the campaign nm to one hot index mapper from S3.

```python

{ 
    "campaign_nm_1":0,
    "campaign_nm_*":5,
    "campaign_nm_n":10
 }
```

2. transform the mapper into a list where each campaign_nm is at the specified index (**categories**).

```bash

["campaign_nm_1", ..., "campaign_nm_*", ..., "campaign_nm_n"]

```

3. initialize the one hot encoder using **categories**

4. iterate over the set_nms

5. load set

```bash
 conversion_status | campaign_nm_at_index_0 | campaign_nm_at_index_* | campaign_nm_at_index_n | journey_id
 ___________________________________________________________________________________________________________

         bool      |           str          |           str          |           str          |   bigint  

pd.DataFrame
```

6. one hot encode the set campaign_nm_at_index_*

```bash

array([
    [0, 0, ..., 1],
    ..............
    [0, 0, ..., 1]
], shape=(m, n))

m: nb_obs
n: nb_campaigns*journey_max_len

```

7. format the results

```bash
 conversion_status | campaign_nm_at_index_0_is_campaign_nm_1 | campaign_nm_at_index_*_is_campaign_nm_* | journey_id
 ___________________________________________________________________________________________________________________

         bool      |                  bool                   |                  bool                   |   bigint  

pd.DataFrame
```


8. save the results to S3

### Minimal configuration

- intance_type: at least ml.c5.xlarge
- instance_count: at least 2

## Training

## Job summary

The third job load data from the feature store preprocessed (train, test, val in s3://**deep-attribution-bucket**/feature_store_preprocessed/set_nm.parquet) using a generator (tensorflow.keras.utils.Sequence) and train the model, optimize the hyperparameters and save the trained attention model to S3 (s3://**deep-attribution-bucket**/model/).

You can read more on the model and the attention model [here](model.md)

### Scripts

There are five scripts associated with this part the:

- [journey_deepnn.py](../deep_attribution/model/journey_deepnn.py): the model implementation script
- [batch_loader.py](../deep_attribution/train/batch_loader.py): the data generator implementation script
- [oversampling.py](../deep_attribution/train/oversampling.py): the oversampling implementation script
- [train.py](../deep_attribution/train/train.py): the training job script
- [training_exec.py](../deep_attribution/train/training_exec.py): the script to launch training

### Job description

0. parse command line args

1. get the number of campaigns from the campaign name to one hot index mapper

2. initialize the model with the passed hyperparameters

3. initialize the data generator for each set

4. fit the model

5. evaluate the model

6. write the evaluation metrics to S3

7. save the attention model to S3

### Minimal configuration

- intance_type: at least ml.m5.2xlarge
- instance_count: at least 2

## Attention prediction

## Job summary

The fourth job load the training data from the feature store preprocessed (s3://**deep-attribution-bucket**/feature_store_preprocessed/train.parquet), predict the attention values for each journeys and save the results to S3 (s3://**deep-attribution-bucket**/attention_report/attention_score.parquet).

### Scripts

There is one script associated with this part the:

- [predict_attention.py](../deep_attribution/predict/predict_attention.py): the prediction script

### Job description

1. get the number of campaigns from the campaign name to one hot index mapper

2. load the training data

```bash
 conversion_status | campaign_nm_at_index_0_is_campaign_nm_1 | campaign_nm_at_index_*_is_campaign_nm_* | journey_id
 ___________________________________________________________________________________________________________________

         bool      |                  bool                   |                  bool                   |   bigint  

pd.DataFrame
```

3. format the training data for prediction

```bash

array([
    [
        [0, 0, ..., 1],
        ..............
        [0, 0, ..., 1],
    ],
    ...................
    [
        [0, 0, ..., 1],
        ..............
        [0, 0, ..., 1],
    ]
], shape=(m, n, l))

m: nb_obs
n: journey_max_len
l: nb_campaigns

```

4. predict the attention scores for the training data

```bash

array([
    [111345753525, 0.32 ..., 0.0],
    ..............
    [111345753523, 0.15 ..., 0.12]
], shape=(m, n))

m: nb_obs
n: journey_max_len + 1

```

5. format the results as pd.DataFrame

```bash
 attention_at_index_0 | attention_at_index_* | attention_at_index_n | journey_id
 _______________________________________________________________________________

         float        |         float        |         float        |   bigint  

pd.DataFrame
```

5. save the attention score to S3


### Minimal configuration

- intance_type: at least ml.c5.xlarge
- instance_count: at least 2

## Generate attention report

## Job summary

The fourth job load the training data from the feature store (s3://**deep-attribution-bucket**/feature_store/train.parquet), load the attention score from the attention report (s3://**deep-attribution-bucket**/attention_report/attention_score.parquet) join them together on the journey_id field, compute the total and average attention by campaigns and save the results to S3 (s3://**deep-attribution-bucket**/attention_report/campaign_attention.parquet).

### Scripts

There are two scripts associated with this part the:

- [generate_attention_report.py](../deep_attribution/generate_attention_report/generate_attention_report.py): the job script
- [generate_attention_report_exec.py](../deep_attribution/generate_attention_report/generate_attention_report_exec.py): the script to launch the job


### Job description

1. load the campaigns at journey level from the feature store

```bash
 conversion_status | campaign_nm_at_index_0 | campaign_nm_at_index_* | campaign_nm_at_index_n | journey_id
 ___________________________________________________________________________________________________________

         bool      |           str          |           str          |           str          |   bigint  

pyspark.sql.DataFrame
```

2. load the attention scores at journey level from the attention report

```bash
 attention_at_index_0 | attention_at_index_* | attention_at_index_n | journey_id
 _______________________________________________________________________________

         float        |         float        |         float        |   bigint  

pyspark.sql.DataFrame
```

3. unpivot the campaigns at journey level on the journey id

```bash
 campaign_nm | campaign_nm_index_in_journey | journey_id
 _______________________________________________________

      str    |             int              |   bigint  

pyspark.sql.DataFrame
```

4. unpivot the attention scores at journey level on the journey id

```bash
 attention | campaign_nm_index_in_journey | journey_id
 ______________________________________________________

    int    |             int              |   bigint  

pyspark.sql.DataFrame
```

5. create an impression_id field for the unpivoted campaigns

```bash
 campaign_nm | impression_id 
 ____________________________

      str    |    bigint 

pyspark.sql.DataFrame
```

6. create an impression_id field for the unpivoted attentions

```bash
 attention | impression_id 
 __________________________

   float   |    bigint 

pyspark.sql.DataFrame
```

7. join the two DataFrames on the impression_id

```bash
 attention | impression_id | campaign_nm | impression_id 
 _______________________________________________________

   float   |     bigint    |     str     |    bigint

pyspark.sql.DataFrame
```

8. compute the total attention by campaign

```bash
 total_attention | campaign_nm 
 _____________________________

       float    |     str 

pyspark.sql.DataFrame
```

9. compute the average attention by campaign

```bash
 average_attention | campaign_nm 
 _______________________________

        float      |     str 

pyspark.sql.DataFrame
```

10. join the total and average attention on the campaign_nm

```bash
 average_attention | campaign_nm | total_attention
 _________________________________________________

        float      |     str     |       float    

pyspark.sql.DataFrame
```

11. save the results to S3


### Minimal configuration

- intance_type: at least ml.c5.xlarge
- instance_count: at least 2





