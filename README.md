# DeepAttribution

![GitHub repo size](https://img.shields.io/github/repo-size/leopoldavezac/DeepAttribution)
![GitHub contributors](https://img.shields.io/github/contributors/leopoldavezac/DeepAttribution)
![GitHub stars](https://img.shields.io/github/stars/leopoldavezac/DeepAttribution?style=social)
![GitHub forks](https://img.shields.io/github/forks/leopoldavezac/DeepAttribution?style=social)

DeepAttribution is a AWS (sagemaker) ML pipeline that allows marketing data scientist and ML engineer to compute multi touch attribution results using the state of the art technique without effort.

## Prerequistes

* A **big** dataset (> 1GB) at impression level (**impressions dataset**) as parquet file with the following schema:

    * uid: the user/client unique identifier
    * timestamp: unix timestamp
    * campaign: the campaign name
    * conversion: whether or not a conversion happened after the impression

```bash
                    uid | timestamp | campaign | conversion 
                    _______________________________________

                    int |    int    |    str   |    bool 
```

* S3 bucket (**deep-attribution bucket**) with the following folder hierarchy

```bash
    .
    ├── raw                         # contains impressions dataset
    ├── feature_store               # empty before pipeline execution
    ├── feature_store_preprocessed  # empty before pipeline execution
    ├── model                       # empty before pipeline execution
    └── attention_report            # empty before pipeline execution

```

* Sagemaker notebook instance with git repo set to this repo (**deep-attribution bucket**).
* AWS role with sagemaker execution permission and read/write permissions on the s3 bucket mentionned above.
* Basic knowledge of LSTM and familiarity with the multi touch attribution model presented in [Deep Neural Net with Attention for Multi-channel Multi-touch Attribution](https://arxiv.org/pdf/1809.02230.pdf)
* Understanding of what is a [journey](./docs/journey.md), the [model](./docs/model.md) and the [pipeline](./docs/pipeline.md)


## Using DeepAttribution

1. Define the journey maximum length. Please refer to this [doc](./docs/journey.md) to define it.
2. Update the config file (config.yaml) with the desired instance type and count, bucket name and the journey maximum length.
3. In the **deep-attribution instance** open the pipeline execution notebook (deep_attribution/pipeline_exec.ipynb)
4. Run all the cells 
5. Get the attribution results in the **deep-attribution bucket** (attention_report/campaign_attention.parquet)

## Contact

If you want to contact me you can reach me at <leopoldavezac@gmail.com>.

## License

This project uses the following license: [MIT](./LICENSE).





