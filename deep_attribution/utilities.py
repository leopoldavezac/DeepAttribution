from typing import Dict, List

def format_config_as_job_args(config: Dict) -> List[str]:

    config_keys_to_exclude = ["feature_engineering", "preprocessing", "training", "attention_report_generation"]

    job_args = []
    for arg_nm, arg_val in config.items():
        if arg_nm not in config_keys_to_exclude:
            job_args += ["--"+arg_nm, arg_val]

    return job_args
