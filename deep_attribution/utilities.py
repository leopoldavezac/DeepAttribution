from typing import Dict, List

def format_config_as_job_args(config: Dict) -> List[str]:

    job_args = []
    for arg_nm, arg_val in config.items():
        job_args += ["--"+arg_nm, arg_val]

    return job_args
