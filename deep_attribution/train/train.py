import os

import subprocess
import sys

from typing import Dict

from json import loads
from yaml import load

import argparse

from deep_attribution.model.journey_based_deepnn import JourneyBasedDeepNN
from deep_attribution.train.batch_loader import BatchLoader
from deep_attribution.train.utilities import write_json_to_s3, get_nb_campaigns_from_s3

TARGET_NM = "conversion_status"

def main():

    install_dependencies()

    config = load_config()

    args = parse_args()

    nb_campaigns = get_nb_campaigns_from_s3(config["bucket_nm"])

    hp_nm_to_val = get_hp_nm_to_val_from(args)
    model = get_model(hp_nm_to_val, nb_campaigns, config["journey_max_len"])

    set_nm_to_batch_loader = {}

    for set_nm in ["train", "test", "val"]:

        set_nm_to_batch_loader[set_nm] = BatchLoader(
            TARGET_NM,
            set_nm,
            nb_campaigns,
            config["journey_max_len"],
            args.sets_parent_dir_path,
            oversample = True if set_nm == "train" else False
        )

    model.batch_fit(
        set_nm_to_batch_loader["train"],
        set_nm_to_batch_loader["test"]
    )

    set_nm_to_score = {}

    for set_nm, set_loader in set_nm_to_batch_loader.items():
        set_nm_to_score[set_nm] = model.batch_evaluate(set_loader)
        print("auc - {}: {}".format(set_nm, set_nm_to_score[set_nm][2]))
    
    write_json_to_s3(set_nm_to_score, config["bucket_nm"], "score.json")

    model.save_attention_model(args.model_dir)


def install_dependencies():

    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyarrow"])

def load_config():

    with open("config/config.yaml", "rb") as f:
        config = load(f)

    return config


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--sets_parent_dir_path',
        type=str,
        default=os.environ.get("SM_CHANNEL_SETS_PARENT_DIR_PATH"))
    parser.add_argument('--model_dir', type=str)

    parser.add_argument("--epochs", type=int)
    parser.add_argument("--n_hidden_units_embedding", type=int)
    parser.add_argument("--n_hidden_units_lstm", type=int)
    parser.add_argument("--dropout_lstm", type=float)
    parser.add_argument("--recurrent_dropout_lstm", type=float)
    parser.add_argument("--learning_rate", type=float)

    return parser.parse_args()


def get_model(
    hp_nm_to_val: Dict,
    nb_campaigns:int,
    journey_max_len:int
    ) -> JourneyBasedDeepNN:
    
    return JourneyBasedDeepNN(
        **hp_nm_to_val,
        n_cmpgns=nb_campaigns,
        max_nb_eng_per_journey=journey_max_len)


def get_hp_nm_to_val_from(args: argparse.Namespace) -> Dict:

    return {
        "epochs":args.epochs,
        "n_hidden_units_embedding":args.n_hidden_units_embedding,
        "n_hidden_units_lstm":args.n_hidden_units_lstm,
        "dropout_lstm":args.dropout_lstm,
        "recurrent_dropout_lstm":args.recurrent_dropout_lstm,
        "learning_rate":args.learning_rate
    }


if __name__ == "__main__":
    main()