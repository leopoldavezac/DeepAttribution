import os

import subprocess
import sys

from typing import Dict

from yaml import load

import argparse

from pandas import read_parquet

from deep_attribution.model.journey_based_deepnn import JourneyBasedDeepNN
from deep_attribution.train.utilities import write_json_to_s3, get_nb_campaigns_from_s3, reshape_X_with_one_hot_along_z
from deep_attribution.train.oversampling import oversample

TARGET_NM = "conversion_status"

def main():

    install_dependencies()

    config = load_config()

    args = parse_args()

    nb_campaigns = get_nb_campaigns_from_s3(config["bucket_nm"])

    hp_nm_to_val = get_hp_nm_to_val_from(args)
    model = get_model(hp_nm_to_val, nb_campaigns, config["journey_max_len"])

    set_nm_to_set_vals = {}

    for set_nm in ["train", "test", "val"]:

        df = read_parquet(args.sets_parent_dir_path+"/%s.parquet" % set_nm)
        df.iloc[:,1:] = df.iloc[:,1:].astype("bool")

        y = df.loc[:,"conversion_status"].values
        X = df.drop(columns=["conversion_status", "journey_id"]).values

        if set_nm == "train":
            X, y = oversample(X, y)

        X = reshape_X_with_one_hot_along_z(X, nb_campaigns, config["journey_max_len"])

        set_nm_to_set_vals[set_nm] = {"X":X, "y":y}
        del X, y


    model.fit(
        set_nm_to_set_vals["train"]["X"],
        set_nm_to_set_vals["train"]["y"]
    )

    set_nm_to_score = {}

    for set_nm, set_vals in set_nm_to_set_vals.items():
        set_nm_to_score[set_nm] = model.evaluate(set_vals["X"], set_vals["y"])
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