
from typing import Dict

import argparse

from model.journey_based_deepnn import JourneyBasedDeepNN
from batch_loader import BatchLoader
from deep_attribution.s3_utilities import write_json_to_s3

JOURNEY_MAX_LENGTH = 10
TARGET_NM = "conversion"
PREDICTOR_NMS = ["eng_at_index_"+str(i) for i in range(JOURNEY_MAX_LENGTH)]

BUCKET_NM = "deep-attribution"
CAMPAIGN_NM_TO_INDEX_PATH = "feature_store/campaign_nm_to_index.json"



def run():

    args, _ = parse_args()


    hp_nm_to_val = get_hp_nm_to_val_from(args)
    model = get_model(hp_nm_to_val)

    train_batch_loader = BatchLoader(TARGET_NM, "train")
    test_batch_loader = BatchLoader(TARGET_NM, "test")
    val_batch_loader = BatchLoader(TARGET_NM, "val")

    model.batch_fit(
        train_batch_loader,
        test_batch_loader
    )

    set_loaders = [train_batch_loader, test_batch_loader, val_batch_loader]
    set_nms = ["train", "test", "val"]

    set_nm_to_score = {}

    for set_nm, set_loader in zip(set_nms, set_loaders):
        set_nm_to_score[set_nm] = model.batch_evaluate(set_loader).tolist()
        print("auc - {}: {}".format(set_nm, set_nm_to_score[set_nm][2]))
    
    write_json_to_s3(set_nm_to_score, BUCKET_NM, "score.json")

    model.save_attention_model(args.model_dir)


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_dir', type=str)

    parser.add_argument("--epochs", type=int)

    parser.add_argument("--journey_max_length", type=int)
    parser.add_argument("--n_campaigns", type=int)
    
    parser.add_argument("--n_hidden_units_embedding", type=int)
    parser.add_argument("--n_hidden_units_lstm", type=int)

    parser.add_argument("--dropout_lstm", type=float)
    parser.add_argument("--recurrent_dropout_lstm", type=float)
    
    parser.add_argument("--learning_rate", type=float)

    return parser.parse_known_args()


def get_model(hp_nm_to_val: Dict) -> JourneyBasedDeepNN:
    
    return JourneyBasedDeepNN(hp_nm_to_val)


def get_hp_nm_to_val_from(args):

    return {
        "epochs":args.epochs,
        "n_hidden_units_embedding":args.n_hidden_units_embedding,
        "n_hidden_units_lstm":args.n_hidden_units_lstm,
        "dropout_lstm":args.dropout_lstm,
        "recurrent_dropout_lstm":args.reccurent_dropout_lstm,
        "learning_rate":args.learning_rate,
        "beta_1":args.beta_1,
        "beta_2":args.beta_2,
        "epsilon":args.epsilon
    }


if __name__ == "__main__":
    run()