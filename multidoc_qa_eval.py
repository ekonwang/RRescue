import argparse
import copy
import json
import os
import sys
from typing import Dict

from sklearn.metrics import confusion_matrix
from tqdm import tqdm

sys.path.append(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_generation")
)
import utils

UNK = "<unk>"
esnli_label_id_dict = {"entailment": 0, "neutral": 1, "contradiction": 2, UNK: 3}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--truncate", type=int, default=None)
    return parser.parse_args()


def main(opt):
    with open(opt.input_file, "r") as f:
        data = json.load(f)

    tot = len(data)
    right = 0

    for idx, item in enumerate(tqdm(data)):
        ans = item["answer"]
        ground_truths = item["labels"]
        has_right_answer = utils.multidoc_qa_eval(ans, ground_truths)
        right += has_right_answer

    print(f"Accuracy: {right / tot}")


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
