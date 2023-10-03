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
    parser.add_argument(
        "--data_path", type=str, default="esnli", choices=["esnli"], help="dataset"
    )
    parser.add_argument("--e_first", type=int, default=1)
    parser.add_argument("--truncate", type=int, default=None)
    return parser.parse_args()


def esnli(output_dict: Dict, opt):
    answer = copy.deepcopy(output_dict["answer"])
    query = copy.deepcopy(output_dict["query"])
    str_label = output_dict["label"]
    answer = answer.replace(query, "")
    extracted_resp = answer
    if extracted_resp is None:
        return str_label, UNK

    str_pred = UNK
    for label in esnli_label_id_dict.keys():
        if label.lower() in extracted_resp.lower():
            str_pred = label
            break

    return str_label, str_pred


def main(opt):
    with open(opt.input_file, "r") as f:
        data = json.load(f)

    tot = len(data)
    right = 0
    pred_list = list()
    label_list = list()
    label_names = list(esnli_label_id_dict.keys())

    for idx, item in enumerate(tqdm(data)):
        if opt.data_path == "esnli":
            label, pred = esnli(item, opt)
            label_list.append(label.lower())
            pred_list.append(pred.lower())
        if opt.truncate is not None and idx + 1 >= opt.truncate:
            break

    if opt.data_path == "esnli":
        # 1. overall calculate accuracy with confusion matrix
        # 2. calculate accuracy and recall for each class
        cm = confusion_matrix(label_list, pred_list, labels=label_names)
        acc = sum(cm[i][i] for i in range(len(cm))) / sum(
            sum(cm[i]) for i in range(len(cm))
        )
        acc_dict = dict()
        rec_dict = dict()
        for i, label_name in enumerate(label_names):
            deno = sum(cm[j][i] for j in range(len(cm)))
            acc_dict[label_name] = cm[i][i] / (deno if deno != 0 else 1)
        for i, label_name in enumerate(label_names):
            deno = sum(cm[i])
            rec_dict[label_name] = cm[i][i] / (deno if deno != 0 else 1)

        print(label_names)
        print("confusion matrix:")
        print(cm)
        print(f"overall accuracy: {acc * 100:.2f}%")
        print(acc_dict)
        print(rec_dict)


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
