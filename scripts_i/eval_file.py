import argparse
import copy
import json
from typing import Dict

from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from utils import stop_response

UNK = "<unk>"
esnli_label_id_dict = {"entailment": 0, "neutral": 1, "contradiction": 2, UNK: 3}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--data_path", type=str, default="esnli", help="dataset")
    parser.add_argument("--e_first", type=int, default=1)
    return parser.parse_args()


def single_word(input):
    return input.strip().split()[0].strip(".").strip("#")


def esnli(output_dict: Dict, opt):
    answer = copy.deepcopy(output_dict["answer"])
    query = copy.deepcopy(output_dict["query"])
    str_label = output_dict["label"]
    if answer.startswith(query):
        answer = answer.replace(query, "")
    answer = stop_response(answer)
    temp = answer.split("####")
    if opt.e_first:
        if len(temp) > 1:
            concise = single_word(temp[1])
        else:
            concise = single_word(temp[0])
    else:
        concise = single_word(temp[0])
    if len(concise) > 40:
        import pdb

        pdb.set_trace()
    print(concise, ">>>", str_label)
    str_pred = UNK
    for cand_pred in esnli_label_id_dict.keys():
        if cand_pred in concise:
            str_pred = cand_pred
            break
    return str_label, str_pred


def gsm8k(output_dict: Dict):
    answer = output_dict["answer"]
    query = output_dict["query"]
    str_label = output_dict["label"]
    if answer.startswith(query):
        answer = answer.replace(query, "")
    answer = stop_response(answer)
    temp = answer.rsplit("#", 1)
    if len(temp) > 1:
        concise = temp[1].strip()
    else:
        concise = temp[0].strip()
    print(concise, ">>>", str_label)
    return str_label, concise


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
            label_list.append(label)
            pred_list.append(pred)
        elif opt.data_path == "gsm8k":
            label, pred = gsm8k(item)
            label_list.append(label)
            pred_list.append(pred)

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
    if opt.data_path == "gsm8k":
        tot = len(label_list)
        right = 0
        for label, pred in zip(label_list, pred_list):
            if label == pred:
                right += 1
        print(f"overall accuracy: {right / tot * 100:.2f}%")


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
