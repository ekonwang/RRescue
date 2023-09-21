import json
import os
import random
import re
import sys
import time

from datasets import load_dataset
import openai
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm
import utils

# --------------------------------------- #
# 08-26 filter and rank the responses
# --------------------------------------- #

# --- rank the responses --- #
def label_rank(responses, multidoc_qa_data_dict):
    scores = [1.0] * len(responses)
    answers = multidoc_qa_data_dict["answers"]
    for index in range(len(responses)):
        resp = responses[index]
        if utils.multidoc_qa_eval(resp, answers):
            scores[index] = 2.0
    return scores


def filter_and_rank(file, func, resp_num_thres=None):
    log_dict = {s: 0 for s in range(0, 10)}
    sample_list = [1000]
    mean_scores = list()
    all_one = 0
    all_two = 0

    with open(file, "r") as f:
        data_list = json.load(f)

    def save_data_list(data_list, file_name):
        dir = os.path.join(os.path.dirname(file), func.__name__)
        output_file = os.path.join(dir, file_name)
        os.makedirs(dir, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(data_list, f, indent=4)

    new_data_list = list()
    for step, data_dict in enumerate(tqdm(data_list)):
        new_data_dict = dict()
        responses = data_dict["responses"]
        scores = func(responses, data_dict["data_dict"])

        new_data_dict["data_dict"] = data_dict["data_dict"].copy()
        new_data_dict["scores"] = scores
        new_data_dict["responses"] = responses

        if step % 1000 == 0:
            save_data_list(new_data_list, "rank_all.json")

        new_data_list.append(new_data_dict)
        mean_scores.append(sum(scores) / len(scores))
        if (sum(scores) / len(scores)) == 1.0:
            all_one += 1
        elif (sum(scores) / len(scores)) == 2.0:
            all_two += 1

    for sample_num in sample_list:
        if sample_num <= len(new_data_list):
            save_data_list(new_data_list[:sample_num], f"rank_{sample_num//1000}k.json")

    dir = os.path.join(os.path.dirname(file), func.__name__)
    with open(os.path.join(dir, "log"), "w") as f:
        # redirect stdout to f
        origin = sys.stdout
        sys.stdout = f
        for k, v in log_dict.items():
            if v and isinstance(k, int):
                print(f"==== responses {k} === {v} / {sum(list(log_dict.values()))}")
        print(f"mean score: {sum(mean_scores) / len(mean_scores)}")
        print(f"all one: {all_one}")
        print(f"all two: {all_two}")
        # restore stdout
        sys.stdout = origin


if __name__ == "__main__":
    file = sys.argv[
        1
    ]  # "/workspace/RRescue/data_generation/output/mix/raw-mixed-39k.json"
    for func in [label_rank]:
        # for func in [gpt_rank]:
        filter_and_rank(file, func, 5)
