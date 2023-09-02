import json
import os
import random
import re
import sys

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm
import utils

# --------------------------------------- #
# 08-26 filter and rank the responses
# --------------------------------------- #


# --- misc --- #
def __create_sim_reward_fn():
    sbert = SentenceTransformer("all-MiniLM-L6-v2")
    sbert.to(torch.cuda.current_device())

    def reward_fn(candidates, reference):
        if isinstance(candidates, list):
            results = list()
            for cand in candidates:
                sim = reward_fn(cand, reference)
                results.append(sim)
            return results
        else:
            cand_embed = sbert.encode(candidates, convert_to_tensor=True).to(sbert.device)
            ref_embed = sbert.encode(reference, convert_to_tensor=True).to(
                sbert.device
            )
            # calculate cosine similarity
            return torch.cosine_similarity(cand_embed, ref_embed, dim=-1).mean().item()
    return reward_fn


# --- rank the responses --- #
def human_rank(responses, esnli_data_dict):
    scores = [1.0] * len(responses)
    scores[0] = 2.0
    return scores


def label_rank(responses, esnli_data_dict):
    scores = [2.0] + [1.0] * (len(responses) - 1)
    gt = esnli_data_dict["label"].lower()
    for index in range(1, len(responses)):
        resp = responses[index]
        label = utils.parse_response(resp)["label"].lower()
        if gt == label:
            scores[index] = 2.0
    return scores


def group_rank(responses, esnli_data_dict):
    scores = [3.0] + [1.0] * (len(responses) - 1)
    gt = esnli_data_dict["label"].lower()
    for index in range(1, len(responses)):
        resp = responses[index]
        label = utils.parse_response(resp)["label"].lower()
        if gt == label:
            scores[index] = 2.0
    return scores


def similarity_rank(responses, esnli_data_dict):
    human_response = responses[0]
    scores = SIM_REWARD_FN(responses, human_response)
    return scores


# --- extract the valid responses --- #
def fetch_valid_response(data_dict):
    "fetch the human response and the valid candidates"
    human_response = utils.make_response(data_dict["data_dict"], captalize=True)
    resp_list = [human_response]
    for lm_resp in data_dict["responses"]:
        extracted = utils.extract_first_response(lm_resp)
        if extracted:
            resp_list.append(extracted)
    return resp_list


def filter_and_rank(file, func, resp_num_thres=None):
    log_dict = {s: 0 for s in range(0, 10)}
    log_dict["strip"] = list()
    log_dict["scores"] = list()
    sample_list = [1000, 2000, 5000, 10000, 20000, 50000]

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
        responses = fetch_valid_response(data_dict)
        scores = func(responses, data_dict["data_dict"])

        new_data_dict["data_dict"] = data_dict["data_dict"].copy()
        new_data_dict["scores"] = scores
        new_data_dict["responses"] = [utils.wrap_response(resp) for resp in responses]

        if len(responses) < resp_num_thres:
            log_dict["strip"].append(data_dict["data_dict"]["index"])
            continue
        if scores is None:
            log_dict["scores"].append(data_dict["data_dict"]["index"])
            continue

        log_dict[len(responses)] += 1
        new_data_list.append(new_data_dict)

        if step % 1000 == 0:
            save_data_list(new_data_list, "rank_all.json")

    # for sample_num in sample_list:
    #     dir = os.path.join(os.path.dirname(file), func.__name__)
    #     output_file = os.path.join(dir, f"rank_{sample_num//1000}k.json")
    #     os.makedirs(os.path.dirname(output_file), exist_ok=True)
    #     with open(output_file, "w") as f:
    #         json.dump(new_data_list[:sample_num], f, indent=4)
    for sample_num in sample_list:
        save_data_list(new_data_list[:sample_num], f"rank_{sample_num//1000}k.json")

    dir = os.path.join(os.path.dirname(file), func.__name__)
    with open(os.path.join(dir, "log"), "w") as f:
        # redirect stdout to f
        origin = sys.stdout
        sys.stdout = f
        print(f"stripped out {len(log_dict['strip'])} responses: {log_dict['strip']}")
        print(f"invalid scores: {log_dict['scores']}")
        del log_dict["strip"]
        del log_dict["scores"]
        for k, v in log_dict.items():
            if v and isinstance(k, int):
                print(f"==== responses {k} === {v} / {sum(list(log_dict.values()))}")
        # restore stdout
        sys.stdout = origin


if __name__ == "__main__":
    SIM_REWARD_FN = __create_sim_reward_fn()
    file = sys.argv[1]  # "/workspace/RRescue/data_generation/output/mix/raw-mixed-39k.json"
    for func in [human_rank, label_rank, group_rank, similarity_rank]:
        filter_and_rank(file, func, 5)
