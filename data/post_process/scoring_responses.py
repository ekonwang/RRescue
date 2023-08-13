#### The code is modified from trlX
import argparse
import json
import math
import os
import sys

from sentence_transformers import SentenceTransformer
import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import utils


def create_reward_fn():
    sbert = SentenceTransformer("all-MiniLM-L6-v2")
    sbert.to(torch.cuda.current_device())

    def reward_fn(candidate, gt_explaination):
        if isinstance(candidate, list):
            results = list()
            for cand in candidate:
                sim = reward_fn(cand, gt_explaination)
                results.append(sim)
            return results
        else:
            embed = sbert.encode(candidate, convert_to_tensor=True).to(sbert.device)
            gt_embed = sbert.encode(gt_explaination, convert_to_tensor=True).to(
                sbert.device
            )
            # calculate cosine similarity
            return torch.cosine_similarity(embed, gt_embed, dim=-1).mean().item()

    return reward_fn


def parse_args():
    parser = argparse.ArgumentParser(description="Parameters")
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--expansion", type=int, required=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    torch.cuda.set_device(args.device_id)
    with open(args.input_file, "r") as f:
        candidates = json.load(f)
    buffer = []
    # proto sample's reponses number
    response_num = len(candidates[0]["responses"])
    reward_fn = create_reward_fn()

    for idx in tqdm(range(len(candidates))):
        label = candidates[idx]["label"]
        inputs = candidates[idx]["inputs"]
        gt_explaination = candidates[idx]["explaination"]
        gt_response = candidates[idx]["gt_response"]
        reponses = candidates[idx]["responses"]
        explainations = candidates[idx]["explainations"]
        explainations = [gt_explaination] + explainations
        scores = reward_fn(explainations, gt_explaination)

        # rank the responses with scores descending order
        scores, explainations = zip(*sorted(zip(scores, explainations), reverse=True))
        scores = list(scores)
        explainations = list(explainations)
        synthesized_responses = [
            utils.make_response(
                dict(
                    explaination=exp,
                    label=label,
                    first="explaination",
                    format="special",
                )
            )
            for exp in explainations
        ]

        buffer.append(
            dict(
                inputs=inputs,
                explainations=explainations,
                synthesized_responses=synthesized_responses,
                scores=scores,
            )
        )

    with open(args.output_file, "w") as f:
        json.dump(buffer, f, indent=2)
