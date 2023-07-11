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


def preprocess(text):
    assert "entailment" in text or "neutral" in text or "contradiction" in text
    processed_text = text.split(".", 1)[1].strip()
    return processed_text


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
            candidate = preprocess(candidate)
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

    if args.expansion > 1:
        args.expansion = 3
    else:
        args.expansion = 1
    return args


if __name__ == "__main__":
    args = parse_args()
    torch.cuda.set_device(args.device_id)
    with open(args.input_file, "r") as f:
        candidates = json.load(f)
    finals = []
    # proto sample's reponses number
    response_num = len(candidates[0][1])
    reward_fn = create_reward_fn()

    for idx in tqdm(range(len(candidates) // args.expansion)):
        prompts = list()
        candidate = list()
        gt_explaination = candidates[idx * args.expansion][2]
        label = candidates[idx * args.expansion][3]
        for inner in range(args.expansion):
            candidate.append(candidates[idx * args.expansion + inner][1])
            prompts.append(candidates[idx * args.expansion + inner][0])
        candidate = sum(candidate, [])
        results_scores = reward_fn(candidate, gt_explaination)
        human_idx = prompts[-1].rfind("Human:")
        question_part = prompts[-1][human_idx:]

        assert args.expansion == len(prompts)
        assert len(results_scores) == args.expansion * response_num

        # add real human response into candidates
        real_human = "It's {}. Because {}.".format(label, gt_explaination)
        candidate = [real_human] + candidate
        results_scores = [1.0] + results_scores

        finals.append(
            dict(
                prompt=prompts,
                question=question_part,
                response=candidate,
                explaination=gt_explaination,
                scores=results_scores,
                label=label,
            )
        )

    with open(args.output_file, "w") as f:
        json.dump(finals, f, indent=2)
