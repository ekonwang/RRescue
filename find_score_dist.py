import argparse
from dataclasses import dataclass, field
import io
import json
import logging
import os
import random
import sys
from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
import transformers
from transformers import Trainer

from train_utils import (
    safe_save_model_for_hf_trainer,
    smart_tokenizer_and_embedding_resize,
    tokenize_fn,
)

sys.path.append(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_generation")
)
import utils

IGNORE_INDEX = -100


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    aa("--output_dir", type=str, default=None, help="output directory")
    aa("--model_name_or_path", type=str, default=None, help="model name or path")
    aa("--cache_dir", type=str, default=None, help="cache directory")
    aa("--model_max_length", type=int, default=512, help="model max length")
    aa("--prompt_file", type=str, default=None, help="prompt file")
    aa("--input_data", type=str, default=None, help="input data")
    aa("--sample_size", type=int, default=1000, help="sample size")
    aa("--human_response", action="store_true", help="human response")
    aa("--length_penalty", type=float, default=1.0, help="length penalty")
    return parser.parse_args()


def main():
    global LOGFILE, VIOLATION_LIST
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    model = (
        transformers.AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
            torch_dtype=torch.float32,
        )
        .cuda()
        .eval()
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    # ---------- load data -------------- #
    with open(args.prompt_file, "r") as f:
        prompt_template = json.load(f)[0]
    with open(args.input_data, "r") as f:
        all_data_list = json.load(f)[: args.sample_size]
    value_list = list()

    # --------- helper function ---------- #
    def gather_logits_labels(logits, labels):
        labels = labels.clone()
        mask = (labels != -100).float()
        new_logits = logits.clone()  # Create a copy to avoid in-place modification
        labels[labels == -100] = 0
        output = torch.gather(new_logits, dim=-1, index=labels.unsqueeze(-1)).squeeze(
            -1
        )
        output = output * mask  # B * L
        return output

    def get_score(logit_label, labels):
        mask = (labels != -100).float()
        length = mask.sum(-1)
        scores = logit_label.sum(-1) / (length**args.length_penalty)
        return scores

    def calculate_score(data_dict, resp):
        query = prompt_template.format_map(data_dict)
        query_input_ids = tokenize_fn(query, tokenizer)
        query_target = torch.LongTensor([IGNORE_INDEX] * (query_input_ids.shape[0] - 1))
        dummy_target = torch.LongTensor([IGNORE_INDEX])
        res_input_ids = tokenize_fn(
            resp + tokenizer.eos_token,
            tokenizer,
            max_len=tokenizer.model_max_length - query_input_ids.shape[0],
        )  # eos here
        input_ids = (
            torch.cat((query_input_ids, res_input_ids), dim=0).unsqueeze(0).cuda()
        )
        labels = (
            torch.cat((query_target, res_input_ids, dummy_target), dim=0)
            .unsqueeze(0)
            .cuda()
        )
        attention_mask = torch.ones_like(input_ids).cuda()

        with torch.no_grad():
            out_logits = model(input_ids.cuda(), attention_mask.cuda()).logits
            out_logits = F.log_softmax(out_logits, dim=-1)
            logit_label = gather_logits_labels(out_logits, labels)
            scores = get_score(logit_label, labels)
        return scores.squeeze().item()

    # --------- calculate the score ---------- #
    for data_dict in tqdm(all_data_list):
        if args.human_response:
            human_response = utils.make_response(data_dict["data_dict"], captalize=True)
            resps = [utils.wrap_response(human_response)]
        else:
            try:
                resps = data_dict["responses"]
            except KeyError:
                resps = data_dict["response"]
                if isinstance(resps, str):
                    resps = [resps]
            resps = [
                utils.wrap_response(utils.extract_first_response(resp))
                for resp in resps
                if utils.extract_first_response(resp) is not None
            ]

        for resp in resps:
            score = calculate_score(data_dict["data_dict"], resp)
            value_list.append(score)

    # --------- data analysis --------- #
    length_mean = np.array(value_list).mean()
    length_std = np.array(value_list).std()

    if args.human_response:
        tag = "human_response"
    else:
        tag = os.path.basename(args.input_data).rsplit(".", 1)[0]

    file = os.path.join(
        args.output_dir,
        f"log_prob_scores_{tag}_length_penalty_{args.length_penalty:.2f}.json",
    )
    os.makedirs(os.path.dirname(file), exist_ok=True)
    with open(file, "w") as f:
        json.dump(value_list, f, indent=2)


if __name__ == "__main__":
    main()
