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
    return parser.parse_args()


def main():
    global LOGFILE, VIOLATION_LIST
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    with open(args.prompt_file, "r") as f:
        prompt_template = json.load(f)[0]

    # ---------- load data -------------- #
    with open(args.input_data, "r") as f:
        all_data_list = json.load(f)[: args.sample_size]
    tokens_list = list()

    for data_dict in all_data_list:
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
            res_input_ids = tokenize_fn(
                resp + tokenizer.eos_token,
                tokenizer,
            )
            tokens_list.append(res_input_ids.shape[0])
            print(resp, res_input_ids.shape[0])

    # --- data analysis --- #
    length_mean = np.array(tokens_list).mean()
    length_std = np.array(tokens_list).std()

    if args.human_response:
        tag = "human_response"
    else:
        tag = os.path.basename(args.input_data).rsplit(".", 1)[0]
    # plot frequency distribution of tokens length
    sns.histplot(tokens_list, bins=100, kde=True)
    plt.title(f"{tag}\nmean = {length_mean:.2f} std = {length_std:.2f}")
    plt.savefig(
        os.path.join(args.output_dir, f"tokens_length_distribution_{tag}.png"), dpi=300
    )


if __name__ == "__main__":
    main()
