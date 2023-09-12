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


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    aa("--output_dir", type=str, default=None, help="output directory")
    aa("--model_name_or_path", type=str, default=None, help="model name or path")
    aa("--cache_dir", type=str, default=None, help="cache directory")
    aa("--model_max_length", type=int, default=512, help="model max length")
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

    # ---------- load data -------------- #
    with open(args.input_data, "r") as f:
        all_data_list = json.load(f)[: args.sample_size]

    acc = 0
    tot = 0
    for data_dict in tqdm(all_data_list):
        label = data_dict["data_dict"]["label"].lower()
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
            pred = utils.safe_parse_response(resp)["label"].lower()
            if pred == label:
                acc += 1
            tot += 1

    # --- data analysis --- #
    print(f"Accuracy: {acc / tot * 100:.2f}%")


if __name__ == "__main__":
    main()
