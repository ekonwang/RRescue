import argparse
import json
from typing import Dict

from tqdm import tqdm
from utils import stop_response


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--data_path", type=str, default="esnli", help="dataset")
    return parser.parse_args()


def esnli(output_dict: Dict) -> bool:
    answer = output_dict["answer"]
    query = output_dict["query"]
    str_label = output_dict["label"]
    if answer.startswith(query):
        answer = answer.replace(query, "")
    answer = stop_response(answer)
    concise = answer.split(".")[0].strip()
    print(concise, ">>>", str_label)
    if str_label in concise:
        return True
    else:
        return False


def main(opt):
    with open(opt.input_file, "r") as f:
        data = json.load(f)

    tot = len(data)
    right = 0
    for item in tqdm(data):
        if opt.data_path == "esnli":
            if esnli(item):
                right += 1
    print(f"Accuracy: {right / tot * 100:.2f}% ({right}/{tot})")


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
