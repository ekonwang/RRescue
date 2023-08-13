import argparse
import json
import sys

from datasets import load_dataset
from tqdm import tqdm
import utils


def parse_args():
    parser = argparse.ArgumentParser(description="Parameters")
    parser.add_argument("--input_file", type=str, default="", required=True)
    parser.add_argument("--output_file", type=str, default="", required=True)
    parser.add_argument("--dataset", type=str, default="Dahoas/rm-static")
    parser.add_argument("--diverse_beam", type=int, required=True)
    parser.add_argument("--expansion", type=int, default=1, required=True)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    print(f"loading dataset {args.dataset}")
    if args.dataset == "gsm8k":
        dataset = load_dataset(args.dataset, "main", split="train")
    else:
        dataset = load_dataset(args.dataset, split="train")
    print(f"dataset size: {len(dataset)}")
    dataset_name = args.dataset.split("/")[-1]
    with open(args.input_file, "r") as f:
        samples = json.load(f)

    print("==" * 10)
    print(samples[-1])
    print(dataset[-1])

    buffer = []
    now_samples = dict()
    now_idx = 0
    for idx in tqdm(range(len(samples))):
        sample = samples[idx]  # a list of (prompt, response, index)
        for prompt, response, index in sample:
            if index != now_idx:
                assert (
                    len(now_samples["responses"]) == args.diverse_beam * args.expansion
                )
                buffer.append(now_samples)
                now_samples = dict()
                now_idx = index
            if now_samples.get("prompt", "") == "":
                # ----- dataset specific ----- #
                if args.dataset == "esnli":
                    explaination = dataset[index]["explanation_1"]
                    label = utils.esnli_label_map[dataset[index]["label"]]
                    inputs = dict(
                        premise=dataset[index]["premise"],
                        hypothesis=dataset[index]["hypothesis"],
                    )
                elif args.dataset == "gsm8k":
                    explaination = dataset[index]["answer"].split("####")[0].strip()
                    label = dataset[index]["answer"].split("####")[1].strip()
                    inputs = dict(question=dataset[index]["question"])
                else:
                    raise ValueError(f"Invalid dataset {args.dataset}")
                gt_response = utils.make_response(
                    dict(
                        explaination=explaination,
                        label=label,
                        first="explaination",
                        format="special",
                    )
                )
                now_samples["explaination"] = explaination
                now_samples["label"] = label
                now_samples["gt_response"] = gt_response
                now_samples["inputs"] = inputs

                # ----- general ----- #
                now_samples["prompt"] = prompt
                now_samples["explainations"] = []
                now_samples["raw_responses"] = []
                now_samples["responses"] = []
            resp = utils.strip_process(response)
            explaination = resp.split("#", 1)[0].strip()
            now_samples["responses"].append(resp)
            now_samples["raw_responses"].append(response)
            now_samples["explainations"].append(explaination)

    if len(now_samples) > 0:
        assert len(now_samples["responses"]) == args.diverse_beam * args.expansion
        buffer.append(now_samples)
    with open(args.output_file, "w") as f:
        json.dump(buffer, f, indent=4)
