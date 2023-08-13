import argparse
import json

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Parameters")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    file = f"{args.output}/scored_beam4_0.json"
    labels = ["entailment", "neutral", "contradiction"]
    label_score_dict = {
        "entailment": list(),
        "neutral": list(),
        "contradiction": list(),
    }

    with open(file, "r") as f:
        samples = json.load(f)
    tot = len(samples)
    for idx, sample in enumerate(tqdm(samples)):
        temp_sd = {"entailment": 0, "neutral": 0, "contradiction": 0}
        responses = sample["response"]
        label = sample["label"]
        scores = sample["scores"][1:]
        responses = responses[1:]
        predicted_label_formax = None

        assert len(scores) == len(responses)

        for response, score in zip(responses, scores):
            for cand_label in labels:
                if cand_label in response.split(".", 1)[0].lower():
                    temp_sd[cand_label] = max(temp_sd[cand_label], score)

        max_score = 0
        for candidate in ["entailment", "neutral", "contradiction"]:
            if temp_sd[candidate] > max_score:
                max_score = temp_sd[candidate]
                predicted_label_formax = candidate

        for label in labels:
            label_score_dict[label].append(temp_sd[label])

    for label in label_score_dict.keys():
        label_score_dict[label] = sum(label_score_dict[label]) / len(
            label_score_dict[label]
        )
    print(label_score_dict)
