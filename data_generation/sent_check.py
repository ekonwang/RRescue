import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser(description="Parameters")
    parser.add_argument("--diverse_beam", type=int, default=0, required=True)
    parser.add_argument("--expansion", type=int, default=1, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    file = f"{args.output}/scored_beam4_0.json"

    if args.expansion > 1:
        with open(file, "r") as f:
            samples = json.load(f)
        hit = 0
        tot = len(samples)
        for idx, sample in enumerate(samples):
            labels = ["entailment", "neutral", "contradiction"]
            score_dict = {"entailment": 0, "neutral": 0, "contradiction": 0}
            responses = sample["response"]
            label = sample["label"]
            scores = sample["scores"][1:]
            responses = responses[1:]
            predicted_label_formax = None

            assert len(scores) == len(responses)

            for response, score in zip(responses, scores):
                for cand_label in labels:
                    if cand_label in response.split(".", 1)[0].lower():
                        score_dict[cand_label] = max(score_dict[cand_label], score)

            max_score = 0
            for candidate in ["entailment", "neutral", "contradiction"]:
                if score_dict[candidate] > max_score:
                    max_score = score_dict[candidate]
                    predicted_label_formax = candidate
            if predicted_label_formax != label:
                print(idx, predicted_label_formax, label)
            else:
                hit += 1
        print(score_dict)
        print("*" * 10, f"{hit/tot*100:.2f}%({hit}/{tot})")
