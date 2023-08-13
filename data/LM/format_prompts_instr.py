import argparse
import json
import os

import datasets

DATA = "/".join(os.path.dirname(os.path.abspath(__file__)).split("/")[:-1])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="esnli")
    parser.add_argument("--examples", type=int, default=3)
    parser.add_argument("--e_first", type=int, default=1)
    return parser.parse_args()


def esnli_label_map(label):
    if label == 0:
        return "entailment"
    elif label == 1:
        return "neutral"
    elif label == 2:
        return "contradiction"
    else:
        raise ValueError(f"Invalid label {label}")


def main(args):
    data_examples_file = os.path.join(DATA, "configs", f"{args.input}_examples.json")
    # ------ load examples ------- #
    if not os.path.exists(data_examples_file):
        data = dict()
        if args.input == "esnli":
            data["esnli"] = dict()
            esnli_dev = datasets.load_dataset("esnli", split="validation")
            for la in ["entailment", "neutral", "contradiction"]:
                data["esnli"][la] = list()
                for i in range(len(esnli_dev)):
                    example = esnli_dev[i]
                    if la == esnli_label_map(example["label"]):
                        example_dict = dict(
                            premise=example["premise"],
                            hypothesis=example["hypothesis"],
                            label=la,
                            idx=i,
                            explanation=example["explanation_1"],
                        )
                        data["esnli"][la].append(example_dict)
                    if len(data["esnli"][la]) >= args.examples:
                        break
        if args.input == "gsm8k":
            data["gsm8k"] = list()
            gsm8k_dev = datasets.load_dataset("gsm8k", "main", split="test")
            for i in range(len(gsm8k_dev)):
                example = gsm8k_dev[i]
                question = example["question"].replace("\u2019", "'")
                explanation, answer = (
                    example["answer"].replace("\u2019", "'").rsplit("####", 1)
                )
                if len(explanation) > 100:
                    continue
                example_dict = dict(
                    question=question,
                    answer=answer.strip(),
                    explanation=explanation.strip(),
                    idx=i,
                )
                data["gsm8k"].append(example_dict)
                if len(data["gsm8k"]) >= args.examples:
                    break
        with open(data_examples_file, "w") as f:
            json.dump(data, f, indent=4)

    with open(data_examples_file, "r") as f:
        all_examples = json.load(f)

    # ------ make prompts ------- #
    example_template = "{description}\n\n" "{examples}" "{input}"
    template = "{description}\n\n" "{input}"
    prompts = list()
    example_prompts = list()
    if args.input == "esnli":
        labels = ["entailment", "neutral", "contradiction"]
        description = "Predict a label for a hypothesis given a premise. The labels are entailment, neutral, and contradiction."
        examples = ""
        for la in labels:
            la_example0 = all_examples[args.input][la][0]
            human = f"Human: Premise is \"{la_example0['premise']}\"\nHypothesis is \"{la_example0['hypothesis']}\"\n\n"
            if args.e_first:
                assistant = f"Assistant: {la_example0['explanation']} #### {la_example0['label']}\n\n"
            else:
                assistant = f"Assistant: {la_example0['label']} #### {la_example0['explanation']}\n\n"
            examples += human + assistant
        input = (
            'Human: Premise is "{premise}"\nHypothesis is "{hypothesis}"\n\nAssistant:'
        )

    elif args.input == "gsm8k":
        description = "The task is to answer basic mathematical problems that require multi-step reasoning."
        examples = ""
        for i, example in enumerate(all_examples["gsm8k"]):
            human = f"Human: {example['question']}\n\n"
            if args.e_first:
                assistant = (
                    f"Assistant: {example['explanation']} #### {example['answer']}\n\n"
                )
            else:
                assistant = (
                    f"Assistant: {example['answer']} #### {example['explanation']}\n\n"
                )
            examples += human + assistant
            if i + 1 >= args.examples:
                break
        input = "Human: {question}\n\nAssistant: "

    prompt = template.format_map(dict(description=description, input=input))
    example_prompt = example_template.format_map(
        dict(description=description, examples=examples, input=input)
    )
    prompts.append(prompt)
    example_prompts.append(example_prompt)
    print(
        example_prompt.format_map(
            dict(hypothesis="hypothesis", premise="premise", question="question")
        )
    )

    with open(
        os.path.join(DATA, "configs", f"{args.input}_prompts_instr.json"), "w"
    ) as f:
        json.dump(prompts, f, indent=4)
    if args.e_first:
        with open(
            os.path.join(DATA, "configs", f"{args.input}_example_prompts_instr.json"),
            "w",
        ) as f:
            json.dump(example_prompts, f, indent=4)
    else:
        with open(
            os.path.join(
                DATA, "configs", "e1", f"{args.input}_example_prompts_instr.json"
            ),
            "w",
        ) as f:
            json.dump(example_prompts, f, indent=4)


if __name__ == "__main__":
    main(parse_args())
