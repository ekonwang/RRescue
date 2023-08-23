# ------- flip the explanation expresstion ------ #
# Author: @ekonwang
import argparse
import json
import multiprocessing

from datasets import load_dataset
import openai
import os
from tqdm import tqdm
import utils
import random

## ----- args ----- ##
def parse_args():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    # --- misc --- #
    aa(
        "--input",
        type=str,
        default=None,
    )
    aa("--dataset", type=str, default="esnli")
    aa("--flip_beam", type=int, default=3)
    # --- model params --- #
    aa("--model_name", type=str, default="gpt-3.5-turbo-0301")  # gpt-4-0314
    aa("--temperature", type=float, default=1)
    aa("--pred_temperature", type=float, default=0)
    aa("--max_tokens", type=int, default=512)
    aa("--top_p", type=float, default=0.9)
    return parser.parse_args()


## ----- openai api ----- ##
def get_gpt_response(params, messages=None, temperature=None):  # model="gpt-4-0314"
    response = openai.ChatCompletion.create(
        model=params.model_name,  # "gpt-4-0314"
        messages=messages,
        temperature=params.temperature if temperature is None else temperature,
        max_tokens=params.max_tokens,
        top_p=params.top_p,
        frequency_penalty=0,
        presence_penalty=0,
    )
    return response.choices[0]["message"]["content"]


def gpt_flip_response(params, Explanation):
    prompt = f"""\
Rewrite this sentence to convey the opposite meaning: ```{Explanation}```
"""
#     prompt = f"""\
# Combine the context and write a sentence to express a very different meaning: ```{Explanation}```
# """
    messages = [{"role": "user", "content": prompt}]
    return get_gpt_response(params, messages)


def gpt_pred_label(params, data_dict, flipped_explanation):
    prompt = f"""\
Classify the relationship between two sentences: a premise and a hypothesis.

Assign one of three labels:
Entailment: The hypothesis is a logical inference that can be derived from the premise.
Contradiction: The hypothesis contradicts the information in the premise.
Neutral: The hypothesis neither logically follows from nor contradicts the premise.

Provide a brief explanation up to 30 words to justify your decision, then add a classification label.

Premise: ```Two women are embracing while holding to go packages.```
Hypothesis: ```Two woman are holding packages.```
Response: ```Saying the two women are holding packages is a way to paraphrase that the packages they are holding are to go packages. #### Entailment```

Premise: ```Two women are embracing while holding to go packages.```
Hypothesis: ```The sisters are hugging goodbye while holding to go packages after just eating lunch.```
Response: ```The to go packages may not be from lunch. #### Neutral```

Premise: ```Two women are embracing while holding to go packages.```
Hypothesis: ```The men are fighting outside a deli.```
Response: ```In the first sentence there is an action of affection between women while on the second sentence there is a fight between men. #### Contradiction```

Premise: ```{data_dict["premise"]}```
Hypothesis: ```{data_dict["hypothesis"]}```
Response: ```{flipped_explanation} ####
"""
    messages = [{"role": "user", "content": prompt}]
    return get_gpt_response(params, messages, temperature=params.pred_temperature)


if __name__ == "__main__":
    args = parse_args()
    new_data_list = list()
    with open(args.input, "r") as f:
        data_list = json.load(f)
    print(f"Loaded {len(data_list)} data from {args.input}")

    for i, data_dict in enumerate(tqdm(data_list)):
        new_data_dict = data_dict.copy()
        flip_resps = list()
        new_data_dict.update({"response": utils.strip_response(data_dict["response"])})

        # --- flip the explanation expresstion --- #
        explanation = utils.parse_response(data_dict["response"])["explanation"]
        for _ in range(args.flip_beam):
            # flipped_explanation = gpt_flip_response(args, explanation)
            flipped_explanation = gpt_flip_response(args, data_dict["data_dict"]["explanation"])
            # --- predict the label --- #
            pred_label = utils.strip_response(gpt_pred_label(args, data_dict["data_dict"], flipped_explanation))
            # --- save the flipped response --- #
            flipped = utils.make_response(dict(explanation=flipped_explanation, label=pred_label))
            flip_resps.append(flipped)
            print(f"{flipped}")

        new_data_dict.update({"flipped_responses": flip_resps})
        del new_data_dict["messages"]
        new_data_list.append(new_data_dict)
        
        if i >= 5:
            break

    # --- save the flipped data --- #
    output = args.input.replace(".json", "-flipped.json")
    with open(output, "w") as f:
        json.dump(new_data_list, f, indent=4)
