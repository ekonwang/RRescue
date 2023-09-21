# ------- flip the explanation expresstion ------ #
# Author: @ekonwang
import argparse
import json
import multiprocessing
import os
import random
import time

from datasets import load_dataset
import openai
from tqdm import tqdm
import utils


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
    count = 0
    while True:
        count += 1
        try:
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
        except Exception as e:
            print(e)
        if count >= 3:
            break
    return None


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
    fail_list = list()
    with open(args.input, "r") as f:
        data_list = json.load(f)
    print(f"Loaded {len(data_list)} data from {args.input}")

    def save_data():
        # --- save the flipped data --- #
        output = args.input.replace(
            ".json", f"-flipped-{len(new_data_list) // 1000}k.json"
        )
        with open(output, "w") as f:
            json.dump(new_data_list, f, indent=4)

    for i, data_dict in enumerate(tqdm(data_list)):
        new_data_dict = dict()
        new_data_dict.update({"data_dict": data_dict["data_dict"]})
        new_data_dict.update({"original_responses": data_dict["responses"]})
        responses = data_dict["responses"].copy()
        flipped_responses = list()
        del data_dict["responses"]

        fail_flag = False
        # --- flip the explanation expresstion --- #
        for resp in responses:
            # --- get explanation --- #
            explanation = utils.parse_response(resp)["explanation"]

            flipped_explanation = gpt_flip_response(args, explanation)
            if flipped_explanation is None:
                fail_flag = True
                break
            # --- predict the label --- #
            pred_label = utils.strip_response(
                gpt_pred_label(args, data_dict["data_dict"], flipped_explanation)
            )
            if pred_label is None:
                fail_flag = True
                break
            # --- make the response --- #
            flipped_resp = utils.wrap_response(
                utils.make_response(
                    dict(explanation=flipped_explanation, label=pred_label)
                )
            )
            # --- save the flipped response --- #
            flipped_responses.append(flipped_resp)
            # --- sleep --- #
            time.sleep(0.2)

        new_data_dict.update({"responses": flipped_responses})
        new_data_list.append(new_data_dict)

        if fail_flag:
            fail_list.append(i)
        if (i + 1) % 5000 == 0:
            save_data()

    print(f"Failed {len(fail_list)} Times\ndata: {fail_list}")
    save_data()
