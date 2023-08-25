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
        "--examples_path",
        type=str,
        default="/home/yw637/workspace/RankRL/data/configs/esnli_examples.json",
    )
    aa("--output_root", type=str, default="./output")
    aa("--num_samples", type=int, default=10000)
    aa("--dataset", type=str, default="esnli")
    aa("--mid", type=int, default=2)
    # --- model params --- #
    aa("--model_name", type=str, default="gpt-3.5-turbo-0301")  # gpt-4-0314
    aa("--temperature", type=float, default=0)
    aa("--max_tokens", type=int, default=512)
    aa("--top_p", type=float, default=0.9)
    return parser.parse_args()


## ----- openai api ----- ##
def get_gpt_response(params, messages=None):  # model="gpt-4-0314"
    response = openai.ChatCompletion.create(
        model=params.model_name,  # "gpt-4-0314"
        messages=messages,
        temperature=params.temperature,
        max_tokens=params.max_tokens,
        top_p=params.top_p,
        frequency_penalty=0,
        presence_penalty=0,
    )
    return response.choices[0]["message"]["content"]


## ----- generate ----- ##
def msg_esnli(examples, data_dict, mid=1):
    # return a messages list
    assert mid in [1, 2]
    message_list = []

    def aa(role, content):
        message_list.append({"role": role, "content": content})

    # --- use case 1: all in one --- #
    if mid == 1:
        aa("user", f"""\
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
Response:
""")

    # --- use case 2: separate with system command --- #
    if mid == 2:
        aa("system", """\
Classify the relationship between two sentences: a premise and a hypothesis.

Assign one of three labels:
Entailment: The hypothesis is a logical inference that can be derived from the premise.
Contradiction: The hypothesis contradicts the information in the premise.
Neutral: The hypothesis neither logically follows from nor contradicts the premise.

Provide a brief explanation up to 30 words to justify your decision, then add a classification label.

""")
        aa("user", """\
Premise: ```Two women are embracing while holding to go packages.```
Hypothesis: ```Two woman are holding packages.```
""")
        aa("assistant", """\
```Saying the two women are holding packages is a way to paraphrase that the packages they are holding are to go packages. #### Entailment```
""")
        aa("user", """\
Premise: ```Two women are embracing while holding to go packages.```
Hypothesis: ```The sisters are hugging goodbye while holding to go packages after just eating lunch.```
""")
        aa("assistant", """\
```The to go packages may not be from lunch. #### Neutral```
""")
        aa("user", """\
Premise: ```Two women are embracing while holding to go packages.```
Hypothesis: ```The men are fighting outside a deli.```
""")
        aa("assistant", """\
```In the first sentence there is an action of affection between women while on the second sentence there is a fight between men. #### Contradiction```
""")
        aa("user", f"""\
Premise: ```{data_dict["premise"]}```
Hypothesis: ```{data_dict["hypothesis"]}```
""")

    return message_list


if __name__ == "__main__":
    args = parse_args()
    esnli = load_dataset("esnli")["train"]
    save_list = list()
    # ---  --- #
    select_list = list(range(len(esnli)))
    random.shuffle(select_list)
    
    for i in tqdm(range(args.num_samples)):
        index = select_list[i]
        data_dict = utils.process_esnli(esnli[index], index)
        messages = msg_esnli(None, data_dict, mid=args.mid)
        response = get_gpt_response(args, messages=messages)
        save_list.append(dict(
            data_dict=data_dict,
            response=response,
            messages=messages,
        ))
        print(response)

    # --- save --- #
    raw_path = os.path.join(args.output_root, args.dataset, f"{args.model_name}.json")
    if args.num_samples >= 1000:
        output_path = raw_path.replace(".json", f"-samples-{args.num_samples//1000}k.json")
    else:
        output_path = raw_path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(save_list, f, indent=4)

    for samples in [1000, 2000, 5000, 10000]:
        if samples < args.num_samples:
            output_path = raw_path.replace(".json", f"-samples-{samples//1000}k.json")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(save_list[:samples], f, indent=4)
