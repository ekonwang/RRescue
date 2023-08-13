import argparse
import json
import multiprocessing

from datasets import load_dataset
import openai
from tqdm import tqdm
import utils


## ----- args ----- ##
def parse_args():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    aa("--model_name", type=str, default="gpt-3.5-turbo-0301")
    aa("--multiprocessing", type=int, default=0)
    aa(
        "--examples_path",
        type=str,
        default="/workspace/Rank/data/configs/esnli_examples.json",
    )
    aa("--output_root", type=str, default=None)
    aa("--dataset", type=str, default="esnli")

    aa("--temperature", type=float, default=0.5)
    aa("--max_tokens", type=int, default=512)
    aa("--top_p", type=float, default=0.9)

    aa("--beam_size", type=int, default=2)
    return parser.parse_args()


## ----- openai api ----- ##
def get_gpt_response(params, messages=None):  # model="gpt-4-0314"
    response = openai.ChatCompletion.create(
        model=params.model_name,  # "gpt-4-0314"
        # messages=[
        #     {
        #     "role": "user",
        #     "content": f"{prompt}"
        #     }
        # ],
        messages=messages,
        temperature=params.temperature,
        max_tokens=params.max_tokens,
        top_p=params.top_p,
        frequency_penalty=0,
        presence_penalty=0,
    )
    return response.choices[0]["message"]["content"]
    # return response.choices


## ----- generate ----- ##
def msg_esnli(examples, data_dict):
    # return a messages list
    message_list = []

    def aa(role, content):
        message_list.append({"role": role, "content": content})

    sys = "Give an explanation and a prediction for a pair of premise and hypothesis. Please follow previous responses' format."
    # sys = f"Give an explanation for a given prediction \"{data_dict['label']}\", make sure the explanation is consistent with the prediction, ignore logically unreasonable parts and maintain formal consistency. Please follow previous reponses' format."
    aa("system", sys)

    for example in examples:
        premise = example["premise"]
        hypothesis = example["hypothesis"]
        human_content = f'Premise is "{premise}"\nHypothesis is "{hypothesis}"'
        aa("human", human_content)

        label = (
            example["label"]
            if isinstance(example["label"], str)
            else utils.esnli_label_map[example["label"]]
        )
        explanation = example["explanation"]
        assistant_content = utils.make_response(
            {"label": label, "explanation": explanation}
        )
        aa("assistant", assistant_content)
    return message_list


def process_esnli(data_dict, index):
    # process data loading from huggingface repo
    return dict(
        premise=data_dict["premise"],
        hypothesis=data_dict["hypothesis"],
        label=utils.esnli_label_map[data_dict["label"]],
        explanation=data_dict["explanation_1"],
        index=index,
    )


def handle_esnli_examples(params):
    with open(params.examples_path, "r") as f:
        data = json.load(f)
    examples = data[params.dataset]
    return list(examples.values())


def robust_generate(examples, data_dict, msg_func, params):
    count = 0
    while count < 3:
        # try:
        messages = msg_func(examples, data_dict)
        responses = get_gpt_response(messages=messages, params=params)
        return responses
        # except Exception as e:
        #     count += 1
        #     print("Error! ", e)
        #     print(data_dict)
    return None


def generate(data_list, examples_list, output_file, msg_func, params):
    results = list()
    for data_dict in tqdm(data_list):
        for examples in examples_list:
            responses = []
            for i in range(params.beam_size):
                response = robust_generate(examples, data_dict, msg_func, params)
                if response is not None:
                    break
                resp = [None, response, data_dict["index"]]  # query, response, index
            responses.append(resp)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)


## ----- main ----- ##
def main(params):
    if params.dataset == "gsm8k":
        dataset = load_dataset(params.dataset, "main")["train"]
    else:
        dataset = load_dataset(params.dataset)["train"]

    if params.dataset == "esnli":
        msg_func = msg_esnli
        datalist = [
            process_esnli(data_dict, index)
            for index, data_dict in enumerate(dataset)
            if index <= 2
        ]
        examples = handle_esnli_examples(params)

        if params.multiprocessing:
            raise NotImplementedError
        else:
            output_file = f"{params.output_root}/raw_generation_{params.dataset}.json"
            generate(datalist, examples, output_file, msg_func, params)


if __name__ == "__main__":
    params = parse_args()
    main(params)
