import argparse
import json

import utils


# -- parse -- #
def parse_args():
    parser = argparse.ArgumentParser(description="Parameters")
    parser.add_argument("--input_file", type=str, default="", required=True)
    parser.add_argument("--output_file", type=str, default="", required=True)
    parser.add_argument("--dataset", type=str, default="esnli")
    parser.add_argument("--first", type=str, default="explaination")
    parser.add_argument("--resp_fmt", type=str, default="special")
    parser.add_argument("--func", type=str, default="simple")
    parser.add_argument("--log_freq", type=int, default=100)
    args = parser.parse_args()

    return args


# -- processing function -- #
def simple_rescore(scores, helper_dict):
    assert len(scores) == helper_dict["resp_num"]
    new_scores = [0] * helper_dict["resp_num"]
    for idx in range(helper_dict["resp_num"]):
        new_scores[idx] = 2.0 if idx == 0 else 1.0
    return new_scores


def get_partial(data_dict, helper_dict, rescore_func):
    partial_dict = data_dict.copy()
    new_scores = rescore_func(partial_dict["scores"], helper_dict)
    partial_dict["scores"] = new_scores
    return partial_dict


# -- utils -- #
def get_data_helper(data_dict, params):
    _, label = utils.parse_response(
        data_dict["synthesized_responses"][0], first=params.first, fmt=params.resp_fmt
    )
    resp_num = len(data_dict["synthesized_responses"])
    return dict(label=label, resp_num=resp_num)


def load_data(params):
    with open(params.input_file, "r") as f:
        data = json.load(f)
    return data


def write_data(params, data):
    with open(params.output_file, "w") as f:
        json.dump(data, f, indent=2)
    file_1k = params.output_file.replace(".json", "_1k.json")
    with open(file_1k, "w") as f:
        json.dump(data[:1000], f, indent=2)


# -- main -- #
def main(params):
    data_list = load_data(params)
    new_list = []
    if params.func == "simple":
        rescore_func = simple_rescore
    else:
        raise ValueError(f"Invalid function {params.func}")

    for i, data_dict in enumerate(data_list):
        helper_dict = get_data_helper(data_dict, params)
        partial_dict = get_partial(data_dict, helper_dict, rescore_func)
        new_list.append(partial_dict)
        if i % params.log_freq == 0:
            print(f"processing {i} / {len(data_list)}")
            print(partial_dict)
    write_data(params, new_list)


if __name__ == "__main__":
    # get parser args / params
    params = parse_args()
    # main function
    main(params)
