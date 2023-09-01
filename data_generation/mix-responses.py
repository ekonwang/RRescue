import argparse
import json
import os

import utils


## ----- args ----- ##
def parse_args():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    # --- misc --- #
    aa("--input_files", type=str, nargs="+", required=True)
    aa("--valid_responses", type=int, nargs="+", required=True)
    aa("--sample_list", type=str, default="./output/index/esnli_seed40.json")
    # --- model params --- #
    aa("--truncate", type=int, default=70000)
    return parser.parse_args()


## ---- utils ---- ##
def fetch_valid_and_wrap_responses(data_dict):
    resp_list = []
    try:
        responses = data_dict["responses"]
    except KeyError:
        responses = data_dict["response"]
        if isinstance(responses, str):
            responses = [responses]
    for lm_resp in responses:
        extracted = utils.extract_first_response(lm_resp)
        if extracted:
            resp_list.append(extracted)
    resp_list = [utils.wrap_response(resp) for resp in resp_list]
    return resp_list


## ----- main ----- ##
def main(args):
    with open(args.sample_list, "r") as f:
        sample_list = json.load(f)
    sample_list = sample_list[: args.truncate]

    assert len(args.input_files) == len(args.valid_responses)
    data_list_dict = dict()
    data_index_dict = dict()
    thres_dict = dict()
    for input_file in args.input_files:
        input_name = input_file.split("/")[-1].rsplit(".", 1)[0]
        with open(input_file, "r") as f:
            data_list = json.load(f)

        thres_dict[input_name] = args.valid_responses[
            args.input_files.index(input_file)
        ]
        data_list_dict[input_name] = data_list
        data_index_dict[input_name] = dict()
        for i, data_dict in enumerate(data_list):
            data_index_dict[input_name][data_dict["data_dict"]["index"]] = i

    new_data_list = list()
    cnt = 0
    for ii, sample_num in enumerate(sample_list):
        found_list = [
            (sample_num in data_index_dict[input_name])
            for input_name in data_index_dict
        ]
        data_dict = dict(responses=list(), sources=list())
        if all(found_list):
            flag = True
            for input_name in sorted(data_index_dict.keys()):
                index = data_index_dict[input_name][sample_num]
                found_dict = data_list_dict[input_name][index]

                valid_wrapped_responses = fetch_valid_and_wrap_responses(found_dict)
                if len(valid_wrapped_responses) < thres_dict[input_name]:
                    flag = False
                    print(
                        f"invalid {input_name} {sample_num} only has {len(valid_wrapped_responses)} valid responses, expected {thres_dict[input_name]}"
                    )
                    # import pdb; pdb.set_trace()
                    break
                else:
                    valid_wrapped_responses = valid_wrapped_responses[
                        : thres_dict[input_name]
                    ]

                data_dict["data_dict"] = found_dict["data_dict"].copy()
                data_dict["responses"].extend(valid_wrapped_responses.copy())
                data_dict["sources"].extend([input_name] * len(valid_wrapped_responses))
                assert found_dict["data_dict"]["index"] == sample_num

            if flag:
                new_data_list.append(data_dict)
                assert (
                    len(data_dict["responses"])
                    == len(data_dict["sources"])
                    == sum([thres_dict[input_name] for input_name in data_index_dict])
                )
                cnt += 1
        if (ii + 1) % 10000 == 0:
            print(f"success {cnt}/({ii-9999}-{ii+1})")
            cnt = 0

    for input_name, data_list in data_list_dict.items():
        print(f"{input_name} has {len(data_list)} samples")

    print(f"total samples: {len(new_data_list)}")
    output_file = f"./output/mix/raw-mixed-{len(new_data_list)//1000}k.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(new_data_list, f, indent=4)


if __name__ == "__main__":
    args = parse_args()
    main(args)
