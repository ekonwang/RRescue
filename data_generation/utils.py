import re

esnli_label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}
unk_label = "<unk>"
# TODO: strip out useless functions

def stop_response(res):
    stops = ["\n\nHuman:", "\n\nAssistant:", "\n\nhuman:", "\n\nassistant:"]
    for stop in stops:
        if res.find(stop) >= 0:
            res = res[: res.find(stop)].strip()
    return res


def strip_response(response):
    # use re to match a the pattern like this:
    # ```{sentence} ### {one-word-label}```

    if isinstance(response, str):
        response = response.strip(' `\n')
        return response
    elif isinstance(response, list):
        return [strip_response(res) for res in response]
    else:
        raise ValueError(f"Invalid response type {type(response)}")
    

def wrap_response(response):
    clean_resp = strip_response(response)
    return f"```{clean_resp}```"


def make_response(input_dict, first="explanation", captalize=True):
    assert first in ["explanation", "label"]
    label = input_dict["label"].capitalize() if captalize else input_dict["label"]
    if first == "explanation":
        return input_dict["explanation"] + " #### " + label
    elif first == "label":
        return label + " #### " + input_dict["explanation"]
    else:
        raise ValueError(f"Invalid order {first}")


def parse_response(response, first="explanation"):
    assert first in ["explanation", "label"]
    splits = strip_response(response).split("####")
    splits = [s.strip() for s in splits]
    try:
        if first == "explanation":
            return dict(explanation=splits[0], label=splits[1])
        else:
            return dict(explanation=splits[1], label=splits[0])
    except IndexError:
        print(f"Invalid response: {response}")
        return dict(explanation=None, label=unk_label)
    

def process_esnli(data_dict, index):
    # process data loading from huggingface repo
    return dict(
        premise=data_dict["premise"],
        hypothesis=data_dict["hypothesis"],
        label=esnli_label_map[data_dict["label"]],
        explanation=data_dict["explanation_1"],
        index=index,
    )
