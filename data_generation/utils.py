esnli_label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}

# TODO: strip out useless functions

def stop_response(res):
    stops = ["\n\nHuman:", "\n\nAssistant:", "\n\nhuman:", "\n\nassistant:"]
    for stop in stops:
        if res.find(stop) >= 0:
            res = res[: res.find(stop)].strip()
    return res


def strip_response(response):
    if isinstance(response, str):
        response = response.strip(' `\n')
        return response
    elif isinstance(response, list):
        return [strip_response(res) for res in response]
    else:
        raise ValueError(f"Invalid response type {type(response)}")


def make_response(input_dict, first="explanation"):
    assert first in ["explanation", "label"]
    if first == "explanation":
        return input_dict["explanation"] + " #### " + input_dict["label"]
    elif first == "label":
        return input_dict["label"] + " #### " + input_dict["explanation"]
    else:
        raise ValueError(f"Invalid order {first}")


def parse_response(response, first="explanation"):
    assert first in ["explanation", "label"]
    splits = response.strip(' `').split("####")
    splits = [s.strip() for s in splits]
    if first == "explanation":
        return dict(explanation=splits[0], label=splits[1])
    else:
        return dict(explanation=splits[1], label=splits[0])
