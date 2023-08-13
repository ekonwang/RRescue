esnli_label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}


def stop_response(res):
    stops = ["\n\nHuman:", "\n\nAssistant:", "\n\nhuman:", "\n\nassistant:"]
    for stop in stops:
        if res.find(stop) >= 0:
            res = res[: res.find(stop)].strip()
    return res


def strip_process(response):
    if isinstance(response, str):
        response = response.strip().lstrip("\n")
        response = stop_response(response)
        response = response.replace("\n", "")
        # response = response.split(".", 1)[1] # no need for split the sentence here.
        return response
    elif isinstance(response, list):
        return [strip_process(res) for res in response]
    else:
        raise ValueError(f"Invalid response type {type(response)}")


def make_response(input_dict):
    # expect a specfication, a answer (label), and a order (whether specification first or answer first)
    # and format (sparate by special characters like #### or using natural language)
    fmt = input_dict.get("format", "special")
    first = input_dict.get("first", "explaination")
    if fmt == "special":
        if first == "explaination":
            return input_dict["explaination"] + " #### " + input_dict["label"]
        elif first == "label":
            return input_dict["label"] + " #### " + input_dict["explaination"]
        else:
            raise ValueError(f"Invalid order {first}")
    elif fmt == "natural":
        if first == "explaination":
            return (
                input_dict["explaination"]
                + "\nThe answer is "
                + input_dict["label"]
                + "."
            )
        elif first == "label":
            return (
                "The answer is "
                + input_dict["label"]
                + ".\n"
                + input_dict["explaination"]
            )
        else:
            raise ValueError(f"Invalid order {first}")
    else:
        raise ValueError(f"Invalid format {fmt}")


def parse_response(response, first="explaination", fmt="special"):
    if fmt == "special":
        if first == "explaination":
            explaination, label = response.split("####")
        elif first == "label":
            label, explaination = response.split("####")
        else:
            raise ValueError(f"Invalid order {first}")
        return explaination.strip(), label.strip()
    elif fmt == "natural":
        if first == "explaination":
            explaination = response.split("\n")[0]
            label = response.split("\n")[1].split("The answer is ")[1].strip(".")
        elif first == "label":
            label = response.split("\n")[0].split("The answer is ")[1].strip(".")
            explaination = response.split("\n")[1]
        else:
            raise ValueError(f"Invalid order {first}")
        return explaination.strip(), label.strip()
    else:
        raise ValueError(f"Invalid format {fmt}")
