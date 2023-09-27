import random
import re
import string
from typing import List

import regex
import torch

# ----- data process ----- #
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
        response = response.strip(" `\n")
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
        return None


def safe_parse_response(response, first="explanation"):
    result = parse_response(response, first)
    if result["label"].lower() not in esnli_label_map.values():
        result["label"] = unk_label
    return result


def extract_first_response(resp):
    # valid format is:
    # `<a sentence> #### <a word>`
    # find the first match within the response
    # if not found, return None

    pattern = re.compile(r"`([^`]*####[^`]*)`")
    match = pattern.search(resp)
    if match:
        return match.group(1)
    else:
        return None


def process_esnli(data_dict, index):
    # process data loading from huggingface repo
    return dict(
        premise=data_dict["premise"],
        hypothesis=data_dict["hypothesis"],
        label=esnli_label_map[data_dict["label"]],
        explanation=data_dict["explanation_1"],
        index=index,
    )


# ----- multi-document QA ----- #
def normalize_answer(s: str) -> str:
    """Normalization from the SQuAD evaluation script.

    See https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
    """

    def remove_articles(text):
        return regex.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def multidoc_qa_eval(prediction: str, ground_truths: List[str]) -> float:
    """
    Code reference from:
    Lost in the Middle: How Language Models Use Long Contexts
    """
    normalized_prediction = normalize_answer(prediction)

    for ground_truth in ground_truths:
        normalized_ground_truth = normalize_answer(ground_truth)
        if normalized_ground_truth.lower() in normalized_prediction.lower():
            return 1.0
    return 0.0


def make_qa_prompt(ctxs, question, use_gold=False):
    prompt = f"""\
Write a high-quality answer for the given question using only the provided search results (some of which might be irrelevant).

"""
    for index, ctx in enumerate(ctxs):
        if use_gold == False or (use_gold == True and ctx["isgold"] == True):
            prompt += f"""\
Document [{index+1}](Title: {ctx['title']}) {ctx['text']}

"""
    prompt += f"""\
Question: {question}
    
Answer: """
    return prompt


# ----- misc ----- #
def set_all_seed(seed):
    """Set all seeds."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
