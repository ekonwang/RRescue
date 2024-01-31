import sys
import os
import json
import time
import warnings
import datetime
from pathlib import Path
from typing import Optional
from tqdm import tqdm

import torch
import openai
import lightning as L

# support running without installing as a package
wd = Path(__file__).parent.resolve()
sys.path.append(str(wd))

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

from lit_llama import Tokenizer, LLaMA
from generate import generate
from lit_llama.lora import lora
from lit_llama.utils import lazy_load, llama_model_lookup
from benchmark.utils import (lora_alpha, lora_dropout, lora_r, lora_config,
                             load_lora_ckpt_from_disk_to_hf_model)

model_configs = {
    'vicuna-7b': "lmsys/vicuna-7b-v1.5",
    'alpaca-7b': 'out/alpaca/lit-llama-lora-finetuned.pth',
    'openai_gpt4_turbo': 'openai_gpt4_1106_turbo',
}
data_configs = {
    'lima': "GAIR/lima"
}

def main(
    model_tag: str = 'vicuna-7b',
    lora_path: Path = None,
    dataset_tag: str = 'lima',
    max_tokens: int = 1024,
    max_new_tokens: int = 1024,
    top_k: int = 200,
    temperature: float = 0.8,
    n_sample: int = 1,
    output_file: str = None,
) -> None:
    assert model_tag in model_configs.keys()
    assert dataset_tag in data_configs.keys()

    model_signature = f"{model_tag}"
    output_file = Path(f"out/sample/"\
                    f"{model_signature}/{dataset_tag}"\
                    f".json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    print(output_file)

    precision = "bf16-true" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "32-true"
    fabric = L.Fabric(devices=1, precision=precision)
    print("Loading model ...", file=sys.stderr)
    t0 = time.time()
    model, tokenizer, prompt_fmt_fn = load_causal_model(model_tag, lora_path, fabric)
    if hasattr(model, 'device'):
        model.eval()
        model = fabric.setup(model)
    print(f"Time to load model: {time.time() - t0:.02f} seconds.", file=sys.stderr)

    collected_responses = list()
    dataset = data_preprocess(dataset_tag)
    acc_cnt = 0
    tot_cnt = 0 
    for sample in tqdm(dataset):
        # prompt = generate_prompt(return_dict['inputs'])
        instruction = sample['instruction']
        prompt = prompt_fmt_fn(instruction)

        sample['responses'] = []
        for i in range(n_sample):
            resp = model_generate(model, tokenizer, prompt, model_tag,
                                  max_tokens=max_tokens, max_new_tokens=max_new_tokens, top_k=top_k, temperature=temperature)
            sample['responses'].append(dict(
                response=resp,
                source=model_configs[model_tag]
            ))

        collected_responses.append(sample)
        
        with open(output_file, "w") as f:
            json.dump(collected_responses, f, indent=4, ensure_ascii=False)
        print(f"Saved to {output_file}", file=sys.stderr)

    if fabric.device.type == "cuda":
        print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB", file=sys.stderr)


def model_generate(model, tokenizer, prompt, model_tag,
                   max_tokens, max_new_tokens, top_k, temperature, llama_stream=False):
    if tokenizer is not None:
        encoded = tokenizer.encode(prompt).to(model.device)[-max_tokens:]
    else:
        assert 'openai' in model_tag

    if 'llama2' in model_tag or 'alpaca' in model_tag:
        output = generate(
            model,
            idx=encoded,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_id=tokenizer.processor.eos_id,
            tokenizer=tokenizer,
            stream=llama_stream
        )
        output = tokenizer.decode(output)
        model.reset_cache()
    elif model_tag == 'mistral-7b' or model_tag == 'vicuna-7b':
        output = model.generate(
            input_ids=encoded,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            do_sample=True,
            num_return_sequences=1,
            eos_token_id=tokenizer.processor.eos_token_id,
            early_stopping=True,
        )
        output = tokenizer.decode(output)
    elif 'openai' in model_tag:
        output = model(prompt, temperature=temperature)

    response = output.replace(prompt, "").strip()
    response = response.replace('<s>', '').replace('</s>', '')
    return response


def load_causal_model(pretrained_model_tag, lora_path, fabric):
    # model tag should be inside the lora path 
    if lora_path is not None:
        assert pretrained_model_tag in str(lora_path)
    fmt_fn = None

    if 'openai' in pretrained_model_tag:
        model = openai_gpt4_1106_turbo
        tokenizer = None 
        fmt_fn = openai_fmt_fn

    else:
        if 'llama2' in pretrained_model_tag or 'alpaca' in pretrained_model_tag:
            tokenizer = Tokenizer('checkpoints/lit-llama/tokenizer.model')
            if '7b' in pretrained_model_tag:
                pretrained_path = 'checkpoints/lit-llama/7B/lit-llama.pth'
            if '13b' in pretrained_model_tag:
                pretrained_path = 'checkpoints/lit-llama/13B/lit-llama.pth'
            if 'alpaca' in pretrained_model_tag:
                lora_path = model_configs[pretrained_model_tag]

            with lazy_load(pretrained_path) as pretrained_checkpoint, lazy_load(lora_path) as lora_checkpoint:
                name = llama_model_lookup(pretrained_checkpoint)
                with fabric.init_module(empty_init=True), lora(r=lora_r, alpha=lora_alpha, dropout=lora_dropout, enabled=True):
                    model = LLaMA.from_name(name)
                    # 1. Load the pretrained weights
                    model.load_state_dict(pretrained_checkpoint, strict=False)
                    # 2. Load the fine-tuned lora weights
                    if lora_checkpoint is not None:
                        model.load_state_dict(lora_checkpoint, strict=False)
            
        elif pretrained_model_tag == 'mistral-7b' or pretrained_model_tag == 'vicuna-7b':
            model_name_or_path = model_configs[pretrained_model_tag]
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            # make lora model
            if lora_path is not None:
                model = load_lora_ckpt_from_disk_to_hf_model(lora_path, model, lora_config=lora_config)

        tokenizer = GeneralTokenizer(tokenizer, pretrained_model_tag)

    
    if 'alpaca' in pretrained_model_tag:
        fmt_fn = alpaca_fmt_fn
    elif pretrained_model_tag == 'vicuna-7b':
        fmt_fn = vicuna_fmt_fn

    return model, tokenizer, fmt_fn


def openai_gpt4_1106_turbo(prompt, temperature=None):
    from openai import OpenAI
    client = OpenAI()
    messages = [dict(role='user', content=prompt)]
    resp = client.chat.completions.create(
        model='gpt-4-1106-preview',
        messages=messages,
        temperature=0 if temperature is None else temperature,
        top_p=0.9,
    )
    return resp.choices[0].message.content


class GeneralTokenizer:
    def __init__(self, tokenizer, model_tag):
        self.processor = tokenizer 
        self.model_tag = model_tag

        if 'alpaca' in self.model_tag:
            self.type = 'lit-llama'
        elif 'mistral' in self.model_tag or 'vicuna' in self.model_tag:
            self.type = 'hf__'

    
    def encode(self, string):
        if self.type == 'lit-llama':
            return self.processor.encode(string, bos=True, eos=False)
        elif self.type == 'hf__':
            return self.processor.encode(string, add_special_tokens=True, return_tensors='pt')

    def decode(self, tokens):
        if self.type == 'lit-llama':
            return self.processor.decode(tokens)
        elif self.type == 'hf__':
            tokens = tokens.squeeze()
            return self.processor.decode(tokens)


def data_preprocess(data_tag):
    if data_tag == 'lima':
        dataset = load_dataset(data_configs[data_tag])['train']
    
    new_data = []
    for sample in tqdm(dataset):
        data_dict = dict()
        if data_tag == 'lima':
            data_dict['instruction'] = sample['conversations'][0]
            data_dict['responses'] = []
    
        new_data.append(data_dict)
    return new_data


def vicuna_fmt_fn(instruction):
    return f"""A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.

USER: {instruction}
ASSISTANT: """


def alpaca_fmt_fn(instruction):
    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{instruction}\n\n### Response:"
    )


def openai_fmt_fn(instruction):
    return instruction

        
if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    warnings.filterwarnings(
        # Triggered internally at ../aten/src/ATen/EmptyTensor.cpp:31
        "ignore", 
        message="ComplexHalf support is experimental and many operators don't support it yet"
    )
    CLI(main)
