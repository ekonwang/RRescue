import argparse
from dataclasses import dataclass, field
import json
import os
import sys
from typing import Dict, Optional, Sequence

from datasets import load_dataset
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import transformers
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          GenerationConfig, HfArgumentParser)

from train_utils import (sequence_gather, smart_tokenizer_and_embedding_resize,
                   tokenize_fn)

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_generation"))
import utils

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"


def parse_args():
    parser = argparse.ArgumentParser(description="Parameters")
    # --- model --- #
    parser.add_argument(
        "--model_name_or_path",
        default="chainyo/alpaca-lora-7b",
        type=str,
        help="model path",
    )
    parser.add_argument("--model_max_length", default=512, type=int, help="model path")
    # --- data --- #
    parser.add_argument("--data_path", default="", type=str, help="config path")
    parser.add_argument("--output_path", default="", type=str, help="config path")
    parser.add_argument("--prompt_file", default="", type=str, help="config path")
    # --- evaluation --- #
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--diverse_beam", type=int, default=1, help="batch size")
    parser.add_argument("--fp16", action="store_true", help="fp16")
    parser.add_argument("--stop_response", action="store_true", help="stop response")
    parser.add_argument(
        "--tag", type=str, default=None, help="tag for the output file name"
    )
    parser.add_argument(
        "--truncate", type=int, default=None, help="truncate the dataset"
    )
    args = parser.parse_args()
    return args


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        args,
    ):
        super(SupervisedDataset, self).__init__()

        if args.data_path == "gsm8k":
            self.dataset_for_eval = load_dataset(args.data_path, "main")["test"]
        elif args.data_path == "esnli":
            self.dataset_for_eval = load_dataset(args.data_path)["validation"]
        else:
            raise NotImplementedError()

        if args.truncate:
            self.dataset_for_eval = [
                self.dataset_for_eval[i]
                for i in range(min(args.truncate, len(self.dataset_for_eval)))
            ]

        self.data_path = args.data_path
        self.tokenizer = tokenizer
        self.input_ids = []
        with open(args.prompt_file, "r") as f:
            self.query_prompt = json.load(f)[0]  # select the first prompt template

    def __len__(self):
        return len(self.dataset_for_eval)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        item = self.dataset_for_eval[i]
        if self.data_path == "esnli":
            premise = item["premise"]
            hypothesis = item["hypothesis"]
            label = item["label"]
            query = self.query_prompt.format_map(
                dict(premise=premise, hypothesis=hypothesis)
            )
        elif self.data_path == "gsm8k":
            question = item["question"]
            answer = item["answer"]
            query = self.query_prompt.format_map(dict(question=question))
            label = answer.rsplit("#", 1)[1].strip()
        else:
            raise NotImplementedError()
        return dict(input_ids=tokenize_fn(query, self.tokenizer), id=i, label=label)


def padding(inputs, padding_token, cutoff=None):
    num_elems = len(inputs)
    if cutoff is None:
        cutoff = max([len(item) for item in inputs])
    tokens = torch.ones(num_elems, cutoff).long().to(inputs[0].device) * padding_token
    for i in range(num_elems):
        toks = inputs[i]
        length = min(cutoff, len(toks))
        tokens[i, -length:] = toks[-length:]
    return tokens


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        results = dict()
        proto = instances[0]
        proto_keys = list(proto.keys())

        for proto_key in proto_keys:
            value_list = list()
            for instance in instances:
                value = instance[proto_key]
                value_list.append(value)
            if isinstance(value, torch.Tensor):
                results[proto_key] = padding(
                    value_list, padding_token=self.tokenizer.pad_token_id
                )
            elif isinstance(value, int) or isinstance(value, float):
                results[proto_key] = torch.tensor(value_list)
            elif isinstance(value, str):
                results[proto_key] = value_list
            else:
                raise NotImplementedError()
            if proto_key == "input_ids":
                attention_mask = results[proto_key].ne(self.tokenizer.pad_token_id)
                results["attention_mask"] = attention_mask
        return results


def make_supervised_data_module(
    args, tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    eval_dataset = SupervisedDataset(tokenizer=tokenizer, args=args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return eval_dataset, data_collator


def main():
    rank = int(os.environ["LOCAL_RANK"])
    world_size = torch.cuda.device_count()
    args = parse_args()
    if world_size > 1:
        dist.init_process_group("nccl")

    os.makedirs(args.output_path, exist_ok=True)
    output_json_file = os.path.join(args.output_path, f"{args.tag}.json")

    # --- load model and tokenizer --- #
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    print(f"Loading {args.model_name_or_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if args.fp16 else torch.float32,
    )
    model = model.cuda().eval()
    torch.cuda.set_device(rank)
    model.to(torch.cuda.current_device())
    if world_size > 1:
        model = DDP(model, device_ids=[torch.cuda.current_device()])
    model.eval()

    # --- load tokenizer --- #
    # tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    tokenizer.truncation_side = "left"
    print(f"{args.model_name_or_path} load completed!!")
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    tokenizer.truncation_side = "left"

    # --- data and tokenizer --- #
    print(f"Loading {args.data_path}...")
    eval_dataset, data_collator = make_supervised_data_module(args, tokenizer)
    if world_size > 1:
        sampler = DistributedSampler(
            eval_dataset, num_replicas=world_size, rank=rank, shuffle=False
        )
    else:
        sampler = None
    batch_size = args.batch_size
    dataloader = DataLoader(
        eval_dataset,
        shuffle=False,
        collate_fn=data_collator,
        batch_size=batch_size,
        sampler=sampler,
    )
    print(f"{args.data_path} load completed!!")

    # --- start evaluation --- #
    generation_config = GenerationConfig(
        # temperature=0.8,
        # num_beam_groups=args.diverse_beam,
        # diversity_penalty=1.0,
        # num_beams=args.diverse_beam,
        temperature=0,
        min_length=1,
        max_new_tokens=128,
        # num_return_sequences=args.diverse_beam,
        num_return_sequences=1,
    )
    all_outputs = []
    for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        input_ids = batch["input_ids"].cuda()
        attention_mask = batch["attention_mask"].cuda()
        labels = batch["label"].unsqueeze(-1).cuda()
        with torch.no_grad():
            if world_size > 1:
                generation_model = model.module
            else:
                generation_model = model
            generation_output = generation_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                return_dict_in_generate=True,
            )
        s = generation_output.sequences
        print(s.shape)

        if world_size > 1:
            num = input_ids.size(0)
            gathered_outputs = sequence_gather(s, world_size, tokenizer.pad_token_id)
            gathered_inputs = sequence_gather(
                input_ids, world_size, tokenizer.pad_token_id
            )
            gathered_labels = sequence_gather(labels, world_size, -1)
            gathered_outputs = torch.stack(gathered_outputs).reshape(
                world_size, num, -1
            )
            gathered_inputs = torch.stack(gathered_inputs)
            gathered_labels = torch.stack(gathered_labels)
            gathered_outputs = gathered_outputs.transpose(0, 1).reshape(
                num * world_size, -1
            )
            gathered_inputs = gathered_inputs.transpose(0, 1).reshape(
                num * world_size, -1
            )
            gathered_labels = gathered_labels.transpose(0, 1).reshape(-1)
        else:
            num = input_ids.size(0)
            gathered_outputs = s.reshape(num, -1)
            gathered_inputs = input_ids.reshape(num, -1)
            gathered_labels = labels.reshape(-1)
        outputs_string = tokenizer.batch_decode(
            gathered_outputs, skip_special_tokens=True
        )
        inputs_string = tokenizer.batch_decode(
            gathered_inputs, skip_special_tokens=True
        )

        labels = gathered_labels.cpu().tolist()
        temp_list = []
        for input_string, output_string, label in zip(
            inputs_string, outputs_string, labels
        ):
            if label == -1:
                continue
            answer = output_string.replace(input_string, "")
            temp_list.append(
                {
                    "query": input_string,
                    "answer": answer,
                    "label": utils.esnli_label_map[label],
                    "original_output": output_string,
                }
            )
        all_outputs += temp_list

    if rank == 0:
        with open(output_json_file, "w") as f:
            json.dump(all_outputs, f, indent=4)


if __name__ == "__main__":
    main()
