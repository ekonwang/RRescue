## model is modified based on Alpaca train.py
import argparse
import copy
from dataclasses import dataclass
import json
import os
from typing import Dict, Sequence
import random

from datasets import load_dataset
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import transformers
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

import utils

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"
DATA = os.path.dirname(os.path.abspath(__file__))


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings, tokenizer: transformers.PreTrainedTokenizer):
    """Tokenize a list of strings."""
    if isinstance(strings, list):
        tokenized_list = [
            tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                max_length=tokenizer.model_max_length,
                truncation=True,
            )
            for text in strings
        ]
        input_ids = [tokenized.input_ids[0] for tokenized in tokenized_list]
        return input_ids
    else:
        tokenized = tokenizer(
            strings,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        input_ids = tokenized.input_ids[0]
        return input_ids


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        args,
    ):
        super(SupervisedDataset, self).__init__()

        if args.data_path == "gsm8k":
            self.dataset_original = load_dataset(args.data_path, "main")["train"]
        else:
            self.dataset_original = load_dataset(args.data_path)["train"]
        self.data_path = args.data_path
        self.tokenizer = tokenizer
        self.input_ids = []
        self.dataset_for_eval = list()

        # elif self.data_path == "gsm8k":
        #     for i, example in enumerate(self.dataset_original):
        #         data = dict(question=example["question"])
        #         add_prompt(data, i)

        def add_prompt(prompt, i):
            self.dataset_for_eval.append(dict(prompt=prompt, index=i))

        with open(args.sample_path, "r") as f:
            idxs = json.load(f)
        if args.truncate is not None:
            idxs = idxs[: args.truncate]

        if self.data_path == "esnli":
            template = """\
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

Premise: ```{premise}```
Hypothesis: ```{hypothesis}```
Response: """
            for idx in idxs:
                example = self.dataset_original[idx]
                data = utils.process_esnli(example, idx)
                prompt = template.format_map(data)
                add_prompt(prompt, idx)


    def __len__(self):
        return len(self.dataset_for_eval)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        item = self.dataset_for_eval[i]
        return dict(
            input_ids=_tokenize_fn(item["prompt"], self.tokenizer), id=item["index"]
        )


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


def sequence_gather(s, world_size, pad_tok_id):
    local_size = torch.tensor(s.size(), device=s.device)
    all_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size)
    max_length = max(size[1] for size in all_sizes)
    length_diff = max_length.item() - local_size[1].item()
    if length_diff:
        pad_size = (*s.shape[:-1], length_diff)
        padding = torch.ones(pad_size, device=s.device, dtype=s.dtype) * pad_tok_id
        s = torch.concat((s, padding), dim=-1)
    gathered_s = [torch.ones_like(s) * pad_tok_id for _ in range(world_size)]
    dist.all_gather(gathered_s, s)

    return gathered_s


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
                    value_list,
                    padding_token=self.tokenizer.pad_token_id,
                    cutoff=self.tokenizer.model_max_length,
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
    tokenizer: transformers.PreTrainedTokenizer, args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    eval_dataset = SupervisedDataset(tokenizer=tokenizer, args=args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return eval_dataset, data_collator


def set_all_seed(seed):
    """Set all seeds."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def main(rank, args):
    set_all_seed(args.seed)
    os.makedirs(args.out_path, exist_ok=True)

    torch.cuda.set_device(rank)
    world_size = torch.cuda.device_count()
    if world_size > 1:
        dist.init_process_group("nccl")
        print(f"===== rank: {rank} ===== world_size: {world_size} =====")
    base_model = args.base_model
    data_path = args.data_path
    batch_size = args.batch_size

    print(f"{data_path} loading..")
    tokenizer = LlamaTokenizer.from_pretrained(
        base_model, model_max_length=args.model_max_length
    )
    eval_dataset, data_collator = make_supervised_data_module(tokenizer, args=args)
    print(f"{data_path} load completed!!")

    print(f"{base_model} loading..")
    if "pytorch_model.bin" not in base_model:
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
        )
        ckpt_state = torch.load(base_model)
        ckpt_state = {
            k[11:]: v for k, v in ckpt_state.items() if k.startswith("base_model.")
        }
        model.load_state_dict(ckpt_state, strict=False)
    print(f"{base_model} load completed!!")

    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    if "llama" in base_model:
        tokenizer.add_special_tokens(
            {
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            }
        )
    tokenizer.truncation_side = "left"

    torch.cuda.set_device(rank)
    model.to(torch.cuda.current_device())
    if world_size > 1:
        model = DDP(model, device_ids=[torch.cuda.current_device()])
    model.eval()

    # --- raw data --- #
    if args.data_path == "gsm8k":
        dataset_original = load_dataset(args.data_path, "main")["train"]
    elif args.data_path == "esnli":
        dataset_original = load_dataset(args.data_path)["train"]
    else:
        raise NotImplementedError()

    if world_size > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(
            eval_dataset, num_replicas=world_size, rank=rank, shuffle=False
        )
    else:
        sampler = None
    dataloader = DataLoader(
        eval_dataset,
        shuffle=False,
        collate_fn=data_collator,
        batch_size=batch_size,
        sampler=sampler,
    )
    generation_config = GenerationConfig(
        temperature=args.temperature,
        num_beam_groups=args.diverse_beam,
        diversity_penalty=1.0,
        num_beams=args.diverse_beam,
        min_length=1,
        max_new_tokens=128,
        num_return_sequences=args.diverse_beam,
    )
    all_outputs = []
    for step, batch in enumerate(tqdm(dataloader)):
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        ids = batch["id"].unsqueeze(-1).to(model.device)
        if step == 0 and rank == 0:
            print(input_ids.size(0))
            print(input_ids[0])
            print(attention_mask[0])
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

        if world_size > 1:
            gather_outputs = sequence_gather(s, world_size, tokenizer.pad_token_id)
            gathered_inputs = sequence_gather(
                input_ids, world_size, tokenizer.pad_token_id
            )
            gathered_ids = sequence_gather(ids, world_size, -1)
            gather_outputs = torch.stack(gather_outputs).reshape(
                world_size, batch_size, args.diverse_beam, -1
            )
            gathered_inputs = torch.stack(gathered_inputs)
            gathered_ids = torch.stack(gathered_ids)
            gather_outputs = gather_outputs.transpose(0, 1).reshape(
                batch_size * world_size * args.diverse_beam, -1
            )
            gathered_inputs = gathered_inputs.transpose(0, 1).reshape(
                batch_size * world_size, -1
            )
            gathered_ids = gathered_ids.transpose(0, 1).reshape(-1)
        else:
            gather_outputs = s.reshape(batch_size * world_size * args.diverse_beam, -1)
            gathered_inputs = input_ids.reshape(batch_size * world_size, -1)
            gathered_ids = ids.reshape(-1)
        outputs_string = tokenizer.batch_decode(
            gather_outputs, skip_special_tokens=True
        )
        inputs_string = tokenizer.batch_decode(
            gathered_inputs, skip_special_tokens=True
        )
        num = (gathered_ids != -1).sum()
        id_list = gathered_ids.tolist()

        for it in range(num):
            idx = id_list[it]
            if args.data_path == "esnli":
                data_dict = utils.process_esnli(dataset_original[idx], idx)
            
            responses = []
            for div_idx in range(args.diverse_beam):
                response = outputs_string[it * args.diverse_beam + div_idx]
                response = response.replace(inputs_string[it], "")
                responses.append(response)
            
            all_outputs.append(dict(
                data_dict=data_dict,
                responses=responses,
            ))

    if rank == 0:
        dataset_name = data_path.split("/")[-1]
        model_name = base_model.split("/")[-1]
        output_path = args.out_path + f"/{model_name}/{model_name}_{dataset_name}_seed{args.seed}.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(all_outputs, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameters")
    parser.add_argument("--base_model", default="chainyo/alpaca-lora-7b", type=str, choices=["chainyo/alpaca-lora-7b", "NousResearch/Llama-2-7b-hf"], help="model path")
    parser.add_argument("--data_path", default="esnli", type=str, choices=["esnli"], help="config path")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--port", type=int, default=0, help="batch size")
    parser.add_argument("--diverse_beam", type=int, default=6, help="batch size")
    parser.add_argument("--out_path", default="./output", type=str, help="config path")
    parser.add_argument(
        "--model_max_length", default=512, type=int, help="token length"
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--truncate", type=int, default=None, help="truncate")
    parser.add_argument("--sample_path", type=str, default=None, help="sample list")
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    if torch.cuda.device_count() > 1:
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        local_rank = 0
    # CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port 7881 alpaca-gen.py --truncate 20000 --sample_path ./output/index/esnli_seed40.json --base_model NousResearch/Llama-2-7b-hf --diverse_beam 3
    main(local_rank, args)
