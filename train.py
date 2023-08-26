from dataclasses import dataclass, field
import io
import json
import logging
import os
import random
from typing import Dict, Optional, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import transformers
from transformers import Trainer
from train_utils import (safe_save_model_for_hf_trainer,
                   smart_tokenizer_and_embedding_resize, tokenize_fn)

# ---------------------------------------------------------------------- #
# train LM with ranking guidance
# RRHF: Rank Responses to Align Language Models with Human Feedback without tears
# code reference https://github.com/GanjinZero/RRHF/blob/main/train.py
# ---------------------------------------------------------------------- #

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"
LOGFILE = None

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    stop_response: bool = field(default=False)
    num_resp: int = field(default=4)
    prompt_file: str = field(default=None)
    use_eos_token: int = field(default=0)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    rrhf_weight: float = field(default=100.0)
    train_strategy: str = field(default="sft")
    length_penalty: float = field(default=1.0)
    logging_file: str = field(default=None)


# ---------- dataset -------------- #
class ScoreDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(ScoreDataset, self).__init__()
        logging.warning("Loading data...")
        with open(data_path, "r") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return dict(input_ids=self.data[i])


# ---------- data collator -------------- #
@dataclass
class DataCollatorForRankDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    stop_response: bool
    data_args: DataArguments

    def __call__(self, instances):
        idxs = []
        all_scores = []
        input_ids = []
        score_mask = []
        labels = []
        for idx, ins in enumerate(instances):
            with open(self.data_args.prompt_file, "r") as f:
                prompt = json.load(f)[0]
            ins = ins["input_ids"]  # hack
            query = prompt.format_map(ins["data_dict"])
            responses = ins["responses"]
            scores = ins["scores"]

            query_input_ids = tokenize_fn(query, self.tokenizer)
            query_target = torch.LongTensor(
                [IGNORE_INDEX] * (query_input_ids.shape[0] - 1)
            )
            if self.data_args.use_eos_token:
                dummy_target = torch.LongTensor([self.tokenizer.eos_token_id])
            else:
                dummy_target = torch.LongTensor([IGNORE_INDEX])

            sel_resps = [responses[0]]
            all_scores.append([scores[0]])
            sample_idxs = random.sample(
                list(range(1, len(responses))), min(self.data_args.num_resp, len(responses)) - 1
            )
            sel_resps += [responses[i] for i in sample_idxs]
            all_scores[-1] += [scores[i] for i in sample_idxs]

            for r in sel_resps:
                res_input_ids = tokenize_fn(
                    r + self.tokenizer.eos_token,
                    self.tokenizer,
                    max_len=self.tokenizer.model_max_length - query_input_ids.shape[0],
                )  # eos here
                input_ids.append(torch.cat((query_input_ids, res_input_ids), dim=0))
                labels.append(
                    torch.cat((query_target, res_input_ids, dummy_target), dim=0)
                )
            idxs.append([idx] * len(sel_resps))

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        return dict(
            input_ids=input_ids,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            labels=labels,
            idxs=torch.LongTensor(idxs),
            scores=torch.FloatTensor(all_scores),
        )


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    stop_response: bool
    data_args: DataArguments

    def __call__(self, instances):
        idxs = []
        all_scores = []
        input_ids = []
        score_mask = []
        labels = []
        for idx, ins in enumerate(instances):
            with open(self.data_args.prompt_file, "r") as f:
                prompt = json.load(f)[0]
            ins = ins["input_ids"]  # hack
            query = prompt.format_map(ins["inputs"])
            responses = ins["synthesized_responses"]
            scores = ins["scores"]
            idxs.append([idx] * len(scores))

            query_input_ids = tokenize_fn(query, self.tokenizer)
            query_target = torch.LongTensor(
                [IGNORE_INDEX] * (query_input_ids.shape[0] - 1)
            )
            if self.data_args.use_eos_token:
                dummy_target = torch.LongTensor([self.tokenizer.eos_token_id])
            else:
                dummy_target = torch.LongTensor([IGNORE_INDEX])

            r = responses[0]  # only choose the dataset given one
            all_scores.append([scores[0]])
            res_input_ids = tokenize_fn(
                r + self.tokenizer.eos_token,
                self.tokenizer,
                max_len=self.tokenizer.model_max_length - query_input_ids.shape[0],
            )  # eos here
            input_ids.append(torch.cat((query_input_ids, res_input_ids), dim=0))
            labels.append(torch.cat((query_target, res_input_ids, dummy_target), dim=0))

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        return dict(
            input_ids=input_ids,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            labels=labels,
            idxs=torch.LongTensor(idxs),
            scores=torch.FloatTensor(all_scores),
        )


class RankTrainer(Trainer):
    def gather_logits_labels(self, logits, labels):
        mask = (labels != -100).float()
        new_logits = logits.clone()  # Create a copy to avoid in-place modification
        labels[labels == -100] = 0
        output = torch.gather(new_logits, dim=-1, index=labels.unsqueeze(-1)).squeeze(
            -1
        )
        output = output * mask  # B * L
        return output

    def get_score(self, logit_label, labels):
        mask = (labels != -100).float()
        length = mask.sum(-1)
        scores = logit_label.sum(-1) / (length**self.args.length_penalty)
        return scores

    def rrhf_loss(self, scores, idxs, rw_scores):
        diff = scores.unsqueeze(0) - scores.unsqueeze(-1)  # b * b
        rw_diff = rw_scores.unsqueeze(0) - rw_scores.unsqueeze(-1)  # b * b
        aval = torch.bitwise_and(rw_diff > 0, diff < 0)
        # original aval have aval = aval[0], IDK why, so I change it to aval = aval
        # reference: https://github.com/GanjinZero/RRHF/blob/main/train.py#L251
        if aval.ndim > 2:
            aval = aval[0]
        return -diff[aval].sum()

    def sft_loss(self, logit_label, labels, rw_scores):
        max_idx = torch.argmax(rw_scores)
        mask = (labels[max_idx] != -100).float()
        return -logit_label[max_idx].sum() / mask.sum()

    def compute_loss(self, model, inputs, return_outputs=False):
        local_rank = int(os.environ["LOCAL_RANK"])
        logits = model(
            input_ids=inputs.get("input_ids"),
            attention_mask=inputs.get("attention_mask"),
        ).logits  # (batch * cand) * L * V
        logits = F.log_softmax(logits, dim=-1)
        logit_label = self.gather_logits_labels(logits, inputs.get("labels").clone())

        if self.args.train_strategy == "rank":
            scores = self.get_score(logit_label, inputs.get("labels"))
            rrhf_loss = self.rrhf_loss(scores, inputs.get("idxs"), inputs.get("scores"))
            sft_loss = self.sft_loss(
                logit_label, inputs.get("labels"), inputs.get("scores")
            )
            loss = self.args.rrhf_weight * rrhf_loss + sft_loss
        elif self.args.train_strategy == "sft":
            sft_loss = self.sft_loss(
                logit_label, inputs.get("labels"), inputs.get("scores")
            )
            loss = sft_loss

        if local_rank == 0:
            if self.args.train_strategy == "rank":
                stats_dict = dict(
                    rrhf_loss=rrhf_loss.item(),
                    sft_loss=sft_loss.item(),
                    loss=loss.item(),
                )
            elif self.args.train_strategy == "sft":
                stats_dict = dict(sft_loss=loss.item(), loss=loss.item())
            with open(LOGFILE, "a") as f:
                f.write(json.dumps(stats_dict) + "\n")

        return (loss, scores) if return_outputs else loss


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args, train_strategy
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = ScoreDataset(tokenizer=tokenizer, data_path=data_args.data_path)
    if train_strategy == "sft":
        data_collator = DataCollatorForSupervisedDataset(
            tokenizer=tokenizer,
            stop_response=data_args.stop_response,
            data_args=data_args,
        )
    elif train_strategy == "rank":
        data_collator = DataCollatorForRankDataset(
            tokenizer=tokenizer,
            stop_response=data_args.stop_response,
            data_args=data_args,
        )
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )


def train():
    global LOGFILE
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    os.makedirs(training_args.output_dir, exist_ok=True)
    logging_file = os.path.join(training_args.output_dir, "training.log")
    with open(logging_file, "w") as f:
        f.write("")
    LOGFILE = logging_file
    all_args = {
        k: v
        for k, v in {
            **vars(model_args),
            **vars(data_args),
            **vars(training_args),
        }.items()
        if isinstance(v, (int, float, str, bool))
    }
    with open(os.path.join(training_args.output_dir, "training_args.json"), "w") as f:
        f.write(json.dumps(all_args, indent=2))

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    if "llama" in model_args.model_name_or_path:
        tokenizer.add_special_tokens(
            {
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            }
        )

    data_module = make_supervised_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
        train_strategy=training_args.train_strategy,
    )
    trainer = RankTrainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )
    trainer.train()
    trainer.save_state()
    local_rank = int(os.environ["LOCAL_RANK"])
    if local_rank == 0:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
