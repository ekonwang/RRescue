import transformers


def tokenize_fn(strings, tokenizer: transformers.PreTrainedTokenizer, max_len=None):
    """Tokenize a list of strings."""
    if max_len is None:
        max_len = tokenizer.model_max_length
    if isinstance(strings, list):
        tokenized_list = [
            tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                max_length=max_len,
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
            max_length=max_len,
            truncation=True,
        )
        input_ids = tokenized.input_ids[0]
        return input_ids


def stop_response(res):
    stops = ["\n\nHuman:", "\n\nAssistant:", "\n\nhuman:", "\n\nassistant:"]
    for stop in stops:
        if res.find(stop) >= 0:
            res = res[: res.find(stop)].strip()
    return res


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict,
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


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa
