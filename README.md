# RRescue: Ranking LLM Responses to Enhance Contextual Understanding

![](https://img.shields.io/static/v1?label=&message=MIPS&color=orange&style=for-the-badge)![](https://img.shields.io/static/v1?label=&message=LLM&color=red&style=for-the-badge)![](https://img.shields.io/static/v1?label=&message=For-ICLR-2024&color=black&style=for-the-badge)

[overleaf link](https://www.overleaf.com/project/64ca9b8ac33902595d5adc01) [google doc link](https://docs.google.com/document/d/1eanF7cs4QSEUCIwU1uDmrvI2nGzhdCUzq1EmDR19oWM/edit)

Effectively using a given context is paramount for large language models (LLMs). Their context window can include task specifications, retrieved documents, previous conversations, and model self-reflections, functioning similarly to episodic memory. While there have been efforts to expand this context window, studies indicate that LLMs do not use their context optimally for response generation. In this paper, we introduce a novel approach to optimize LLMs using ranking metrics, which teaches LLMs to rank a collection of contextually-grounded candidate responses. Rather than a traditional full ordering, we advocate for a partial ordering. This is because achieving consensus on the perfect order for system responses can be challenging. Our partial ordering is more robust, less sensitive to noise, and can be acquired through human labelers, heuristic functions, or model distillation. We test our system's improved contextual understanding using the latest benchmarks, including a new multi-document question answering dataset. We conduct comprehensive ablation studies to investigate crucial factors such as the number of ranked responses, their sources, and how to best balance supervised fine-tuning with the ranking metrics. Our approach, named **RRescue**, suggests a promising avenue for enhancing LLMs' contextual understanding via response ranking.

## setup

This is a revised-version of the original code from branch master.

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
git submodule update --init --recursive

python -m pip install -e ./transformers
python -m pip install -e ./accelerate
python -m pip install -r requirements.txt
python -m pip install xformers
python -m pip install pre-commit

pre-commit install --install-hooks --all

# --- for pre-commit all files --- #
pre-commit run --all-files
```
## data generation

```bash
## gpt-3.5-turbo-0613
cd data_generation
python gpt-series-gen.py --model_name gpt-3.5-turbo-0613 --num_samples 20000
python gpt-series-flip.py --input <file>

## alpaca-7b
cd data_generation
CUDA_VISIBLE_DEVICES=1,2,3,4,5 torchrun --nproc_per_node=5 --master_port 7881 gen-alpaca.py --truncate 20000 --sample_path ./output/index/esnli_seed40.json
python alpaca-rank.py --input <generated_file>
```

## run training and evaluation (alpaca-7b / llama2-7b)

```bash
export MODEL_FOR_TRAIN=<target_model>
bash train.sh <output_path> <data_path> <num_response>
bash esnli_eval.sh <output_path> <num_eval_samples>
```
