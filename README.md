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
```
## data generation

```bash
## gpt-3.5-turbo-0631
cd data_generation
python gpt-series-gen.py --model_name gpt-3.5-turbo-0301 --num_samples 20000
python gpt-series-flip.py --input <file>

## alpaca-7b
cd data_generation
CUDA_VISIBLE_DEVICES=1,2,3,4,5 torchrun --nproc_per_node=5 --master_port 7881 gen-alpaca.py --truncate 20000 --sample_path ./output/index/esnli_seed40.json
python alpaca-rank.py --input <generated_file>
```

## run training (alpaca-7b)

```bash
bash train.sh <output_path> <data_path> <num_response>
bash esnli_eval.sh <output_path> <num_eval_samples>
```
