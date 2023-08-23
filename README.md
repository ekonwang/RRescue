## setup

This is a revised-version of the original code from branch master.

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
git submodule update --init --recursive

python -m pip install -e ./transformers
python -m pip install -e ./accelerate
python -m pip install -r requirements.txt
python -m pip install xformers

pip install -r requirements.txt
```
## data generation

```bash
cd data_generation
python gpt-series-gen.py --model_name gpt-3.5-turbo-0301 --num_samples 20000
python gpt-series-flip.py --input <file>
```


