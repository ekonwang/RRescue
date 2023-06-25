#!/bin/bash 
# for innitializing new environment
conda create -n rrhf python=3.8
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
git clone https://github.com/huggingface/transformers.git
pip install -e ./transformers
pip install -r requirements.txt

cd ./data_generation/
bash response_gen.sh chainyo/alpaca-lora-7b ../generated_data
