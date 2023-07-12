# please run outside of the directory
# conda create -n rank python=3.8
# conda activate rank

python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
git submodule update --init --recursive

python -m pip install -e ./transformers
python -m pip install -e ./accelerate
python -m pip install -r requirements.txt
python -m pip install xformers
