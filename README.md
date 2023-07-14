### Jun23 - July23 一月工作计划 🚀

目前的语言模型虽然强大，但语言模型在生成文本时依然容易受到无关context的影响，或者生成不基于 context 的输出。我们希望将模型输出的 explaination 与相应 query 具备的 context 对齐，以保证大语言模型 answer 的良好效果。

包含项目细节的[腾讯文档](https://docs.qq.com/doc/DWnBIcGZVc3R6d0Nl)

## 数据集

- [E-SNLI](https://docs.qq.com/doc/DWnBIcGZVc3R6d0Nl)

- 一些 QA 数据集

## 运行

0. 配置新环境

conda create -n rank python=3.8
conda activate rank

1. 首先安装对应 cuda 版本的 torch：

```shell
cd <DIR>

python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

2. 安装 `dev0` 版本 `transformers` 以满足 Llama 🦙 模型依赖，以及训练要求的 `dev` 版本 `accelerate`

```shell
git submodule update --init --recursive

pip install -e ./transformers

pip install -e ./accelerate
```

3. 安装其他依赖

```shell
pip install -r requirements.txt
python -m pip install xformers
```

4. 开始数据生成，根据空闲 GPU 数量自行修改 NPROC 变量


```shell
cd data_generation

chmod +x gen_response.sh

./gen_response.sh
```

## 数据生成

- [x]  SNLI 数据生成 (Jun23 - Jun27)
    - [x] 模型 Alpaca-lora-7B 或者更小的 Flan-T5 模型(可能不行因为无法对话).
    - [x] `response_gen` 模块修改.
    - [x] A40 单机单卡测试, 基于 beam search 生成若干回答.
    - [x] `scoring_responses` 模块修改，加入 SBERT.
    - [x] 针对 `e-snli` 任务提升 Alpaca-7B 生成质量.
    - [x] 测试 `data_generation` 模块.

## 训练 pipeline

- [x]  训练 pipeline (Jun28 - July14)
    - [x] SFT 代码以及 RankSFT 代码
    - [x] 生成 1k 条 proof-of-concept 训练样本数据
    - [x] 八卡训练模型保存 checkpoint
    - [ ] 分别 inference 并比较 E-SNLI 数据集 performance
    - [ ] 看论文思考下一步方向
