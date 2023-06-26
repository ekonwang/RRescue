### Jun23 - July23 一月工作计划 🚀

目前的语言模型虽然强大，但语言模型在生成文本时依然容易受到无关context的影响，或者生成不基于 context 的输出。我们希望将 explaination 与相应的 context 对齐，以保证最终 output 的良好效果。

## 运行

1. 首先安装对应 cuda 版本的 torch：

`pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116`

2. 安装 dev 版本 `transformers`

```
wget https://github.com/huggingface/transformers/archive/refs/heads/main.zip && unzip main.zip
mv ./transformers-main ./transformers
pip install -e ./transformers
```

3. 安装其他依赖

`pip install -r requirements.txt`

4. 开始数据生成，根据空闲 GPU 数量自行修改 NPROC 变量

```
cd data_generation
chmod +x response_gen.sh
./response_gen.sh
```


## 数据增强

- [ ]  SNLI 数据生成 (Jun23 - Jun25)
    - [x] 模型 Alpaca-lora-7B 或者更小的 Flan-T5 模型(可能不行因为无法对话).
    - [x] `response_gen` 模块修改.
    - [x] A40 单机单卡测试, 基于 beam search 生成若干回答.
    - [x] `scoring_responses` 模块修改，加入 SBERT.
    - [ ] 针对 `e-snli` 任务提升 Alpaca-7B 生成质量.
    - [ ] 测试 `data_generation` 模块.

- [ ]  训练 pipeline (Jun26 - Jun27)