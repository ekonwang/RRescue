### Jun23 - July23 一月工作计划 🚀

目前的语言模型虽然强大，但语言模型在生成文本时依然容易受到无关context的影响，或者生成不基于 context 的输出。我们希望将 explaination 与相应的 context 对齐，以保证最终 output 的良好效果。

## 数据增强

- [ ]  SNLI 数据生成 (Jun23 - Jun25)
    - [x] 模型 Alpaca-lora-7B 或者更小的 Flan-T5 模型(可能不行因为无法对话).
    - [x] `response_gen` 模块修改.
    - [x] A40 单机单卡测试, 基于 beam search 生成若干回答.
    - [x] `scoring_responses` 模块修改，加入 SBERT.
    - [ ] 针对 `e-snli` 任务提升 Alpaca-7B 生成质量.
    - [ ] 测试 `data_generation` 模块.

- [ ]  训练 pipeline (Jun26 - Jun27)