import json

with open("/workspace/Rank/data/configs/gsm8k_example_prompts_instr.json", "r") as f:
    data = json.load(f)
print(data[0])
