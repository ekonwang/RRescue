import json
import os
import sys

with open(sys.argv[1], "r") as f:
    samples = json.load(f)
print(f"number of samples {len(samples)}")
os.makedirs(os.path.dirname(sys.argv[2]), exist_ok=True)
with open(sys.argv[2], "w") as f:
    new_samples = samples[: len(samples) // 10]
    json.dump(new_samples, f, indent=4)
print(f"number of new samples {len(new_samples)}")
