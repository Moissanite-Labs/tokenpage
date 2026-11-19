"""Mean Qwen2.5 tokens per message, split by the dependency field; 600 messages."""
import numpy as np
from transformers import AutoTokenizer
from messages import make, FORMATS
q = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct"); msgs = make(600)
for f, fn in FORMATS.items():
    n = {d: [len(q(fn(m))["input_ids"]) for m in msgs if m["dependency"] == d] for d in (0, 1)}
    print(f"{f} dep=0: {np.mean(n[0]):.1f} dep=1: {np.mean(n[1]):.1f}")
