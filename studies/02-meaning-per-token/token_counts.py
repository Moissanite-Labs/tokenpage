"""Token counts per format under three tokenizers (Qwen2.5, o200k_base, cl100k_base); 600 messages."""
import numpy as np, tiktoken
from transformers import AutoTokenizer
from messages import make, FORMATS
q = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct"); o = tiktoken.get_encoding("o200k_base"); c = tiktoken.get_encoding("cl100k_base")
msgs = make(600); H = 11.492
print("| format | qwen2.5 tok | o200k tok | cl100k tok | H/qwen tok (bits/token if fully decodable) | H/o200k tok |"); print("|---|---|---|---|---|---|")
for f, fn in FORMATS.items():
    t = [fn(m) for m in msgs]; a = np.mean([len(q(x)["input_ids"]) for x in t]); b = np.mean([len(o.encode(x)) for x in t]); d = np.mean([len(c.encode(x)) for x in t])
    print(f"| {f} | {a:.1f} | {b:.1f} | {d:.1f} | {H/a:.3f} | {H/b:.3f} |")
