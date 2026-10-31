"""Mean-pooled hidden states (all layers) for every message in every format. Message text only, no chat template."""
import sys, json, time, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from messages import make, FORMATS
MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B-Instruct"; N = int(sys.argv[2]) if len(sys.argv) > 2 else 600
OUT = sys.argv[3] if len(sys.argv) > 3 else "states_qwen0.5b.npz"
torch.set_num_threads(8); torch.manual_seed(0)
tok = AutoTokenizer.from_pretrained(MODEL); model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
msgs = make(N); out = {"labels": np.array([[m[k] for k in ["title", "priority", "scope", "criteria", "dependency"]] for m in msgs])}
t0 = time.time()
for fname, f in FORMATS.items():
    texts = [f(m) for m in msgs]; feats = []; ntok = []; ids_all = []
    for i in range(0, N, 32):
        batch = texts[i:i + 32]
        enc = tok(batch, return_tensors="pt", padding=True)
        with torch.no_grad(): hs = model(**enc, output_hidden_states=True).hidden_states
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = torch.stack([(h * mask).sum(1) / mask.sum(1) for h in hs], 1)  # [B, L+1, D]
        feats.append(pooled.numpy().astype(np.float32)); ntok += enc["attention_mask"].sum(1).tolist()
        ids_all += [" ".join(map(str, tok(t)["input_ids"])) for t in batch]
    out[f"X_{fname}"] = np.concatenate(feats); out[f"ntok_{fname}"] = np.array(ntok); out[f"ids_{fname}"] = np.array(ids_all)
    out[f"text_{fname}"] = np.array(texts)
    print(fname, "mean_tokens", np.mean(ntok), "elapsed", round(time.time() - t0), flush=True)
np.savez_compressed(OUT, **out); print("saved", OUT)
