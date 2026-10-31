"""Zero-shot reading test: can an instruct model answer field questions from each format? Multiple choice scored by
option log-likelihood (sum over option tokens) after a chat-templated prompt. Optional legend for ABIOS and G1."""
import sys, json, time, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from messages import make, FORMATS, TITLES, PRI, SCOPE, CRIT, GLYPH_LEGEND, ABIOS_LEGEND
MODEL = sys.argv[1]; NMSG = int(sys.argv[2]); OUT = sys.argv[3]
torch.set_num_threads(8)
tok = AutoTokenizer.from_pretrained(MODEL); model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
Q = {"title": ("What is the task's title?", [t[0] for t in TITLES]),
     "priority": ("What is the task's priority?", [p[0] for p in PRI]),
     "scope": ("Which module is in scope?", [s[0] for s in SCOPE]),
     "criteria": ("What is the acceptance criterion?", [c[0] for c in CRIT]),
     "dependency": ("Does the task depend on another task?", ["no", "yes"])}
CONDS = [(f, None) for f in FORMATS] + [("abios_refs", ABIOS_LEGEND), ("glyph_g1", GLYPH_LEGEND)]
msgs = make(600)[:NMSG]; t0 = time.time()
import os
done = set()
if os.path.exists(OUT):  # resume: skip rows already written
    for l in open(OUT):
        r = json.loads(l); done.add((r["fmt"], r["legend"], r["msg"], r["field"]))
fout = open(OUT, "a", buffering=1)
for fmt, legend in CONDS:
    for i, m in enumerate(msgs):
        text = FORMATS[fmt](m)
        for field, (q, opts) in Q.items():
            if (fmt, legend is not None, i, field) in done: continue
            user = (legend + "\n\n" if legend else "") + f"Message:\n{text}\n\nQuestion: {q}\nOptions: {', '.join(opts)}\nReply with exactly one option."
            prompt = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
            p_ids = tok(prompt)["input_ids"]; seqs = [p_ids + tok(o)["input_ids"] for o in opts]
            L = max(map(len, seqs)); ids = torch.full((len(seqs), L), tok.pad_token_id); att = torch.zeros_like(ids)
            for k, s in enumerate(seqs): ids[k, :len(s)] = torch.tensor(s); att[k, :len(s)] = 1
            with torch.no_grad(): lp = torch.log_softmax(model(input_ids=ids, attention_mask=att).logits.float(), -1)
            scores = [sum(lp[k, t - 1, s[t]].item() for t in range(len(p_ids), len(s))) for k, s in enumerate(seqs)]
            pred = int(np.argmax(scores)); probs = np.exp(np.array(scores) - max(scores)); probs /= probs.sum()
            fout.write(json.dumps(dict(fmt=fmt, legend=legend is not None, msg=i, field=field, gold=m[field], pred=pred,
                                       p_gold=float(probs[m[field]]), text=text)) + "\n")
    print(fmt, "legend" if legend else "", "done", round(time.time() - t0), "s", flush=True)
