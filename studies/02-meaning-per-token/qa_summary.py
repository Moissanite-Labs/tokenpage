import json, sys, collections, numpy as np
rows = [json.loads(l) for l in open(sys.argv[1])]; MAXM = int(sys.argv[2]) if len(sys.argv) > 2 else 10**9; rows = [r for r in rows if r["msg"] < MAXM]; F = ["title", "priority", "scope", "criteria", "dependency"]
K = {"title": 12, "priority": 4, "scope": 6, "criteria": 5, "dependency": 2}
g = collections.defaultdict(list)
for r in rows: g[(r["fmt"], r["legend"], r["field"])].append(r)
print("| format | legend | " + " | ".join(f"{f} acc" for f in F) + " | mean acc | bits recovered (sum over fields, H - CE of option probs) |")
print("|---" * (len(F) + 4) + "|")
for fmt, lg in dict.fromkeys((r["fmt"], r["legend"]) for r in rows):
    accs = [np.mean([x["pred"] == x["gold"] for x in g[(fmt, lg, f)]]) for f in F]
    bits = sum(max(0, np.log2(K[f]) + np.mean([np.log2(max(x["p_gold"], 1e-12)) for x in g[(fmt, lg, f)]])) for f in F)
    print(f"| {fmt} | {lg} | " + " | ".join(f"{a:.2f}" for a in accs) + f" | {np.mean(accs):.3f} | {bits:.2f} |")
print(f"\nchance acc: " + ", ".join(f"{f}={1/K[f]:.2f}" for f in F) + f"; n per cell = {len(g[(rows[0]['fmt'], rows[0]['legend'], 'title')])}")
