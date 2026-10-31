"""Linear probes: bits of the 5 key fields recoverable from (a) hidden states, (b) bag-of-tokens baseline,
(c) English-trained probe transferred to each format. Bits = H(field) - held-out cross-entropy (bits), floored at 0."""
import sys, numpy as np, warnings
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
warnings.filterwarnings("ignore")
D = np.load(sys.argv[1] if len(sys.argv) > 1 else "states_qwen0.5b.npz"); LAYER = int(sys.argv[2]) if len(sys.argv) > 2 else 16
FIELDS = ["title", "priority", "scope", "criteria", "dependency"]; FMTS = ["english", "json", "abios_refs", "chinese", "glyph_g1"]
Y = D["labels"]; N = len(Y); H = [np.log2(len(np.unique(Y[:, j]))) for j in range(5)]  # uniform design -> H = log2(K)
kf = list(KFold(5, shuffle=True, random_state=0).split(np.arange(N)))

def ce_bits(P, y): return float(-np.mean(np.log2(np.clip(P[np.arange(len(y)), y], 1e-12, 1))))
def clf(): return LogisticRegression(C=0.05, max_iter=500)

def cv_bits(Xtr_all, Xte_all=None, center=False, pca=True):
    """Returns per-field (bits recovered, accuracy). If Xte_all given: train on Xtr_all[train], test on Xte_all[test]."""
    Xte_all = Xtr_all if Xte_all is None else Xte_all
    res = []
    for j in range(5):
        y = Y[:, j]; P = np.zeros((N, len(np.unique(y))))
        for tr, te in kf:
            A, B = Xtr_all[tr], Xte_all[te]
            if center: A = A - Xtr_all[tr].mean(0); B = B - Xte_all[tr].mean(0)  # unlabeled per-format centering
            m = (make_pipeline(StandardScaler(), PCA(128, random_state=0), clf()) if pca else make_pipeline(StandardScaler(), clf())).fit(A, y[tr]); P[te] = m.predict_proba(B)
        res.append((max(0.0, H[j] - ce_bits(P, y)), float((P.argmax(1) == y).mean())))
    return res

def bow_bits(ids):
    X = CountVectorizer(token_pattern=r"\S+", ngram_range=(1, 2)).fit_transform(ids).toarray().astype(float)
    return cv_bits(X, pca=False)

Htot = sum(H); print(f"probe: standardize -> PCA(128, fit on train fold) -> L2 logistic C=0.05; 5-fold CV\nlayer={LAYER}  N={N}  H_total={Htot:.3f} bits/message  per-field H={[round(h,3) for h in H]}\n")
print("| format | qwen tokens/msg | probe acc (title,pri,scope,crit,dep) | bits recovered (state probe) | bits/token (state) | bits recovered (bag-of-tokens) | EN->fmt transfer bits (raw) | EN->fmt transfer bits (centered) | transfer acc centered |")
print("|---|---|---|---|---|---|---|---|---|")
XE = D["X_english"][:, LAYER]
for f in FMTS:
    X = D[f"X_{f}"][:, LAYER]; nt = D[f"ntok_{f}"].mean()
    r = cv_bits(X); b = bow_bits(D[f"ids_{f}"])
    tr_raw = cv_bits(XE, X); tr_c = cv_bits(XE, X, center=True)
    bits = sum(x[0] for x in r)
    print(f"| {f} | {nt:.1f} | {', '.join(f'{x[1]:.2f}' for x in r)} | {bits:.2f} | {bits/nt:.3f} | {sum(x[0] for x in b):.2f} | "
          f"{sum(x[0] for x in tr_raw):.2f} | {sum(x[0] for x in tr_c):.2f} | {', '.join(f'{x[1]:.2f}' for x in tr_c)} |")
