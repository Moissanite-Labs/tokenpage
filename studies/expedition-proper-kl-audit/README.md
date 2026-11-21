# Expedition Study: Proper-KL Legend Audit (PR `#5` / Issue `#9`) & Exact Zero-Float Sub-Text Channel Theorems

- **Target:** `Moissanite-Labs/tokenpage` PR `#5` (`study/02-meaning-per-token`, commit `4d28be2`) and Issue `#9` (plus exact algebraic bridge to `Moissanite-Labs/nullmodem` CHARTER Q2/Q4/Q5 & PR `#3`).
- **Status:** `analysis-verified` (independent R1–R5 verification audit `AUDIT-024`; 100% pure Python 3 stdlib, zero external dependencies, zero bare `assert`s, bit-identical across all 4 isolated `-I` interpreter modes).
- **Disclosure:** Scripts, exact rational proofs, and audit outputs were produced and independently audited with AI assistance as part of the Foundation-0 / Moissanite-Labs cross-repo verification sprint.

---

## 1. Live Audit of `tokenpage` PR `#5` (`4d28be2`) & Resolution of Issue `#9`

The committed PR `#5` raw artifacts (`raw_qa_qwen1.5b.jsonl` and `results_probe_qwen0.5b_L{4,8,12,16,20,24}.txt`, ~`260 KB` total) are bundled read-only under `data/tokenpage_pr5/` so this audit runs 100% standalone prior to PR `#5` merging.

### Findings
1. **Unpaired Data Hygiene Catch (`raw_qa_qwen1.5b.jsonl`):**
   - `raw_qa_qwen1.5b.jsonl` contains **`112` unpaired extra `english` rows** (`msg 24..46`, including truncated `msg = 46` with only `2/5` fields), while all other 6 cells (`json`, `abios_refs`, `chinese`, `glyph_g1`, `abios_refs+legend`, `glyph_g1+legend`) cover `msg 0..23`.
   - Filtering strictly to the common 24 paired messages (`msg 0..23`) reproduces PR `#5`'s committed `results_qa_qwen1.5b.md` table verbatim across all 7 cells.
2. **Non-Proper Floor Inverts the Legend Effect on `glyph_g1` (Sharpening Issue `#9`):**
   - PR `#5`'s per-field metric `max(0, log2 K - CE)` is **not a proper scoring rule**; clipping negative per-field scores reported that adding the legend *hurt* `glyph_g1` by **`-1.97 bits`** (`+2.58 -> +0.62`).
   - Evaluating the **exact proper (unfloored) empirical KL skill** $\sum_f \bigl(H_{\mathrm{emp}}(Y_f) - \mathrm{CE}_f\bigr)$ shows `glyph_g1` is severely overconfident in both cells (`-6.65` bits without legend vs **`-4.88`** bits with legend) — adding the legend actually **improves calibration by `+1.7670 bits`** (`min p_gold = 2.43e-6`, so `+1.7670` bits holds verbatim with zero probability clipping!).
   - Supplemented with order-invariant discrete mutual information $I(Y; \hat Y)$, Wilson 95% confidence intervals, and per-field majority baselines.
3. **Position-0/1 Collapse on `glyph_g1 + legend`:**
   - For the 12-option `title` field, `glyph_g1 + legend` collapses predictions onto option positions `0` or `1` on **`19/24` messages vs `1/24` gold** (confirming Issue `#9`'s readout/order-bias diagnosis and motivating mandatory $\mathcal{S}_K$ option-permutation symmetrization).
4. **Cross-Format Hidden-State Gauge Rotation (`Qwen2.5-0.5B`, `L4..L24`):**
   - Across all 6 live probe layers, `glyph_g1` retains **`11.32–11.40 / 11.49` bits** (`0.962–0.970 bits/tok`) of linearly decodable meaning, whereas raw English-probe transfer collapses to **`0.00` bits** at `L8` and `L12` (`3.69–4.79` bits after mean-centering) — evidencing an inter-format representation gauge rotation rather than information destruction.

---

## 2. Exact Zero-Float Sub-Text Channel Theorems (`nullmodem` Bridge, Pure $\mathbb{Q} / \mathbb{Q}(i)$)

Part 2 of `probe_s2_nullmodem_tokenpage_channel.py` establishes three zero-float rational theorems (with `8/8` strict R3 type/domain negative controls):

- **Theorem NM-1 ($\mathcal{G}_{\mathrm{obs}}$ Gauge-Invariant Sub-Text Codes):** Exact Householder $O(4, \mathbb{Q})$ cross-model basis rotations shift raw coordinates strictly (`> 0`), while exact Gram inner-product codes $K = Z Z^\top$ and 1D $PGL(2, \mathbb{Q})$ cross-ratios remain invariant with exact **`0`** residual; a non-orthogonal shear negative control breaks Gram invariance.
- **Theorem NM-2 (Exact $16 \times 16$ Quantum-Channel GKSL Structural Analogue of `nullmodem#3` Context-Hash Anchoring):** Explicitly scoped as a **quantum-channel GKSL structural analogue** of `nullmodem#3`'s classical `RAM == WIRE == HASH` UDP gossip (not a physical claim about classical UDP packets): strong context anchoring $\gamma_{\mathrm{ctx}} \mathcal{D}[Z_{\mathrm{ctx}} \otimes I_{\mathrm{msg}}]$ ($\gamma_{\mathrm{ctx}} = 25, g = 3/2, \gamma_{\mathrm{tid}} = 1/50$) commutes identically with Tomita–Takesaki modular flow $[\mathcal{D}_{\mathrm{ctx}}, \Delta_{\rho_S}] \equiv 0$, Zeno-seals latent message coherence at exact rational Schur rate `11/50`, and holds audit pointer `IZ` at exact **`0`** drift; wrong-basis local hash (`I x Z`) degrades coherence by `> 200x` and entangling hash (`Z x X`) corrupts `IZ` (`0 -> 50`).
- **Theorem NM-3 (Exact Cross-Model Adapter Bregman/KL Hidden-Channel Latch):** Proves over exact `Fraction` that both Euclidean and nonlinear cubic-Bregman ($F(q) = \sum q_i^3$) adapters admit covert perturbations in $\ker(\Phi_{\mathrm{audit}}^\top)$ with `0` audit gradient and exact Pythagorean inflation $D_F(p \| q_{\mathrm{adapt}}) = D_F(p \| q^*) + D_F(q^* \| q_{\mathrm{adapt}})$ (`0` residual).

---

## 3. How to Rerun (4-Mode Isolated R4 Parity)

From the repository root (pure Python 3 stdlib, zero external dependencies):

```bash
for flags in "-I -B" "-I -O -B" "-I -B -W error" "-I -O -B -W error"; do
  python3 $flags studies/expedition-proper-kl-audit/probe_s2_nullmodem_tokenpage_channel.py
done
```

Expected deterministic artifact SHA-256 digest:
- `probe_s2_nullmodem_tokenpage_results.json`: `b0cba027918e623b6f1169499eb49693aa47bede08c0c58411542bd91c9408f8`

---

## 4. Limits

- **Sample size:** PR `#5`'s paired QA evaluation contains `24` messages per cell (`Qwen2.5-1.5B`) and `6` hidden-state probe checkpoints (`Qwen2.5-0.5B`); full $\mathcal{S}_K$ option-permutation symmetrization requires rerunning model inference on permuted option orders (not recoverable post-hoc from single-order logs).
- **Structural scope of NM-2:** Theorem NM-2 is an exact algebraic operator model over $\mathbb{Q}(i)$ showing how commuting pointer-basis anchoring decouples audit observables from latent channels; it serves as a structural analogue rather than a direct simulation of `nullmodem`'s classical UDP transport.
