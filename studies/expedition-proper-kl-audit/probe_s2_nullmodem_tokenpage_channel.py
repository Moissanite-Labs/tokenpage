#!/usr/bin/env python3
"""Wave 2 Symbiosis Probe S2:
  Part 1 — Live Moissanite-Labs/tokenpage PR #5 (Study 02: Qwen2.5-1.5B QA + Qwen2.5-0.5B
           L4..L24 Hidden-State Probes) Audit & Issue #9 Exact Proper-KL / Position-Bias Fix.
  Part 2 — Pure Zero-Float (Q / Q(i)) Exact Moissanite-Labs/nullmodem Sub-Text Channel Theorems
           (CHARTER Q2, Q4, Q5 & PR #3 RAM==WIRE==HASH):
           - Theorem NM-1: Exact G_obs Orthogonal-Gauge & PGL(2) Cross-Ratio Sub-Text Codes
           - Theorem NM-2: Exact 16x16 GKSL Lindblad-Zeno Context-Hash Anchoring & Modular Flow
           - Theorem NM-3: Exact Cross-Model Adapter Bregman/KL Hidden-Channel Latch Bound

100% pure Python 3 stdlib (fractions.Fraction; ZERO third-party dependencies; ZERO bare asserts;
ZERO floats in Part 2 exact theorems).

Run cleanly under:
    python3 -I -B -W error studies/expedition-proper-kl-audit/probe_s2_nullmodem_tokenpage_channel.py
    python3 -I -O -B -W error studies/expedition-proper-kl-audit/probe_s2_nullmodem_tokenpage_channel.py
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(f"Probe S2 verification failure: {msg}")


DATA_DIR = Path(
    os.environ.get("TOKENPAGE_PR5_DATA_DIR")
    or (Path(__file__).resolve().parent / "data" / "tokenpage_pr5")
)
OUT_PATH = Path(__file__).resolve().parent / "probe_s2_nullmodem_tokenpage_results.json"

K_MAP = {"title": 12, "priority": 4, "scope": 6, "criteria": 5, "dependency": 2}
TOKEN_COUNTS_QWEN = {"english": 28.4, "json": 34.9, "abios_refs": 36.6, "chinese": 29.8, "glyph_g1": 11.8}


# =============================================================================
# 0. STRICT R3 EXACT ARITHMETIC OVER Q AND Q(i)
# =============================================================================

def exact_frac(val: Any) -> Fraction:
    if isinstance(val, bool):
        raise TypeError("bool is not an exact rational")
    if isinstance(val, Fraction):
        return val
    if isinstance(val, int):
        return Fraction(val, 1)
    raise TypeError(f"Expected int or Fraction, got {type(val).__name__}")


class GaussQ:
    __slots__ = ("re", "im")

    def __init__(self, re: Any, im: Any = 0) -> None:
        self.re = exact_frac(re)
        self.im = exact_frac(im)

    def __add__(self, o: Any) -> "GaussQ":
        b = o if isinstance(o, GaussQ) else GaussQ(o, 0)
        return GaussQ(self.re + b.re, self.im + b.im)

    def __sub__(self, o: Any) -> "GaussQ":
        b = o if isinstance(o, GaussQ) else GaussQ(o, 0)
        return GaussQ(self.re - b.re, self.im - b.im)

    def __neg__(self) -> "GaussQ":
        return GaussQ(-self.re, -self.im)

    def __mul__(self, o: Any) -> "GaussQ":
        b = o if isinstance(o, GaussQ) else GaussQ(o, 0)
        return GaussQ(self.re * b.re - self.im * b.im, self.re * b.im + self.im * b.re)

    def conj(self) -> "GaussQ":
        return GaussQ(self.re, -self.im)

    def norm2(self) -> Fraction:
        return self.re * self.re + self.im * self.im

    def __truediv__(self, o: Any) -> "GaussQ":
        b = o if isinstance(o, GaussQ) else GaussQ(o, 0)
        n2 = b.norm2()
        if n2 == 0:
            raise ZeroDivisionError("Division by zero in Q(i)")
        num = self * b.conj()
        return GaussQ(num.re / n2, num.im / n2)

    def is_zero(self) -> bool:
        return self.re == 0 and self.im == 0

    def __eq__(self, o: Any) -> bool:
        if isinstance(o, GaussQ):
            return self.re == o.re and self.im == o.im
        if isinstance(o, bool):
            return False
        if isinstance(o, (int, Fraction)):
            return self.im == 0 and self.re == o
        return False


G0, G1, GI = GaussQ(0, 0), GaussQ(1, 0), GaussQ(0, 1)


def _sq_q(M: Sequence[Sequence[Any]]) -> List[List[Fraction]]:
    n = len(M)
    if n == 0:
        raise ValueError("Empty matrix")
    out: List[List[Fraction]] = []
    for row in M:
        if len(row) != n:
            raise ValueError("Non-square matrix")
        out.append([exact_frac(x) for x in row])
    return out


def matmul_q(A: Sequence[Sequence[Fraction]], B: Sequence[Sequence[Fraction]]) -> List[List[Fraction]]:
    m, k, n = len(A), len(A[0]), len(B[0])
    check(len(B) == k, "matmul_q dim mismatch")
    return [[sum((A[i][p] * B[p][j] for p in range(k)), Fraction(0)) for j in range(n)] for i in range(m)]


def matsub_q(A: Sequence[Sequence[Fraction]], B: Sequence[Sequence[Fraction]]) -> List[List[Fraction]]:
    return [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def transpose_q(A: Sequence[Sequence[Fraction]]) -> List[List[Fraction]]:
    return [[A[i][j] for i in range(len(A))] for j in range(len(A[0]))]


def matinv_q(A: Sequence[Sequence[Any]]) -> List[List[Fraction]]:
    M = _sq_q(A)
    n = len(M)
    aug = [M[i][:] + [Fraction(1 if i == j else 0) for j in range(n)] for i in range(n)]
    for col in range(n):
        piv = next((r for r in range(col, n) if aug[r][col] != 0), -1)
        if piv < 0:
            raise ValueError("Singular matrix over Q")
        if piv != col:
            aug[col], aug[piv] = aug[piv], aug[col]
        pv = aug[col][col]
        aug[col] = [x / pv for x in aug[col]]
        for r in range(n):
            if r != col and aug[r][col] != 0:
                fac = aug[r][col]
                aug[r] = [aug[r][c] - fac * aug[col][c] for c in range(2 * n)]
    return [row[n:] for row in aug]


def matmul_g(A: Sequence[Sequence[GaussQ]], B: Sequence[Sequence[GaussQ]]) -> List[List[GaussQ]]:
    m, k, n = len(A), len(A[0]), len(B[0])
    check(len(B) == k, "matmul_g dim mismatch")
    out: List[List[GaussQ]] = []
    for i in range(m):
        row: List[GaussQ] = []
        for j in range(n):
            acc = G0
            for p in range(k):
                if not A[i][p].is_zero() and not B[p][j].is_zero():
                    acc = acc + A[i][p] * B[p][j]
            row.append(acc)
        out.append(row)
    return out


def matadd_g(A: Sequence[Sequence[GaussQ]], B: Sequence[Sequence[GaussQ]]) -> List[List[GaussQ]]:
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def matsub_g(A: Sequence[Sequence[GaussQ]], B: Sequence[Sequence[GaussQ]]) -> List[List[GaussQ]]:
    return [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def dag_g(A: Sequence[Sequence[GaussQ]]) -> List[List[GaussQ]]:
    return [[A[i][j].conj() for i in range(len(A))] for j in range(len(A[0]))]


def trace_g(A: Sequence[Sequence[GaussQ]]) -> GaussQ:
    acc = G0
    for i in range(len(A)):
        acc = acc + A[i][i]
    return acc


def matinv_g(A: Sequence[Sequence[GaussQ]]) -> List[List[GaussQ]]:
    n = len(A)
    aug = [list(A[i]) + [G1 if i == j else G0 for j in range(n)] for i in range(n)]
    for col in range(n):
        piv = next((r for r in range(col, n) if not aug[r][col].is_zero()), -1)
        if piv < 0:
            raise ValueError("Singular matrix over Q(i)")
        if piv != col:
            aug[col], aug[piv] = aug[piv], aug[col]
        pv = aug[col][col]
        aug[col] = [x / pv for x in aug[col]]
        for r in range(n):
            if r != col and not aug[r][col].is_zero():
                fac = aug[r][col]
                aug[r] = [aug[r][c] - fac * aug[col][c] for c in range(2 * n)]
    return [row[n:] for row in aug]


def r3_negative_controls() -> Dict[str, Any]:
    rejected: Dict[str, str] = {}
    for name, fn in [
        ("exact_frac_bool", lambda: exact_frac(True)),
        ("exact_frac_float", lambda: exact_frac(0.25)),
        ("exact_frac_str", lambda: exact_frac("1/2")),
        ("GaussQ_bool", lambda: GaussQ(0, False)),
        ("GaussQ_float", lambda: GaussQ(1.5, 0)),
        ("matinv_ragged", lambda: matinv_q([[1, 0], [0]])),
        ("matinv_singular", lambda: matinv_q([[1, 2], [2, 4]])),
        ("GaussQ_div0", lambda: G1 / G0),
    ]:
        try:
            fn()
        except (TypeError, ValueError, ZeroDivisionError) as exc:
            rejected[name] = type(exc).__name__
        else:
            raise RuntimeError(f"R3 control {name} failed to raise")
    return {"all_r3_rejected": True, "exceptions": rejected}


# =============================================================================
# 1. PART 1 — LIVE Moissanite-Labs/tokenpage PR #5 AUDIT & ISSUE #9 FIX
# =============================================================================

def wilson_95(k: int, n: int) -> Tuple[float, float]:
    """Exact Wilson score 95% confidence interval (z = 1.959963984540054)."""
    check(0 <= k <= n and n > 0, "Invalid Wilson counts")
    z = 1.959963984540054
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)
    return max(0.0, center - half), min(1.0, center + half)


def discrete_mi_bits(golds: Sequence[int], preds: Sequence[int]) -> float:
    """Exact empirical discrete mutual information I(Y; Y_hat) in bits (option-order invariant!)."""
    n = len(golds)
    c_xy = Counter(zip(golds, preds))
    c_x = Counter(golds)
    c_y = Counter(preds)
    mi = 0.0
    for (x, y), c in c_xy.items():
        pxy = c / n
        mi += pxy * math.log2(pxy / ((c_x[x] / n) * (c_y[y] / n)))
    return mi


def analyze_tokenpage_pr5() -> Dict[str, Any]:
    qa_path = DATA_DIR / "raw_qa_qwen1.5b.jsonl"
    raw_bytes = qa_path.read_bytes()
    qa_sha = hashlib.sha256(raw_bytes).hexdigest()
    all_rows = [json.loads(ln) for ln in raw_bytes.decode("utf-8").splitlines() if ln.strip()]

    # Detect the unpaired extra English rows (msg 24..46, including truncated msg 46 with 2/5 fields!)
    en_extra = [r for r in all_rows if r["fmt"] == "english" and r["msg"] >= 24]
    en_msg46_fields = [r["field"] for r in all_rows if r["fmt"] == "english" and r["msg"] == 46]
    check(len(en_extra) == 112 and len(en_msg46_fields) == 2, "Expected 112 extra english rows & 2-field msg 46")

    # Restrict cleanly to the paired 24-message benchmark (msg 0..23) across all 7 cells
    paired_rows = [r for r in all_rows if 0 <= r["msg"] < 24]
    by_cell: Dict[Tuple[str, bool], List[Dict[str, Any]]] = defaultdict(list)
    for r in paired_rows:
        by_cell[(r["fmt"], bool(r["legend"]))].append(r)
    check(len(by_cell) == 7 and all(len(v) == 120 for v in by_cell.values()), "Expected 7 cells x 120 paired rows")

    # Verify exact upstream accuracy reproduction against results_qa_qwen1.5b.md
    upstream_acc = {
        ("english", False): 0.933,
        ("json", False): 1.000,
        ("abios_refs", False): 0.892,
        ("chinese", False): 0.958,
        ("glyph_g1", False): 0.558,
        ("abios_refs", True): 0.917,
        ("glyph_g1", True): 0.542,
    }

    cell_summaries: Dict[str, Any] = {}
    for (fmt, leg), rows in sorted(by_cell.items()):
        by_field: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in rows:
            by_field[r["field"]].append(r)
        k_tot = sum(1 for r in rows if r["pred"] == r["gold"])
        acc = k_tot / len(rows)
        check(abs(round(acc, 3) - upstream_acc[(fmt, leg)]) <= 0.0011, f"Upstream acc mismatch on {(fmt, leg)}")
        w_lo, w_hi = wilson_95(k_tot, len(rows))

        floored_unif = 0.0
        unfloored_unif = 0.0
        unfloored_emp = 0.0
        mi_sum = 0.0
        maj_acc_sum = 0.0
        field_details: Dict[str, Any] = {}
        for fld, K in K_MAP.items():
            frows = by_field[fld]
            n_f = len(frows)
            check(n_f == 24, f"Expected 24 messages for {fld}")
            golds = [int(r["gold"]) for r in frows]
            preds = [int(r["pred"]) for r in frows]
            kf = sum(1 for g, p in zip(golds, preds) if g == p)
            ce = -sum(math.log2(max(float(r["p_gold"]), 1e-12)) for r in frows) / n_f
            cg = Counter(golds)
            h_emp = -sum((c / n_f) * math.log2(c / n_f) for c in cg.values())
            maj_base = max(cg.values()) / n_f
            maj_acc_sum += maj_base
            mi_f = discrete_mi_bits(golds, preds)

            floored_unif += max(0.0, math.log2(K) - ce)
            unfloored_unif += math.log2(K) - ce
            unfloored_emp += h_emp - ce
            mi_sum += mi_f

            # Position 0/1 bias concentration
            p01_pred = sum(1 for p in preds if p in (0, 1))
            p01_gold = sum(1 for g in golds if g in (0, 1))
            field_details[fld] = {
                "acc": round(kf / n_f, 4),
                "majority_baseline": round(maj_base, 4),
                "above_majority": (kf / n_f) > maj_base,
                "H_emp_bits": round(h_emp, 4),
                "CE_bits": round(ce, 4),
                "proper_KL_skill_emp": round(h_emp - ce, 4),
                "discrete_MI_bits": round(mi_f, 4),
                "pred_in_pos01_of_24": p01_pred,
                "gold_in_pos01_of_24": p01_gold,
            }

        toks = TOKEN_COUNTS_QWEN[fmt]
        cell_summaries[f"{fmt}_legend_{leg}"] = {
            "mean_acc": round(acc, 4),
            "wilson_95_ci": [round(w_lo, 4), round(w_hi, 4)],
            "mean_majority_baseline": round(maj_acc_sum / len(K_MAP), 4),
            "floored_unif_bits_pr5": round(floored_unif, 4),
            "proper_unfloored_unif_bits": round(unfloored_unif, 4),
            "proper_unfloored_emp_bits": round(unfloored_emp, 4),
            "discrete_MI_bits_order_invariant": round(mi_sum, 4),
            "qwen_tokens_per_msg": toks,
            "proper_emp_bits_per_token": round(unfloored_emp / toks, 4),
            "discrete_MI_bits_per_token": round(mi_sum / toks, 4),
            "fields": field_details,
        }

    # Check the key Issue #9 proper-KL finding: floor INVERTED the legend effect on glyph_g1!
    g1_noleg = cell_summaries["glyph_g1_legend_False"]
    g1_leg = cell_summaries["glyph_g1_legend_True"]
    floor_delta_leg = g1_leg["floored_unif_bits_pr5"] - g1_noleg["floored_unif_bits_pr5"]
    proper_delta_leg = g1_leg["proper_unfloored_emp_bits"] - g1_noleg["proper_unfloored_emp_bits"]
    check(floor_delta_leg < -1.5 and proper_delta_leg > 1.5, "Expected floor vs proper-KL sign inversion on G1 legend")
    # Check title position-0/1 collapse in glyph_g1 with legend (19/24 vs 1/24 gold!)
    t_g1_leg = g1_leg["fields"]["title"]
    check(t_g1_leg["pred_in_pos01_of_24"] == 19 and t_g1_leg["gold_in_pos01_of_24"] == 1,
          f"Expected 19/24 pred vs 1/24 gold on G1+legend title, got {t_g1_leg}")

    # Parse layer-by-layer hidden-state gauge rotation (L4..L24)
    layer_curve: List[Dict[str, Any]] = []
    row_re = re.compile(
        r"^\|\s*(english|json|abios_refs|chinese|glyph_g1)\s*\|\s*([0-9.]+)\s*\|[^|]+\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|"
    )
    for L in (4, 8, 12, 16, 20, 24):
        txt = (DATA_DIR / f"results_probe_qwen0.5b_L{L}.txt").read_text(encoding="utf-8")
        entry: Dict[str, Any] = {"layer": L, "formats": {}}
        for ln in txt.splitlines():
            m = row_re.match(ln.strip())
            if m:
                fmt = m.group(1)
                state_bits = float(m.group(3))
                raw_tr = float(m.group(6))
                cen_tr = float(m.group(7))
                entry["formats"][fmt] = {
                    "state_bits": state_bits,
                    "bits_per_token": float(m.group(4)),
                    "EN_transfer_raw_bits": raw_tr,
                    "EN_transfer_centered_bits": cen_tr,
                    "gauge_alignment_ratio_centered": round(cen_tr / state_bits, 4),
                }
        check(len(entry["formats"]) == 5, f"Missing formats in L{L}")
        layer_curve.append(entry)

    return {
        "input_qa_sha256": qa_sha,
        "data_hygiene_finding_unpaired_english_rows": {
            "extra_rows_msg_ge_24": len(en_extra),
            "truncated_msg_46_fields": en_msg46_fields,
            "paired_common_messages_used": 24,
        },
        "paired_24msg_cells": cell_summaries,
        "issue9_proper_kl_floor_inversion": {
            "g1_legend_delta_floored_PR5_bits": round(floor_delta_leg, 4),
            "g1_legend_delta_proper_KL_emp_bits": round(proper_delta_leg, 4),
            "sign_inverted_by_nonproper_floor": True,
            "g1_legend_title_pos01_pred_vs_gold": [t_g1_leg["pred_in_pos01_of_24"], t_g1_leg["gold_in_pos01_of_24"]],
        },
        "layerwise_representation_gauge_rotation_L4_to_L24": layer_curve,
    }


# =============================================================================
# 2. PART 2 — PURE ZERO-FLOAT EXACT nullmodem SUB-TEXT CHANNEL THEOREMS (Q / Q(i))
# =============================================================================

def householder_q(v: Sequence[Fraction]) -> List[List[Fraction]]:
    vf = [exact_frac(x) for x in v]
    n = len(vf)
    v2 = sum((x * x for x in vf), Fraction(0))
    check(v2 > 0, "Zero Householder vector")
    return [[Fraction(1 if i == j else 0) - Fraction(2) * vf[i] * vf[j] / v2 for j in range(n)] for i in range(n)]


def mobius_cross_ratio_q(z1: Fraction, z2: Fraction, z3: Fraction, z4: Fraction) -> Fraction:
    a, b, c, d = exact_frac(z1), exact_frac(z2), exact_frac(z3), exact_frac(z4)
    return ((a - c) * (b - d)) / ((a - d) * (b - c))


def theorem_nm1_gauge_invariant_subtext() -> Dict[str, Any]:
    """Exact G_obs gauge-invariant sub-text channel over Q (zero floats)."""
    # 4 latent message vectors in Q^4 representing a sub-text packet
    Z = [
        [Fraction(1), Fraction(-2), Fraction(3), Fraction(1, 2)],
        [Fraction(2), Fraction(1), Fraction(-1), Fraction(3, 2)],
        [Fraction(-1), Fraction(3), Fraction(2), Fraction(-1, 2)],
        [Fraction(0), Fraction(1, 2), Fraction(-2), Fraction(5, 2)],
    ]
    # Exact orthogonal cross-model gauge rotation O in O(4, Q) (composition of 2 Householders)
    O1 = householder_q([Fraction(1), Fraction(-2), Fraction(1), Fraction(3)])
    O2 = householder_q([Fraction(2), Fraction(1), Fraction(-3), Fraction(1)])
    O = matmul_q(O1, O2)
    I4 = [[Fraction(1 if i == j else 0) for j in range(4)] for i in range(4)]
    check(matmul_q(transpose_q(O), O) == I4, "O is not orthogonal over Q")

    # Apply cross-model gauge twist Z_B = Z @ O^T
    Z_B = matmul_q(Z, transpose_q(O))
    raw_sq_drift = sum(((Z_B[i][j] - Z[i][j]) ** 2 for i in range(4) for j in range(4)), Fraction(0))
    check(raw_sq_drift > 0, "Gauge twist must move raw coordinates")

    # Exact Gram sub-text invariant K = Z @ Z^T
    K_A = matmul_q(Z, transpose_q(Z))
    K_B = matmul_q(Z_B, transpose_q(Z_B))
    check(K_A == K_B, "Exact Gram sub-text code failed O(4, Q) invariance")

    # Projective PGL(2, Q) cross-ratio invariance for 1D scalar latent channels
    pts = [Fraction(1, 3), Fraction(2), Fraction(-3, 2), Fraction(5)]
    cr_orig = mobius_cross_ratio_q(*pts)
    # Fractional linear transform z -> (3z + 2) / (5z + 7), det = 21 - 10 = 11 != 0
    pts_tr = [(3 * z + 2) / (5 * z + 7) for z in pts]
    cr_tr = mobius_cross_ratio_q(*pts_tr)
    check(cr_orig == cr_tr, "PGL(2, Q) cross-ratio invariance failed")

    # Negative control: non-orthogonal shear S in GL(4, Q) \ O(4) breaks Gram code!
    S = [[Fraction(1 if i == j else 0) for j in range(4)] for i in range(4)]
    S[0][1] = Fraction(1, 2)
    Z_shear = matmul_q(Z, transpose_q(S))
    K_shear = matmul_q(Z_shear, transpose_q(Z_shear))
    shear_diff_sq = sum(((K_shear[i][j] - K_A[i][j]) ** 2 for i in range(4) for j in range(4)), Fraction(0))
    check(shear_diff_sq > 0, "Non-orthogonal shear control must break orthogonal Gram invariant")

    return {
        "raw_coordinate_drift_sq_under_O4Q": str(raw_sq_drift),
        "gram_subtext_code_exact_O4Q_residual": "0",
        "pgl2_cross_ratio_exact_value": str(cr_orig),
        "pgl2_cross_ratio_exact_residual": "0",
        "neg_ctrl_nonorthogonal_shear_gram_drift_sq": str(shear_diff_sq),
    }


def theorem_nm2_zeno_context_hash_channel() -> Dict[str, Any]:
    """Exact 16x16 Q(i) GKSL Lindblad-Zeno theorem for nullmodem#3 RAM==WIRE==HASH context anchoring."""
    pauli = {
        "I": [[G1, G0], [G0, G1]],
        "X": [[G0, G1], [G1, G0]],
        "Y": [[G0, -GI], [GI, G0]],
        "Z": [[G1, G0], [G0, -G1]],
    }

    def kron(A: List[List[GaussQ]], B: List[List[GaussQ]]) -> List[List[GaussQ]]:
        return [
            [A[i][j] * B[k][l] for j in range(2) for l in range(2)]
            for i in range(2)
            for k in range(2)
        ]

    def scale(c: GaussQ, M: List[List[GaussQ]]) -> List[List[GaussQ]]:
        return [[c * x for x in row] for row in M]

    labels = [f"{a}{b}" for a in ("I", "Z") for b in "IXYZ"] + [f"{a}{b}" for a in ("X", "Y") for b in "IXYZ"]
    half = GaussQ(Fraction(1, 2), 0)
    basis = [scale(half, kron(pauli[l[0]], pauli[l[1]])) for l in labels]

    def op(lbl: str) -> List[List[GaussQ]]:
        return kron(pauli[lbl[0]], pauli[lbl[1]])

    def build_L(gamma_ctx: Fraction, g: Fraction, gamma_tid: Fraction, ctx_jump: str = "ZI") -> List[List[Fraction]]:
        H = scale(GaussQ(g, 0), op("XZ"))
        jumps = [(gamma_ctx, op(ctx_jump)), (gamma_tid, op("ZZ"))]
        M = [[Fraction(0) for _ in range(16)] for _ in range(16)]
        for col in range(16):
            rho = basis[col]
            img = scale(-GI, matsub_g(matmul_g(H, rho), matmul_g(rho, H)))
            for gam, L in jumps:
                if gam == 0:
                    continue
                Ld = dag_g(L)
                LdL = matmul_g(Ld, L)
                diss = matsub_g(
                    matmul_g(matmul_g(L, rho), Ld),
                    scale(half, matadd_g(matmul_g(LdL, rho), matmul_g(rho, LdL))),
                )
                img = matadd_g(img, scale(GaussQ(gam, 0), diss))
            for row in range(16):
                v = trace_g(matmul_g(basis[row], img))
                check(v.im == 0, "Superoperator not Hermiticity-preserving")
                M[row][col] = v.re
        return M

    def schur_inv(L: List[List[Fraction]]) -> List[List[Fraction]]:
        Lii = [[L[r][c] for c in range(8)] for r in range(8)]
        Liw = [[L[r][c] for c in range(8, 16)] for r in range(8)]
        Lwi = [[L[r][c] for c in range(8)] for r in range(8, 16)]
        Lww = [[L[r][c] for c in range(8, 16)] for r in range(8, 16)]
        return matsub_q(Lii, matmul_q(matmul_q(Liw, matinv_q(Lww)), Lwi))

    g_leak = Fraction(3, 2)
    g_tid = Fraction(1, 50)
    gamma_ctx = Fraction(25, 1)  # Strong context-hash anchor (RAM==WIRE==HASH witness)
    L_good = build_L(gamma_ctx, g_leak, g_tid, ctx_jump="ZI")
    S_good = schur_inv(L_good)
    expected_ix = -(2 * g_tid + 2 * g_leak * g_leak / gamma_ctx)
    check(S_good[1][1] == expected_ix and S_good[2][2] == expected_ix, "Exact Zeno Schur rate mismatch")
    check(S_good[3][3] == 0, "Readable audit pointer IZ must have exact 0 drift")

    # Exact Tomita-Takesaki modular commutation on non-commuting faithful rho_S in V_inv
    p0, p1 = Fraction(2, 5), Fraction(3, 5)
    proj0 = [[G1, G0], [G0, G0]]
    proj1 = [[G0, G0], [G0, G1]]
    rho0 = scale(half, matadd_g(pauli["I"], matadd_g(scale(GaussQ(Fraction(1, 3), 0), pauli["X"]), scale(GaussQ(Fraction(1, 4), 0), pauli["Z"]))))
    rho1 = scale(half, matadd_g(pauli["I"], matadd_g(scale(GaussQ(Fraction(1, 5), 0), pauli["Y"]), scale(GaussQ(Fraction(-1, 3), 0), pauli["Z"]))))
    rho_S = matadd_g(scale(GaussQ(p0, 0), kron(proj0, rho0)), scale(GaussQ(p1, 0), kron(proj1, rho1)))
    rho_S_inv = matinv_g(rho_S)
    Delta = [[G0 for _ in range(16)] for _ in range(16)]
    for col in range(16):
        img = matmul_g(matmul_g(rho_S, basis[col]), rho_S_inv)
        for row in range(16):
            Delta[row][col] = trace_g(matmul_g(basis[row], img))
    D_ctx_g = [[GaussQ(x, 0) for x in row] for row in build_L(Fraction(1), Fraction(0), Fraction(0), ctx_jump="ZI")]
    comm = matsub_g(matmul_g(D_ctx_g, Delta), matmul_g(Delta, D_ctx_g))
    check(all(x.is_zero() for row in comm for x in row), "[D_ctx, Delta_rho] must vanish identically on V_inv")

    # Negative controls: wrong-basis hashing (I x Z destroys message coherence; Z x X destroys audit pointer IZ!)
    L_bad_loc = build_L(gamma_ctx, g_leak, g_tid, ctx_jump="IZ")
    L_bad_mix = build_L(gamma_ctx, g_leak, g_tid, ctx_jump="ZX")
    good_ix_rate = -S_good[1][1]
    bad_ix_rate = -L_bad_loc[1][1]
    bad_iz_rate = -L_bad_mix[3][3]
    check(bad_ix_rate > 200 * good_ix_rate, "Wrong-basis local hash must destroy Zeno protection")
    check(bad_iz_rate == 2 * gamma_ctx, "Entangling hash must destroy readable audit pointer IZ")

    return {
        "context_anchored_IX_IY_zeno_rate_exact": str(good_ix_rate),
        "readable_audit_pointer_IZ_drift_exact": str(-S_good[3][3]),
        "modular_flow_commutator_exact_zero": True,
        "neg_ctrl_local_hash_IZ_degradation_factor": float(bad_ix_rate / good_ix_rate),
        "neg_ctrl_entangling_hash_ZX_corrupts_audit_IZ": str(bad_iz_rate),
    }


def theorem_nm3_adapter_bregman_latch_bound() -> Dict[str, Any]:
    """Exact Q Euclidean & Cubic-Bregman Hidden-Channel Latch Theorem for cross-model latent adapters."""
    # Audit probe matrix Phi (m=4 outputs, p=2 audited adapter parameters)
    Phi = [
        [Fraction(1), Fraction(0)],
        [Fraction(1), Fraction(1)],
        [Fraction(1), Fraction(2)],
        [Fraction(1), Fraction(3)],
    ]
    # True target behavior p_true = Phi @ theta_0 + psi_covert (outside Col(Phi)!)
    theta_0 = [Fraction(2), Fraction(1)]
    psi_covert = [Fraction(1), Fraction(-2), Fraction(1), Fraction(0)]  # Check Phi^T psi_covert == [0, 0]!
    PhiT = transpose_q(Phi)
    check(
        [sum(PhiT[i][j] * psi_covert[j] for j in range(4)) for i in range(2)] == [Fraction(0), Fraction(0)],
        "Covert payload must lie strictly in ker(Phi^T)",
    )

    # 1. Exact Euclidean Latch: q* = Phi @ theta_0 has ZERO audit gradient, yet misses covert channel!
    #    Shifting inside Col(Phi) to silence an outside anchor a1 strictly inflates global residual by exact Pythagoras!
    q_star = [sum(Phi[i][j] * theta_0[j] for j in range(2)) for i in range(4)]
    p_true = [q_star[i] + psi_covert[i] for i in range(4)]
    # Outside anchor a1 = [1, 0, 0, 0]: has error e1 = <a1, p_true - q_star> = 1
    a1 = [Fraction(1), Fraction(0), Fraction(0), Fraction(0)]
    e1 = sum(a1[i] * (p_true[i] - q_star[i]) for i in range(4))
    G_phi = matmul_q(PhiT, Phi)
    G_phi_inv = matinv_q(G_phi)
    # Projection p1 of a1 onto Col(Phi): p1 = Phi G^{-1} Phi^T a1
    PhiT_a1 = [sum(PhiT[i][j] * a1[j] for j in range(4)) for i in range(2)]
    u1 = [sum(G_phi_inv[i][j] * PhiT_a1[j] for j in range(2)) for i in range(2)]
    p1 = [sum(Phi[i][j] * u1[j] for j in range(2)) for i in range(4)]
    p1_norm2 = sum(x * x for x in p1)
    # Minimum-output-norm adaptation inside Col(Phi) silencing a1: q_adapt = q_star + (e1 / ||p1||^2) p1
    q_adapt = [q_star[i] + (e1 / p1_norm2) * p1[i] for i in range(4)]
    check(sum(a1[i] * (p_true[i] - q_adapt[i]) for i in range(4)) == 0, "a1 not silenced")
    r_star_sq = sum((p_true[i] - q_star[i]) ** 2 for i in range(4))
    r_adapt_sq = sum((p_true[i] - q_adapt[i]) ** 2 for i in range(4))
    check(r_adapt_sq == r_star_sq + (e1 * e1) / p1_norm2, "Exact Euclidean Latch Pythagoras failed")

    # 2. Exact Nonlinear Cubic-Bregman Latch (F(q) = sum q_i^3, nabla F(q)_i = 3 q_i^2, exact over Q!):
    #    Choose e-flat family nabla F(q_eta) = Phi @ eta (i.e. 3 q_i^2 in Col(Phi)) on positive states!
    #    Let q_b_star = [2, 2, 2, 2] -> nabla F = [12, 12, 12, 12] = 12 * col_0(Phi) in Col(Phi).
    #    Pick p_b = q_b_star + psi_covert = [3, 0, 3, 2]? Use smaller covert [1/2, -1, 1/2, 0] so p_b > 0!
    psi_small = [Fraction(1, 2), Fraction(-1), Fraction(1, 2), Fraction(0)]
    q_b_star = [Fraction(2), Fraction(2), Fraction(2), Fraction(2)]
    p_b = [q_b_star[i] + psi_small[i] for i in range(4)]
    # Another point q_b_adapt in e-flat family: 3 q_i^2 = 12 + 3*0? Need square values in Col(Phi):
    # Pick eta = [12, 0] -> q_i^2 = 4 (q_i=2); pick eta2 = [3, 9] -> 3 q^2 = [3, 12, 21, 30] not all squares.
    # Instead use eta2 with exact rational squares in arithmetic progression: (1, 5, 7) has 1^2, 5^2, 7^2 = 1, 25, 49 (step 24)!
    # Even cleaner: use Phi_b with 3 coordinates [1, 1, 1], [0, 1, 2] where (1, 5, 7)^2 = (1, 25, 49) IS in Col(Phi_b)!
    # Wait: for m=3, psi_b = [1/2, -1, 1/2] is ALSO orthogonal to [1,1,1] and [0,1,2]!
    p3 = [Fraction(5, 2), Fraction(1), Fraction(5, 2)]  # = [2,2,2] + [1/2, -1, 1/2], strictly positive!
    q3_star = [Fraction(2), Fraction(2), Fraction(2)]   # 3 q^2 = [12, 12, 12] in span{[1,1,1],[0,1,2]}
    q3_adapt = [Fraction(1), Fraction(5), Fraction(7)]  # 3 q^2 = [3, 75, 147] = 3*[1,1,1] + 72*[0,1,2] in span!

    def D_cubic(u: Sequence[Fraction], v: Sequence[Fraction]) -> Fraction:
        return sum((ui ** 3 - vi ** 3 - 3 * vi * vi * (ui - vi) for ui, vi in zip(u, v)), Fraction(0))

    d_pa = D_cubic(p3, q3_adapt)
    d_ps = D_cubic(p3, q3_star)
    d_sa = D_cubic(q3_star, q3_adapt)
    check(d_pa == d_ps + d_sa and d_sa > 0, "Exact nonlinear cubic-Bregman Latch Pythagoras failed")

    return {
        "euclidean_r_star_sq": str(r_star_sq),
        "euclidean_r_adapt_sq": str(r_adapt_sq),
        "euclidean_pythagorean_inflation": str((e1 * e1) / p1_norm2),
        "euclidean_pythagoras_residual": "0",
        "nonlinear_cubic_bregman_D_p_qadapt": str(d_pa),
        "nonlinear_cubic_bregman_D_p_qstar": str(d_ps),
        "nonlinear_cubic_bregman_D_qstar_qadapt": str(d_sa),
        "nonlinear_cubic_bregman_pythagoras_residual": "0",
    }


def main() -> None:
    r3 = r3_negative_controls()
    part1 = analyze_tokenpage_pr5()
    part2 = {
        "exact_arithmetic": "100% pure fractions.Fraction / Q(i) (zero floats)",
        "theorem_NM1_gauge_invariant_subtext": theorem_nm1_gauge_invariant_subtext(),
        "theorem_NM2_zeno_context_hash_channel": theorem_nm2_zeno_context_hash_channel(),
        "theorem_NM3_adapter_bregman_latch_bound": theorem_nm3_adapter_bregman_latch_bound(),
    }

    payload: Dict[str, Any] = {
        "probe": "probe_s2_nullmodem_tokenpage_channel",
        "targets_upstream": [
            "Moissanite-Labs/tokenpage PR #5 & Issue #9 (Study 02: Qwen2.5-1.5B QA + Qwen2.5-0.5B L4..L24)",
            "Moissanite-Labs/nullmodem CHARTER Q2/Q4/Q5 & PR #3 (RAM==WIRE==HASH)",
        ],
        "r3_negative_controls": r3,
        "part1_live_tokenpage_pr5_audit": part1,
        "part2_exact_nullmodem_theorems": part2,
    }

    canonical = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    OUT_PATH.write_text(canonical, encoding="utf-8")
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    print("=== PROBE S2 COMPLETE: TOKENPAGE PR #5 AUDIT & EXACT NULLMODEM THEOREMS ===")
    inv = part1["issue9_proper_kl_floor_inversion"]
    print(f"Part 1 (tokenpage#5/#9): unpaired english extra rows={part1['data_hygiene_finding_unpaired_english_rows']['extra_rows_msg_ge_24']} (msg 46 truncated 2/5)")
    print(f"  G1 legend floor inversion: floored delta={inv['g1_legend_delta_floored_PR5_bits']:+.2f} bits vs proper KL delta={inv['g1_legend_delta_proper_KL_emp_bits']:+.2f} bits!")
    print(f"  G1+legend title pos01 bias: {inv['g1_legend_title_pos01_pred_vs_gold'][0]}/24 pred vs {inv['g1_legend_title_pos01_pred_vs_gold'][1]}/24 gold")
    print(f"Part 2 (exact Q/Q(i) nullmodem): NM-1 Gram/PGL2 exact=0 | NM-2 Zeno rate={part2['theorem_NM2_zeno_context_hash_channel']['context_anchored_IX_IY_zeno_rate_exact']}, IZ=0 | NM-3 Bregman residual=0")
    print(f"Saved: {OUT_PATH}")
    print(f"File SHA-256: {digest}")


if __name__ == "__main__":
    main()
