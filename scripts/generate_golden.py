"""
Generate golden/golden.json from the oracle.

Each case records: id, category (discriminating | control | edge), the twist it
targets, inputs, expected output (oracle, 4 decimals), the called genotype
(number of "x" alleles, for reviewers only), the naive baseline (both twists
off), each single-twist ablation (t1_off, t2_off), the recall baseline
(MAQ-literal) and a short note.

Usage:  python scripts/generate_golden.py
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "oracle"))
sys.path.insert(0, os.path.join(ROOT, "grader"))

import oracle  # noqa: E402
import baselines  # noqa: E402

Q40x6 = [40, 40, 40, 40, 40, 40]
Q30x8 = [30] * 8

# (id, category, twist, value_a, value_b, value_c, value_d, note)
CASES = [
    # --- Twist 1: base quality capped by mapping quality ---
    ("D1", "discriminating", "T1", Q40x6, "xxxxyy", 20, 0.001,
     "Cap 40 -> 20 flips the call from het to hom; naive stays het"),
    ("D2", "discriminating", "T1", Q40x6, "xxxxyy", 10, 0.001,
     "Deeper cap: output is not monotone in value_c"),
    ("D3", "discriminating", "T1", [35, 38, 40, 25, 30], "xxyyy", 15, 0.2,
     "All reads capped to 15: raw qualities become irrelevant"),
    ("D4", "discriminating", "T1", [45, 20, 50, 15, 38, 42], "xyxyxy", 25, 0.05,
     "Partial cap: only reads with Q > 25 change"),
    ("C1", "control", "T1", Q40x6, "xxxxyy", 60, 0.001,
     "value_c above every quality: cap inactive (pair with D1)"),
    ("C2", "control", "T1", [20, 25, 18], "xyy", 30, 0.001,
     "value_c above every quality, different pileup"),
    # --- Twist 2: error dependency inside each allele group ---
    ("D5", "discriminating", "T2", Q30x8, "xxxxyyyy", 60, 0.001,
     "4+4 equal reads: independent model overstates het quality"),
    ("D6", "discriminating", "T2", Q30x8, "xxxxxyyy", 60, 0.001,
     "5+3 equal reads"),
    ("D7", "discriminating", "T2", [35, 32, 28, 25, 22, 20, 38, 36], "xxxxxyyy", 60, 0.001,
     "Unequal qualities: sort order inside a group matters"),
    ("D8", "discriminating", "T2", [30, 30, 30, 30, 30, 30], "xxxxyy", 60, 0.2,
     "Known-site prior r = 0.2"),
    ("C3", "control", "T2", [30, 30], "xy", 60, 0.001,
     "One read per group: rank 0 only, dependency inactive"),
    ("C4", "control", "T2", [35], "y", 60, 0.2,
     "Single read: dependency inactive"),
    # --- Interaction: both twists on together ---
    ("D9", "discriminating", "T1+T2", [40] * 8, "xxxxyyyy", 20, 0.2,
     "Cap makes the group all-equal, then dependency decays it"),
    ("D10", "discriminating", "T1+T2", [50, 45, 40, 35, 30, 25, 20, 15], "xyxyxyxy", 30, 0.001,
     "Cap changes the sort keys used by the dependency"),
    ("D11", "discriminating", "T1+T2", [60] * 10, "xxxxxxxyyy", 25, 0.01,
     "High raw quality, cap 25, 7+3 reads"),
    # --- Freeze-vary ladders (vary one input, others frozen) ---
    ("F1", "discriminating", "T1", Q40x6, "xxxxyy", 5, 0.001, "value_c ladder step 1"),
    ("F2", "discriminating", "T1", Q40x6, "xxxxyy", 30, 0.001, "value_c ladder step 2"),
    ("F3", "control", "T1", Q40x6, "xxxxyy", 40, 0.001, "value_c ladder step 3 (cap = Q, inactive)"),
    ("F4", "discriminating", "T2", Q30x8, "xxxxxxyy", 60, 0.001, "group-size ladder: 6+2"),
    ("F5", "control", "T2", Q30x8, "xxxxxxxy", 60, 0.001,
     "group-size ladder: 7+1 (only the x group decays; small effect)"),
    # --- Edge cases ---
    ("E1", "edge", "none", [1], "x", 1, 0.5, "Minimal input: one read, lowest qualities"),
    ("E2", "edge", "none", [60] * 50, "x" * 50, 60, 0.001,
     "Maximum depth, one allele only: no errors under the call, twists invisible"),
    ("E3", "edge", "T1", [60, 60, 60, 60, 60, 60], "xxxyyy", 1, 0.001,
     "value_c = 1: every read nearly uninformative"),
    ("E4", "edge", "T2", [30, 30, 30, 30], "xxyy", 60, 0.99,
     "Prior near 1: heterozygote dominates"),
    ("E5", "edge", "T1+T2", Q40x6, "yyyyxx", 20, 0.001,
     "Label swap of D1: output is symmetric in x/y"),
]


def build():
    out = []
    for cid, cat, twist, a, b, c, d, note in CASES:
        expected = oracle.f(a, b, c, d)
        _, g_hat = oracle._compute(a, b, c, d, want_genotype=True)
        out.append({
            "id": cid,
            "category": cat,
            "twist": twist,
            "inputs": {"value_a": a, "value_b": b, "value_c": c, "value_d": d},
            "expected": {"output": expected},
            "called_genotype_x_count": g_hat,
            "t1_off": round(oracle._compute(a, b, c, d, cap=False), 4),
            "t2_off": round(oracle._compute(a, b, c, d, dependency=False), 4),
            "naive_baseline": round(baselines.naive(a, b, c, d), 4),
            "recall_baseline": round(baselines.recall(a, b, c, d), 4),
            "note": note,
        })
    return out


def main():
    cases = build()
    path = os.path.join(ROOT, "golden", "golden.json")
    payload = {
        "task": "task-02",
        "output_key": "output",
        "rounding": 4,
        "tolerance": {"rel": 1e-3, "abs": 1e-4},
        "cases": cases,
    }
    with open(path, "w", encoding="ascii") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    print(f"wrote {len(cases)} cases -> {path}")


if __name__ == "__main__":
    main()
