"""
Freeze-vary evidence for each twist (numbers, not words).

For each twist: hold all inputs fixed except one, and print the output with
the twist ON vs OFF (ablation switches in oracle._compute). A twist is
load-bearing when the ON/OFF gap exceeds 10% in its firing regime.

Usage:  python scripts/freeze_vary.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "oracle"))

import oracle  # noqa: E402


def row(label, a, b, c, d):
    full, g = oracle._compute(a, b, c, d, want_genotype=True)
    no_t1 = oracle._compute(a, b, c, d, cap=False)
    no_t2 = oracle._compute(a, b, c, d, dependency=False)
    naive = oracle._compute(a, b, c, d, cap=False, dependency=False)
    qs = f"{a[0]}x{len(a)}" if len(set(a)) == 1 else ",".join(map(str, a))
    print(f"{label:16} a=[{qs:>18}] b={b:10} c={c:<3} d={d:<6} full={full:9.4f} (g={g})  "
          f"T1 off={no_t1:9.4f} ({(no_t1 / full - 1) * 100:6.1f}%)  "
          f"T2 off={no_t2:9.4f} ({(no_t2 / full - 1) * 100:6.1f}%)  naive={naive:9.4f}")


def main():
    print("TWIST 1 (MAPQ cap): vary value_c, others frozen")
    for c in (5, 10, 20, 30, 40, 50, 60):
        row("  vary value_c", [40] * 6, "xxxxyy", c, 0.001)
    print("TWIST 1: value_c above every quality -> flat (control)")
    for c in (45, 60):
        row("  vary value_c", [20, 25, 18], "xyy", c, 0.001)
    print("TWIST 2 (error dependency): vary group sizes at fixed depth 8, Q = 30")
    for k in range(1, 5):
        row("  vary value_b", [30] * 8, "x" * (8 - k) + "y" * k, 60, 0.001)
    print("TWIST 2: one read per group -> no dependency (control)")
    row("  control", [30, 30], "xy", 60, 0.001)
    row("  control", [35], "y", 60, 0.2)
    print("TWIST 2: permute qualities inside the y group (independent model is order-free too;")
    print("         dependency ranks by quality, so reordering inputs must NOT change output)")
    row("  permute", [30, 30, 30, 30, 40, 25, 20], "xxxxyyy", 60, 0.001)
    row("  permute", [30, 30, 30, 30, 20, 40, 25], "xxxxyyy", 60, 0.001)
    print("value_d: vary prior, others frozen (both twists on)")
    for d in (0.0001, 0.001, 0.05, 0.2, 0.5, 0.9):
        row("  vary value_d", [30] * 6, "xxxxyy", 60, d)


if __name__ == "__main__":
    main()
