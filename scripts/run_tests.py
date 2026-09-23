"""
Execute all golden tests against the oracle (and the edge-case invariants).

Usage:  python scripts/run_tests.py
Exit code 0 when every check passes.
"""

import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "oracle"))

import oracle  # noqa: E402


def main():
    with open(os.path.join(ROOT, "golden", "golden.json"), encoding="ascii") as fh:
        gold = json.load(fh)

    failures = []
    for case in gold["cases"]:
        got = oracle.f(**case["inputs"])
        exp = case["expected"]["output"]
        ok = got == exp
        print(f"{case['id']:5} {case['category']:15} expected={exp:<12} got={got:<12} "
              f"{'OK' if ok else 'FAIL'}")
        if not ok:
            failures.append(case["id"])

    q6 = [40] * 6
    # Invariants
    checks = {
        "one read, r = 0.5 -> 10log10(2)": math.isclose(
            oracle.f([1], "x", 1, 0.5), round(10 * math.log10(2), 4)),
        "x/y label swap symmetric": oracle.f(q6, "xxxxyy", 20, 0.001) == oracle.f(q6, "yyyyxx", 20, 0.001),
        "read order irrelevant": oracle.f([40, 25, 20, 30], "yyyx", 60, 0.01)
                                 == oracle.f([30, 20, 40, 25], "xyyy", 60, 0.01),
        "T1 flat when value_c >= max(value_a)": oracle.f(q6, "xxxxyy", 40, 0.001)
                                                == oracle.f(q6, "xxxxyy", 60, 0.001),
        "T1 cap == lowering qualities": oracle.f(q6, "xxxxyy", 30, 0.001)
                                        == oracle.f([30] * 6, "xxxxyy", 60, 0.001),
        "T2 inactive with one read per group": math.isclose(
            oracle._compute([30, 30], "xy", 60, 0.001),
            oracle._compute([30, 30], "xy", 60, 0.001, dependency=False), rel_tol=1e-12),
        "theta cited value": oracle.THETA == 0.85,
        "output finite at max depth": math.isfinite(oracle.f([60] * 50, "x" * 50, 60, 0.001)),
    }
    for bad in [([], "", 30, 0.1), ([30], "xy", 30, 0.1), ([0], "x", 30, 0.1), ([30], "z", 30, 0.1),
                ([30], "x", 0, 0.1), ([30], "x", 30, 0.0), ([30], "x", 30, 1.0), ([61], "x", 30, 0.1)]:
        try:
            oracle.f(*bad)
            checks[f"rejects {bad}"] = False
        except ValueError:
            checks[f"rejects {bad}"] = True

    for name, ok in checks.items():
        print(f"invariant: {name:40} {'OK' if ok else 'FAIL'}")
        if not ok:
            failures.append(name)

    total = len(gold["cases"]) + len(checks)
    print(f"\n{total - len(failures)}/{total} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
