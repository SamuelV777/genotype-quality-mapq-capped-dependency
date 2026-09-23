"""
Verify task structure and integrity.

Checks:
  1. Required files exist.
  2. All text files are ASCII only (Standing Rule 2).
  3. golden.json is well-formed: >= 5 cases, >= 1 control per twist,
     output key "output", inputs named value_a..value_d only.
  4. Golden expected values reproduce exactly from the oracle.
  5. Solution module does not import the oracle at module level for f(),
     and scores 100% under the grader.
  6. Naive baseline (textbook form) and recall baseline (MAQ-literal) each
     score <= 50% on discriminating cases (textbook / recall test).

Usage:  python scripts/verify_task.py
"""

import ast
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "oracle"))
sys.path.insert(0, os.path.join(ROOT, "grader"))

import oracle  # noqa: E402
import grader  # noqa: E402

REQUIRED = [
    "PROPOSAL.md",
    "oracle/oracle.py",
    "solution/solution.py",
    "grader/grader.py",
    "grader/baselines.py",
    "golden/golden.json",
    "scripts/run_tests.py",
    "scripts/verify_task.py",
    "scripts/generate_golden.py",
    "scripts/freeze_vary.py",
]
TEXT_EXT = (".py", ".md", ".json", ".txt")


def check(results, name, ok, detail=""):
    results.append((name, ok, detail))


def main():
    results = []

    for rel in REQUIRED:
        check(results, f"exists {rel}", os.path.isfile(os.path.join(ROOT, rel)))

    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if not d.startswith(".") and d != "__pycache__"]
        for fn in filenames:
            if fn.endswith(TEXT_EXT):
                p = os.path.join(dirpath, fn)
                with open(p, "rb") as fh:
                    data = fh.read()
                bad = [i for i, ch in enumerate(data) if ch > 127]
                check(results, f"ascii {os.path.relpath(p, ROOT)}", not bad,
                      f"first non-ASCII byte at {bad[0]}" if bad else "")

    with open(os.path.join(ROOT, "golden", "golden.json"), encoding="ascii") as fh:
        gold = json.load(fh)
    cases = gold["cases"]
    check(results, "golden >= 5 cases", len(cases) >= 5, str(len(cases)))
    check(results, "output key is 'output'",
          gold["output_key"] == "output" and all(list(c["expected"]) == ["output"] for c in cases))
    check(results, "inputs are value_a..value_d",
          all(sorted(c["inputs"]) == ["value_a", "value_b", "value_c", "value_d"] for c in cases))
    for tw in ("T1", "T2"):
        check(results, f"control case for {tw}",
              any(c["category"] == "control" and c["twist"] == tw for c in cases))
    disc = [c for c in cases if c["category"] == "discriminating"]
    check(results, "discriminating cases >= 5", len(disc) >= 5, str(len(disc)))
    mism = [c["id"] for c in cases if oracle.f(**c["inputs"]) != c["expected"]["output"]]
    check(results, "golden reproduces from oracle", not mism, str(mism))

    sol_path = os.path.join(ROOT, "solution", "solution.py")
    with open(sol_path, encoding="ascii") as fh:
        tree = ast.parse(fh.read())
    top_imports = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_imports += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            top_imports.append(node.module or "")
    check(results, "solution does not import oracle at top level", "oracle" not in top_imports)

    rep = grader.grade(sol_path, quiet=True)
    check(results, "solution scores 100%", rep["score"] == 1.0, f"{rep['score']:.3f}")
    check(results, "naive baseline <= 50% on discriminating",
          rep["naive_baseline_discriminating"] <= 0.5,
          f"{rep['naive_baseline_discriminating']:.3f}")
    check(results, "recall baseline <= 50% on discriminating",
          rep["recall_baseline_discriminating"] <= 0.5,
          f"{rep['recall_baseline_discriminating']:.3f}")

    fails = 0
    for name, ok, detail in results:
        print(f"[{'OK' if ok else 'FAIL'}] {name} {detail}")
        fails += not ok
    print(f"\n{len(results) - fails}/{len(results)} checks passed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
