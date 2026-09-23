"""
Grader for Blackbox Task 02.

Loads a submission module that defines f(value_a, value_b, value_c, value_d)
(or blackbox(inputs) -> {"output": float}), runs every golden case, and reports
pass/fail per case plus difficulty metrics:
  - overall score, score on discriminating cases, score on control/edge cases
  - naive-baseline score (independent reads, no MAPQ cap: both twists off)
  - recall-baseline score (MAQ-literal caller an expert could name)
  - per-twist score

It also cross-checks every golden expected value against the oracle.

Usage:
  python grader/grader.py                         # grades solution/solution.py
  python grader/grader.py path/to/submission.py
  python grader/grader.py path/to/submission.py --json
"""

import importlib.util
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "oracle"))

import oracle  # noqa: E402

GOLDEN = os.path.join(ROOT, "golden", "golden.json")
DEFAULT_SUBMISSION = os.path.join(ROOT, "solution", "solution.py")


def load_submission(path):
    spec = importlib.util.spec_from_file_location("submission", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if hasattr(mod, "f"):
        return lambda i: mod.f(list(i["value_a"]), i["value_b"], i["value_c"], i["value_d"])
    if hasattr(mod, "blackbox"):
        return lambda i: mod.blackbox(dict(i))["output"]
    raise AttributeError("submission must define f(...) or blackbox(inputs)")


def close(got, exp, rel, abs_tol):
    if got is None or not isinstance(got, (int, float)) or not math.isfinite(got):
        return False
    return abs(got - exp) <= max(abs_tol, rel * abs(exp))


def grade(submission_path, quiet=False):
    with open(GOLDEN, encoding="ascii") as fh:
        gold = json.load(fh)
    rel = gold["tolerance"]["rel"]
    abs_tol = gold["tolerance"]["abs"]
    sub = load_submission(submission_path)

    rows = []
    oracle_mismatch = []
    for case in gold["cases"]:
        inp = case["inputs"]
        exp = case["expected"]["output"]
        if oracle.f(**inp) != exp:
            oracle_mismatch.append(case["id"])
        try:
            got = sub(inp)
            got = round(float(got), 4)
        except Exception as exc:  # submission crash counts as fail
            got = None
            err = repr(exc)
        else:
            err = ""
        rows.append({
            "id": case["id"],
            "category": case["category"],
            "twist": case["twist"],
            "expected": exp,
            "got": got,
            "pass": close(got, exp, rel, abs_tol),
            "naive_pass": close(case["naive_baseline"], exp, rel, abs_tol),
            "recall_pass": close(case["recall_baseline"], exp, rel, abs_tol),
            "error": err,
        })

    def frac(sel, key="pass"):
        sel = list(sel)
        return (sum(r[key] for r in sel) / len(sel)) if sel else float("nan")

    disc = [r for r in rows if r["category"] == "discriminating"]
    ctrl = [r for r in rows if r["category"] in ("control", "edge")]
    twists = sorted({r["twist"] for r in rows})
    report = {
        "submission": os.path.relpath(submission_path, ROOT),
        "n_cases": len(rows),
        "score": frac(rows),
        "score_discriminating": frac(disc),
        "score_control_edge": frac(ctrl),
        "naive_baseline_score": frac(rows, "naive_pass"),
        "naive_baseline_discriminating": frac(disc, "naive_pass"),
        "recall_baseline_score": frac(rows, "recall_pass"),
        "recall_baseline_discriminating": frac(disc, "recall_pass"),
        "per_twist": {t: frac(r for r in rows if r["twist"] == t) for t in twists},
        "oracle_golden_mismatches": oracle_mismatch,
        "cases": rows,
    }
    report["verdict"] = "PASS" if report["score"] == 1.0 and not oracle_mismatch else "FAIL"

    if not quiet:
        print(f"Submission: {report['submission']}")
        print(f"{'id':5} {'category':15} {'twist':6} {'expected':>12} {'got':>12}  result")
        for r in rows:
            got = "ERR" if r["got"] is None else f"{r['got']:.4f}"
            print(f"{r['id']:5} {r['category']:15} {r['twist']:6} {r['expected']:12.4f} {got:>12}  "
                  f"{'PASS' if r['pass'] else 'FAIL'} {r['error']}")
        print("-" * 64)
        print(f"Score (all)               : {report['score']:.3f}")
        print(f"Score (discriminating)    : {report['score_discriminating']:.3f}")
        print(f"Score (control + edge)    : {report['score_control_edge']:.3f}")
        print(f"Naive baseline (all)      : {report['naive_baseline_score']:.3f}")
        print(f"Naive baseline (discrim.) : {report['naive_baseline_discriminating']:.3f}")
        print(f"Recall baseline (all)     : {report['recall_baseline_score']:.3f}")
        print(f"Recall baseline (discrim.): {report['recall_baseline_discriminating']:.3f}")
        for t, s in report["per_twist"].items():
            print(f"  twist {t:6}: {s:.3f}")
        if oracle_mismatch:
            print(f"WARNING golden/oracle mismatch: {oracle_mismatch}")
        print(f"VERDICT: {report['verdict']}")
    return report


def main(argv):
    args = [a for a in argv if not a.startswith("--")]
    path = os.path.abspath(args[0]) if args else DEFAULT_SUBMISSION
    report = grade(path, quiet="--json" in argv)
    if "--json" in argv:
        print(json.dumps(report, indent=2))
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
