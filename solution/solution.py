"""
Example solver for Blackbox Task 02.

Part A (run as a script): shows the probe sequence a skilled solver would use,
querying the blackbox and reasoning about each response.
Part B: the final reconstructed function f(...). It does NOT call the oracle;
the grader imports this module and uses f directly.

Usage:
  python solution/solution.py        # replay probes + self-check vs oracle
"""

import math
import os
import sys

# ---------------------------------------------------------------------------
# Part B: reconstructed implementation (what the solver submits)
# ---------------------------------------------------------------------------
# Probes showed: value_a behaves like Phred qualities (one read of Q gives a
# Phred-scaled posterior), value_b splits reads into two alleles, the output is
# symmetric under swapping x <-> y (symmetric homozygote prior), value_d acts
# as a heterozygote prior r with homozygotes at (1 - r)/2, and the output is
# the Phred-scaled error of the MAP genotype.  Two hidden behaviours:
#   * value_c caps every quality (flat once value_c >= max(value_a))
#   * the k-th error inside one allele group counts only theta**k as much;
#     theta recovered from the ratio of successive decrements = 0.85
THETA = 0.85
LN10 = math.log(10.0)


def _ln1m(x):
    return math.log(-math.expm1(x)) if x > -0.6931 else math.log1p(-math.exp(x))


def f(value_a, value_b, value_c, value_d):
    quals, alle, mapq, r = list(value_a), value_b, value_c, value_d
    if not (1 <= len(quals) <= 50) or len(alle) != len(quals) or set(alle) - {"x", "y"}:
        raise ValueError("invalid input")
    if not (1 <= mapq <= 60) or not (0.0 < r < 1.0) or any(not (1 <= q <= 60) for q in quals):
        raise ValueError("invalid input")
    n = len(quals)
    capped = [min(q, mapq) for q in quals]

    def group(tag):
        return sorted((q for q, a in zip(capped, alle) if a == tag), reverse=True)

    def ln_errors(qs, dependent):
        return sum(-(q * (THETA ** i if dependent else 1.0) / 10.0) * LN10 for i, q in enumerate(qs))

    def ln_correct(qs):
        return sum(_ln1m(-(q / 10.0) * LN10) for q in qs)

    gx, gy = group("x"), group("y")
    # log posterior weights; the common factor 2**-n cancels only for the
    # homozygotes, so keep it explicit: hom L = prod(correct)*prod(error), het L = 2**-n
    w = {
        "xx": math.log((1 - r) / 2) + ln_correct(gx) + ln_errors(gy, True),
        "yy": math.log((1 - r) / 2) + ln_correct(gy) + ln_errors(gx, True),
        "xy": math.log(r) - n * math.log(2.0),
    }
    best = max(w, key=w.get)
    top = w[best]
    rest = math.log(sum(math.exp(v - top) for k, v in w.items() if k != best)) + top
    total = math.log(sum(math.exp(v - top) for v in w.values())) + top
    return round(-10.0 * (rest - total) / LN10, 4)


# ---------------------------------------------------------------------------
# Part A: discovery replay (uses the oracle only as the blackbox being probed)
# ---------------------------------------------------------------------------
def _replay():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(root, "oracle"))
    import oracle
    bb = oracle.f

    def show(tag, *args):
        y = bb(*args)
        print(f"  {tag:4} f{args} = {y}")
        return y

    print("Step 1 - single read: output tracks value_a -> Phred-like quality")
    for q in (10, 20, 30, 40):
        show("P1", [q], "y", 60, 0.001)
    show("P2", [30], "x", 60, 0.001)
    print("       x and y give the same value -> prior symmetric in the two alleles")

    print("Step 2 - value_d: single read, vary d")
    for d in (0.001, 0.2, 0.5):
        show("P3", [30], "y", 60, d)
    print("       fits P(hom) = (1-d)/2 each, P(het) = d; output = -10log10(1 - P(MAP))")

    print("Step 3 - value_c: 6 reads Q40, 4x + 2y, vary c")
    for c in (60, 40, 30, 20, 10):
        show("P4", [40] * 6, "xxxxyy", c, 0.001)
    print("       flat for c >= 40 = max(Q); below that the result equals c used as every Q:")
    print(f"       f([30]*6, xxxxyy, 60) = {bb([30] * 6, 'xxxxyy', 60, 0.001)}  vs  "
          f"f([40]*6, xxxxyy, 30) = {bb([40] * 6, 'xxxxyy', 30, 0.001)}")
    print("       -> q = min(Q, c)")

    print("Step 4 - group size: depth 8, Q30, move reads from x to y")
    ind = []
    for k in (1, 2, 3, 4):
        y = show("P5", [30] * 8, "x" * (8 - k) + "y" * k, 60, 0.001)
        ind.append(y)
    print("       an independent-read model gives 21.07, 9.48, 38.95, 65.94 for these; the")
    print("       blackbox matches only when one read is in the minority group")

    print("Step 5 - recover theta from homozygote-vs-het log-odds increments")
    # Use a huge prior so the MAP call is het; then Q tracks the hom-x weight,
    # and each added y error at Q30 adds about 30*theta**i Phred units.
    prev = None
    for k in (1, 2, 3, 4):
        y = bb([30] * 10, "x" * (10 - k) + "y" * k, 60, 0.9)
        if prev is not None:
            print(f"       k={k}: increment = {y - prev:.4f}")
        prev = y
    print("       increments shrink by ~0.85 per extra error -> theta = 0.85 (MAQ, Eq. 2)")

    print("Self-check of reconstructed f against the blackbox:")
    worst = 0.0
    for args in [([40] * 8, "xxxxyyyy", 20, 0.2), ([50, 45, 40, 35, 30, 25, 20, 15], "xyxyxyxy", 30, 0.001),
                 ([60] * 50, "x" * 50, 60, 0.001), ([1], "x", 1, 0.5), ([30] * 4, "xxyy", 60, 0.99)]:
        a, b = f(*args), bb(*args)
        worst = max(worst, abs(a - b) / max(1.0, abs(b)))
        print(f"  {str(args)[:60]:60}: mine={a}  blackbox={b}")
    print(f"  worst relative error = {worst:.2e}")


if __name__ == "__main__":
    _replay()
