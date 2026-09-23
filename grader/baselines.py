"""
Reference baselines used to measure difficulty (not solutions).

naive(...)  : textbook caller. Phred errors from raw base qualities (no MAPQ
              cap), independent reads, Li (2011) Eq. 2 diploid likelihoods,
              MAQ prior and Q_g. Both twists off.
recall(...) : "MAQ-literal" caller an expert could write by naming MAQ
              (Li, Ruan & Durbin 2008): MAPQ cap, Eq. 2 dependency with
              c'_nk = 1 applied ONLY to the reads counted as errors, and the
              binomial heterozygote likelihood C(n, k) / 2**n.
Both return unrounded floats; callers round to 4 decimals.
"""

import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "oracle"))

import oracle  # noqa: E402


def naive(value_a, value_b, value_c, value_d):
    return oracle._compute(list(value_a), value_b, value_c, value_d, cap=False, dependency=False)


def recall(value_a, value_b, value_c, value_d):
    oracle._validate(list(value_a), value_b, value_c, value_d)
    quals = [min(q, value_c) for q in value_a]
    n = len(quals)
    ln10 = math.log(10.0)

    def ln_alpha(err_quals):
        # Eq. 2: prod_i eps_(i+1) ** (theta ** i), eps ascending, c'_nk = 1
        s = 0.0
        for i, q in enumerate(sorted(err_quals, reverse=True)):
            s += -(q / 10.0) * ln10 * oracle.THETA ** i
        return s

    qx = [q for q, ch in zip(quals, value_b) if ch == "x"]
    qy = [q for q, ch in zip(quals, value_b) if ch == "y"]
    ll = {
        2: ln_alpha(qy),                                        # <x,x>: y reads are errors
        0: ln_alpha(qx),                                        # <y,y>: x reads are errors
        1: math.log(math.comb(n, len(qx))) - n * math.log(2.0),  # binomial heterozygote
    }
    q, _ = oracle.consensus_quality(ll, oracle.log_prior(value_d))
    return q
