"""
Oracle (reference implementation) for Blackbox Task 02.

Diploid consensus-genotype quality at one site from a pileup of read bases:
  - Step 1  TWIST 1 (MAQ coupling): each base quality is capped by the read
            mapping quality, q_i = min(Q_i, MAPQ)   (Li, Ruan & Durbin 2008,
            "Consensus genotype calling"); Phred error eps = 10**(-q/10)
            (SAM spec v1.6, s1.2 "Phred scale")
  - Step 2  TWIST 2 (MAQ error dependency): inside each allele group the i-th
            smallest error (0-based) is raised to theta**i, theta = 0.85
            (Li, Ruan & Durbin 2008, Eq. 2, with c'_nk := 1)
  - Step 3  Genotype likelihoods, ploidy m = 2 (Li 2011, s2.2, Eq. 2). Authored
            coupling: the dependent errors replace eps only in the factors
            where a read is an error under the genotype (a homozygote of the
            other allele); the (1 - eps) factors keep the independent eps.
  - Step 4  Posterior with MAQ prior P(hom) = (1 - r)/2 each, P(het) = r,
            called genotype = argmax, quality Q_g = -10 log10[1 - P(g_hat|D)]
            (Li, Ruan & Durbin 2008, "Consensus genotype calling")

Public entry point: f(value_a, value_b, value_c, value_d) -> float (4 decimals)

  value_a : list[int], 1..50 entries, each 1..60   per-read base quality
  value_b : str, same length, chars in {"x","y"}   x = reference base, y = other
  value_c : int 1..60                              read mapping quality (all reads)
  value_d : float in (0, 1)                        heterozygote prior r
"""

import math

# ---------------------------------------------------------------------------
# Constants (every value is cited or exact; see PROPOSAL.md s8)
# ---------------------------------------------------------------------------
# Li, Ruan & Durbin (2008) Genome Res 18:1851, text after Eq. 2:
# theta = 0.85 selected for Illumina Genetic Analyzer data
THETA = 0.85
# Ploidy (diploid; MAQ assumes a diploid sample by default)
PLOIDY = 2
# Phred scale base: SAM spec v1.6 s1.2, "-10 log10 p" (definitional)
PHRED = 10.0

ALLELES = ("x", "y")
MAX_READS = 50
QMIN, QMAX = 1, 60


def _validate(value_a, value_b, value_c, value_d):
    if not isinstance(value_a, (list, tuple)) or not (1 <= len(value_a) <= MAX_READS):
        raise ValueError("value_a must be a list of 1..50 integers")
    for q in value_a:
        if isinstance(q, bool) or not isinstance(q, int) or not (QMIN <= q <= QMAX):
            raise ValueError("value_a entries must be integers in 1..60")
    if not isinstance(value_b, str) or len(value_b) != len(value_a):
        raise ValueError("value_b must be a string with one char per value_a entry")
    if any(ch not in ALLELES for ch in value_b):
        raise ValueError('value_b chars must be "x" or "y"')
    if isinstance(value_c, bool) or not isinstance(value_c, int) or not (QMIN <= value_c <= QMAX):
        raise ValueError("value_c must be an integer in 1..60")
    if isinstance(value_d, bool) or not isinstance(value_d, (int, float)) or not (0.0 < value_d < 1.0):
        raise ValueError("value_d must be a number in (0, 1)")


def effective_qualities(value_a, value_c, cap=True):
    """Step 1: q_i = min(Q_i, MAPQ) (TWIST 1). Returns Phred-scale floats."""
    return [float(min(q, value_c)) if cap else float(q) for q in value_a]


def dependent_log_errors(quals, value_b, dependency=True):
    """Step 2: natural-log error per read. Inside each allele group, sort by
    error ascending (quality descending); the i-th gets eps**(theta**i)
    (TWIST 2). In Phred terms q_eff = q * theta**i.   Returns list aligned
    with the input order."""
    ln_eps = [0.0] * len(quals)
    for allele in ALLELES:
        idx = [j for j, ch in enumerate(value_b) if ch == allele]
        idx.sort(key=lambda j: -quals[j])
        for rank, j in enumerate(idx):
            w = THETA ** rank if dependency else 1.0
            ln_eps[j] = -(quals[j] * w / PHRED) * math.log(10.0)
    return ln_eps


def _log1m_exp(x):
    """log(1 - exp(x)) for x < 0, numerically stable."""
    return math.log(-math.expm1(x)) if x > -0.6931 else math.log1p(-math.exp(x))


def log_likelihoods(ln_eps, ln_eps_dep, value_b):
    """Step 3: Li (2011) Eq. 2 with m = 2, g = number of reference alleles.
    L(g) = m**-k prod_ref[(m-g) eps + g (1-eps)] prod_alt[(m-g)(1-eps) + g eps]
    A read that is an error under g (ref read at g = 0, alt read at g = m)
    uses the dependent error from step 2; every other factor uses eps.
    Returns {g: ln L(g)} for g in (0, 1, 2)."""
    m = PLOIDY
    k = len(ln_eps)
    out = {}
    for g in range(m + 1):
        s = -k * math.log(m)
        for le, ld, ch in zip(ln_eps, ln_eps_dep, value_b):
            ref = ch == "x"
            # homozygous terms in log space so tiny eps keeps full precision
            if g == 0:
                s += math.log(m) + (ld if ref else _log1m_exp(le))
            elif g == m:
                s += math.log(m) + (_log1m_exp(le) if ref else ld)
            else:
                eps = math.exp(le)
                if ref:
                    s += math.log((m - g) * eps + g * (1.0 - eps))
                else:
                    s += math.log((m - g) * (1.0 - eps) + g * eps)
        out[g] = s
    return out


def log_prior(r):
    """MAQ prior: P(<b,b>) = P(<b',b'>) = (1 - r)/2, P(<b,b'>) = r."""
    return {0: math.log((1.0 - r) / 2.0), 1: math.log(r), 2: math.log((1.0 - r) / 2.0)}


def consensus_quality(ll, lp):
    """Step 4: posterior, g_hat = argmax, Q_g = -10 log10[1 - P(g_hat|D)],
    computed as -10 log10(sum_{g != g_hat} w_g / sum_g w_g) in log space."""
    lw = {g: ll[g] + lp[g] for g in ll}
    g_hat = max(lw, key=lambda g: lw[g])
    top = max(lw.values())

    def lse(vals):
        return top + math.log(sum(math.exp(v - top) for v in vals))

    ln_rest = lse([lw[g] for g in lw if g != g_hat])
    ln_all = lse(list(lw.values()))
    return -PHRED * (ln_rest - ln_all) / math.log(10.0), g_hat


def _compute(value_a, value_b, value_c, value_d, cap=True, dependency=True, want_genotype=False):
    """Unrounded core with ablation switches (used by scripts/freeze_vary.py)."""
    _validate(value_a, value_b, value_c, value_d)
    quals = effective_qualities(value_a, value_c, cap=cap)
    ln_eps = dependent_log_errors(quals, value_b, dependency=False)
    ln_eps_dep = dependent_log_errors(quals, value_b, dependency=dependency)
    ll = log_likelihoods(ln_eps, ln_eps_dep, value_b)
    q, g_hat = consensus_quality(ll, log_prior(value_d))
    return (q, g_hat) if want_genotype else q


def f(value_a, value_b, value_c, value_d):
    """Blackbox function. Returns {"output": ...}-compatible float, 4 decimals."""
    return round(_compute(list(value_a), value_b, value_c, value_d), 4)


def blackbox(inputs):
    """Dict interface: {"value_a":..,"value_b":..,"value_c":..,"value_d":..} -> {"output": float}"""
    return {"output": f(inputs["value_a"], inputs["value_b"], inputs["value_c"], inputs["value_d"])}


if __name__ == "__main__":
    for args in [([30, 30, 30, 30, 30, 30], "xxxyyy", 60, 0.001),
                 ([30, 30, 30, 30, 30, 30], "xxxxxy", 60, 0.001),
                 ([40, 40, 40, 40], "yyyy", 20, 0.2)]:
        print(args, "->", f(*args), _compute(*args, want_genotype=True)[1])
