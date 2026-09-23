# Blackbox Task 02 - Diploid Consensus-Genotype Quality with Mapping-Quality Capping and Rank-Decayed Error Dependency

**Domain:** Biology / genomics (short-read variant calling, genotype likelihoods)
**Tier:** Med-High (floor: 5h, 2 genuine twists, 4 substantive steps)
**Honest counts:** 4 substantive steps, 2 genuine twists

---

## Design record (Stages 1-4)

### Stage 1 - Candidates considered

| # | Candidate | Outcome |
|---|-----------|---------|
| A | LD decay: expected r**2 from bp distance via a Kosambi map function, then Hill & Weir (1988) with sample size n | Rejected: at the 4-decimal output scale the Kosambi vs Haldane gap is invisible unless Ne is tiny, and Hill-Weir LD decay is a recallable standard fit. |
| B | Codon Adaptation Index with an organism-specific reference set | Rejected: it needs a large external reference table (the "empirical dataset" red flag), and the twist would only be which table was picked. |
| C | Microsatellite heterozygosity under the stepwise mutation model (Ohta & Kimura 1973) vs the infinite-alleles model | Rejected: a one-formula map from 4Nu. The mode flag would be the only twist, and it is definitional. |
| **D** | **Diploid consensus-genotype quality from a read pileup: Phred errors, MAPQ cap, MAQ rank-decayed error dependency, Li (2011) likelihoods, MAQ prior and Q_g** | **Selected.** Every constant is exact or cited. Both twists are hidden behaviours that move output by more than 10%. The composition matches neither MAQ nor modern callers, and both baselines score 0/14 on the discriminating cases. |

### Stage 2 - GATE 1 screen (candidate D): **PASS**

1. **Algebraic collapse: PASS.** out = -10 log10( sum_{g != g_hat} w_g / sum_g w_g ), with w_g = prior(r, g) * L(g | min(Q, MAPQ), groups). Here r (value_d) sits inside a normalised mixture with the likelihoods. The MAP index g_hat switches with every input. value_c enters through a min() on each read, and value_b decides the grouping and ranking. Nothing separates additively.
2. **Domain recall: PASS.** An expert who names "genotype likelihood + het prior" writes the independent-read caller, which scores 0/14 on the discriminating cases (naive baseline). An expert who names "MAQ consensus model" writes the MAQ-literal caller, with the binomial het likelihood C(n,k)/2**n, errors-only homozygote terms and c'_nk = 1. That also scores 0/14 (recall baseline). No published caller computes this exact function.
3. **Twists survive simplification: PASS.** In the het likelihood, both twists cancel exactly: Li Eq. 2 with m = 2, g = 1 gives each read the factor 1 and L = 2**-n. The homozygote likelihoods keep both: min(Q_i, MAPQ) cannot be absorbed by any other input, and the rank weights theta**i are irreducible (they depend on group sizes and on the sorted order).
4. **Genuine vs definitional:** Phred conversion, Li Eq. 2, the Bayes posterior and the Phred-scaled Q_g are definitional steps and are **not** counted as twists. T1 (MAPQ cap) and T2 (rank-decayed error dependency) are the only counted twists.
5. **Non-sequential: PASS.** value_d feeds step 4 only, so it can be probed on a single read. value_c feeds step 1 only. value_b feeds steps 2 and 3 in parallel (the grouping and ranking, and which factor applies). value_a feeds steps 1 to 3.
6. **Tier fit: PASS.** 4 steps and 2 twists, against a Med-High floor of 4 and 2.

### Stage 4 - GATE 2 verification: **PASS** (see the table at the end of this document)

---

## 1. Pick Your Algorithm (textbook test)

**Standard algorithm.** This is Bayesian diploid genotype calling at a single site, as in SAMtools, GATK and MAQ. Each read base carries a Phred quality Q, with error eps = 10**(-Q/10). Reads are independent. The genotype likelihoods are L(g) = prod over reads of P(base | g), multiplied by a genotype prior. The reported value is the Phred-scaled probability that the called genotype is wrong. An expert who implements this textbook form scores **0/14 on the discriminating cases** (naive baseline, computed by `grader/baselines.py`).

**Twists (authored composition of published parts):**

- **T1 - mapping-quality cap.** Every base quality is replaced by min(Q_i, MAPQ) before anything else. This is taken from MAQ (Li, Ruan & Durbin 2008). Modern callers either ignore MAPQ at the likelihood stage or use it only as a filter. The blackbox is flat in value_c above max(value_a) and strongly non-monotone below it, because the capped qualities flip the called genotype.
- **T2 - rank-decayed error dependency.** Inside each allele group, reads are sorted by capped quality in descending order. The i-th read (0-based) that counts as an error under a homozygote of the other allele has its error raised to theta**i, with theta = 0.85 (MAQ Eq. 2, taking c'_nk = 1). In Phred terms this is q_eff = q * theta**i. The authored scope: the decayed error replaces eps only in those error factors, inside the Li (2011) per-read likelihood. The (1 - eps) factors keep the raw eps. The het likelihood is Li's ordered form, not MAQ's binomial.

## 2. Ensure Multi-Step Computation

```text
INPUT  value_a : list[int]   Q_1..Q_n       (1..50 reads, each 1..60)
       value_b : str         a_1..a_n       (each "x" or "y")
       value_c : int         M              (1..60)
       value_d : float       r              (0 < r < 1)

STEP 1 (T1)  q_i   = min(Q_i, M)                                  # MAQ, cap
             e_i   = 10 ** (-q_i / 10)                            # SAM v1.6 s1.2

STEP 2 (T2)  for each allele group G in {x, y}:
                 sort G by q descending  (= e ascending)
                 for rank i = 0, 1, 2, ... in that order:
                     d_j = e_j ** (THETA ** i)                    # MAQ Eq. 2, c'_nk = 1

STEP 3       m = 2, g = number of "x" alleles in the genotype     # Li 2011 Eq. 2
             L(2) = 2**-n * prod_{x reads} 2 (1 - e_j) * prod_{y reads} 2 d_j
             L(1) = 2**-n * prod_{all reads} 1                     (= 2**-n)
             L(0) = 2**-n * prod_{x reads} 2 d_j     * prod_{y reads} 2 (1 - e_j)

STEP 4       prior P(2) = P(0) = (1 - r) / 2,  P(1) = r          # MAQ prior
             w_g   = L(g) * P(g);   g_hat = argmax_g w_g
             output = -10 log10( 1 - w_{g_hat} / sum_g w_g )      # MAQ Q_g
             round to 4 decimals
```

The substantive steps are 1, 2, 3 and 4, and each has its own input channel: value_c only in step 1, value_b's grouping in steps 2 and 3, value_d only in step 4. Rounding and the log-space evaluation (log-sum-exp, log1m-exp) are numerics and do not count as steps.

## 3. Ground Every Step in Literature

| Step | Source | Locator | Verbatim quote (< 15 words) | Status |
|------|--------|---------|-----------------------------|--------|
| 1 (Phred) | SAM/BAM Format Specification v1.6 (hts-specs, printing b5341fb, 12 Aug 2025), https://samtools.github.io/hts-specs/SAMv1.pdf | s1.2 "Terminologies and Concepts", entry "Phred scale" | "the phred scale of p equals -10 log10 p" | confirmed (PDF fetched) |
| 1 (T1 cap), 2 (theta, Eq. 2), 4 (prior, Q_g) | Li H, Ruan J, Durbin R (2008) Mapping short DNA sequencing reads and calling variants using mapping quality scores. Genome Res 18:1851-1858. doi:10.1101/gr.078212.108, PMC2577856 | Methods, "Consensus genotype calling": cap (paragraph 1); prior r and Q_g = -10log10[1 - P(g_hat \| D)] (paragraph 2); Eq. 2 alpha_nk ~ c'_nk prod_{i=0}^{k-1} eps_{i+1}^(theta^i) with eps_i the i-th smallest error, theta = 0.85 (paragraph 3) | "the base quality used in SNP calling cannot exceed the mapping quality" | confirmed (HTML and Eq. 2 image fetched) |
| 3 (likelihood) | Li H (2011) A statistical framework for SNP calling, mutation discovery, association mapping and population genetical parameter estimation from sequencing data. Bioinformatics 27:2987-2993. doi:10.1093/bioinformatics/btr509, PMC3198575 | s2.2 "Computing genotype likelihoods", Eq. 2: L(g) = m**-k prod_{j<=l}[(m-g) e_j + g(1-e_j)] prod_{j>l}[(m-g)(1-e_j) + g e_j] | "Assuming error independency, we can derive that" | confirmed (HTML and Eq. 2 image fetched) |

**Citation-to-code honesty.**
- Li Eq. 2 is the independent-error likelihood. Replacing e_j with the dependent d_j in the error factors is the authored coupling (T2 scope). The source grounds the likelihood form, not the substitution.
- MAQ Eq. 2 is an approximation to alpha_nk, the probability of k errors among n bases. Setting c'_nk = 1 is authored; the paper states c'_nk varies little with eps_i and derives it only in the supplement. We quote the paper's own ranking convention (i-th smallest error gets exponent theta**(i-1), 1-based).
- In MAQ, b and b' are the two most frequent nucleotides. Here they are the "x" and "y" read groups. The prior is symmetric, so the labelling cannot change the result (invariant tested in `scripts/run_tests.py`).

## 4. Function Description (for the reviewer)

Computes the Phred-scaled confidence of a diploid genotype call at one site, from the reads covering it: per-read qualities, which allele each read shows, a single mapping quality, and a heterozygote prior. It composes the Li (2011) likelihood with two MAQ-era corrections that modern callers dropped. One caps base quality at mapping quality. The other discounts repeated errors within an allele group geometrically (theta = 0.85). The solver must discover the cap threshold and the rank-decay law, and notice that the decay applies only to reads that count as errors under a homozygote.

## 5. Output Schema

```json
{"output": 8.48}
```

- A single float, rounded to 4 decimals, in [0, inf). In practice it runs from about 2 (weak data) to about 180 (50 concordant Q60 reads).
- It is the MAQ Q_g on its natural Phred scale, not pre-normalised. The output is not monotone in any single input (value_c: 5.01, 12.45, 8.48, 10.84, 28.96 for c = 5, 10, 20, 30, 40), because the called genotype switches.

## 6. Inputs (4 inputs, 4 different types)

| Key | Type | Allowed | Neutral description given to the solver |
|-----|------|---------|-------------------------------------------|
| value_a | list[int] | length 1..50, entries 1..60 | a list of positive integers |
| value_b | str | same length as value_a, chars "x"/"y" | one label per list entry |
| value_c | int | 1..60 | a positive integer |
| value_d | float | open interval (0, 1) | a number between 0 and 1 |

The schema contains no domain words or units. Invalid inputs raise ValueError.

## 7. What Makes This Hard?

**T1 - MAPQ cap**
1. *Which inputs expose it:* value_c below max(value_a). With [40]*6 and "xxxxyy" (r = 0.001) the output is 28.9601 for c = 40, 50 and 60, then 10.8431, 8.4800, 12.4548 and 5.0143 for c = 30, 20, 10 and 5. The naive model is flat at 34.9560 for every c. The clearest single probe: f([40]*6, c = 30) equals f([30]*6, c = 60) exactly.
2. *Why a naive solver fails:* the common guess is that MAPQ enters as an extra independent error (1 - (1 - e_b)(1 - e_m)), as a filter, or not at all. None of these gives a hard flat region above max(Q), and none reproduces the genotype flip at c = 20 (het to hom, D1).
3. *Why limited probes matter:* the response is flat whenever c >= max(Q), which is exactly where a solver using default "high MAPQ" probes will sit. value_c has to be pushed below the qualities on purpose. After that the non-monotone response confounds with the genotype switch, so the solver has to separate "cap" from "prior" by holding value_d fixed.

**T2 - rank-decayed error dependency**
1. *Which inputs expose it:* value_b group sizes of 2 or more on both sides. At depth 8, Q30, r = 0.001, moving reads from x to y gives 21.0714, 5.7881, 26.1379 and 41.5384 for 1 to 4 y reads. The independent model gives 21.0714, 9.4780, 38.9545 and 65.9393, so it matches only when one group has a single read (+0%, +64%, +49%, +59%).
2. *Why a naive solver fails:* an independent-read caller adds one full Q per extra error. The blackbox adds about 30 * 0.85**i Phred units per extra Q30 error (measured increments 25.26, 21.67, 18.41 in the solver replay). A MAQ-literal reimplementation gets the decay but uses the binomial het likelihood and errors-only homozygotes, so it misses as well (D5: 59.97 vs 41.54).
3. *Why limited probes matter:* the decay is invisible on single reads and on pure one-allele pileups (E2: both twists off gives the same 177.5001). The solver has to design mixed pileups. The decay also ranks by *capped* quality, so it interacts with T1 (D9, D10, D11). Probing the two together before isolating each one gives confounded ratios.

**Red flags avoided:** no matrix operations, no external empirical percentiles, no uncited thresholds (the only threshold is the input value_c itself), 2 modifications (not more than 4), and nothing stochastic.

## 8. No Magic Numbers

| Constant | Value | Status |
|----------|-------|--------|
| THETA | 0.85 | cited: MAQ (2008), text after Eq. 2 (theta = 0.85 selected for Illumina data) |
| c'_nk | 1 | authored simplification of MAQ Eq. 2 (flagged; the paper states it varies little) |
| ploidy m | 2 | cited: MAQ, Consensus genotype calling, paragraph 1 (diploid by default); also definitional of the task |
| Phred base 10, factor -10 | exact | definitional (SAM v1.6 s1.2) |
| prior split (1 - r)/2, r | exact given r | cited: MAQ prior; r is the input value_d |
| 2**-n factor | exact | Li Eq. 2 with m = 2 (it cancels in the posterior only for the het term) |
| 0.6931 in log1m_exp | ln 2 | numerical branch point only; changes nothing beyond float round-off |
| input bounds 1..60, 1..50 | - | input domain only, never used in the arithmetic |

The only hidden element with a numeric value is THETA, which is cited. The authored parts are structural (the cap is applied first, the decay covers error factors only and ranks by capped quality, c'_nk = 1), not invented numbers.

## 9. Edge Cases (25 golden cases: 14 discriminating, 6 control, 5 edge)

Columns: expected (oracle), g = called genotype (number of x alleles), T1off and T2off = oracle with one twist switched off, naive = both twists off, recall = MAQ-literal.

| id | cat | twist | value_a | value_b | c | d | expected | g | T1off | T2off | naive | recall |
|----|-----|-------|---------|---------|---|---|----------|---|-------|-------|-------|--------|
| D1 | disc | T1 | [40]*6 | xxxxyy | 20 | 0.001 | 8.4800 | 2 | 28.9601 | 6.0958 | 34.9560 | 5.2462 |
| D2 | disc | T1 | [40]*6 | xxxxyy | 10 | 0.001 | 12.4548 | 2 | 28.9601 | 17.7403 | 34.9560 | 11.3395 |
| D3 | disc | T1 | [35,38,40,25,30] | xxyyy | 15 | 0.2 | 10.1890 | 1 | 49.5939 | 12.4704 | 54.9293 | 19.3943 |
| D4 | disc | T1 | [45,20,50,15,38,42] | xyxyxy | 25 | 0.05 | 24.7657 | 1 | 41.9995 | 32.0744 | 49.1619 | 37.7108 |
| C1 | ctrl | T1 | [40]*6 | xxxxyy | 60 | 0.001 | 28.9601 | 1 | 28.9601 | 34.9560 | 34.9560 | 40.7141 |
| C2 | ctrl | T1 | [20,25,18] | xyy | 30 | 0.001 | 14.7175 | 0 | 14.7175 | 15.2783 | 15.2783 | 11.0881 |
| D5 | disc | T2 | [30]*8 | xxxxyyyy | 60 | 0.001 | 41.5384 | 1 | 41.5384 | 65.9393 | 65.9393 | 59.9717 |
| D6 | disc | T2 | [30]*8 | xxxxxyyy | 60 | 0.001 | 26.1379 | 1 | 26.1379 | 38.9545 | 38.9545 | 43.5876 |
| D7 | disc | T2 | [35,32,28,25,22,20,38,36] | xxxxxyyy | 60 | 0.001 | 32.0268 | 1 | 32.0268 | 42.9846 | 42.9846 | 49.4538 |
| D8 | disc | T2 | [30]*6 | xxxxyy | 60 | 0.2 | 34.4464 | 1 | 34.4464 | 38.9458 | 38.9458 | 46.1885 |
| C3 | ctrl | T2 | [30]*2 | xy | 60 | 0.001 | 2.2173 | 0 | 2.2173 | 2.2173 | 2.2173 | 1.7624 |
| C4 | ctrl | T2 | [35] | y | 60 | 0.2 | 6.9842 | 0 | 6.9842 | 6.9842 | 6.9842 | 6.9853 |
| D9 | disc | T1+T2 | [40]*8 | xxxxyyyy | 20 | 0.2 | 33.8059 | 1 | 97.3637 | 50.0716 | 129.8987 | 52.0805 |
| D10 | disc | T1+T2 | [50,45,40,35,30,25,20,15] | xyxyxyxy | 30 | 0.001 | 30.9408 | 1 | 50.8983 | 48.5845 | 68.9385 | 49.3132 |
| D11 | disc | T1+T2 | [60]*10 | xxxxxxxyyy | 25 | 0.01 | 17.4387 | 1 | 107.3010 | 28.0540 | 132.9510 | 38.0559 |
| F1 | disc | T1 | [40]*6 | xxxxyy | 5 | 0.001 | 5.0143 | 2 | 28.9601 | 7.5162 | 34.9560 | 7.4625 |
| F2 | disc | T1 | [40]*6 | xxxxyy | 30 | 0.001 | 10.8431 | 1 | 28.9601 | 15.1063 | 34.9560 | 22.2393 |
| F3 | ctrl | T1 | [40]*6 | xxxxyy | 40 | 0.001 | 28.9601 | 1 | 28.9601 | 34.9560 | 34.9560 | 40.7141 |
| F4 | disc | T2 | [30]*8 | xxxxxxyy | 60 | 0.001 | 5.7881 | 1 | 5.7881 | 9.4780 | 9.4780 | 18.9594 |
| F5 | ctrl | T2 | [30]*8 | xxxxxxxy | 60 | 0.001 | 21.0714 | 2 | 21.0714 | 21.0714 | 21.0714 | 12.3004 |
| E1 | edge | none | [1] | x | 1 | 0.5 | 3.0103 | 1 | 3.0103 | 3.0103 | 3.0103 | 1.9238 |
| E2 | edge | none | [60]*50 | "x"*50 | 60 | 0.001 | 177.5001 | 2 | 177.5001 | 177.5001 | 177.5001 | 177.5004 |
| E3 | edge | T1 | [60]*6 | xxxyyy | 1 | 0.001 | 2.9963 | 0 | 106.2926 | 2.9948 | 131.9426 | 3.0078 |
| E4 | edge | T2 | [30]*4 | xxyy | 60 | 0.99 | 63.4238 | 1 | 63.4238 | 67.9238 | 67.9238 | 71.1967 |
| E5 | edge | T1+T2 | [40]*6 | yyyyxx | 20 | 0.001 | 8.4800 | 0 | 28.9601 | 6.0958 | 34.9560 | 5.2462 |

What each case tests:
- **D1/C1 and F1-F3** form the value_c ladder at fixed pileup. T1 is off exactly when c >= 40 = max(Q) (T1off equals expected).
- **C2** is a second T1 control on a different pileup.
- **D5, D6, F4, F5** form the group-size ladder at fixed depth. T2 is off exactly when one group has a single read (F5, C3, C4).
- **D7** checks that ranking uses sorted quality, not input order (reordering the list leaves the output unchanged; see `scripts/freeze_vary.py`).
- **D9-D11** check the interaction: the cap equalises qualities, then the decay runs on the capped ranks.
- **E1** is the minimum input: one read, r = 0.5, exactly 10 log10 2.
- **E2** is maximum depth with one allele only. No read counts as an error under the called genotype, so both twists are invisible.
- **E3** has c = 1, so every read is nearly uninformative. The two homozygotes tie, and the output does not depend on the tie-break.
- **E4** has a prior near 1.
- **E5** swaps the x/y labels of D1 and gives the same output (symmetry).

Baseline scores (grader): naive 5/25 overall, **0/14 discriminating**; MAQ-literal recall 2/25 overall, **0/14 discriminating**; `solution/solution.py` 25/25.

---

## GATE 2 - Verification table

| Check | Result | Evidence |
|-------|--------|----------|
| Twist integrity | PASS | Freeze-vary, value_c only: T1 on/off gap 0% for c >= 40, then +132% to +478% for c <= 30. Group size only: T2 gap 0% with one minority read, then +49% to +64% with 2 to 4. Each twist has a regime where it is exactly 0 while the other is active (C1: T2 +20.7%, T1 0%; D5: T1 0%, T2 +58.7%), so they are disjoint and each can be isolated through the inputs. |
| No invented constant | PASS | See s8. theta cited; ploidy cited; Phred exact; prior uses the input r; c'_nk = 1 flagged as an authored simplification. |
| Recall-reconstructability | PASS | Textbook caller 0/14 discriminating; MAQ-literal caller 0/14 discriminating. |
| Citations fetched | PASS | SAM v1.6 PDF, MAQ PMC2577856 (HTML and Eq. 2 image), Li 2011 PMC3198575 (HTML and Eq. 2 image) all fetched on 2026-09-23. One quote per source, each under 15 words, all confirmed. |
| Citation matches code | PASS (with disclosure) | The cap, Eq. 2 form, theta, prior and Q_g match the MAQ text. The Li Eq. 2 likelihood matches the step 3 code. Substituting d_j into the error factors is disclosed as authored (s3). |
| Output key + neutralization | PASS | The key is "output". Inputs are value_a..value_d with neutral descriptions. Cold read: "list of positive ints + x/y labels + int + fraction" hints at some kind of scoring, but does not name a caller, and the cap and decay stay hidden. |
| Tier floor on honest counts | PASS | 4 substantive steps and 2 genuine twists, against a Med-High floor of 4 and 2. |
| Edge cases | PASS | 25 cases: 14 discriminating, 6 control (at least 2 per twist), 5 edge. The discriminating inputs sit where the twists fire (c < max Q; both groups of size 2 or more). |

**Verdict: PASS.**

---

## Files

```text
oracle/oracle.py           reference implementation (f, blackbox, _compute with ablation switches)
solution/solution.py       example solver: probe replay + independent reimplementation
grader/grader.py           scores a submission against golden/, reports naive and recall baselines
grader/baselines.py        naive (both twists off) and MAQ-literal recall baselines
golden/golden.json         25 cases with expected outputs, ablations and baselines
scripts/generate_golden.py regenerates golden/golden.json from the oracle
scripts/run_tests.py       golden vs oracle + invariants (symmetry, order, cap, bounds)
scripts/verify_task.py     structure, ASCII, golden integrity, solver 100%, baselines <= 50%
scripts/freeze_vary.py     numeric freeze-vary evidence per twist
```

Run: `python scripts/run_tests.py && python scripts/verify_task.py && python grader/grader.py`
