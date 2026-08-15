# CLAIM

**Registered 2026-08-03, before any measurement code was written.**

This file is a pre-registration. The text above the `## Amendments` heading is
frozen as of the first commit that produces data. Changes go in a dated
amendment at the bottom — never by editing the claim to match the result.

---

## C1 — the claim

Tensor-program transformations that are valid under real arithmetic — the class
accepted by Axon's operator propagation and Prism's parallelization axioms —
admit floating-point error exceeding the `rtol = atol = 1e-4` FP32 threshold
used to validate them, and the gap widens under the reduced precisions
(bf16/fp16) that production inference runs in.

## A1 — the alternative

No such transformation exists at realistic ML kernel shapes and activation
distributions. Real-arithmetic proof plus `1e-4` FP32 sampling is empirically
adequate, and reduced precision does not change the verdict.

**Both answers are results.** A1 winning means the field's shortcut is justified,
which is currently assumed rather than measured. The project is not built to make
C1 true.

---

## Pre-registered thresholds

Fixed now. Not movable after seeing data.

**T1 — failure criterion.** A transformation *fails* at a given (shape, precision,
input-distribution) cell if **relative error exceeds `1e-4` on ≥1% of output
elements**.

**T2 — realistic inputs.** "Realistic" means activations sampled from a trained
model, not `torch.randn`. Both are measured; the distinction is reported, never
collapsed.

**T3 — catch rate, not existence.** The headline number is the *rate at which
uniform random sampling catches a failure*, not merely whether a failure exists.
A failure that uniform sampling finds 1 time in 10⁴ is a different finding from
one it finds every time, and the difference is the point.

**T4 — float64 is truth.** Soundness of any individual finding is checked against
a float64 reference computation, never against the other variant. A disagreement
between two low-precision variants is not evidence about either one.

---

## Operational definitions

Pre-registration is worthless if the metric has free parameters left in it. These
are pinned now, with the same numbers, so that "exceeds 1e-4" cannot be quietly
reinterpreted later.

**Reference.** For a transformation pair `(f, g)` and input `x`, truth is
`T = f(x₆₄)` computed in float64 on inputs upcast from the precision under test.
Both `f` and `g` are evaluated at the test precision on the *same* rounded inputs.

**Per-element relative error.**

```
rel(v, T)ᵢ = |vᵢ − Tᵢ| / (|Tᵢ| + δ),    δ = 1e-30
```

`δ` guards division by zero only; it is 20+ orders below any activation magnitude
in the corpus and cannot mask a real error.

**Gate (the thing being tested).** The literal Axon/`torch.allclose` predicate at
the tolerance the literature uses:

```
pass  ⟺  ∀i.  |gᵢ − Tᵢ| ≤ atol + rtol·|Tᵢ|,    rtol = atol = 1e-4
```

Reported per cell: max relative error, fraction of elements over `1e-4`, the full
error distribution, and the boolean gate outcome.

**Why two metrics.** T1 (≥1% of elements over `1e-4`) is the pre-registered
failure criterion and is deliberately *stricter to trip* than the allclose gate,
which any single bad element fails. Reporting both prevents a one-element outlier
from being sold as a systemic result, and prevents a systemic result from hiding
behind an `allclose` that happens to pass.

**Scope of `g`.** Only transformations that are *exactly* equivalent over ℝ are in
the corpus. Every pair must agree to `1e-12` in float64 (Phase 1 exit criterion)
before it is eligible to be measured. A pair that disagrees in float64 is a bug in
my implementation, not a finding.

---

## What this is not

Not a claim that Mirage, Prism, or Axon is wrong. All three are explicit that they
operate over an idealized arithmetic — finite fields, axiomatic algebra, and ℝ
respectively — and all three say plainly that the shipped kernel is checked by
random testing. The contribution is measuring what that costs, which none of them
measures, and which each paper's own limitations section implicitly invites.

---

## Verdict — 2026-08-04

Registered 2026-08-03, measured across Phases 1–5. **Neither C1 nor A1 is correct as
written.** The answer is a third thing neither anticipated.

### C1, clause 1 — *transformations admit FP error exceeding the 1e-4 FP32 threshold*

**Satisfied, but by a mechanism that does not support its implied narrative.**

`matmul_k_tiling` exceeds the gate on **14% of draws [95% CI 9–22%]** under
gaussian sampling at the measured activation scale (σ≈2.7) at fp32, at a realistic
shape — no seeding required. At unit input scale the rate is 0/100, so the catch
rate is itself scale-dependent, which is the mechanism restated *(scale condition
added 2026-08-04 — [`AUDIT.md`](AUDIT.md) Finding 1)*. That is an existence proof
at realistic inputs.

But its **relative** error is 8.65e-07, a hundredfold *inside* the tolerance. It
fails because absolute error is relative error × output magnitude, and matmul output
grows as σ²√K. The transformation is not numerically dangerous; the threshold is
scale-dependent.

### C1, clause 2 — *the gap widens under reduced precision*

**True and uninformative.** Floors move ~1e-6 → ~1e-2 → ~1e-1, but `d/floor ≈ 1` for
two-thirds of the corpus, so the transformation contributes nothing beyond what
precision itself costs. And Axon states its tolerance for FP32 only, so the clause
measures against a gate nobody applies at bf16.

### A1 — *no such transformation at realistic shapes and activation distributions*

**Holds for distributions. Fails for shapes.**

- *Activation distributions* — **A1 holds.** Every real site passes with ~10×
  headroom, including `post_ln`, zero-mean by construction and reaching row condition
  number **368,927** — 400× more ill-conditioned than uniform sampling produces.
- *Kernel shapes* — **A1 fails.** `matmul_k_tiling` fails at K=512 stochastically and
  at K=2048 outright. Production LLMs run K = 4096–16384.

### What is actually true

**The gate is dimensionally incoherent.** `atol = 1e-4` is an absolute constant
applied to tensors of arbitrary magnitude. Whether a *correct* ℝ-equivalent
transformation passes depends on output scale — which depends on K and input
variance — and not on its soundness. The same code, unchanged:

| K | abs err | verdict |
|---|---|---|
| 128 | 1.34e-05 | pass |
| 512 | 7.63e-05 | pass (0.8× atol) |
| 2048 | 2.37e-04 | **fail** |

Across all six pairs the relative errors span 3.3e-07 – 1.0e-06, a factor of 3. The
absolute errors span four orders of magnitude, entirely because the output magnitudes
do. The gate measures how large the numbers are, not whether the rewrite is sound.

**The actionable consequence:** a relative-only gate, or an `atol` scaled to output
magnitude, would make the check measure the transformation. A fixed absolute
tolerance becomes progressively unusable as models scale.

### Retracted along the way

- **"The field states 1e-4 but practises 1e-2, a 100× gap."** Category error — Axon
  (AWS/UIUC) compared against MPK runtime-kernel tests (CMU): different groups,
  different subsystems, different artifact. See ERRATA §1.5.
- **"C1 survives at fp32 via seeding."** The seeded arrangements sit **+10.4σ**
  outside the real distribution. They show these transformations *can* diverge, not
  that they *do*.

### Confidence

Highest on the scale-dependence result — it follows from measurement plus dimensional
analysis, so it cannot be undone by having misread anyone's code. Lowest on anything
concerning whether real workloads reach adverse arrangements, which this project did
not establish and did not attempt to.

---

## Amendments

### 2026-08-03 — Phase 1 float64 bug check is gated on scale-relative error

**What changed.** "Every pair must agree to `1e-12` in float64" (above, under *Scope
of `g`*) did not say *which* error measure. It is now gated on

```
max_i |g_i − T_i|  /  max_i |T_i|          (scale-relative)
```

at the same `1e-12`, not on max per-element relative error.

**Why.** Per-element relative error is not well posed as an agreement check when an
output passes through cancellation: the denominator can be arbitrarily near zero and
the ratio is then unbounded *for a bit-correct implementation*. Found on the first
run — `scalar_past_matmul` at the MLP shape reported `7.5e-11` and tripped the gate.
The element behind it has magnitude `1.3e-05` in a tensor whose max is `4.88`, with
an absolute difference of `9.8e-16` — float64 rounding noise at the scale of the
terms being summed. Under the scale-relative measure every pair reads `1e-15`–`3e-15`,
a few float64 eps, with three orders of headroom to the gate.

This is the same reasoning as the `atol` term in the registered gate above, applied
to the reference check. Both measures are printed by `tests/test_corpus.py` so the
choice stays visible.

**What did not change.** T1–T4 and the `rtol = atol = 1e-4` gate are untouched. Those
govern fp32/fp16 measurement against float64 truth; this concerns only the Phase 1
check that my two implementations are the same function over ℝ. No threshold that
bears on C1 vs A1 has moved.

**Related bug found by the same run.** Inputs were being drawn at float32 and cast
*up* to float64, leaving 29 spare mantissa bits — which made float64 summation of a
few thousand such values exact in every order, so the reduction pairs reported a
perfect `0.000e+00`. Passing for the wrong reason. `corpus._randn` now draws at
float64 and rounds down; `test_float64_reference_actually_rounds` pins it.

### 2026-08-08 — detection arm registered (before any mutant is run)

**Trigger.** External review (agent-notes, 2026-08-08): the study measures false
rejection of *valid* rewrites but never measures detection of *invalid* ones, so
"catch rate" lacked its complement. This amendment registers the missing arm.

**Question.** At the literal Axon gate (`rtol = atol = 1e-4`, fp32, vs the correct
baseline), what fraction of random draws detects each of six plausible injected
bugs (mutants): matmul dropped last tile; matmul dropped last column; LayerNorm
Bessel divisor (n−1); LayerNorm eps added to std instead of variance; online
softmax missing the rescale correction; reduction dropping its last element.

**Metric.** Per-mutant per-draw detection rate over 100 independent draws, Wilson
95% CI. A mutant must first differ from the correct baseline in float64 (sanity:
it is a real bug, not a no-op).

**Prediction (registered now).** Gross mutants (dropped tile/column/element,
missing rescale, Bessel) are detected on ≥95% of draws. **At least one plausible
mutant — eps-to-std — evades detection on ≥95% of draws**, because its output
shift (~1e-5 relative at unit variance) sits below the gate. If that holds, the
gate's failure is discrimination, not leniency: it rejects valid rewrites at
large K while passing a real bug class.

**No prior threshold moves.** T1–T4 and both gate definitions are untouched; this
arm adds a complement, it does not reinterpret the existing one.

### 2026-08-08 (second amendment) — frontier, K extension, decomposition, library reference

**Trigger.** External review #2 (agent-notes). Registered before any of these run.

**E1 — tolerance frontier.** Sweep `atol, rtol ∈ {1e-8 … 1e-1}` (log grid). Per
grid point, over ≥50 draws per program: rejection rate on the valid corpus (six
pairs at fp32 sweep shapes, plus the K=2048 tiling cell) and miss rate on a
mutant severity continuum (drop-j elements/columns for j ∈ {1,2,4,8,16,32};
divisor n−j for j ∈ {1,2,8}; eps-on-std for eps ∈ {1e-5,1e-4,1e-3}).
**Prediction:** no grid point achieves zero valid-rejection and zero mutant-miss
simultaneously. Specifically, any point that passes the K=2048 valid cell misses
the eps-1e-5 mutant.

**E2 — K extension.** K ∈ {2048, 4096, 11008} at M=N=64, unit scale, 100 draws.
**Prediction:** failure rate ≥95% of draws at K=4096 and K=11008.

**E3 — per-element decomposition.** At K=512, σ=2.67, 100 draws: per-element
exceedance rate p̂ and observed tensor-level rate. **Prediction:** the iid model
`1−(1−p̂)^4096` predicts the observed tensor rate within its 95% CI — i.e., the
output-count effect is extreme-value aggregation, not a new numerical effect.

**E4 — library reference.** Re-run the softmax reference comparison with
`torch.sum` (a real library order) as the normalizer. **Prediction:** floor
within 2× of the strided-tree reference's floor, same verdicts as tree.

No prior threshold moves. Analyses in the paper are marked registered, amended,
or exploratory per this file's history.

### 2026-08-03 — the gate compares against the baseline, not against truth

**What changed.** The *Gate* defined above compares the variant against float64
truth `T`. That is not what the systems do. The gate is now the **differential**,

```
pass  ⟺  ∀i.  |gᵢ − bᵢ| ≤ atol + rtol·|bᵢ|,    rtol = atol = 1e-4
```

where `b` is the **baseline evaluated at the same precision as `g`**. The
as-registered form (against `T`) is **retained and co-reported**, never replaced.

**Why — verified against the paper, not from memory.** Axon §4.6 states its check
literally as `|emitted_code − reference| ≤ atol + rtol|reference|` with
`rtol = atol = 10⁻⁴`, on FP32 — comparing the emitted kernel against a *reference
implementation*, not against a high-precision oracle. Mirage tests a candidate
μGraph against the input program; Prism tests generated kernels against the
reference. All three compare **candidate against original**, both in floating point.

C1 is a claim about "the `rtol = atol = 1e-4` FP32 threshold **used to validate
them**." An instrument that compares against float64 truth measures a gate nobody
uses, so leaving this in place would mean not testing C1 at all.

**This is a correction of a factual error about the systems, not a re-tuned
parameter.** The trigger was textual — readable in Axon §4.6 with no data — and it
should have been caught in Phase 0. `1e-4`, the `≥1%` bar, and T1–T4 are all
unchanged. Only *what is compared* changes.

**Direction of effect: measured, and it cuts both ways.** At `split_reduction/fp32`
the differential (1.53e-06) is *larger* than variant-vs-truth (2.59e-07), making C1
easier there. At `layernorm_variance/bf16` it is 1.60e-02 against 3.45e-01, making
C1 far harder. The fp32 verdict does not move: all six pairs still pass. The
correction did not manufacture a positive result.

**Dual-reporting rule.** Every cell reports the gate under **both** definitions.
Both quantities are computed anyway, so this costs nothing and makes the amendment
auditable rather than something a reader must take on trust. Any headline cell where
the two disagree is itself reported.

**Third quantity, added.** With the baseline now evaluated at test precision, its
own deviation from truth — the **precision floor** — is available for free. It is
recorded per cell, along with `d/floor`. This is what separates "this precision is
inaccurate" from "this transformation is inaccurate," and at reduced precision the
distinction is the entire result: a differential that merely tracks the floor says
nothing about the transformation.

### 2026-08-14 — the verdict's K=2048 line is superseded (retraction #4)

**What changed.** The verdict of 2026-08-04 (above) states that
`matmul_k_tiling` "fails at K=2048 outright" and its table marks K=2048
**fail**. That reading compared the maximum absolute difference against
`atol` alone and ignored the `rtol` term — the quantity confusion this
project itself criticizes. Under the literal elementwise rule the corrected
result (E2, registered by the 2026-08-08 amendment before its run; 100 draws
per point, M=N=64, unit scale) is:

| K | rejected | 95% CI |
|---|---|---|
| 2048 | 0/100 | [0%, 4%] |
| 4096 | 48/100 | [38%, 58%] |
| 11008 | 100/100 | [96%, 100%] |

Retraction #4, [`ERRATA.md`](ERRATA.md) §2.

**What did not change.** No threshold moved. The verdict's mechanism — the
absolute term makes the outcome scale-dependent — is unchanged, and the
corrected onset lands on Llama-2-7B's contraction widths (4096 projection,
11008 MLP) with no extrapolation. The frozen text above stays as written;
this amendment is the forward pointer the house rules require.

### 2026-08-14 — C1-C3 registered (before any run)

**C1 — reference sensitivity with draws.** 100 draws per precision
(fp16, bf16) at the mlp shape (512, 1024). Prediction: at fp16 the
online-softmax candidate fails against the sequential reference and
passes against the tree and torch.sum references on >=95% of draws
each, and the floor ordering seq > tree > torch.sum holds on >=95% of
draws. At bf16 the candidate fails against all three references on
>=95% of draws: the precision floor alone exceeds a 1e-4-class gate.

**C2 — detection at activation scale.** The six-mutant arm at
sigma = 2.67. Prediction: the five gross mutants stay at >=95%
detection; ln_eps_to_std still evades on >=95% of draws. LayerNorm
normalizes scale away, and larger row variance shrinks the eps shift
relative to std.

**C3 — rejection across K and scale.** The valid K-tiled matmul,
M=N=64, fp32, 100 draws per cell. Axis 1: K in {512, 1024, 2048, 4096}
at sigma = 2.67. Axis 2: sigma in {0.5, 1.0, 2.0, 2.67, 4.0} at K=512
(AUDIT step 5). Prediction: rejection is non-decreasing in K and in
sigma; at sigma = 2.67, K = 2048 is rejected on >=95% of draws; the
sigma sweep at K = 512 reads 0/100 at sigma <= 1.0 and contains the
committed 14% at sigma = 2.67 inside its CI.

No prior threshold moves.

### 2026-08-14 (second amendment) — exact tolerance separability registered

**Trigger.** External review #4: the frontier's 64-point grid does not by itself
establish that the rule family cannot separate the classes; a separator could sit
between grid points. The reviewer sketches the exact test; this registers it.

**E5 — exact separability on the recorded corpus.** For each recorded element,
acceptance at (atol, rtol) is the half-plane atol + rtol*|b| >= |g-b|. Per draw,
the acceptance boundary is the upper envelope max_i(|g-b|_i - rtol*|b|_i),
computed exactly from the convex hull of that draw's (|b|, |g-b|) points. A
separator exists iff some rtol in [0, 100] has max(F(rtol), 0) < G(rtol), where
F is the pooled valid envelope and G the minimum mutant-draw envelope. Between
sample points, no-separation is certified per interval using convexity (each
envelope lies below its chord; F is non-increasing).

**Prediction.** No separator exists for any atol >= 0, rtol in [0, 100]: every
interval is certified, and sup(G - max(F,0)) < 0. Mechanism: the valid corpus
contains near-zero-|b| violations whose required atol exceeds every eps-mutant
draw's entire envelope at every rtol before both fall below zero.

No prior threshold moves. Scope: the recorded corpus, rtol <= 100.

### 2026-08-14 (third amendment) — cross-draw quantifiers and grid completion

**Trigger.** External review #5 (Rev. 5 direction): E5 tested one cross-draw
semantics only. The cross-draw aggregation rule is itself an unspecified
protocol coordinate, so both plausible quantifiers must be computed.

**E6 — two-criterion envelope separability.** On the recorded corpus, with
AV(r) the pooled valid envelope: criterion EVERY (each mutant draw rejected)
uses AM_every(r) = min over mutant draws of the draw envelope; criterion SOME
(each mutant caught on at least one draw) uses AM_some(r) = min over mutants
of the max over that mutant's draw envelopes. A separator exists under a
criterion iff some rtol in [0, 100] has max(AV(r), 0) < AM(r). Interval
certificates as in E5 (chord bounds; AV non-increasing).
**Prediction:** no separator under either criterion; sup gaps negative under
both, with the SOME gap less negative than the EVERY gap (AM_some >= AM_every
by construction). Either outcome is reportable; if SOME separates, the
finding becomes: the unspecified cross-draw quantifier changes separability.

**E7 — grid completion.** Two cells to complete Table 1's rectangle, 100
draws each, literal rule, M=N=64, fp32. **Predictions:** unit scale K=1024:
0/100 [0-4%]. Scale-matched gaussian (sigma 2.67) K=11008: 100/100 [96-100%].

No prior threshold moves.

### 2026-08-14 (fourth amendment) — full cross-draw semantics box and tail closure

**Trigger.** External review #6: (a) naming the cross-draw semantics
operationally exposed that E6's EVERY criterion paired the strictest rule on
both classes (valid must pass all draws AND each mutant must fail all draws),
which is the separator-hardest corner, not a single coherent validator; the
separator-easiest corner (valid accepted if any draw passes, mutant caught if
any draw fails) was never computed. (b) The rtol <= 100 cutoff can be closed
analytically if the mutant envelopes are nonpositive at r = 100.

**E8 — the 2x2 semantics box.** With F_all(r) = max over valid draws of T_d(r)
(all-draws-must-pass for valid), F_any(r) = max over valid programs of the min
over that program's draws (any-draw-may-pass), G_every(r) = min over mutant
draws (mutant caught only if every draw fails), G_some(r) = min over mutants of
the max over that mutant's draws (mutant caught if at least one draw fails):
compute the separation gap sup_r (G - max(F, 0)) and interval certificates for
all four (F, G) corners on the recorded separability corpus. All envelopes are
non-increasing, and each per-program/per-mutant envelope is convex
piecewise-linear, so the single-witness endpoint bound plus F-monotonicity
certificate applies at every corner.

**Tail closure.** If G_every(100) <= 0 and G_some(100) <= 0, then by
monotonicity G(r) <= 0 <= max(F(r), 0) for all r >= 100 under every corner,
removing the rtol <= 100 cutoff.

**Predictions.** All four corners: no separator; every interval certified. The
separator-easiest corner (F_any, G_some) has the least negative sup gap, on
the order of -1e-4 near r = 0. Both G values at r = 100 are nonpositive, so
the headline claim holds for all rtol >= 0.

No prior threshold moves. Corpus: the E5/E6 separability corpus (the six
pairs at their mlp shapes plus the K=2048 tiling cell, 50 unit-scale draws
each; the 16 parameterized frontier mutant instances, 50 draws each). Adding
mutants can only lower G, so non-separability over the 16 extends a fortiori
to supersets including the two gross detection-arm instances.
