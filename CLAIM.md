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
