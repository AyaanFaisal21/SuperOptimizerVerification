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

## Amendments

*(none yet — dated entries only, appended)*
