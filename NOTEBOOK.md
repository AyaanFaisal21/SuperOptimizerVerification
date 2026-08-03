# Notebook

Dated entries. **Prediction before each run, outcome after.** The prediction is
written and committed before the run executes — that ordering is the whole point
of the file, and it is visible in git history whether or not it was honored.

Format:

```
## YYYY-MM-DD — short title

**Setup.** What is being run, on what.
**Prediction.** What I expect, with a number where possible. Written first.
**Outcome.** What happened.
**Read.** What it means, including "nothing" if that is the answer.
```

---

## 2026-08-03 — Phase 0, project registered

**Setup.** `cornfieldV2` as the repo root; it already held the six-paper working
set (`papers/README.md` is the committed index, PDFs gitignored). Consolidated
`BACKGROUND.md` and `ROADMAP.md` in from `~/Documents/fp-verification-gap/`, which
was sitting inside the home-directory git repo and would never have been
separately publishable. `CLAIM.md` written and dated before any measurement code.

Toolchain on this machine: Python 3.13.1, torch 2.8.0, numpy 2.3.1, mpmath 1.3.0.
**No CUDA** — `torch.cuda.is_available()` is False; MPS only. The 2060 (Turing
sm_75) named in the roadmap is a different box, the Windows one that
`cornfield/winbuild.bat` targets.

**Prediction.** None — this is setup, nothing is being measured. Recording it so
the first real entry has a baseline to point at.

**Outcome.** Phase 0 complete. Repo has a dated claim in history.

**Read.** One thing that surfaced early and matters later: Phases 0–1 are pure
float64 CPU work and are unaffected by the missing GPU, but **Phase 2's fp16
native-vs-simulated validation gate assumes CUDA**. On CPU, torch's float16 ops
frequently upcast to fp32 internally, which would make "native fp16" here itself a
kind of simulation — and comparing a simulation against a simulation proves
nothing. That gate has to run on the 2060, or the bf16 claim gets dropped per the
roadmap's own kill criterion. Not a blocker now; flagged at the Phase 1→2 boundary
so it is decided deliberately rather than discovered.

---

## 2026-08-03 — Phase 1, corpus built and both exit criteria run

**Setup.** Six pairs in `fpgap/corpus.py`, each exactly equal over ℝ, each with a
record of which system accepts it and on what grounds: split reduction
(part/red/comb), reassociation (seq vs tree), scalar-past-matmul (Axon §4.2),
LayerNorm two-pass vs fused one-pass (from `cornfield/autotune_layernorm.py`),
softmax naive vs online (from `TransformerOp/kernels/attn_ext.cu:81`), and matmul
K-tiling. Accumulation order is controlled explicitly in `fpgap/accumulate.py`
rather than delegated to `torch.sum` — torch uses a blocked pairwise cascade on CPU
and a tree reduction on CUDA, and measuring torch against torch would measure
torch's internals invisibly. `tree_sum` is strided halving specifically because
that is what `ln_kernel`'s `seg_s[lane] += seg_s[lane + st]` does.

**Prediction (written before the first run).** All 18 cells (6 pairs × 3 shape
classes) agree in float64 at ≤1e-14, comfortably inside the 1e-12 gate. The
negative control shows the three accumulation orders differing in fp32 at roughly
1e-7 — a few hundred × fp32 eps for reductions of this length.

**Outcome.** Negative control: 1.0e-06 seq-vs-tree, 9.6e-07 seq-vs-chunked. About
10× larger than predicted, which is the right direction for length-4096 rows and
means the instrument is, if anything, more sensitive than assumed.

The equivalence check **failed on the first run** — `scalar_past_matmul/mlp` at
7.5e-11. Two distinct problems behind it, both mine, neither a finding:

1. *The metric was underspecified.* Max per-element relative error is unbounded at
   cancellation-produced near-zeros. The offending element has magnitude 1.3e-05 in
   a tensor whose max is 4.88, with an absolute difference of 9.8e-16 — float64
   rounding noise at the scale of the summands. Gate moved to scale-relative error
   at the same 1e-12; see CLAIM.md amendment. Both measures are still printed.
2. *The reference was not rounding.* Inputs were drawn at float32 and cast **up** to
   float64, leaving 29 spare mantissa bits, so float64 summation of a few thousand
   such values is exact in every order. `split_reduction` and `reassociation` read
   exactly `0.000e+00`. They were passing because the reference had too much
   headroom to round — not because the pair is equivalent. Fixed by drawing at
   float64 and rounding down, which also gives every precision the same underlying
   real values. Pinned by `test_float64_reference_actually_rounds`.

Post-fix: all 18 cells at 6e-16 – 3.5e-15 scale-relative, three orders inside the
gate, and the reduction pairs now show genuine float64 rounding instead of zeros.
4 tests pass.

**Read.** The second problem is the one worth remembering. It produced a *cleaner*
result than the truth — a perfect zero — and a perfect zero on an equivalence check
reads as success. If the same flaw had survived into Phase 4 it would have
suppressed real error at the reduction pairs and pushed the whole sweep toward A1,
which is the direction I would be least likely to interrogate. That is the argument
for the negative control being a first-class exit criterion rather than a nicety:
check 1 alone could not have caught it, because check 1 is satisfied by an
instrument that reads zero.

Not a result, but recorded because it was measured before the fix and I do not want
to rediscover it later: no pair is degenerate in fp32 — all 18 cells diverge
(8e-08 – 1.2e-06 scale-relative), so nothing in the corpus is dead weight. Fraction
of elements over 1e-4 relative is nonzero in 10 cells but peaks at 0.20%, under
T1's 1% bar. Those are the same near-zero denominators, on `randn`, at fp32, with
no seeding — i.e. the condition under which the field's assumption is *most* likely
to hold. Phases 3–5 are what test it.

---
