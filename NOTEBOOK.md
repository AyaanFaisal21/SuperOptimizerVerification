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

## 2026-08-03 — A10 acquired; hardware characterisation (prediction only)

**Setup.** Hardware decision resolved: a Lambda `gpu_1x_a10`, us-east-1. Verified
NVIDIA A10, **compute capability 8.6** (Ampere), 23 GB, driver 580.105.08, CUDA
12.8, Python 3.10.12, torch **2.7.0**. Note the version skew against this Mac
(torch 2.8.0, Python 3.13.1) — recorded because reduced-precision defaults have
changed between torch releases before, and any cross-machine comparison has to
account for it.

Ampere settles the bf16 question: native in hardware, so nothing is simulated. The
roadmap's Phase 2 simulation sub-task, its stated bf16 limitation, the
corresponding Phase 6 threat-to-validity, and the second kill criterion are all
moot. `tools/probe_hardware.py` is the replacement — it characterises the machine's
arithmetic so that record can travel with every result.

**Prediction (written and committed before the probe runs).**

1. **TF32.** `torch.backends.cuda.matmul.allow_tf32` reads **False** — the default
   flipped in torch 1.12 — and `cudnn.allow_tf32` reads **True**. Empirically, fp32
   matmul vs float64 lands near **1e-7** with TF32 off and near **1e-3** with it on,
   a ratio of roughly 10³–10⁴. If the flag instead reads True on this build, every
   fp32 number this box would have produced is a ~10-bit-mantissa baseline and the
   whole fp32 arm would have been quietly wrong.
2. **Storage rounding.** The stagnation probe returns exactly 1.0 for both fp16 and
   bf16 on device, matching the Mac. Elementwise accumulation is genuinely in-type.
3. **Matmul accumulate.** fp16 and bf16 matmul error against exact arithmetic on the
   same rounded inputs is ~1e-3, dominated by rounding the *output* to the narrow
   type rather than by accumulation. I therefore expect the two settings of
   `allow_*_reduced_precision_reduction` to differ **little or not at all** at
   K=4096 — the tensor-core MMA accumulates in fp32 either way and the flag governs
   only the split-k reduction. If that holds, "accumulation width as a switchable
   experimental axis" is weaker than I claimed when recommending this box, and I
   should say so rather than quietly drop it.
4. **Determinism.** Repeated identical matmuls are bitwise equal within a process.
5. **float64.** Works; roughly 20–40× slower than fp32 (A10 is nominally 1:32).

**Outcome.** Probe run on the A10. Predictions 1, 2, 4, 5 held; 3 held in the
direction I flagged against myself, and a fourth thing turned up that neither the
prediction nor the roadmap anticipated.

| Check | Predicted | Measured |
|---|---|---|
| `cuda.matmul.allow_tf32` | False | **False** ✓ |
| `cudnn.allow_tf32` | True | **True** ✓ |
| fp32 matmul, TF32 off | ~1e-7 | **3.98e-07** ✓ |
| fp32 matmul, TF32 on | ~1e-3 | **2.99e-04** (ratio 750×, I said 10³–10⁴) |
| fp16/bf16 stagnation probe | stagnates | **stagnates, both** ✓ |
| reduced-precision-reduction flags | little or no difference | **bit-identical** ✓ |
| matmul repeatable bitwise | yes | **yes** ✓ |
| fp64 slowdown | 20–40× | **35.2×** ✓ |

Corpus re-checked on CUDA: **18/18 cells pass** the float64 equivalence gate at
2.8e-16 – 2.4e-15, same magnitudes as CPU. Equivalence over ℝ is not a
platform-specific accident.

**Read — two claims of mine are now falsified, and both were things I told Ayaan
when recommending this machine.**

*First: the A10 does not make accumulation width a switchable experimental axis.*
I argued `allow_fp16_reduced_precision_reduction` and its bf16 twin would give the
in-type vs fp32-accumulate contrast on real hardware. Measured, the two settings
are **bit-identical** — 3.972063525748143e-04 either way. Almost certainly cuBLAS
never selected a split-k kernel at 512×4096×512, so the flag was a no-op; the
tensor-core MMA accumulates in fp32 regardless. Precise statement: the flag has no
observable effect *at this shape*, which is not the same as accumulation width
never mattering. But the switchable axis I promised is not there.

*Second: CPU does not overstate error relative to tensor cores.* I claimed torch
CPU accumulating bf16 in bf16 would inflate error versus a GPU accumulating in
fp32. Paired test, identical input bits shipped to both machines:

| dtype | Mac M3 CPU | A10 CUDA |
|---|---|---|
| fp32 | 2.6421e-06 | 8.7677e-07 |
| fp16 | **3.9533e-04** | **3.9533e-04** |
| bf16 | **3.0960e-03** | **3.0960e-03** |

Identical to every printed digit at fp16 and bf16. The reason is that **rounding
the output to the narrow type dominates everything else**: half an ulp is 4.88e-04
for fp16 and 3.9e-03 for bf16, and the measurements sit right at that scale. The
accumulation pathway is entirely masked. Only fp32 shows a genuine platform gap
(CPU 3× worse), because fp32 output has enough mantissa not to mask it.

This narrows to matmul specifically. The reduction pairs produce one scalar per row
from thousands of accumulation steps, so output quantisation cannot dominate there
the way it does here — that has to be measured separately, not assumed either way.

**The thing nobody predicted, and it may reframe half the claim.** At fp16 and
bf16, an **untransformed matmul already exceeds the `1e-4` gate** — 3.95e-04 and
3.10e-03 against exact arithmetic on its own inputs. Before any transformation is
applied. The identity fails.

C1's second clause says the gap "widens under the reduced precisions that
production inference runs in." That is at risk of being trivially true: precision
alone blows the tolerance, no superoptimizer required. The sharp question is
whether a transformation adds error **beyond what the baseline already suffers at
that precision** — i.e. err(variant vs truth) against err(baseline vs truth), not
just against the absolute gate.

No threshold moves. T1 and the registered gate stand exactly as written. What this
changes is the *analysis plan*: the differential has to be co-reported with the
absolute gate, or the bf16 column will be a row of failures that says nothing about
transformations. And the observation stands on its own as a result — Axon validates
at `rtol=atol=1e-4` in FP32, and that gate is simply **inapplicable** at the
precision production actually runs at, independent of any superoptimizer. What the
field substitutes at bf16 is not stated in any of the three papers.

**Also worth recording.** TF32 on costs 750× accuracy and lands at 2.99e-04 — over
the gate by itself. `cudnn.allow_tf32` is True by default and left that way; it does
not touch matmul, but any future conv path would need the same treatment. Every
script in this repo pins `matmul.allow_tf32 = False` explicitly rather than
inheriting it.

---

## 2026-08-03 — Phase 2 closed; Phase 4 synthetic arm (prediction)

**Setup.** Three things landed before this run. `CLAIM.md` amended: the gate is now
the differential (variant vs baseline at the same precision), verified against
Axon §4.6's literal text, with the as-registered form retained and co-reported.
`fpgap/harness.py` emits floor / total / differential per cell plus both gate
readings and both T1 readings. `tools/validate_reference.py` confirms float64 is
adequate as truth — worst drift against mpmath at 50 digits is **8.2e-16, 4× float64
eps**, twelve orders below the 1e-4 gate. That closes Phase 2's exit criteria.

Now the Phase 4 synthetic arm: 6 pairs × 3 shape classes × 3 precisions = 54 cells,
`randn` inputs. The realistic-activation arm waits on the checkpoint currently
training on the A10.

**Disclosure.** The MLP-shape column is *not* a blind prediction — I measured those
18 cells while answering a question about the gate correction, and they are quoted
in the previous entry. The predictions below are therefore about what I have **not**
seen: the small and attention shape classes, and how the ratios move with shape.

**Prediction.**

1. **fp32: all 18 cells pass the Axon gate**, at every shape class. No fp32 cell
   trips T1 (≥1% of elements over 1e-4 relative). The floor is ~1e-6 against a 1e-4
   gate — two orders of headroom — and `randn` cannot supply the ~100× amplification
   needed to close it.
2. **fp16 and bf16: all 36 cells fail the Axon gate.** Not because the
   transformations are bad but because the differential tracks the precision floor.
3. **`d/floor ≈ 1` for the reduction pairs at every shape.** Both sides scale with
   the same accumulation length, so the ratio should be roughly shape-invariant even
   as the absolute numbers grow with row length. If this instead drifts with shape,
   my reading of why the reductions behave this way is wrong.
4. **`matmul_k_tiling` keeps `d/floor > 1` and it tracks K.** K is 256 / 768 / 512
   for small / mlp / attention, and d/floor was 3.48 at fp16-mlp, so I expect the
   attention and small cells to come in lower than mlp — ordering by K rather than
   by shape-class name.
5. **The two gate definitions agree on the verdict in the large majority of cells.**
   Where they disagree, it should be `layernorm_variance`, whose floor is enormous
   at reduced precision while its differential is small.

**Outcome.** 54 cells in 7 s (CPU). Raw records in `results/sweep_randn.json`.

| Prediction | Result |
|---|---|
| 1. fp32: all 18 pass, no T1 failures | **correct** — 18/18 pass, 0 T1 |
| 2. fp16/bf16: all 36 fail | **correct** — 36/36 fail, all trip T1 |
| 3. `d/floor ≈ 1` for reductions, shape-invariant | **correct** — split 0.82–1.10, reassoc 0.94–1.09 |
| 4. `matmul_k_tiling` `d/floor` orders by K | **FALSIFIED** |
| 5. gates mostly agree; disagreements at layernorm | **half** — 51/54 agree, but the 3 are softmax |

**Prediction 4 is wrong and I do not have an explanation.** I expected `d/floor` to
track K, so attention (K=512) should sit below mlp (K=768). Measured: attention
**4.01** at both fp16 and bf16, against mlp 3.48/2.10 and small (K=256) 2.07/2.08.
K=512 exceeds K=768. Whatever drives the amplification, it is not K-tile count
alone — output width (N = 3072 mlp vs 256 attention) and the max-based normaliser
are both confounded with it here. Recorded as open; it needs a controlled sweep
over K with M and N pinned, which the current shape classes cannot separate.

**Prediction 5 was half-reasoned.** The majority-agreement part held (51/54). My
guess at *which* pair would disagree was wrong — I said `layernorm_variance` because
its floor is enormous at reduced precision. The three disagreements are all
`softmax_online` at fp16, one per shape class.

**The disagreement cells are the headline result.** All three read
`axon = FAIL, as-registered = PASS`. The online-softmax variant is close enough to
float64 truth to pass a 1e-4 gate against *truth*, and fails only when compared
against the naive baseline — because the baseline is the inaccurate side. At
mlp/fp16 the variant is **15× closer to truth** than the baseline it is checked
against (6.03e-04 vs 8.94e-03).

That is the "gate rejects the more accurate kernel" claim, demonstrated in a cell
where the two gate definitions actually diverge — which is precisely why the
dual-reporting rule from this morning's amendment was worth adopting. Under the
as-registered definition alone this result is invisible.

**Scope it honestly.** Softmax outputs have magnitude ~1/N, so `atol = 1e-4` is
comparable to the output values themselves and the gate is unusually permissive
*against truth* for this operator. The differential still fails because the
baseline's own error is large in absolute terms. So the mechanism is partly an
atol artifact specific to small-magnitude outputs, not purely a statement about
reordering. The finding stands; the explanation needs that caveat attached.

**`d/floor` splits the corpus into three classes,** which is the metric doing the
job it was introduced for (mean over fp16+bf16 cells):

| class | pairs | `d/floor` |
|---|---|---|
| reordering only | `reassociation` 1.00, `split_reduction` 0.97, `softmax_online` 0.88 | ≈ 1 |
| genuinely amplifying | `scalar_past_matmul` 1.80, `matmul_k_tiling` 2.96 | > 1 |
| suppressed | `layernorm_variance` 0.40 | < 1 |

The first class adds nothing beyond what precision costs — the differential just
tracks the floor, and a gate at 1e-4 fails them for being *different*, not wrong.
The second genuinely amplifies. The third is the interesting one: at bf16-mlp
`layernorm_variance` has a floor of 3.55e-01 and a differential of 1.60e-02, so
both forms are catastrophically wrong and wrong *together*. On zero-mean `randn`
the one-pass form is not the weaker variant — `E[x²] − μ²` has no cancellation when
μ ≈ 0. Its instability needs biased inputs, which is what the activation fixtures
and Phase 5 are for.

**Read.** At fp32 with synthetic inputs, **A1 holds cleanly** — every pair passes
with ~2 orders of headroom and nothing approaches T1. Whether C1 survives at fp32
now rests entirely on Phase 5 seeding, exactly as predicted when the gate was
corrected. At reduced precision the interesting quantity is not the gate at all but
`d/floor`, because 36/36 failures where two-thirds of the corpus sits at
`d/floor ≈ 1` is a fact about the tolerance, not about the transformations.

---
