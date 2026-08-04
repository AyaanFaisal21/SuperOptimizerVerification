# Progress against the roadmap

Tracks [`ROADMAP.md`](ROADMAP.md) phase by phase, against that document's own exit
criteria. Updated 2026-08-03.

The purpose of this file is to be honest about what is *not* done and what changed
relative to the plan. Where the outline turned out to be wrong, that is recorded as
a finding rather than quietly absorbed.

| Phase | Outline estimate | Status | Exit criterion met? |
|---|---|---|---|
| 0 — Setup | ½ day | **complete** | yes — claim committed dated, `f9adefc` |
| 1 — Transformation corpus | 3–5 days | **complete** | yes — 6 pairs, 18/18 cells inside 1e-12 |
| 2 — Reference & precision harness | 3–4 days | **complete** | yes — harness built, float64 validated vs mpmath |
| 3 — Realistic inputs | 2–3 days | **complete** | yes — fixture + regeneration script committed |
| 4 — Main sweep | 4–6 days | **complete** | yes — 54 synthetic + 24 activation + 16 matched-control cells |
| 5 — Seeded-input experiment | 3–4 days | **complete** | yes — catch rates, uniform vs seeded, per transformation |
| 6 — Writeup | 5–7 days | Background only | no |

**Five of six phases complete**, well ahead of the outline's calendar. Only the
writeup remains.

**A1 largely holds at fp32; C1 is not established.** Every real activation site
passes with ~10× headroom — including `post_ln`, which is zero-mean by construction
and reaches row condition number **368,927**, 400× what uniform sampling produces.
Seeded inputs do trip the gate at 1600–2100× tolerance, but those arrangements sit
**+10.4σ** outside the real distribution, so they show these transformations *can*
diverge, not that they *do*.

**The one C1-supporting result that survived stress-testing:** `matmul_k_tiling`
fails the gate **14% of the time under plain uniform sampling** at fp32 — and Phase
4's single draw missed it. Single-sample validation misses a real failure five times
in six.

Errors and falsified claims are collected by failure mode in
[`ERRATA.md`](ERRATA.md), including four that would have produced clean wrong
numbers.

---

## The question that reframed C1 — resolved into the design

**Settled 2026-08-03** by a dated `CLAIM.md` amendment: the gate is now the
differential (variant vs baseline at the same precision, which is what Axon §4.6
literally specifies), the as-registered form is co-reported on every cell, and the
precision floor is recorded alongside. No registered threshold moved. The reasoning
that got there is kept below because it is the argument the writeup has to make.

Measured on the A10 and reproduced bit-identically on the Mac: **at fp16 and bf16,
an untransformed matmul already exceeds the `1e-4` gate** — 3.95e-04 and 3.10e-03
against exact arithmetic on its own inputs, before any transformation is applied.
The identity fails.

C1's second clause says the gap "widens under the reduced precisions that production
inference runs in." That is at risk of being **trivially true**: precision alone
blows the tolerance, no superoptimizer required. The question with content is
whether a transformation adds error *beyond what the baseline already suffers at
that precision* — `err(variant vs truth)` against `err(baseline vs truth)`, not just
against the absolute gate.

**No threshold moves.** T1 and the registered gate stand exactly as written. What
changes is the analysis plan: the differential must be co-reported alongside the
absolute gate, or the bf16 column is a row of failures that says nothing about
transformations.

The observation is also a result in its own right. Axon validates at
`rtol = atol = 1e-4` in FP32, and that gate is simply **inapplicable** at the
precision production runs at — independent of any superoptimizer. None of the three
papers states what replaces it at bf16.

---

## Phase 0 — Setup ✅

**Exit criterion:** *the claim is committed with a date you cannot edit without it
showing in history.* **Met** — `f9adefc`, CLAIM.md registered 2026-08-03.

All listed items done: repo, MIT license, `.gitignore`, `CLAIM.md`, `NOTEBOOK.md`.

**Deviation from outline.** The repo root is `cornfieldV2`, which already held the
six-paper working set, rather than a fresh repo. `BACKGROUND.md` and `ROADMAP.md`
were consolidated in from `~/Documents/fp-verification-gap/`, which was sitting
inside the home-directory git repo and could never have been published on its own.

**Added beyond the outline.** The roadmap lists four pre-registered thresholds.
`CLAIM.md` also pins the metric's free parameters — the reference computation, the
per-element relative-error formula, and the literal `allclose` predicate. A
pre-registration that leaves the metric open is not one: "exceeds 1e-4" has to mean
the same thing after the data as before it. That turned out to matter within hours
(see Phase 1).

---

## Phase 1 — Transformation corpus ✅

**Exit criterion:** *≥6 pairs, each verified identical under float64 within 1e-12.*
**Met** — 6 pairs × 3 shape classes = 18 cells, all at **6e-16 – 3.5e-15**
scale-relative error, three orders inside the gate.

All six transformations named in the roadmap's table are implemented; none was
substituted or dropped. Each carries the required metadata record — which system
accepts it, which paper section, and the claimed justification — plus a `hazard`
field noting how each pair could produce a misleading result.

**Added beyond the outline.**

- [`fpgap/accumulate.py`](fpgap/accumulate.py) — order-pinned summation. Not called
  for by the roadmap, but necessary: `torch.sum` is a blocked pairwise cascade on
  CPU and a tree reduction on CUDA, so delegating to it would have measured torch's
  internals rather than the transformation, and would have done so invisibly.
- A **negative control** promoted to a first-class exit criterion — proof the
  instrument has a nonzero reading before any null result is trusted.

**Deviation from outline.** The float64 gate is now scale-relative error rather
than max per-element relative error, at the same `1e-12`. Recorded as a dated
amendment in [`CLAIM.md`](CLAIM.md); the C1 thresholds are untouched.

**Two instrument bugs found, both recorded in [`NOTEBOOK.md`](NOTEBOOK.md).** The
second is the one worth carrying forward: inputs were drawn at float32 and cast
*up* to float64, leaving 29 spare mantissa bits, so float64 summation was exact in
every order and the reduction pairs reported a perfect `0.000e+00`. It produced a
*cleaner* result than the truth, and a perfect zero on an equivalence check reads
as success. Had it survived into Phase 4 it would have suppressed real error and
pushed the sweep toward A1 — the direction least likely to be interrogated.

---

## Phase 2 — Reference and precision harness ✅

**Exit criterion:** *the harness takes (transformation, shape, precision, input
tensor) and returns per-element relative error against float64 truth.* **Met** —
[`fpgap/harness.py`](fpgap/harness.py). It returns more than the criterion asks for:
three quantities per cell (floor / total / differential), both gate definitions, and
both T1 readings. `inputs` is overridable, which is the hook Phase 3 fixtures and
Phase 5 seeding plug into.

**float64-as-truth is now checked, not assumed.**
[`tools/validate_reference.py`](tools/validate_reference.py) recomputes every corpus
baseline against mpmath at 50 digits: worst drift **8.2e-16, 4× float64 eps**, twelve
orders below the 1e-4 gate. That was the outline's remaining Phase 2 item.

**Hardware settled: Lambda `gpu_1x_a10`, us-east-1.** Verified NVIDIA A10, compute
capability 8.6 (Ampere), 23 GB, driver 580.105.08, CUDA 12.8, Python 3.10.12, torch
2.7.0. Version skew against the Mac (torch 2.8.0 / Python 3.13.1) is recorded
because reduced-precision defaults have changed between torch releases before.

**Done:** [`tools/probe_hardware.py`](tools/probe_hardware.py) characterises the
machine's arithmetic so that record travels with every result, and
[`tools/check_corpus_device.py`](tools/check_corpus_device.py) re-runs the Phase 1
equivalence gate per device — **18/18 pass on CUDA** at 2.8e-16 – 2.4e-15, so
equivalence over ℝ is not a platform-specific accident.

**Two of my own claims were falsified by the probe** (full detail in
[`NOTEBOOK.md`](NOTEBOOK.md)):

- *The A10 does not give a switchable accumulation-width axis.* Both settings of
  `allow_fp16_reduced_precision_reduction` are bit-identical at 512×4096×512 —
  cuBLAS evidently never picks a split-k kernel there, so the flag is a no-op and
  the tensor-core MMA accumulates in fp32 either way.
- *CPU does not overstate error relative to tensor cores.* Paired test on identical
  input bits gives **3.9533e-04 (fp16) and 3.0960e-03 (bf16) on both machines**,
  identical to every printed digit, because rounding the output to the narrow type
  dominates. Only fp32 shows a real platform gap (CPU 3× worse).

**Still outstanding:** nothing in Phase 2.

**The outline's premise here is already known to be wrong.** It says *"bf16 is not
native on your card — simulate by rounding through `torch.bfloat16` and computing
in fp32,"* and requires validating that simulation against native fp16.

Measured on the M3 Pro: CPU fp16 **and** bf16 both accumulate genuinely in-type.
The stagnation probe `1 + 4096×(eps/4)` returns exactly `1.0` in both, against
exact answers of 2.000448 and 8.997440 — rounding to storage type at every step,
no hidden wider accumulator. An A10 (Ampere, sm_86) has native bf16 in hardware.

So on either candidate machine there is **no bf16 simulation to validate**, which
deletes rather than solves:

- the "validate the simulation" sub-task in Phase 2;
- the stated bf16 limitation in Phase 2;
- the "simulated bf16" entry in the Phase 6 threats-to-validity list;
- the second kill criterion ("Phase 2's fp16 simulation doesn't track native").

**A replacement threat, which must not be lost.** Accumulation *width* differs by
platform: torch CPU accumulates bf16 matmuls in bf16, while NVIDIA tensor cores
accumulate bf16 products in fp32. CPU results would therefore **overstate** error
for matmul-shaped ops relative to production GPUs. On Ampere this is switchable
(`allow_fp16_reduced_precision_reduction` and its bf16 twin), which turns the
confound into a measured axis — in-type is what a naive hand-written kernel does,
fp32-accumulate is what production does, and the two bracket reality.

**New hazard identified, not in the outline.** On Ampere, **TF32 must be pinned
off** (`torch.backends.cuda.matmul.allow_tf32 = False`). TF32 has a 10-bit
mantissa; left enabled, the fp32 *baseline* is silently not fp32 and every
matmul-involving cell shows inflated fp32 error for a reason unrelated to the
transformation — making C1 look more true. Same class of silent instrument error as
the float64-headroom bug, and it needs an assertion, not an assumption.

**Still outstanding regardless of hardware:** the mpmath 50-digit cross-check
confirming float64 is itself adequate as truth. `mpmath 1.3.0` is available.

---

## Phase 3 — Realistic inputs ✅

**Exit criterion:** *a fixture file of real activation tensors, committed, with a
script that regenerates it.* **Met** — `fixtures/activations.pt` (11.5 MB, tracked
deliberately) plus [`tools/dump_activations.py`](tools/dump_activations.py).

**The outline assumed a checkpoint that did not exist.** It says to dump activations
from the TransformerOp checkpoint; there was none, and no downloaded data. So Phase 3
had an unbudgeted training step in front of it: `get_data.py` → `train.py` → dump.
Trained on the A10 — 5000 iters, tiny-shakespeare, val loss **1.5034**, ~8 minutes.
Not optional, since T2 is pre-registered.

**Nine sites captured** at blocks 0 and 5, 512 rows each: residual stream entering
LayerNorm, LayerNorm output, post-GELU, MLP output, and attention scores.

**Per-row cancellation is the number that matters**, since LayerNorm reduces per row.
`μ²/E[x²] → 1` is catastrophic for the one-pass variance form:

| site | row mean | row p99 | row max |
|---|---|---|---|
| `resid_pre_ln_L0` — *what LayerNorm actually consumes* | 0.0020 | 0.0117 | **0.0158** |
| `resid_pre_ln_L5` | 0.0007 | 0.0056 | 0.0077 |
| `post_gelu_L5` | 0.0654 | 0.2683 | **0.3602** |
| `attn_scores` | 0.0466 | 0.2989 | **0.3608** |
| `post_ln_*` (control) | 0.0000 | 0.0000 | 0.0000 |

**Against my own prediction.** I expected post-GELU to supply the badly-conditioned
LayerNorm inputs that `randn` cannot. It is indeed the most biased site (row max
0.36) — but **in a pre-norm transformer LayerNorm never sees it.** What LayerNorm
consumes is the residual stream, whose row max is 0.0158. So on real activations the
one-pass variance form looks well conditioned, and that is a fact about pre-norm
architecture rather than about `randn` being zero-mean. It cuts toward A1, and it is
a stronger version of that result than the synthetic one.

Caveat: one small char-level model. A post-norm architecture, or LayerNorm applied
to something other than a residual stream, could differ.

---

## Phase 4 — Main sweep ✅

**Exit criterion:** *raw per-cell results on disk, and the headline table exists.*
**Met** — 54 synthetic + 24 activation + 16 shape-matched control cells in
`results/`.

**fp32: all pass except one.** `matmul_k_tiling` fails **14% of uniform draws** —
missed by Phase 4's single draw per cell, caught by Phase 5's 100 trials. The T3
catch-rate phenomenon appearing inside the experiment meant to measure it.

**fp16/bf16: all fail, but `d/floor ≈ 1`** for two-thirds of the corpus, so the
differential merely tracks the precision floor. A fact about the tolerance, not the
transformations. `d/floor` splits the corpus three ways: reordering-only
(`reassociation` 1.00, `split_reduction` 0.97, `softmax_online` 0.88), amplifying
(`scalar_past_matmul` 1.80, `matmul_k_tiling` 2.96), suppressed
(`layernorm_variance` 0.40).

**Real vs synthetic, shape pinned.** Real activations produce **2.3–3.6×** more error
than `randn` at biased sites, are indistinguishable at centred ones, and change no
verdict. The roadmap's "validates on synthetic, ships on real" hypothesis holds at
the ~3× level, not the orders-of-magnitude level the framing invites.

**The outline's estimate was wrong in a useful direction.** It budgets "4–6 days,
mostly compute." The harness is launch-overhead bound by design; the full cross
product runs in **7 seconds**. The days are analysis.

**Deviation recorded:** the roadmap says *"No rental, no clock"* and `$0`. The A10 was
rented — a defensible trade for native bf16, but a deviation.

---

## Phase 5 — Seeded-input experiment ✅

**Exit criterion:** *a catch-rate comparison — uniform vs seeded — per
transformation.* **Met** — `results/seeded_catch_rates.json`, 100 trials × 6 pairs ×
5 strategies.

Seeded inputs trip the gate on 4 of 6 pairs at 100%, at 1600–2100× tolerance. **But
stress-testing showed the seeds are not realistic.** The `cancellation` strategy's
median row condition number is 4.5e6 against real rows' 42.8 — **+10.4σ** on a log
scale, ~600× worse than the *worst* real row. Bounding values inside the observed
range does not make the arrangement realistic, and the arrangement is the mechanism.

The decisive control: **real activations at their worst still pass.** `post_ln_L5`
reaches row condition number **368,927** — zero-mean by construction, so `Σx ≈ 0` —
and its differential is 7.77e-06 against a 1e-4 gate. Condition rose 400× over
uniform; the differential rose 11×.

So Phase 5 shows these transformations **can** diverge, not that they **do**. What
survives is `matmul_k_tiling` at 14% under plain uniform sampling — and the
explanation for it became the project's headline. See the Verdict in
[`CLAIM.md`](CLAIM.md).

---

## Phase 6 — Writeup ◐ background only

1. **Background** — drafted, [`BACKGROUND.md`](BACKGROUND.md)
2. **Method** — not started; the bf16-simulation subsection is likely moot
3. **Results** — not started
4. **Threats to validity** — needs revision before it is written. "Simulated bf16"
   is gone (nothing is simulated). Its intended replacement, "accumulation width
   differs from production tensor cores," is **also gone** — measured identical on
   CPU and A10 at fp16/bf16. What replaces both: output quantisation to the narrow
   type dominates matmul error, so matmul results say little about accumulation;
   and one shape class was tested for that, not all
5. **What it means** — not started
6. **Related work** — citation spine exists in [`papers/README.md`](papers/README.md)

---

## Kill criteria — current standing

| Criterion | Standing |
|---|---|
| Phase 1 stalls — corpus cannot be constructed | **did not fire.** All six built, all exactly equivalent over ℝ |
| Phase 2's fp16 simulation doesn't track native | **retired.** A10 is Ampere; bf16 is native and nothing is simulated |
| Phase 4 shows error 3+ orders inside tolerance everywhere | **did not fire, narrowly** |

On the third: it came close. At fp32 the differentials sit **1–2 orders** inside the
gate, not 3+, and one pair (`matmul_k_tiling`) crosses it on 14% of draws. So A1 does
not win decisively and the short-paper exit was not taken — but it was nearer than the
Phase 5 headline suggested before that headline was stress-tested.

The Phase 1 note kept below for the record, since it was written before any result and
should not be re-read as one:

> an fp32 sanity check was run during Phase 1 to confirm no corpus pair
is degenerate. All 18 cells diverge, and the fraction of elements over `1e-4`
relative peaked at **0.20%**, under T1's 1% bar. **This is not a result and must not
be cited as one.** It is `randn` input, fp32 only, no seeding, no realistic
activations — i.e. precisely the condition under which the field's assumption is
most likely to hold, and the per-element outliers behind it are the same near-zero
denominators discussed above. Phases 3–5 are what actually test C1.

---

## Open decisions

1. ~~**Hardware**~~ — **resolved.** Lambda A10, us-east-1. Native bf16 was the
   deciding factor and it held up. The secondary argument I made for it — a
   switchable accumulation-width axis — did not; see Phase 2.
2. ~~**Differential vs absolute reporting**~~ — **resolved.** Dated `CLAIM.md`
   amendment: gate is the differential, as-registered form co-reported on every cell.
3. ~~**Phase 3 prerequisite**~~ — **resolved.** Checkpoint trained on the A10, val
   loss 1.5034; fixtures committed.
4. **Deviation to record:** the roadmap says `$0 / no rental, no clock`. The A10 was
   rented. Noted in Phase 4 rather than amended into the roadmap, which is left as the
   historical plan.
5. **Threats-to-validity list** — still open, and now needs *more* rewriting than
   before. Two entries vanished (simulated bf16; CPU-vs-tensor-core accumulation, both
   measured false) and the ones that replace them are different in kind: seeded
   arrangements +10.4σ from real, one small char-level model, torch reimplementations
   rather than the actual systems, and no evidence about end-model quality.

### Remaining work

Only Phase 6. The measurement is done and the verdict is written
([`CLAIM.md`](CLAIM.md)); what is left is prose.

The one experiment that would most strengthen the result, not attempted here: measure
how often real workloads reach adverse arrangements. This project showed the seeded
inputs are +10.4σ from the observed distribution but never established the frequency
with which production traffic approaches them. That is the question a reviewer will
press hardest, and it is answerable.
