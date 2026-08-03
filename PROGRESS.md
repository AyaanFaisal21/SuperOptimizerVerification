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
| 2 — Reference & precision harness | 3–4 days | **in progress** | no — hardware settled, harness not built |
| 3 — Realistic inputs | 2–3 days | **blocked** | no — prerequisite missing, see below |
| 4 — Main sweep | 4–6 days | not started | no |
| 5 — Seeded-input experiment | 3–4 days | not started | no |
| 6 — Writeup | 5–7 days | Background only | no |

Two of six phases complete, roughly on the outline's calendar. Phase 3 is gated on a
prerequisite the outline did not anticipate; Phases 4–6 follow from it.

---

## Open question that may reframe C1

Not a phase item, and it needs a decision before Phase 4 is designed.

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

## Phase 2 — Reference and precision harness ◐ in progress

**Exit criterion:** *the harness takes (transformation, shape, precision, input
tensor) and returns per-element relative error against float64 truth.* **Not met** —
the harness itself is not written. Hardware and machine characterisation are done.

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

**Still outstanding:** the harness itself, and the mpmath 50-digit cross-check
confirming float64 is adequate as truth (`mpmath 1.3.0` available).

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

## Phase 3 — Realistic inputs ⛔ blocked on a missing prerequisite

**Exit criterion:** *a fixture file of real activation tensors, committed, with a
script that regenerates it.* Not started.

**The outline assumes a checkpoint that does not exist.** It says "dump real
activations from your TransformerOp checkpoint." There is no checkpoint and no
downloaded data — `TransformerOp/train.py` writes to `checkpoints/{model}.pt`, and
`TransformerOp/data/` contains only `get_data.py`.

Phase 3 therefore has a training step in front of it: `get_data.py` → `train.py` →
dump activations. Feasible locally on the M3, but it is real work the outline did
not budget. Not optional either — T2 is pre-registered: *"realistic inputs means
activations sampled from a trained model, not `torch.randn`."*

---

## Phase 4 — Main sweep ⬜ not started

Depends on Phases 2 and 3.

**The outline's estimate looks wrong in a useful direction.** It budgets "4–6 days,
mostly compute." The harness is Python-loop and kernel-launch-overhead bound, not
FLOP-bound — `accumulate.py` walks the reduction dim one vectorized op at a time by
design. Actual GPU time for the full cross product is minutes; the days are
analysis. A larger GPU than an A10 would buy nothing.

**Conflict to resolve if the A10 is chosen.** The roadmap says *"Runs on your 2060,
or CPU. No rental, no clock,"* and the totals line says `$0`. Renting breaks that.
It is a defensible trade — native bf16 is the single most load-bearing hardware
fact for a claim explicitly about bf16 — but it should be an amendment or a
recorded deviation, not a silent change.

---

## Phase 5 — Seeded-input experiment ⬜ not started

Unaffected by the open decisions; the roadmap calls it the sharper half of the
paper. Phase 4 asks whether the gap exists, Phase 5 asks whether the sampling
method everyone uses would find it.

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
| Phase 4 shows error 3+ orders inside tolerance everywhere | **unknown — not yet measured** |

On the third: an fp32 sanity check was run during Phase 1 to confirm no corpus pair
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
2. **Amend the roadmap's `$0 / no rental, no clock` line**, or record the deviation.
   Still open. The A10 is rented; the roadmap says it would not be.
3. **Differential vs absolute reporting** — see the open question above. Needs
   settling before Phase 4 is designed, because it determines what the headline
   table's columns are. It does *not* move any registered threshold.
4. **Threats-to-validity list** — revise before Phase 6. Two entries are now gone
   rather than replaced, which leaves the section thinner than the roadmap assumed
   and means it needs new entries, not edits to old ones.
5. **Phase 3 prerequisite** — the TransformerOp checkpoint still has to be trained
   before any realistic-activation work can start. Now the critical path.
