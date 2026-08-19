# Progress against the roadmap

Historical. Tracks [`ROADMAP.md`](ROADMAP.md) phase by phase against its
own exit criteria, so the plan and the work can be compared. Deviations
are recorded, not absorbed. Current results live in
[`PAPER.md`](PAPER.md); corrections in [`ERRATA.md`](ERRATA.md).

| Phase | Estimate | Status | Exit criterion |
|---|---|---|---|
| 0 Setup | 0.5 day | Complete | Met: claim committed dated |
| 1 Corpus | 3-5 days | Complete | Met: 6 pairs, 18/18 cells inside 1e-12 |
| 2 Harness | 3-4 days | Complete | Met: harness built; float64 checked vs mpmath |
| 3 Realistic inputs | 2-3 days | Complete | Met: fixture and regeneration script committed |
| 4 Main sweep | 4-6 days | Complete | Met: 54 + 24 + 16 cells on disk |
| 5 Seeded inputs | 3-4 days | Complete | Met: catch rates, uniform vs seeded |
| 6 Writeup | 5-7 days | Complete | Met: paper, artifact, and number checker committed |

## Phase 0: Setup

- Met by commit `f9adefc`, claim registered 2026-08-03 before any code.
- Deviation: repo root is `cornfieldV2`; BACKGROUND and ROADMAP were
  consolidated in from a folder inside the home-directory git repo.
- Added: CLAIM.md pins the metric's free parameters, not only the
  thresholds. That mattered within hours.

## Phase 1: Corpus

- Met: 18/18 cells at 6e-16 to 3.5e-15 scale-relative in float64.
- Added: order-pinned summation (`fpgap/accumulate.py`) and a negative
  control as a first-class exit criterion.
- Deviation: the float64 gate is scale-relative at the same 1e-12. Dated
  CLAIM amendment; C1 thresholds untouched.
- Two instrument bugs found and fixed (ERRATA 1.1).

## Phase 2: Harness and hardware

- Met: `fpgap/harness.py` returns floor, total, differential, and the
  gate readings per cell.
- Float64 as truth is checked: worst drift vs 50-digit mpmath is
  8.2e-16, 4x float64 eps.
- Hardware: Lambda A10 (Ampere, CUDA 12.8, torch 2.7.0) plus M3 Pro
  (torch 2.8.0). Version skew recorded. Corpus equivalence re-checked on
  CUDA: 18/18.
- TF32 measured at 750x accuracy cost; pinned off in every script.
- The roadmap's bf16-simulation premise was false on both machines.
  Nothing is simulated; that kill criterion is retired.

## Phase 3: Realistic inputs

- Met: `fixtures/activations.pt` (11.5 MB, tracked) plus
  `tools/dump_activations.py`.
- Deviation: the roadmap assumed a checkpoint that did not exist. One
  was trained on the A10 (5000 iters, val loss 1.5034, about 8 minutes).
- Nine sites captured at blocks 0 and 5, 512 rows each. Per-row
  cancellation statistics ship with the fixture in
  `fixtures/activations.json`.
- Read: post-GELU is the most biased site, but a pre-norm architecture
  never feeds it to LayerNorm, so the one-pass variance form is well
  conditioned on this model's real data. One small char-level model, one
  batch.
- 2026-08-04: the checkpoint existed only on the A10, now unreachable.
  The committed fixture is canonical; regeneration is equivalent, not
  bit-identical (AUDIT step 11).

## Phase 4: Main sweep

- Met: 54 synthetic + 24 activation + 16 matched-control cells in
  `results/`.
- fp32: all pass except `matmul_k_tiling`. Phase 4's single draw missed
  its failure; Phase 5's repeated draws caught it.
- fp16/bf16: all 36 cells fail a 1e-4-class rule.
- The per-cell `d/floor` classes recorded in this phase were later
  retracted as direction-blind and replaced by `total/floor` (ERRATA 2).
  The phase's verdicts stand; that metric's interpretation does not.
- The "4-6 days, mostly compute" estimate was wrong. The sweep runs in 7
  seconds. The days are analysis.
- Deviation: the roadmap says no rental. The A10 was rented, for native
  bf16.

## Phase 5: Seeded inputs

- Met: `results/seeded_catch_rates.json`, 100 trials x 6 pairs x 5
  strategies.
- Seeded strategies trip the gate on 4 of 6 pairs at 100% [96-100%].
  Elementwise violations are near 2x tolerance (ERRATA 5.2).
- Other nonzero rates with Wilson 95% CIs: uniform on `matmul_k_tiling`
  14% [9-22%]; wide_range on the same pair 31% [23-41%]; cancellation on
  `scalar_past_matmul` 28% [20-38%]. All other cells read 0% [0-4%].
- The seeds are not realistic: median seeded row condition 4.5e6 against
  42.8 for real rows, 10.4 sigma out on a log scale.
- Phase 5 shows the transformations can diverge, not that they do. The
  surviving result is the repeated-draw catch rate at activation scale,
  whose mechanism became the study's central finding.

## Phase 6: Writeup

- Met. Nine review passes; each round's corrections are in
  [`ERRATA.md`](ERRATA.md) and dated in [`NOTEBOOK.md`](NOTEBOOK.md).
- Added beyond the roadmap: a pre-registered mutant arm, a 64-point
  tolerance grid, an exact separability analysis with an independent
  reimplementation, two diagnostics, an audit of the Mirage artifact, and
  a script that checks the paper's numbers against the records.

## Kill criteria

| Criterion | Standing |
|---|---|
| Phase 1 stalls | Did not fire. All six pairs built |
| fp16 simulation does not track native | Retired. Nothing is simulated |
| Error 3+ orders inside tolerance everywhere | Did not fire, narrowly |

Guard note, written before any result: the Phase 1 fp32 sanity check
(0.20% of elements over 1e-4) is not a result and must not be cited as
one.

## Open

1. Deviation on record: the roadmap's "$0, no rental" line. The A10 was
   rented.
2. AUDIT steps 4 (the GPU half of the bitwise cross-platform test) and
   9-11 remain.
3. Queued and named in the paper: asymmetric operand scales (F1), a
   per-cell gate margin (F6), a k-of-n sweep (F2), per-operator
   separability (F2).
4. The strongest missing experiment: how often real workloads approach
   adverse arrangements. Not attempted here.
