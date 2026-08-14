# Progress against the roadmap

Tracks [`ROADMAP.md`](ROADMAP.md) phase by phase, against its own exit criteria.
Deviations from the plan are recorded, not absorbed. Updated 2026-08-14.

| Phase | Estimate | Status | Exit criterion |
|---|---|---|---|
| 0 Setup | 0.5 day | Complete | Met: claim committed dated |
| 1 Corpus | 3-5 days | Complete | Met: 6 pairs, 18/18 cells inside 1e-12 |
| 2 Harness | 3-4 days | Complete | Met: harness built; float64 checked vs mpmath |
| 3 Realistic inputs | 2-3 days | Complete | Met: fixture and regeneration script committed |
| 4 Main sweep | 4-6 days | Complete | Met: 54 + 24 + 16 cells on disk |
| 5 Seeded inputs | 3-4 days | Complete | Met: catch rates, uniform vs seeded |
| 6 Writeup | 5-7 days | Rev. 3 in revision toward rev. 4 | Not met: tex regeneration and the measure.md queue remain |

Verdict summary. Neither C1 nor A1; see [`CLAIM.md`](CLAIM.md) and its 2026-08-14 amendment.
A1 holds for real activation distributions: every real site passes with 10x headroom.
A1 fails for shapes: the valid tiling is rejected on 48/100 draws at K=4096 and 100/100 at K=11008 (unit scale).
The earlier "fails at K=2048 outright" used the wrong predicate. Retraction #4.
Seeded violations are near 2x tolerance and sit 10.4 sigma outside the real distribution.
The surviving C1 result: 14% [CI 9-22%] failure rate at activation input scale, missed 86% of the time by one draw.

## Phase 0: Setup

- Met by commit `f9adefc`, claim registered 2026-08-03 before any code.
- Deviation: repo root is `cornfieldV2`; BACKGROUND and ROADMAP consolidated in from a folder inside the home-directory git repo.
- Added: CLAIM.md pins the metric's free parameters, not only the thresholds. This mattered within hours.

## Phase 1: Corpus

- Met: 18/18 cells at 6e-16 to 3.5e-15 scale-relative in float64.
- All six roadmap transformations implemented; each carries provenance and a hazard note.
- Added: order-pinned summation (`fpgap/accumulate.py`) and a negative control as a first-class exit criterion.
- Deviation: the float64 gate is scale-relative, same 1e-12. Dated CLAIM amendment; C1 thresholds untouched.
- Two instrument bugs found and fixed. See ERRATA 1.1 and the CLAIM amendment.

## Phase 2: Harness and hardware

- Met: `fpgap/harness.py` returns floor, total, differential, both gate readings, both T1 readings per cell.
- Float64 as truth is checked: worst drift vs 50-digit mpmath is 8.2e-16, 4x float64 eps.
- Hardware: Lambda A10 (Ampere, CUDA 12.8, torch 2.7.0) plus M3 Pro (torch 2.8.0). Version skew recorded.
- Corpus equivalence re-checked on CUDA: 18/18.
- TF32 measured at 750x accuracy cost; pinned off in every script.
- The roadmap's bf16-simulation premise was false on both machines. Nothing is simulated. Its kill criterion is retired.
- Two claims from this phase were later falsified; one more was revised on 2026-08-04. See ERRATA 2.

## Phase 3: Realistic inputs

- Met: `fixtures/activations.pt` (11.5 MB, tracked) plus `tools/dump_activations.py`.
- The roadmap assumed a checkpoint that did not exist. Trained on the A10: 5000 iters, val loss 1.5034, about 8 minutes.
- Nine sites captured at blocks 0 and 5, 512 rows each.
- Per-row cancellation, u^2/E[x^2], is the governing statistic:

| Site | Row mean | Row p99 | Row max |
|---|---|---|---|
| resid_pre_ln_L0 (what LayerNorm consumes) | 0.0020 | 0.0117 | 0.0158 |
| post_gelu_L5 | 0.0654 | 0.2683 | 0.3602 |
| attn_scores | 0.0466 | 0.2989 | 0.3608 |
| post_ln (control for cancellation) | 0.0000 | 0.0000 | 0.0000 |

- Key read: post-GELU is the most biased site, but pre-norm architecture never feeds it to LayerNorm. The one-pass variance form is well conditioned on real data. This is an architecture fact, not a sampling fact.
- Caveat: one small char-level model, one batch.
- 2026-08-04: the checkpoint existed only on the A10, now unreachable. The committed fixture is canonical. Regeneration requires retraining and is equivalent, not bit-identical. AUDIT step 11.

## Phase 4: Main sweep

- Met: 54 synthetic + 24 activation + 16 matched-control cells in `results/`.
- fp32: all pass except `matmul_k_tiling`, which fails 14% of draws at activation scale. Phase 4's single draw missed it; Phase 5's 100 trials caught it.
- fp16/bf16: all 36 cells fail, but d/floor is near 1 for two-thirds of the corpus. Those failures measure precision, not transformations.
- d/floor classes: reordering near 1 (`reassociation` 1.00, `split_reduction` 0.97, `softmax_online` 0.88); amplifying (`scalar_past_matmul` 1.80, `matmul_k_tiling` 2.96); suppressed (`layernorm_variance` 0.40).
- Real vs randn at pinned shape: 2.3-3.6x more error at biased sites, no difference at centred sites, no verdict changes.
- The "4-6 days, mostly compute" estimate was wrong. The sweep runs in 7 seconds. The days are analysis.
- Deviation: the roadmap says no rental. The A10 was rented, for native bf16.

## Phase 5: Seeded inputs

- Met: `results/seeded_catch_rates.json`, 100 trials x 6 pairs x 5 strategies.
- Seeded strategies trip the gate on 4 of 6 pairs at 100% [96-100%]. Elementwise violations are near 2x tolerance (ERRATA 5.2).
- Other nonzero rates, with Wilson 95% CIs (AUDIT step 2): uniform on `matmul_k_tiling` 14% [9-22%]; wide_range on `matmul_k_tiling` 31% [23-41%]; cancellation on `scalar_past_matmul` 28% [20-38%]. All other cells read 0% [0-4%].
- The seeds are not realistic: median seeded row condition 4.5e6 vs real 42.8, which is 10.4 sigma out on a log scale.
- Control: the worst real site (post_ln, row condition 368,927) still passes at 7.77e-06. Condition rose 400x; error rose 11x.
- Phase 5 shows the transformations can diverge, not that they do.
- The surviving result is the 14% uniform catch rate at activation scale, and its mechanism became the project verdict.

## Phase 6: Writeup

External review received 2026-08-08 (archived in agent-notes). It found one
factual error (Mirage, ERRATA 1.7), one analysis error (d/floor, ERRATA 2), and
motivated two new measurements: the tree-reference re-run (AUDIT step 3) and the
mutant detection arm (AUDIT step 12, registered in CLAIM before running).

Review #2 (2026-08-08) motivated the frontier, the K extension, the
decomposition, and the library-reference runs (E1-E4). Review #3 (2026-08-14)
re-verified every reference against its full text; its corrections are in
ERRATA §2 and the NOTEBOOK entry of that date. The paper stands at rev. 3,
corrected in PAPER.md toward rev. 4; the tex regeneration is pending the
measure.md queue.

1. Background: drafted; Prism limitation sentence corrected (review #3).
2. Method: drafted, revised per reviews; mutant bookkeeping corrected.
3. Results: drafted; detection arm, frontier, and K extension added; F2
   restated as a plateau, F5 range recomputed (review #3).
4. Threats to validity: list rebuilt in rev. 3.
5. What it means: drafted (Implications, rev. 3).
6. Related work: TASO, TTrace, FPRev, FTTN, NNSmith, Minotaur, SATIRE added
   in rev. 3; citations swept against DBLP in review #3.

## Kill criteria

| Criterion | Standing |
|---|---|
| Phase 1 stalls | Did not fire. All six pairs built |
| fp16 simulation does not track native | Retired. Nothing is simulated |
| Error 3+ orders inside tolerance everywhere | Did not fire, narrowly. fp32 sits 1-2 orders inside, and one pair crosses |

Guard note, written before any result: the Phase 1 fp32 sanity check (0.20% of elements over 1e-4) is not a result and must not be cited as one.

## Open

1. Deviation on record: the roadmap's "$0, no rental" line. The A10 was rented.
2. AUDIT steps 4 (GPU half), 5, 6, and 9-11 remain. Steps 3 and 12-14 closed 2026-08-08; step 2 closed 2026-08-14. Step 5 is C3 in [`measure.md`](measure.md).
3. The measurement queue for the M3 is [`measure.md`](measure.md): P0 artifact completeness, C1-C3 registered extensions, P2 stretch items.
4. The strongest missing experiment: how often real workloads approach adverse arrangements. Not attempted here. A reviewer will ask.
