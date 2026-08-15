# Notebook

Dated run record. The prediction is written and committed before each run.
Git history shows whether that order was honored.
Entries are chronological. Full prose versions of all entries are preserved in git history.

---

## 2026-08-03: Phase 0, project registered

- Setup: repo consolidated; CLAIM.md written and dated before any measurement code.
- Toolchain: Python 3.13.1, torch 2.8.0, no CUDA on this machine (M3 Pro).
- Flag for later: the roadmap's fp16 native-vs-simulated gate assumes CUDA.

## 2026-08-03: Phase 1, corpus built

- Setup: six pairs; order-pinned summation; float64 equivalence gate at 1e-12.
- Prediction: all 18 cells at or under 1e-14; negative control near 1e-7.
- Outcome: the check failed on first run. Two instrument bugs, both mine:
  - The metric was underspecified. Per-element relative error is unbounded at near-zero outputs. Gate moved to scale-relative, same 1e-12. CLAIM amendment.
  - Inputs were drawn at float32 and cast up. Float64 sums were exact in every order and reduction pairs read exactly zero. ERRATA 1.1.
- Post-fix: 18/18 cells at 6e-16 to 3.5e-15. Control: seq vs tree 1.0e-06.
- Read: a zero-reading instrument passes an equivalence check. The negative control is not optional.

## 2026-08-03: A10 characterisation

- Prediction: TF32 flag reads False; fp32 matmul near 1e-7 off, near 1e-3 on; stagnation probes stagnate; reduced-precision-reduction flags change little; fp64 20-40x slow.
- Outcome: all held. TF32 off 3.98e-07, on 2.99e-04 (750x). Flags bit-identical (no-op at the test shape). fp64 35.2x. Corpus 18/18 on CUDA.
- Falsified, both mine, both predicted against myself first: the switchable accumulation-width axis; CPU overstating error vs tensor cores. Paired inputs gave 3.9533e-04 (fp16) and 3.0960e-03 (bf16) on both machines to five printed digits. See 2026-08-04 revision.
- Unpredicted: an untransformed matmul already exceeds 1e-4 at fp16 and bf16. This forced the differential gate and the floor metric.

## 2026-08-03: Phase 2 closed; Phase 4 predictions

- Float64 confirmed adequate as truth: worst drift vs 50-digit mpmath 8.2e-16.
- Disclosure: the MLP-shape column was already seen; predictions cover the unseen cells.
- Predictions: fp32 all pass, no T1; fp16/bf16 all fail; d/floor near 1 for reductions, shape-invariant; `matmul_k_tiling` d/floor orders by K; gate definitions mostly agree, disagreements at layernorm.

## 2026-08-03: Phase 4 synthetic arm, outcome

- 54 cells in 7 s. fp32 18/18 pass. fp16/bf16 36/36 fail, all T1.
- d/floor classes: reordering near 1; matmul pairs 1.8-3.0; layernorm 0.05-0.43.
- Predictions 1-3 held. Prediction 4 falsified: K=512 reads d/floor 4.01 against K=768 at 3.48. Cause unknown; confounded shapes. Open.
- Prediction 5 half held: 51/54 agree, but all three disagreements are `softmax_online` fp16: fail vs baseline, pass vs truth. The variant is 15x closer to truth than its reference.
- Read: at fp32 on randn, A1 holds with two orders of headroom. The gate-direction result is visible only because both definitions are reported.

## 2026-08-03: RETRACTED ENTRY, kept as a marker

- Claimed: the literature states 1e-4 but shipping code uses 1e-2, a 100x gap.
- Retracted the same day. Four reasons: different groups (Axon is AWS/UIUC; Mirage is CMU); Mirage's paper states no float tolerance; the 1e-2 sites are MPK runtime-kernel tests, not superoptimizer validation; the repo at HEAD is MPK, not the OSDI '25 system.
- Verified instead: Mirage's superoptimizer path has no float check; its verification is exact, as the paper says. Axon states 1e-4 at FP32. Prism states nothing and benchmarks half precision.
- Lesson: a number that supports the thesis deserves more scrutiny, not less. ERRATA 1.5.

## 2026-08-03: Phase 4 activation arm

- Setup: 24 cells on real activations, mapped to the tensors each kernel actually sees, plus 16 randn controls at identical shapes to unconfound distribution from shape.
- Prediction: real behaves like randn at fp32; no verdict flips.
- Outcome: correct. 8/8 fp32 pass, 16/16 reduced-precision fail, d/floor structure reproduces. Floor ratios real/randn: biased sites 2.3-3.6x, centred sites 0.96-1.49x.
- Read: "validate on synthetic, ship on real" is a 3x effect at biased sites, not orders of magnitude.

## 2026-08-03: Phase 5 predictions

- Strategy: seed the arrangement, not the magnitudes. All values clamp into the observed real range.
- Predictions: uniform catches nothing; cancellation breaks reductions above 80%; shifted breaks layernorm near 100%; dynamic_mix weak at fp32; C1 survives at fp32 via seeding.
- Pre-stated kill outcome: if seeding also finds nothing, A1 wins decisively.

## 2026-08-03: Phase 5 outcome

- Catch rates (gate fail fraction, 100 trials): cancellation 100% on both reductions and on `matmul_k_tiling`; shifted 100% on layernorm; softmax 0% under every strategy; uniform 0% except `matmul_k_tiling` at 14%.
- Prediction 1 wrong in the useful direction: uniform catches `matmul_k_tiling` 14% of the time. Phase 4's single draw had missed it.
- Prediction 4 wrong: dynamic_mix 0% everywhere at fp32.
- Severity figures reported here as 1600-2100x were later corrected to near 2x elementwise. See 2026-08-04 audit.
- Limitation stated at once: values shown realistic, arrangements not shown to occur.

## 2026-08-03: Phase 5 stress-tested

- Question: how far outside the real distribution do the seeds sit?
- Outcome: cancellation median row condition 4.5e6 vs real 42.8. That is 10.4 sigma out on a log scale.
- Control: the worst real site, post_ln at row condition 368,927, still passes at 7.77e-06. Condition rose 400x over uniform; the differential rose 11x.
- Read: A1 mostly holds at fp32. The seeds prove "can diverge," not "does." The surviving C1 result is the uniform 14%.

## 2026-08-03: why only `matmul_k_tiling` breaks

- The failing trial has 1 bad element of 4096. Its magnitude is 750x below the tensor median, so only atol protects it.
- Failure rate tracks output count: 2% at 256 outputs, 48% at 16384.
- Across pairs, relative errors span 3x; absolute errors span four orders, because output magnitudes do.
- Controlled K sweep at unit scale, M and N pinned: pass at K=128 (1.34e-05), pass at K=512 (7.63e-05), fail at K=2048 (2.37e-04).
- Read: atol is a constant applied to arbitrary scale. The gate measures magnitude, not soundness. Production K is 4096-16384.

## 2026-08-04: full-method audit

- Three measured checks, predictions committed in the run script:
  - A. The 14% catch rate is scale-contingent. Unit scale 0/100; activation scale (sigma 2.7) 14/100, CI 9-22%. Confirmed, stronger than predicted.
  - B. CPU bf16 matmul equals fp32-accumulate-then-round: falsified. It equals neither simple model.
  - C. Cancellation gate exceedance: 2.0x true, against 322x by the scale-relative summary. Confirmed, worse than predicted.
- Six further findings without new measurement: seq baseline vs the field's tree references; the cross-platform claim was a statistic, not bits; softmax mechanism misstated; no confidence limits and a 5/6-vs-6/7 slip; unverified "systems test once"; narrow evidence base.
- Read: results moved away from the thesis again. Full list: AUDIT.md.

## 2026-08-04: cross-platform test, half run

- Predictions: outputs not bit-identical across machines; statistics match; GPU near the fp32-accumulate model.
- Outcome: the local half ran; the A10 stopped answering before the remote half. CPU matmul agrees with fp32-accumulate-then-round on 99.4% (fp16) and 99.96% (bf16) of elements at 512x4096x512.
- Read: CPU narrow matmul is fp32 accumulation in a blocked order. The earlier "CPU accumulates bf16 in bf16" line held only for elementwise adds. The platforms agreed because they run nearly the same computation.
- Checkpoint accounting: gpt.pt existed only on the instance. The committed fixture stays canonical; regeneration requires retraining and is equivalent, not bit-identical. AUDIT step 11.

## 2026-08-08: external review, two verifications, two new measurements

- An external review of the paper draft found nine issues. Full text in agent-notes.
- Verified point 1: Mirage v3 section 5.2 states FP filtering of uGraphs. Our PDF
  was a stale revision with no such sentence. Retraction #3, ERRATA 1.7.
- Verified point 3 in our own records: total/floor decouples from d/floor.
  Reordering variants are 8-16x more accurate while d/floor reads ~1. The
  d/floor interpretation is retracted; acc_ratio (total/floor) added to the harness.
- Tree-reference run (prediction: 15x shrinks to <=2x; held): softmax fp16
  floor 9.69e-04 vs online total 6.03e-04, ratio 0.62; bf16 ratio 1.92, direction
  inverts. The seq-to-tree switch flips the fp16 gate verdict FAIL to pass.
- Detection arm (registered in CLAIM before running; prediction held): five gross
  mutants detected 100/100 [96%, 100%]. ln_eps_to_std, a real bug with f64
  divergence 4.1e-06, evades 100/100 draws [0%, 4%]. Raw: results/mutant_detection.json.
- Read: the gate fails a valid rewrite at K=2048 and passes a real bug. The
  reviewer's reframing (validator discrimination) is now measured, not argued.

## 2026-08-08: review #2; frontier, K extension, decomposition, library reference

- External review #2 archived in agent-notes. TTrace (arXiv 2506.09280), FPRev
  (ATC '25, 2411.00442), FTTN (2403.00232) verified and added to related work.
  Nautilus, Propilot, Kernel Contracts left uncited pending verification.
- E1 frontier (predicted: no separator; any point passing k2048 misses eps-1e-5):
  half held. 0 separators in 64 points. But the optimum is exactly (1e-4, 1e-4):
  0.0% valid rejection at unit scale, 6.2% mutant miss, all of it eps-1e-5.
  The published constant is the optimum of its family; the family cannot separate.
- E2 K extension (predicted: K=4096 and 11008 fail >=95%): half held, and it
  falsified our own committed K=2048 result. Under the literal rule: 0/100 at
  K=2048, 48/100 [38-58%] at K=4096, 100/100 at K=11008. The old "fails at
  K=2048" compared max abs diff to atol alone, ignoring rtol. Retraction #4,
  ERRATA. Corrected result lands exactly on Llama-2-7B widths; no extrapolation.
- E3 decomposition (predicted: iid model inside CI): held. p_elem 3.42e-05,
  predicted 13.1%, observed 14/100 [9-22%]. The output-count effect is
  extreme-value aggregation.
- E4 library reference (predicted: torch.sum within 2x of tree): falsified.
  torch.sum floor 3.67e-04 vs tree 9.69e-04 at fp16 (2.6x better), consistent
  with wider internal accumulation (FPRev). Online candidate now 1.6x WORSE
  than the library reference at fp16. F3 is a three-reference result: 0.07,
  0.62, 1.64.
- Paper rebuilt as rev. 3: measurement-study identity, 2x2 semantic-vs-numerical
  labels, Axon draw-count wording narrowed, prereg language marked
  registered/amended/exploratory, TASO/TTrace/FPRev/FTTN/NNSmith/Minotaur/SATIRE
  added. Prediction record now nine falsified by their own runs.

## 2026-08-14: review #3; full-text re-verification of every reference

- Scope: full text of Mirage v1/v2/v3 (diffed), Prism, and Axon; 14 secondary
  citations checked against DBLP/ACM/USENIX/arXiv; every committed results file
  recomputed; git history checked. Review by agent, verified against sources.
- Confirmed verbatim: Axon §4.6 gate sentence (reference implementation,
  rtol=atol=1e-4, FP32; draw count unstated; the 100 repetitions are timing).
  Prism's one-clause random testing, no tolerance anywhere, fp16-only
  evaluation. Mirage v3 §5.2 numerical-stability sentences, no protocol.
  Every recomputable number in the paper matched the committed records.
- Corrections applied (each in ERRATA §2; corpus, README, BACKGROUND, PAPER):
  - Prism never states the FP limitation. "Each paper states this limitation"
    was wrong for one of three. The corrected sentence is stronger.
  - The Lax/softmax exclusion is our inference, and the mechanism is the
    running max (not a Lax operator), not exp count. Mirage never classifies
    softmax.
  - layernorm_variance is not in Prism (RMSNorm only). Provenance narrowed to
    Axon-in-principle plus cornfield's shipped kernel.
  - The frontier optimum is a three-point plateau: (1e-4, {1e-4, 1e-3, 1e-2}).
    "Exactly (1e-4, 1e-4)" was first-wins tie-breaking. atol is the binding
    coordinate, which restates the near-zero-element mechanism.
  - Mutant count: eighteen instances, sixteen on the frontier. The 8.2e-1
    endpoint is the detection-arm rescale mutant. Column grid {1,2,8,32}
    deviates from E1 as registered; ERRATA §6.
  - F5's "8-50x" recomputed from committed records: 3-73x, 8-62x at mlp shapes.
  - Mirage primes: p=227, q=113; the product fits 16 bits.
  - Bibliography: Alive-FP third author is Gupta, not Martin.
- Record note: the 2026-08-08 work was committed in one batch on 2026-08-14
  with identical commit timestamps. Commit order preserves the register-then-run
  sequence; wall-clock times do not corroborate it. Disclosed here; from this
  entry on, work is committed when it is done.
- Two review-#3 corrections move toward the thesis (Prism's silence; the wider
  F5 range), the first such movement since registration. The rest are neutral
  or narrowing. Still open from review #3: README/CLAIM/PROGRESS carry the
  retracted K=2048 verdict pending the consistency pass; E2-E4 and the
  tree-baseline runs dump no JSON; committed sweeps predate acc_ratio.

## 2026-08-14: measurement pass (measure.md queue)

- Environment: torch 2.8.0, 6 threads, M3 Pro CPU. Matches the committed-record
  conditions; nothing to disclose under ground rule 3. probe_hardware: storage
  rounding in-type, as before.
- P0.1 (110bff7): both sweeps regenerated. Field-aware diff vs the committed
  records: 0 drift across 94 cells; acc_ratio now first-class in all records.
- P0.2 (bd07e26): k_extension.json committed. E2 reproduces 0/100, 48/100,
  100/100. E3 reproduces p_elem 3.42e-05, iid 13.1%, inside CI.
- P0.3 (c6f497f): tree_baseline.json committed. Same numbers as 2026-08-08.
- C1-C3 registered before running (60b68c4).
- P1.1 / C1 (3c29480): fp16 seq FAIL 100%, tree pass 98%, torch.sum pass 99%;
  bf16 all three FAIL 100%. Three clauses held. One clause FALSIFIED: the
  per-draw floor ordering seq > tree > torch.sum holds on 80/100 (fp16) and
  85/100 (bf16), not >=95%. The tree and torch.sum floor distributions overlap;
  the mean ordering holds, the per-draw ranking does not. Falsified prediction
  #10, and a sharpening of F3: even the ranking of references is draw-dependent.
- P1.2 / C2 (8768fda): held in full. Five gross mutants 100% detection;
  ln_eps_to_std evades 100/100 at sigma 2.67, f64 divergence 2.46e-06, smaller
  than unit scale (4.08e-06), matching the registered mechanism.
- P1.3 / C3 (5542ef2): held in full. K axis at sigma 2.67: 11, 69, 100, 100
  per 100 at K = 512, 1024, 2048, 4096. Sigma axis at K=512: 0, 0, 1, 16, 74
  per 100 at sigma 0.5, 1.0, 2.0, 2.67, 4.0. Monotone on both axes; the
  committed 14% sits inside the new 16/100 CI [10, 24]. The rejection onset at
  activation scale is K near 1024, four times below the unit-scale onset.
- P2 not attempted this pass (optional per the queue): the d/floor anomaly
  sweep (AUDIT step 6) and pretrained activations remain open.
- measure.md retired in this commit per its own rule.
