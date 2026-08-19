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

## 2026-08-14: review #4; exact separability (E5)

- External review #4 archived in agent-notes. Eight fixes; two important: the
  frontier's grid result did not license a family-level claim, and the 6.25%
  selection-amplification extrapolation was statistically meaningless.
- E5 registered in CLAIM before running; prediction held exactly. Separators
  at sampled rtols: 0/401. All 400 inter-sample intervals certified
  separator-free by convexity. sup(G - max(F,0)) = -3.487e-05. F(0) =
  5.035e-04 (the valid corpus's flattest atol requirement) against G(0) =
  4.101e-05 (the eps mutant's envelope ceiling): a 12x structural gap that
  never closes. The family-level claim is now exact on the recorded corpus
  for rtol <= 100.
- Citations: three of review #4's five 2026 works verified and added
  (2606.20128, 2607.16228, 2606.27396); Propilot and Kernel Contracts remain
  unverified and uncited.
- All eight review-4 items applied to PAPER.md and paper.tex: scale-matched
  gaussian phrasing, 6.25% defined as a corpus miss fraction, the torch.sum
  tensor-core explanation replaced, the selection-amplification extrapolation
  deleted, TensorRight split out, justified-budget wording, frontier heatmap
  added to the tex.

## 2026-08-14: review #5; rev 5a consolidation (E6, E7)

- External review #5 (Rev. 5 direction) archived in agent-notes. Adopted
  nearly wholesale: protocol-coordinate framing, evidence-class verbs,
  two-criterion envelope analysis, F3 reorder, F4 two-quantifier form,
  8-item reporting standard, second figure, grid completion.
- E6 registered before running; prediction held. No separator under EITHER
  cross-draw quantifier: EVERY sup gap -3.5e-05 (400/400 certified), SOME
  sup gap -4.0e-06 (400/400 certified). The quantifier moves the margin
  ninefold; SOME comes within 4e-06 of separating but never does. The
  cross-draw aggregation rule is now a measured protocol coordinate.
- E7 registered before running; predictions held: unit K=1024 0/100 [0-4%],
  sigma-2.67 K=11008 100/100 [96-100%]. Table 1 is a complete rectangle.
- Paper rebuilt as rev. 5a: abstract in the review's target structure,
  protocol taxonomy in the introduction with finding-to-coordinate map,
  evidence-class verb conventions, F1 mechanism/onset split with the full
  two-scale table and a K-by-scale figure, F2 with both quantifiers and
  the mutant-instance miss fraction, F3 led by the tree/torch.sum sign
  reversal, F4 with the labeled hypothetical IID calculations, conclusion
  as the 8-item protocol reporting standard. paper.tex mirrors, two
  figures. Next per the review's sequencing: outside expert review, then
  rev 5b only if justified.

## 2026-08-14: Mirage implementation audit (review #5 item 15)

- Prediction (before the audit): no FP-tolerance filter in the released
  search path. HELD, with positive control (the method finds the known
  demo/test allclose sites).
- The released superoptimize path is: search, exact finite-field fingerprint
  verification (no float comparison anywhere), transpile, performance-only
  selection over random inputs. The v3-stated numerical-stability filter is
  not locatable in the released search path at commit 5c28cc6 or on the
  evaluation branch.
- Surprise: the finite-field constants in the released code are FP_P = 167,
  FP_Q = 83 (product 13,861), on both HEAD and the evaluation branch, while
  the paper states p = 227, q = 113. Under the paper's own 1/q bound, the
  released single-test configuration is ~1.4x weaker than stated. The
  paper's primes appear nowhere in the code.
- Written up as Appendix A (artifact/appendix evidence per review #5);
  full method in agent-notes/MIRAGE-AUDIT-2026-08-14.md.

## 2026-08-14 - E8: the semantics box, run and partially falsified

Prediction, registered before the run: no separator at all four corners;
tail closes. Outcome: the three corners that include a coherent Boolean
validator semantics hold (all-draws-must-pass -4.0e-06, any-draw-may-pass
-4.2e-06, pessimal pairing -3.5e-05; 400/400 intervals certified each;
both bug envelopes nonpositive at rtol=100, so all three claims close
over rtol >= 0). The fourth corner is FALSIFIED: best-draw-per-valid
against worst-draw-per-bug admits separators at 110/401 samples, a
contiguous band rtol in [7.8e-03, 7.9], maximum margin +9.5e-06 at
rtol=0.056. The certificate correctly declines the 112 intervals that
touch the band. Reading: the recorded errors permit separation under
per-program draw selection; both coherent uniform semantics forbid it.
The impossibility is a property of the cross-draw rule, not of the data.
Paper F2 is rewritten around this. Two same-day corrections enter the
errata: the Rev. 5a abstract said eighteen bugs where the recorded
envelopes cover the sixteen frontier instances, and it attributed the K
rejection rates to an Ampere backend where they ran on the Apple M3 CPU.

Same day: Mirage recency check. The repository default branch is mpk and
its tip is the audited 5c28cc6, so the audit covered the newest
default-branch code. The main branch is older (ffe38df, 2026-04-17) and
shows the same constants (167/83) and the same performance-only
selection. Three branches now confirm. Appendix A updated.

## 2026-08-15 - Review 7 fact-check before application

Strenuous check of seven items; all seven survive, one against us in a
form worth recording. Item 4: the v3 HTML source gives Theorem 2's
bound verbatim as 8dk^4/q + q^(-1/k^2). Our appendix's 1.4x treated it
as 1/q; the second term moves by (113/83)^(1/k^2), so no universal
factor exists. Retracted (errata 2.10), replaced with the monotone
loosening statement the theorem supports. The same fetch reconfirmed
p=227/q=113, the single uniterated test, all-element comparison, and
the 5.2 filter sentence, all verbatim. Item 1 confirmed: the T(r)
definition needed the atol >= 0 clamp explicit; the computation already
applied max(F, 0), so no rerun. Item 2 confirmed: k-of-n rules are
coherent Boolean aggregations we did not analyze; claims rescoped to
the two analyzed extremes, and F2 now names k-of-n as unanalyzed
specification space (they evaluate the k-th order statistic of the same
per-draw envelopes; genuinely open whether some intermediate k
separates). Item 3 confirmed: the fourth pairing is class-conditioned;
demoted from the abstract, reframed in F2. Items 5, 6, 7 confirmed:
intro no longer claims variation of held-fixed coordinates, the
tensor-scale statistic is named, references demonstrate rather than
bound, and the appendix states what the audit found rather than what
exists. No new experiments. Process note: the previous commit pushed
the PAPER.md half alone after a patch-script anchor failed mid-run;
this commit completes the tex, errata, and notebook half.

## 2026-08-15 - Pre-expert clarity pass

Full read of both paper files after seven review cycles of surgical
patches. Fixed: mutant/bug terminology unified in the envelope machinery
(mutant for corpus objects, erroneous candidate in the operational
definitions, bug for semantic class); E8 added to the protocol-labels
registry; two stale embedded counts corrected (the errata no longer
carries a fixed count, the falsified-prediction count now includes E8);
"closed over rtol >= 0" in the contributions reworded, since the fourth
pairing separates; the F2 heading now says recorded corpus; md/tex
mirrors aligned on the F5 closing sentence, the F6 scale-vs-distribution
sentence, the Axon reporting list, F7 regimes, and the enumerated Method
integrity paragraph, which now includes the Theorem 2 misread. No claim
changed strength; the seven protected claims verified present after the
pass.

## 2026-08-15 - Registered before running: V1, D1, D2

Three checks from the pre-expert triage, predictions first.

V1, independent verification of E5/E8. A second implementation computes
all four class envelopes by direct max over every recorded element at
every grid point, no Pareto prefilter, no shared envelope code, straight
from the Method definitions. Prediction: agreement with
results/separability.json to 1e-12 absolute on every envelope value at
every grid point, identical separator and certificate counts at all four
corners, identical sup gaps to 1e-12. Synthetic known-answer tests: the
machinery classifies a constructed separator, a constructed
non-separator, and a hidden between-samples separator correctly, the
last landing uncertified. If any disagreement: report before use, fix,
rerun, record here.

D1, K-differential diagnostic for the F1 mechanism attribution.
Per K in {512, 1024, 2048, 4096, 11008}, 20 unit-scale draws at the
pinned M=N=64 geometry, recording per-draw max abs diff between the
K-tiled variant and the untransformed reference and the reference
magnitude at the argmax element. Prediction: the median max-diff grows
broadly and monotonically with K, no isolated jump at one width, and
argmax elements sit well below the tensor median magnitude.

D2, oracle spot-check for the F3 headline cell. On 3 draws of the F3
softmax cell, recompute floor and total against a 50-digit mpmath
oracle for the tree, torch.sum, and sequential references. Prediction:
the float64-oracle error statistics agree with the 50-digit ones to
better than 1e-6 relative, and every F3 ratio is unchanged at the
reported precision.

## 2026-08-15 - V1, D1, D2 outcomes; full claim audit

V1 outcome: prediction held exactly. The independent no-Pareto
recomputation reproduces every envelope value with zero deviation and
identical corner statistics; five synthetic known-answer tests pass,
including the hidden-separator case landing uncertified. E5/E8 is
independently verified. D1 outcome: prediction held. Median max
disagreement grows smoothly and roughly proportionally with K, 6.1e-05
at 512 to 1.3e-03 at 11008, no isolated jump, argmax at large
rtol-protected elements; with the fixed-K sigma progression this makes
the F1 mechanism data-backed. D2 outcome: prediction held. Worst
relative deviation of the float64-oracle statistics from the 50-digit
oracle is 1.7e-13 across three draws and four computations; the seq
floor 8.938e-03 reproduces the sweep cell exactly, pinning F3's source.

Claim audit: tools/check_paper_numbers.py asserts 71 paper numbers
against the committed records; 65 recompute, 6 resolve to quoted lines
of this log, and three failures were real paper errors, now errata 2.11
to 2.13: the F3 tree direction is draw-dependent and was presented as
characteristic; the F5 control floors were stale values matching no
committed cell; the F6 cancellation statistic 0.016 matched no recorded
quantity. Also caught and fixed: F2 called five detection-arm mutants
gross where the Method defines two gross classes, and F5's reordering
set is now named (split reduction, reassociation, online softmax).
Provenance table published in BACKGROUND.md from the corpus records; the
intro now marks the online-softmax pair's unclassified status instead of
claiming blanket acceptance.

## 2026-08-15 - Reader pass per review 8; internal loop closed

Four edits, each checked against its audited source before the wording
moved. The abstract and F3 heading now state the reference finding at
both granularities: single-draw reversal, draw-dependent tree direction,
11x mean for sequential (1/0.0938 from the 100-draw mean ratio, the same
number F3's body carries). The abstract drops the corpus parenthesis
(Method carries it) and three CI brackets (F1 and F4 carry them); no
number changed. F2 opens with an operational orientation that defines
separation before the envelope machinery. The eleven-falsified count
leaves the contribution list; the run log keeps it. Central numbers
untouched. Per the review, this closes the internal loop; next is the
external expert read.

## 2026-08-16 - Rev. 6: structure pass against venue conventions

- Trigger: the compiled rev. 5a.4 reads too dense (author review). Six
  comparable papers were profiled for structural norms before editing:
  FPRev (ATC 25), TTrace, Mirage (OSDI 25), NNSmith (ASPLOS 23), FTTN,
  and Mytkowicz (ASPLOS 09), measured from their full texts.
- Norms found: abstracts run 132-215 words for five of six (FTTN's 307
  is the outlier); measurement-genre abstracts carry roughly 4-10
  measured numbers, systems abstracts 0-3. Results paragraphs average
  2-5 sentences, with the measurement papers at 2-3; none of the six
  uses finding boxes or bold-sentence leads; the working device is the
  bold run-in heading. Headline results live in one float and are
  restated once inline (0.7-1.2 floats per body page). No paper in the
  set appendixes formal machinery (proofs are omitted instead), and only
  one of six has a dedicated limitations section.
- Changes (structure and pacing only; no measured value moved, and
  tools/check_paper_numbers.py is untouched): abstract cut from ~280 to
  ~200 words and from twelve quantitative tokens to four; Background
  split one paragraph per system; the envelope formalism moved from
  Method to a new Appendix B behind a five-sentence summary; F2 split
  into four led sub-paragraphs (exact result; the pairing that
  separates; the 64-point grid; the evading bug); F3's twelve inline
  numbers moved to a full-width reference table; F4-F7 split into two
  paragraphs each; Related work into three paragraphs.
- Two deliberate deviations from the profiled norms, kept on purpose:
  the full envelope construction stays in the paper (Appendix B) rather
  than being omitted, because this project's verifiability standard
  wants the exact analysis reproducible from the paper alone; and the
  dedicated threats-to-validity section stays, because an unaffiliated
  preprint substitutes disclosed limitations for venue review.
- Wording constraints from ERRATA 2.8-2.13 were re-checked and
  preserved: the exact-analysis claim stays count-free (sixteen frontier
  instances), the backend attribution stays "Apple M3 (CPU)", F3 states
  both the registered-draw and 100-draw directions, F5 floors carry the
  mlp shape, F6 keeps the bulk-of-rows statistics.
- Gate for this pass: recompile the tex and re-run
  tools/check_paper_numbers.py on the M3; both must be green before
  rev. 6 is called done.

## 2026-08-19 - Adversarial test of the Mirage absence claim

- Trigger: asked whether we know for certain that Mirage never specified
  its post-compilation floating-point check. We did not. The claim rested
  on the paper text and on one code audit whose method was in a
  gitignored file, so a reader could not check it.
- Prediction, written before the sweep: no stated protocol exists in any
  authored channel, but the code audit is the weak link and something
  tolerance-shaped will turn up in the test suite. Both held.
- Method: the search was instructed to DISPROVE the claim, not to confirm
  it. Channels: the USENIX camera-ready (full text, not a summary), the
  CMU and NSF PAR deposits, artifact-evaluation instructions, MPK, the
  ITCS 26 theory paper behind section 5, repository documentation, talks,
  and GitHub code at seven refs plus issues, PRs, and all discussions.
  Full record: audits/mirage-fp-filter.md.
- Outcome: the claim survives. No threshold, reference, or draw count in
  any channel searched. New supporting facts: the camera-ready carries no
  appendix; the sentence is absent from both the v1 PDF and the NSF PAR
  deposit, so it enters at the camera-ready; and the artifact evaluation
  that earned Results Reproduced asks for performance and search time
  only, with no numerical check.
- The counter-evidence, now disclosed rather than missed: is_closed() in
  tests/python/test_tensor_program.py pairs torch references with a 1e-1
  threshold on uGraph outputs. It is not the filter (hand-built graphs,
  never calls superoptimize, rejects nothing, conjunctive criterion).
  Recorded in Appendix A because a reader will find it, and because
  claiming it as Mirage's tolerance would repeat ERRATA 1.5 exactly.
- Three load-bearing facts were re-fetched and read a second time from
  source before entering the paper: is_closed, the fingerprint equality
  in probabilistic_verifier.cc, and ae.md.
- Fixed while here: Appendix A cited "the repository audit notes," which
  were gitignored. The method is now tracked at audits/mirage-fp-filter.md
  and the paper cites that path.
- Residual gap, stated in the paper and the errata: two recorded talks
  were not transcribed. A threshold named aloud would not be seen.
