# Errata

Every error and false claim, by failure mode. [`NOTEBOOK.md`](NOTEBOOK.md) holds the chronology.

Two patterns organise this file.
The dangerous errors made results look better, not worse.
Through review #2, every claim that moved under scrutiny moved away from the thesis: six for six.
Review #3 (2026-08-14, §2 below) broke the pattern: two of its corrections strengthen the thesis, the rest are neutral.
The largest corrections came only after an external prompt to verify.

## 1. Silent corruption: would have produced clean, wrong numbers

| # | Error | Cost if unnoticed | Caught by | Lesson |
|---|---|---|---|---|
| 1.1 | Inputs drawn at float32, cast up to float64. 29 spare mantissa bits made float64 sums exact in every order. Reduction pairs read 0.000e+00 | Clean zeros pass an equivalence check. The sweep tilts toward A1 | Asking why two pairs were exactly zero when a third was not | A zero-reading instrument passes equivalence checks. Require a negative control |
| 1.2 | Gate defined as variant vs float64 truth. Axon 4.6 compares against a reference implementation | Measures a gate nobody uses. Correct numbers, wrong question | Re-reading Axon 4.6 | Any "the field does X" premise needs a line number |
| 1.3 | `grep` classified extracted paper text as binary and returned no hits. First sweep found zero mentions of "tolerance" in six papers | A false absence claim that supported our own thesis | Zero hits for the word "the" in a 107 KB file | An absence claim needs a positive control |
| 1.4 | Cancellation statistic computed globally. LayerNorm reduces per row | Understates severity by up to 40x | Asking which axis the operator reduces over | Match the statistic to the reduction axis |
| 1.5 | The "100x tolerance gap" claim. Compared Axon's stated threshold (AWS/UIUC) against MPK runtime-kernel tests (CMU): different group, subsystem, and artifact. Mirage's paper states no float tolerance (note: this part was itself wrong, from a stale paper version; superseded by 1.7) | A public critique of named researchers, built on a category error | An external request to verify, not an internal check | A number that supports the thesis deserves more scrutiny, not less |
| 1.6 | TF32 on Ampere: 10-bit mantissa. Measured cost 750x, landing at 2.99e-04, over the gate by itself | Every fp32 matmul cell inflated toward C1 | Checked before running; pinned off in every script | Pin precision flags. Do not inherit them |
| 1.7 | Claimed Mirage has "no floating-point check at all." Our PDF was a stale arXiv revision; v3 section 5.2 states Mirage employs FP tests to filter uGraphs with significant numerical errors. Retraction #3 | A factual error about a named system, in the paper draft | External review, verified against arXiv v3 | Pin the version of any paper you quote. Check for revisions before asserting absence |

## 2. Claims falsified after being stated

| Claim | Status | Evidence |
|---|---|---|
| The A10 gives a switchable accumulation-width axis | False | Both flag settings bit-identical at 512x4096x512. The flag is a no-op there |
| CPU overstates matmul error vs tensor cores | False | Paired test: 3.9533e-04 (fp16), 3.0960e-03 (bf16) on both machines to 5 digits. See 2026-08-04 note below |
| `matmul_k_tiling` d/floor orders by K | False | K=512 reads 4.01; K=768 reads 3.48. Unexplained; AUDIT step 6 |
| Gate disagreements appear at `layernorm_variance` | False | All three are `softmax_online` at fp16 |
| Post-GELU supplies ill-conditioned LayerNorm inputs | Weakened | Pre-norm architecture feeds LayerNorm the residual stream, row max 0.0158 |
| Phase 5 seeds are "adversarial but realistic" | False | Median seeded row condition 4.5e6 vs real 42.8: 10.4 sigma out |
| "C1 survives at fp32; uniform sampling is a null test" | Overstated | Worst real site passes with 10x headroom. Violations near 2x tolerance, not 1600x |
| "The two machines agree bit-identically; output rounding masks accumulation" | Revised 2026-08-04 | The equality was a 5-digit statistic, not tensors. CPU matmul agrees with fp32 accumulation on 99.4-99.96% of elements. The platforms run nearly the same computation |
| "d/floor ~ 1 means the transformation adds nothing beyond precision" | False (2026-08-08) | d/floor is direction-blind. total/floor shows reordering variants are 8-16x MORE accurate while d/floor reads ~1. External review point 3, confirmed in our own records |
| "Pre-norm architecture shields LayerNorm structurally" | Overclaimed (2026-08-08) | Pre-LN does not center the residual stream, and residual magnitude grows with depth. Our near-zero row means are an empirical property of one 6-layer model. Post-LN "zero-mean by construction" also ignored the learned affine bias |
| "Single-draw validation, which is what these systems run" | Regression (2026-08-08) | Audit finding 7 removed this; the paper draft reintroduced it. No paper states its FP draw count. Corrected to a protocol-underspecification finding |
| "The valid tiling fails at K=2048" | Wrong predicate (2026-08-08 #2), retraction #4 | The K-sweep compared max abs diff against atol alone, ignoring the rtol term, the exact quantity confusion this paper criticizes. Under the literal elementwise rule: K=2048 rejected 0/100 [0-4%], K=4096 48/100 [38-58%], K=11008 100/100 [96-100%]. The corrected result is stronger and needs no extrapolation |
| "torch.sum reference within 2x of the tree reference" | Falsified (2026-08-08 #2) | torch.sum floor is 2.6x better than our strided tree at fp16 (3.67e-04 vs 9.69e-04), consistent with wider internal accumulation. Verdicts matched |
| "Each paper states this limitation" (rev. 3 intro) | Corrected (2026-08-14, review #3) | Prism's paper never mentions floating point. Only Axon and Mirage state the gap. The corrected sentence is stronger for the thesis |
| "Online softmax is outside Lax because of more than one exp per path" | Corrected (2026-08-14) | Mirage never classifies softmax. Plain softmax has one exp per input-output path and appears to satisfy Def. 5.1. The defensible mechanism is the running max, which is not a Lax operator. Restated as our inference |
| "layernorm_variance is accepted by Prism" | Retracted (2026-08-14) | Prism contains no LayerNorm and no variance rewrite. Every normalization it touches is RMSNorm |
| "Prism accepts online softmax by axiom (streaming/scan axioms)" | Corrected (2026-08-14) | Prism verifies the chunked form without the running max (its Fig. 2). It has no streaming or scan axioms. The stabilized form is not in its paper |
| "The frontier optimum is at exactly (1e-4, 1e-4)" | Imprecise (2026-08-14) | Three-way tie with (1e-4, 1e-3) and (1e-4, 1e-2) at 0% + 6.25% in the committed JSON. First-wins tie-breaking picked the published constant. Now stated as a plateau |
| "Sixteen mutants spanning 4.1e-6 to 8.2e-1" | Corrected (2026-08-14) | Eighteen instances across six classes. The 8.2e-1 endpoint is the detection-arm rescale mutant, not one of the frontier's sixteen |
| "Reordering candidates are 8-50x closer to the oracle" (rev. 3 F5) | Corrected (2026-08-14) | The committed records give 3-73x across cells and 8-62x at the mlp shapes. The selection is now stated |
| "Mirage tests with 16-bit primes" | Imprecise (2026-08-14) | The primes are p=227 and q=113. Their product fits in 16 bits |
| "The ~1e-2 sites are MPK runtime-kernel tests" (detail of 1.5) | Corrected (2026-08-14) | The 2e-2/5e-2 checks sit in the mirage repo's transpiler tests. The runtime tests print ratios with no tolerance |
| Alive-FP cited as "Menendez, Nagarakatte, Martin" | Corrected (2026-08-14) | The third author is Aarti Gupta (DBLP) |

The first two were predicted against myself in the notebook before the falsifying run.
The 2026-08-14 rows come from review #3: a full-text re-verification of every reference.

### 2.8 Abstract corpus count (Rev. 5a, corrected same day)

Rev. 5a said the exact analysis separates six rewrites from eighteen
injected bugs. The recorded envelopes cover the sixteen frontier
instances; the two gross detection-arm instances are excluded. Their
inclusion could only strengthen non-separation, but the published count
was wrong. Corrected in Rev. 5a.1.

### 2.9 Backend attribution (Rev. 5a, corrected same day)

Rev. 5a attributed the K rejection rates to "our PyTorch/Ampere
backend." The K extension and scale grids ran in float32 PyTorch on the
Apple M3 CPU. The Ampere A10 verified corpus equivalence and hardware
characterization only. Corrected in Rev. 5a.1.
### 2.10 Mirage bound factor (Rev. 5a.1, corrected next day)

Appendix A said the released q weakens Mirage's per-test acceptance
bound by roughly 1.4x, treating the bound as proportional to 1/q. The
paper's Theorem 2 bound, re-extracted verbatim from the v3 source, is
8dk^4/q + q^(-1/k^2). The second term moves by (113/83)^(1/k^2), about
1.02 to 1.08 for k in 2..4, so no universal factor exists. Corrected to
the monotone statement the theorem does support: both terms grow as q
falls, so the released configuration loosens the stated bound at every
fixed (d, k). External review 7 caught this.

## 3. Tooling

| Issue | Fix |
|---|---|
| Home directory is itself a git repo | Project moved to a standalone repo |
| `git add -A` swept in 72 KB of copyrighted paper text | Untracked; `papers/*.txt` ignored |
| Same-quote f-string nesting fails on Python 3.10 | Write tools as files, not heredocs |
| `nohup ... &` over ssh holds the channel open | Redirect every fd, or expect the hang |
| 56.6 MB fixture for 11 MB of data | `.contiguous()` is a no-op on row slices; use `.clone()` |
| Instance terminated with the only checkpoint copy | Copy artifacts off rented machines first. AUDIT step 11 |

## 4. Open

- `matmul_k_tiling` d/floor does not order by K. Needs a controlled K sweep with M and N pinned.
- Softmax gate immunity is partly an atol artifact of small outputs. Caveat travels with the finding.
- The seq-order baseline is not the field's tree-order reference. Reduction differentials may overstate. AUDIT step 3.

## 5. Findings of the 2026-08-04 method audit

Nine findings; three measured before writing. Full text: [`AUDIT.md`](AUDIT.md).

| # | Finding | State |
|---|---|---|
| 1 | The 14% catch rate holds at activation scale (sigma 2.7) only. Unit scale: 0/100 | Corrected |
| 2 | The 1600-2100x severity figures had near-zero denominators. True exceedance near 2x | Corrected |
| 3 | Seq baseline is not the field's reference kernel | Open |
| 4 | Cross-platform "identical" was a statistic, not bits | Measured; GPU half open |
| 5 | Softmax immunity mechanism misstated. True cause: rel err 100x under rtol, outputs at most 1 | Corrected |
| 6 | No confidence limits on rates. "Five in six" should read "six in seven" | Corrected |
| 7 | "These systems test once" was never verified | Removed |
| 8 | Narrow evidence: one seed per cell, one model, one activation site for the seed box | Disclosed |
| 9 | d/floor compares maxima at possibly different elements | Disclosed |

## 6. Protocol deviations

Found in review #3 (2026-08-14).

| Registered | Implemented | Note |
|---|---|---|
| E1 mutant grid: dropped columns for j in {1,2,4,8,16,32} | j in {1,2,8,32} | The sparser grid removes the j=4 and j=16 column mutants only. The implemented set still brackets the severity range (j = 1 to 32). No threshold moved |
