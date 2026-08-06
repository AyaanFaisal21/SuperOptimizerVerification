# Errata

Every error and false claim found so far, what it would have cost, and how it was
caught. [`NOTEBOOK.md`](NOTEBOOK.md) is chronological; this is organised by failure
mode, because the *kind* of mistake is the transferable part.

Nothing here is hidden elsewhere in the repo. It is collected because a project whose
entire output is "how wrong are these numbers" has no standing to be quiet about its
own.

**The organising observation:** the dangerous errors are the ones that make results
look *better*. A broken instrument that reads zero passes an equivalence check. A
tolerance measured against the wrong reference still produces a clean table. Loud
failures — crashes, timeouts, syntax errors — cost minutes. The silent ones in §1
would have cost the entire result and left no trace.

**And the pattern held all the way through.** Every claim in this project that moved
under scrutiny moved the *same direction* — away from the thesis. Nothing supporting
the null hypothesis ever shrank when pressed. Two of the largest corrections came only
after someone asked me to verify a result, not from any internal check. That is what
confirmation bias looks like from the inside, and it is the strongest available
evidence that the surviving claims were tested rather than merely confirmed.

---

## 1. Silent corruption — would have produced clean, wrong numbers

### 1.1 The float64 reference had too much headroom to round

**What.** `corpus._randn` drew inputs at float32 and cast them *up* to float64. Values
carrying only 24 mantissa bits, summed in a 53-bit accumulator, are exact **in every
order** — so `split_reduction` and `reassociation` reported `0.000e+00` disagreement
in the Phase 1 equivalence check.

**Cost if unnoticed.** A perfect zero on an equivalence check reads as success. It
would have survived into Phase 4, suppressed real error at the reduction pairs, and
pushed the sweep toward A1 — the direction I was least likely to interrogate.

**Caught by.** Not the equivalence check, which it passed. By reading the per-cell
table and asking why two pairs were *exactly* zero when a third was not.

**Fixed.** Draw at float64, round down to the precision under test — which also gives
every precision the same underlying real values. Pinned by
`test_float64_reference_actually_rounds`.

**Generalisation.** An equivalence check is satisfied by an instrument that reads
zero. That is why the negative control is a first-class exit criterion and not a
nicety: something must independently prove the instrument has a nonzero reading.

### 1.2 The gate compared against the wrong reference

**What.** `CLAIM.md` defined the validation gate as variant-vs-float64-truth. Axon
§4.6 states it as `|emitted_code − reference| ≤ atol + rtol|reference|` — against a
*reference implementation*, not an oracle. Mirage and Prism likewise compare
candidate against input program.

**Cost if unnoticed.** C1 is a claim about "the threshold **used to validate them**."
Measuring against truth measures a gate nobody uses. Every number would have been
correct and irrelevant.

**Caught by.** Re-reading Axon §4.6 while answering a question about direction. It
was readable in the paper with no data at all and should have been caught in Phase 0.

**Fixed.** Dated `CLAIM.md` amendment; gate is now the differential, as-registered
form co-reported on every cell so the change is auditable rather than trusted.

**Generalisation.** The error was textual, not numerical — a claim about what other
systems do, never checked against their text. Any "the field does X" premise needs a
line number before it is built on.

### 1.3 `grep` silently skipped the paper text

**What.** pypdf-extracted text is classified `data` by `file`, so grep exits 1 with no
output rather than searching. The first literature sweep returned **zero hits for
"tolerance"** across all six papers.

**Cost if unnoticed.** I was one step from concluding the papers never mention a
validation tolerance — a *false negative in the literature check that would have
supported our thesis*. The correct sweep found Axon's tolerance stated explicitly.

**Caught by.** Zero hits for the word "the" in a 107 KB document. The absurdity of the
control result, not the result itself.

**Fixed.** `LC_ALL=C grep -a`, plus a sanitising pass over the extracted text.

**Generalisation.** An absence claim needs a positive control. Search for something
that *must* be there before believing something isn't.

### 1.4 Cancellation measured globally instead of per row

**What.** The activation fixture reported `μ²/E[x²]` over the whole tensor. LayerNorm
reduces **per row**, so per-row statistics govern cancellation.

**Cost if unnoticed.** Global understates severity by up to **40×** — `attn_scores`
reads `0.0091` globally against a row max of `0.3608`; the residual stream reads
`0.0000` globally against a row max of `0.0158`. It would have supported "real
activations are perfectly conditioned" more strongly than the data warrants.

**Caught by.** Asking which axis the operator actually reduces over.

**Fixed.** Per-row mean / p99 / max recorded alongside the global figure.

**Generalisation.** Match the statistic to the axis the operator reduces over. A
summary over the wrong axis is not a weaker measurement, it is a different one.

### 1.5 The "100× tolerance gap" — asserted publicly, then retracted

**What.** Claimed that the literature states `1e-4` while the code shipping
reduced-precision kernels uses `1e-2`, an undocumented 100× gap. Committed to a
public repo (`3adb846`) and stated to a collaborator before being checked.

**Why it is wrong.** Four independent reasons:

- **Different groups.** Axon is AWS/UIUC; Mirage is CMU. One group's stated threshold
  against another group's code is not a gap.
- **Mirage's paper states no float tolerance at all** — its only "threshold" is the
  PIT error probability δ.
- **Different subsystems.** All 66 `1e-2` sites are `tests/runtime_python/` and
  `demo/` — hand-written MPK runtime kernels checked against PyTorch. None validates
  superoptimizer output.
- **Different artifact.** The repo at HEAD is MPK, the megakernel successor, not the
  OSDI '25 superoptimizer.

**Cost if unnoticed.** This was a public, specific critique of named researchers,
built on a category error. It would have been the most prominent claim in the writeup
and the first thing a reviewer — or the authors — would have dismantled.

**Caught by.** Being asked to verify it *because* it was a critique of real work.
Not by any internal check.

**What survives.** The census numbers are accurate; every inference drawn from them
was not. Narrowly: Axon states 1e-4 at FP32, Prism states nothing while benchmarking
at half precision, and Mirage's superoptimizer path has no float check because its
verification is exact by construction. MPK's 1e-2 on bf16 kernels is corroborating
context about what practitioners accept, not a contradiction.

**Generalisation.** A measurement that supports the thesis deserves *more* scrutiny
than one that doesn't. This is the second absence-shaped error after §1.3, and both
went the same direction — toward the conclusion I wanted. Before asserting anything
about someone else's work: check authorship, subsystem, and artifact identity.

### 1.6 TF32 — caught before it did damage

**What.** On Ampere, TF32 carries a 10-bit mantissa. Left enabled for matmul, the fp32
*baseline* is not fp32.

**Cost if unnoticed.** Measured: **750×** accuracy loss, landing at `2.99e-04` — over
the 1e-4 gate by itself. Every matmul-shaped fp32 cell inflated, in the direction that
makes C1 look true.

**Caught by.** Checking before running, because §1.1 had already established that this
project's failure mode is silent instrument error. The torch default is safe
(`allow_tf32 = False`), but every script pins it explicitly rather than inheriting it.

---

## 2. Claims of mine that were falsified

Recorded because they were stated to a human before being checked.

| Claim | Status | Evidence |
|---|---|---|
| The A10 makes accumulation width a switchable experimental axis | **false** | Both settings of `allow_fp16_reduced_precision_reduction` are bit-identical at 512×4096×512 — cuBLAS never selects a split-k kernel there, so the flag is a no-op |
| CPU overstates error vs tensor cores for matmul | **false** | Paired test, identical input bits: `3.9533e-04` (fp16) and `3.0960e-03` (bf16) on *both* machines, identical to every printed digit. Output quantisation dominates |
| `matmul_k_tiling`'s `d/floor` orders by K | **false** | attention (K=512) reads 4.01 against mlp (K=768) at 3.48. **Still unexplained** — output width and the max normaliser are confounded with K in the current shapes |
| Gate disagreements would appear at `layernorm_variance` | **false** | All three are `softmax_online`/fp16. The majority-agreement half was right (51/54) |
| Post-GELU activations would supply badly-conditioned LayerNorm inputs | **weakened** | Post-GELU row max cancellation is 0.36, but in a pre-norm transformer LayerNorm consumes the *residual stream*, whose row max is 0.0158 |
| Phase 5's seeded inputs are "adversarial but realistic" | **false** | `cancellation`'s median row condition number is 4.5e6 against real rows' 42.8 — **+10.4σ** on a log scale, ~600× worse than the *worst* real row. Bounding values inside the observed range does not make the arrangement realistic, and the arrangement is the entire mechanism |
| "C1 survives at fp32; uniform sampling is a null test" | **overstated** | Real activations at their worst — `post_ln_L5`, row condition 368,927 — still pass with 10× headroom. Phase 5 shows these transformations *can* diverge, not that they *do* |

The first two were predicted-against-myself in `NOTEBOOK.md` *before* the run that
falsified them, which is the only reason they are corrections rather than silent
retractions.

---

## 3. Tooling and environment

Cost minutes each; recorded so they are not re-debugged.

| Issue | Symptom | Cause / fix |
|---|---|---|
| Home directory is a git repo | `fp-verification-gap/` sat inside a repo of LeetCode commits | Never separately publishable. Consolidated into a standalone repo |
| Copyrighted paper text committed | `git add -A` swept in a 72 KB full-text extraction, into a repo headed for public GitHub | Untracked, `.gitignore` extended to `papers/*.txt`. The repo's own convention already said PDFs stay out |
| f-string nesting | `SyntaxError` on the A10 | Same-quote nesting is 3.12+; the box runs 3.10. Write tools as files, not heredocs |
| `nohup … &` over ssh | Command hung 600 s | Backgrounded process holds the ssh channel unless every fd is redirected. Training was running fine the whole time |
| 56.6 MB fixture for 11 MB of data | — | `.contiguous()` is a no-op on a row slice of a row-major tensor, so `torch.save` wrote the full underlying storage. `.clone()` forces fresh storage |
| pytest missing on the A10 | — | Installed; the test file also runs standalone via `__main__` |

---

## 4. Open

- **`matmul_k_tiling`'s `d/floor` does not order by K** (§2). Needs a controlled sweep
  with M and N pinned; the current shape classes confound K with output width. Logged
  as unexplained rather than rationalised.
- **`softmax_online`'s gate disagreement is partly an `atol` artifact.** Softmax
  outputs have magnitude ~1/N, so `atol = 1e-4` is comparable to the values
  themselves, making the gate unusually permissive against truth for this operator.
  The finding stands; the mechanism needs that caveat attached.
- **Whether Prism's artifact states a validation tolerance.** Verified absent from the
  *paper*; the code may set one. Checking it would strengthen or cleanly refute the
  sharpest observation in the project.

---

## 5. Findings of the 2026-08-04 full-method audit

An adversarial pass over our own methodology; nine findings, three verified by new
measurement before writing. The audit itself, with remaining steps, is
[`AUDIT.md`](AUDIT.md); the measurements are in `NOTEBOOK.md` (2026-08-04).

| # | Finding | Class |
|---|---|---|
| 1 | The 14% catch rate is input-scale-contingent: 14/100 [CI 9–22%] at the measured activation scale σ≈2.7, **0/100 at unit scale**. No document stated the scale | measured; corrected |
| 2 | The "1600–2100× the gate" severity figures were `scale_rel/GATE` with near-zero denominators; true elementwise exceedance is **≈2×** | measured; corrected |
| 3 | The seq-order baseline is not the field's tree-order reference kernel; reduction-pair differentials may overstate the field's comparator | open — AUDIT step 3 |
| 4 | The cross-platform "identical" claim compared 5-digit statistics, not tensors; CPU bf16 matmul matches neither naive accumulator model | measured; mechanism open |
| 5 | Softmax immunity mechanism misstated: median output 5.9e-4 is 6× *larger* than atol. True mechanism: rel err 100× under rtol, outputs ≤ 1 | corrected |
| 6 | No confidence limits on any rate; "five times in six" should be "six times in seven" (86%) | corrected |
| 7 | "These systems test once" was never verified against Axon/Prism | removed from current docs |
| 8 | Narrow evidence: 3×-spread claim from one seed/shape; mpmath check small shapes only; fixtures from one model/batch; REAL_BOX from one site | disclosed |
| 9 | `d/floor` is a ratio of maxima at possibly different elements — compares magnitudes, not element-level coupling | disclosed |

The pattern of §2 continues: every number that supported the thesis shrank under
scrutiny. This is now six for six.
