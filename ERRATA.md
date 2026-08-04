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

### 1.5 TF32 — caught before it did damage

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
