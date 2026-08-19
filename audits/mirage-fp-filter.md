# Mirage: where is the floating-point filter?

Audit of 2026-08-14, extended 2026-08-19. This file is the method and the
evidence for Appendix A of the paper. It replaces a local-only note.

This document uses short sentences. Each sentence gives one idea.

## The question

Mirage v3, section 5.2, states one sentence about floating-point testing:

> Mirage employs floating-point tests to filter out uGraphs with
> significant numerical errors.

The paper states no threshold. It states no reference implementation. It
states no draw count. The question of this audit: is that protocol stated
anywhere else, in any channel the authors control?

An absence claim is weaker than a presence claim. This project has
published a false absence claim before (ERRATA 1.3) and a false absence
claim about Mirage in particular (ERRATA 1.7). So the audit is written to
find the protocol, not to confirm that it is missing.

## What was searched

| Channel | Scope | Result |
|---|---|---|
| arXiv 2405.05751 | v1, v2, v3 full text, diffed | Sentence appears in v3 only. No threshold |
| USENIX OSDI '25 camera-ready | Full published text, pp. 221-238 | Identical sentence. No appendix. No threshold |
| CMU-hosted PDF, NSF PAR deposit | Full text | Both predate the sentence. It is absent |
| OSDI '25 artifact evaluation | `ae.md`, evaluation branch | Performance and search time only |
| MPK (arXiv 2512.22219) | Full text | No tolerance or correctness content |
| Li and Wu, ITCS '26 (arXiv 2506.04529) | Full text | The theory behind section 5. No FP content |
| Repository docs and README | 51 files, main branch | No verification page exists |
| Talks | OSDI '25 and EGRAPHS 2025-10-16 | Descriptions only. Captions not retrieved |
| GitHub code | 70 branches, 6 tags, 6 releases enumerated; trees scanned at `ffe38df` (main), `68ff606` (v0.2.4), `e8980e1` (evaluation), and four verification-adjacent branches | No float comparison in `src/search/**` at any scanned ref |
| GitHub issues, PRs, discussions | Searched for tolerance, numerical, allclose, floating point, filter out, verification | No stated threshold. All 9 discussions read |

Search terms, applied together: `toleran`, `atol`, `rtol`, `allclose`,
`epsilon`, `1e-`, `10^`, `relative error`, `reference implementation`,
`numerical`, `assert_close`. The published text was extracted and matched
directly, not summarised, because a summariser can silently drop a
negative.

## What the acceptance path actually does

Read directly at `ffe38df`, `src/search/verification/probabilistic_verifier.cc`:

```cpp
if (!fingerprints[match[i]].has_same_fingerprint(
        input_graph_fingerprints[i])) {
  return false;
}
```

That is exact equality of finite-field fingerprints. There is no float
comparison in it, and no tolerance.

The only floating-point execution of candidates in the search path is
`src/search/profile.cc`, `profile_run()`. It draws fp16 inputs from a
fixed seed, runs 16 warmup and 100 timed iterations, and reads elapsed
time. It never compares the outputs to anything. It measures speed.

## The strongest counter-evidence, and why it is not the filter

`tests/python/test_tensor_program.py` contains `is_closed()`. It is the
only place in the repository that puts a reference implementation and a
threshold together on uGraph outputs:

```python
if (rel_error > 1e-1) & (abs_error > 1e-1):
    err += 1
```

It compares against torch: `silu`, `softmax`, `matmul`, `RMSNorm`. It is
present on main, on v0.2.4, and on the evaluation branch.

It is not the filter of section 5.2. Four reasons:

1. It builds its graphs by hand. It never calls `superoptimize()`.
2. It never reaches `ProbabilisticVerifier`. It filters no candidate.
3. It is a pytest case in CI. It exercises transpilation and execution.
4. Its criterion is conjunctive. A failure needs both errors above 1e-1.
   That is looser than `allclose`, which any one error trips.

Calling this "Mirage's tolerance" would repeat the exact category error
this project retracted in ERRATA 1.5: a number from one subsystem,
attributed to another. It is recorded here so that a reader who finds it
knows it was found and adjudicated, not missed.

Weaker cases, same reasoning: `demo/demo_jit.py` at `atol=1e-3`,
`tests/transpiler/lib.h` at 2e-2 and 5e-2, and an RMSNorm epsilon of
1e-6 emitted by MPK codegen. None sits in the acceptance path.

## Artifact evaluation

The OSDI '25 badges are Available, Functional, and Results Reproduced.
The instructions that earned them are `ae.md` on the evaluation branch.
They ask the evaluator to install the system, run benchmarks, and read
optimization time and performance. They ask for no numerical check. So
the reproduced result is performance, not numerical acceptance.

## Positive controls

An absence claim needs proof that the search can find a positive.

- The section 5.2 sentence itself was found by the same text search that
  found no threshold. The search reaches floating-point content.
- The code scan found the known tolerance sites in `demo/`, in
  `tests/transpiler/`, and in `tests/python/`. It reaches tolerances.
- The scan found `is_closed()`, which is the hardest case to find and the
  one most likely to refute the claim.

## Independently re-verified 2026-08-19

Three load-bearing facts were fetched and read a second time, from the
source, after the first pass reported them:

- `is_closed()`: thresholds, conjunction, hand-built graphs, torch
  references. Confirmed.
- `probabilistic_verifier.cc`: exact fingerprint equality. Confirmed.
- `ae.md`: performance and optimization time only. Confirmed.

## What this audit does not cover

The claim is bounded by these gaps. They are stated so that the bound is
visible.

1. **Spoken content.** Two recorded talks exist. Captions were not
   retrievable. A threshold named aloud in a talk or in Q&A would not
   appear in any text channel. This is the largest gap.
2. **Full commit history.** Trees were scanned at seven refs, not at
   every commit. A tolerance added and removed between those snapshots
   would not be seen. 66 of 70 branches were not scanned.
3. **Non-public material.** Reviews, author correspondence, and internal
   documents are out of reach by construction.
4. **Forks.** Not searched.

## Conclusion

The protocol for the section 5.2 filter is not stated in any channel
searched. The sentence enters the record at the camera-ready and arrives
without a threshold, a reference, or a draw count. In the released code,
at every scanned ref, candidate acceptance is exact fingerprint equality
and the only floating-point run of a candidate measures time.

The paper states this as a reporting gap, not as a defect. Stating the
threshold, the reference, and the draw count would close it.
