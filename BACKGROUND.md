# Background

## The problem superoptimizers create for themselves

A kernel autotuner searches over *how* a fixed computation is scheduled — tile sizes, threads per row, vector widths. The semantics are pinned by a human-written template, so one reference implementation stays valid across the entire search space, and `torch.allclose` against that reference is an adequate gate.

A superoptimizer removes the template. It searches over program *structure*, inventing decompositions no human wrote. The input program remains available as a reference, so comparison is still possible — but three things change at once: the candidate count rises from tens to millions, the search actively selects for anything that passes the gate, and floating-point reassociation noise becomes indistinguishable in magnitude from genuine small errors. A sampled `allclose` with no error bound is no longer a gate; it is a smoke test being asked to hold back a search.

The field's response has been to replace numerical spot-checking with reasoning over exact arithmetic. Three systems, three approaches:

**Mirage** (OSDI '25) restricts candidates to the **Lax fragment** — multi-linear operators, division, and at most one exponentiation on any input→output path. Under that restriction every output entry has the closed form `Σᵢ (fᵢ/gᵢ)·e^(hᵢ/uᵢ)` with *f, g, h, u* polynomial in the inputs, so equivalence reduces to polynomial identity testing. Random evaluation over **finite fields** — exact modular arithmetic, no rounding — then yields a quantified error bound (≈ `8dk⁴/q`) drivable arbitrarily low by repetition. Operators outside Lax are not verified; the program is partitioned at them and each Lax fragment optimized in isolation.

**Prism** (arXiv 2604.15272) replaces probabilistic testing with e-graph reasoning over ~70 hand-written algebraic axioms describing parallelization operators (`part`, `comb`, `red`, `repl`). The paper is explicit about the resulting guarantee: *"A formal soundness proof is beyond the scope of this paper. Instead, we rely on careful manual review of the ∼70 axioms, and also subject all generated kernels to random equivalence testing."* Completeness is disclaimed, with a named gap (`T+T` = `2·T`), and whether a complete recursively-enumerable axiom set exists is left open.

**Axon** (arXiv 2606.26344) encodes tensor operations as SMT constraints over uninterpreted functions, proving equivalence for tensors of arbitrary size with known rank. Its two stated limitations are the relevant ones: nonlinear functions are treated as uninterpreted and valid transformations involving them are conservatively rejected; and *"the equivalence checking operates over real-valued arithmetic and does not account for floating-point precision."* The reason is cost — one operator swap took 247.75 s under Z3's floating-point theory versus 0.15 s under real theory, a 1650× slowdown. Floating-point equivalence is instead *"validated via testing on random floating-point inputs"* at `rtol = atol = 1e-4` in FP32.

## The gap

Read those three together and a pattern emerges that none of them is individually hiding, because in each paper it is a footnote to a different contribution:

| System | What is bounded or proven | Over what arithmetic | How the shipped kernel is checked |
|---|---|---|---|
| Mirage | `Pr[accepting a non-equivalent μGraph]`, quantified | finite fields (exact, mod *p*) | random testing |
| Prism | nothing formally; soundness by manual review | exact algebra under axioms | random equivalence testing |
| Axon | full proof of equivalence | ℝ (real theory) | random FP inputs, `rtol=atol=1e-4`, FP32 |

Every system verifies an **exact-arithmetic idealization**. Every system ships a **floating-point kernel**. In all three, the distance between the two is closed by sampling, with no error bound and no characterization of what the sampling might miss.

This is a reasonable engineering choice — exact FP reasoning is demonstrably too slow, and reassociation error is genuinely benign in most cases. But "most cases" is an empirical claim that has not been measured, and two trends are pushing against it simultaneously. Transformations are becoming more aggressive: megakernel compilers split reductions across streaming multiprocessors and reorder accumulation in ways that preserve real-arithmetic semantics but change summation order. And precision is falling: production inference runs bf16 or narrower, while the validation tolerance in the literature is calibrated at FP32.

## What this project measures

Whether the sampled floating-point check that every verified superoptimizer relies on is empirically adequate for the transformations those systems actually accept, and how its adequacy scales with reduced precision.

The result is informative either way. If transformation-induced error stays well inside validation tolerance across realistic shapes and precisions, the field's shortcut is justified and that fact is worth establishing rather than assuming. If it does not, the gap between what is proven and what is executed is a correctness concern rather than a precision footnote.

## References

- Mirage — arXiv [2405.05751](https://arxiv.org/abs/2405.05751), OSDI '25 · §5.1 (PIT theorem), §7 (non-Lax partitioning)
- Prism — arXiv [2604.15272](https://arxiv.org/abs/2604.15272) · §4 + Table 1 (axioms), §6
- Axon — arXiv [2606.26344](https://arxiv.org/abs/2606.26344) · §5.2 (guarantees and limitations)
- TensorRight — POPL '25 · the SMT methodology Axon adapts
- Ruler — arXiv [2108.10436](https://arxiv.org/abs/2108.10436), OOPSLA '21 · §6.2 (validation strategies, cvec seeding)
