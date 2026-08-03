# Papers

Working set for the kernel-verification project. PDFs are gitignored — this index is the committed artifact.

| File | Paper | Venue / date | Read |
|---|---|---|---|
| `mirage.pdf` | [Mirage: A Multi-Level Superoptimizer for Tensor Programs](https://arxiv.org/abs/2405.05751) | OSDI '25 | **§5.1** (the PIT theorem), §4.3 (pruning axioms A_eq/A_sub), §7 (non-Lax partitioning) |
| `prism.pdf` | [Prism: Symbolic Superoptimization of Tensor Programs](https://arxiv.org/abs/2604.15272) | arXiv, 16 Apr 2026 | **§4 + Table 1** (the ~70 axioms), §5 (instantiation = autotuning), §6 (axioms → directional rewrite rules) |
| `axon.pdf` | [Axon: A Synthesizing Superoptimizer for Tensor Programs](https://arxiv.org/abs/2606.26344) | arXiv, 24 Jun 2026 | **§5.2** (guarantees + the two limitations), §5.1 (SMT encoding), §4.2.1 (per-operator semantics specs), App. A (threats to validity) |
| `ruler.pdf` | [Rewrite Rule Inference Using Equality Saturation](https://arxiv.org/abs/2108.10436) | OOPSLA '21 | **§6.2** (validation methods — fuzzing as `is_valid`), §7 (limitations: no precondition inference) |
| `tensorright.pdf` | [TensorRight: Automated Verification of Tensor Graph Rewrites](https://arxiv.org/abs/2511.17838) | POPL '25 | The SMT-over-unbounded-tensors method Axon reimplements |
| `mpk.pdf` | [MPK: A Compiler and Runtime for Mega-Kernelizing Tensor Programs](https://arxiv.org/abs/2512.22219) | OSDI '26 | Context — the execution-side sequel to TransformerOp's Phase 5 |

## The through-line

Three systems, three ways of not handling nonlinearity, and all three fall back to random testing for the program that actually ships:

- **Mirage** restricts to the Lax fragment and *partitions around* non-Lax operators.
- **Prism** hand-writes ~70 axioms, states no formal soundness proof, admits incompleteness.
- **Axon** proves over ℝ (FP theory is 1650× slower) and treats nonlinear functions as uninterpreted.

**Ruler** is the socket: domain-general rule inference needing a grammar, an interpreter, and an `is_valid` oracle — never pointed at the parallelization-operator domain.

## Start here

`axon.pdf` §5.2 and `prism.pdf` §4. About three pages combined, and they are the whole argument.
