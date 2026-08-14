# Background

## The problem

An autotuner tunes a human-written template. The template pins the semantics.
One reference implementation stays valid for the whole search, so `allclose` is an adequate gate.

A superoptimizer removes the template and searches program structure.
Three things change at once.
Candidate counts rise from tens to millions.
The search selects for whatever passes the gate.
Reassociation noise becomes the same size as real small errors.
A sampled `allclose` with no error bound is then a smoke test asked to hold back a search.

## Three systems, three exact-arithmetic answers

| System | What is proven | Arithmetic | Shipped kernel checked by |
|---|---|---|---|
| Mirage (OSDI '25) | Probability of accepting a wrong uGraph, quantified | Finite fields, exact | Random testing |
| Prism (arXiv 2604.15272) | Nothing formally; manual review of ~70 axioms | Exact algebra | Random equivalence testing |
| Axon (arXiv 2606.26344) | Full equivalence proof | Reals | Random FP inputs, `rtol = atol = 1e-4`, FP32 |

Mirage restricts search to the Lax fragment and partitions around other operators.
Prism writes ~70 algebraic axioms by hand and disclaims a formal soundness proof.
Axon uses the real theory because Z3's FP theory is 1650x slower on its own example.
Mirage (v3, section 5.2) also employs floating-point tests to filter uGraphs with
significant numerical errors. It states no threshold or protocol for those tests.

## The gap

Every system verifies an exact-arithmetic idealization.
Every system ships a floating-point kernel.
Sampling closes the distance in all three, with no error bound.

The authors state this limitation themselves.
None of them measures it.
Two trends press on it: transformations reorder more accumulation, and production precision falls to bf16 and below, while the one stated tolerance is calibrated at FP32.

## What this project measures

Whether the sampled floating-point check is adequate for the transformations these systems accept, and how that changes with precision.
Either answer is informative.
If error stays inside tolerance, the shortcut is justified and now established.
If it does not, the gap is a correctness concern.

## References

- Mirage: arXiv [2405.05751](https://arxiv.org/abs/2405.05751), OSDI '25. Sections 5.1, 7
- Prism: arXiv [2604.15272](https://arxiv.org/abs/2604.15272). Section 4, Table 1
- Axon: arXiv [2606.26344](https://arxiv.org/abs/2606.26344). Section 5.2
- TensorRight: POPL '25. The SMT method Axon adapts
- Ruler: arXiv [2108.10436](https://arxiv.org/abs/2108.10436), OOPSLA '21. Section 6.2
