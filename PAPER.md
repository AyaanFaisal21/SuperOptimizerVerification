# Characterizing Floating-Point Validation Rules for Tensor Superoptimization

Ayaan Faisal. Draft of 2026-08-08 (rev. 3, after external reviews 1 and 2).
Artifact: this repository.

## Abstract

Recent tensor superoptimizers specify their symbolic equivalence mechanisms in
detail, then apply a comparatively unspecified floating-point acceptance rule
to the compiled kernel: compare against a reference implementation on random
inputs, elementwise, within `|g_i - b_i| <= atol + rtol*|b_i|`. Axon states
`rtol = atol = 1e-4` on FP32; Prism and Mirage describe floating-point testing
without threshold or protocol. We characterize this validation stage as a
measurement problem: what does its verdict depend on, and what would need to
be reported for the verdict to be interpretable? We measure a representative
rule on six rewrites that are algebraic identities over the reals and sixteen
injected bugs spanning float64 divergences from 4.1e-6 to 8.2e-1, across
contraction sizes, precisions, reference implementations, input scales, and
draw counts, under a pre-registered protocol with dated amendments. Findings:
(1) the rule's rejection of valid rewrites is governed by near-zero output
elements meeting accumulation-dependent error growth, and reaches the
contraction widths of deployed models: at M=N=64 the valid tiled matmul is
rejected on 0/100 draws at K=2048, 48/100 [38-58%] at K=4096, and 100/100
[96-100%] at K=11008, the projection and MLP widths of Llama-2-7B; (2) a
tolerance frontier over 64 (atol, rtol) points contains no separator: the
optimum is Axon's own constant, which rejects no valid rewrite at unit scale
yet passes a real eps-placement bug on every draw, so the miss is structural
to the rule family, not a calibration error; (3) the verdict and even the
direction of the accuracy comparison depend on the reference implementation:
against sequential, tree, and `torch.sum` references, the same online-softmax
candidate is 15x closer, 1.6x closer, and 1.6x farther from a float64 oracle,
respectively; (4) detection is a rate that current papers do not permit
computing: a valid rewrite trips the rule on 14% [9-22%] of draws at measured
activation scale and 0/100 at unit scale, and the tensor-level rate is
predicted by per-element exceedance under independence (13.1% predicted,
observed CI 9-22%). Real activations pass everywhere measured. The study's
conclusion is a specification, not an indictment: a reported reference order,
draw count, precision scope, and error budget would make this stage as
interpretable as the verifiers above it.

## 1. Introduction

A kernel autotuner tunes a fixed computation against a template that pins
semantics; one reference validates every candidate. A superoptimizer searches
program structure across millions of candidates, and the search selects for
whatever passes the acceptance rule. Mirage, Prism, and Axon verify candidate
equivalence over exact or idealized arithmetic (Section 2) and close the
distance to the compiled floating-point kernel with a sampled comparison
against a reference implementation. Each paper states this limitation. None
reports enough about the sampled rule, reference, threshold, draw count,
precision scope, for a reader to compute what it accepts or rejects.

We treat that stage as an object of measurement. The question is not whether
any system has shipped a wrong kernel (we find no evidence of that), but what
the acceptance rule's verdict depends on: problem shape, output multiplicity,
reference implementation, precision, input scale and distribution, and number
of draws. Two failure directions are measured on the same rule at the same
tolerance: rejection of rewrites that are algebraic identities over the
reals, and acceptance of injected bugs.

Scope, stated first: we characterize the rule, not the systems' outputs. The
rewrites are PyTorch reimplementations of transformations these systems
accept, run on an Ampere GPU and an Apple M3 with TF32 disabled. No kernel
emitted by Mirage, Prism, or Axon is executed; Axon's deployment target
(Trainium/NKI) differs from our backend in accumulation and lowering, and
FPRev and FTTN document that accumulation order and extra-precision behavior
vary across real libraries and accelerators and are largely unspecified.
Findings transfer to the extent the rule, not the backend, drives them;
Section 5 marks which findings the caveat touches.

A note on labels, following the distinction this study itself surfaces:
"valid" below means equivalent over the reals, and "bug" means inequivalent
over the reals. Real-equivalence is not numerical acceptability. A real
identity can be numerically unstable, and a semantic bug can be numerically
negligible on a given input domain; our corpus contains both cases, and
Section 4 keeps the two axes separate: semantic class is fixed by
construction, numerical quality is measured against a float64 oracle.

Contributions:

1. A two-class corpus: six real-equivalent rewrites with per-system
   provenance (identities established symbolically; implementations validated
   to 3.5e-15 against a 50-digit-validated float64 oracle), and sixteen
   injected bugs whose float64 divergences form a severity continuum from
   4.1e-6 to 8.2e-1.
2. An instrument that reports, per cell, the reference's oracle error
   (floor), the candidate's oracle error (total), their disagreement
   (differential, what the rule tests), the direction ratio total/floor with
   absolute errors alongside, and the rule's verdict under three reference
   implementations.
3. A tolerance-frontier measurement over the full (atol, rtol) grid,
   answering whether observed failures are a calibration problem or a
   rule-family problem.
4. An audit trail: pre-registration with dated amendments, predictions
   committed before every run (the log records nine falsified by their own
   runs, including one this revision), and an errata with four retracted or
   corrected results of our own.

## 2. Background

Mirage restricts search to its Lax fragment and probabilistically verifies
candidate equivalence by evaluation over random finite-field values; the
theorem bounds the probability of accepting a non-equivalent program, and
the implementation runs a single trial with 16-bit primes, which the authors
acknowledge. Separately, Mirage v3 (section 5.2) states that floating-point
tests filter muGraphs with "significant numerical errors"; threshold,
reference, and draw protocol are unstated. Prism reasons over roughly 70
hand-written axioms in an e-graph; the axioms are intended to be sound, a
formal proof is stated to be beyond scope, and every generated kernel is
subjected to random equivalence testing with no stated tolerance, in an
evaluation that is entirely half-precision. Axon proves equivalence over the
reals with Z3 (1650x faster than Z3's floating-point theory on its own
example) and validates compiled NKI kernels on Trainium against a reference
implementation on random FP32 inputs at `rtol = atol = 1e-4`. Axon does not
state how many independent inputs the numerical gate uses; its evaluation
separately reports 100 timing repetitions on random inputs, so the draw
count for correctness may be one, one hundred, or something else. The three
mechanisms also occupy different roles, a candidate-selection gate (Axon), a
post-generation validation check (Prism), and a numerical-stability filter
(Mirage); we measure the shared rule shape, not any one system's pipeline.

Two structural properties of the rule drive the measurements. It is
reference-relative: the target is another floating-point implementation, not
a higher-precision oracle, so the rule cannot observe which side of a
disagreement is more accurate. And it is mixed absolute-relative: `rtol`
scales with each output element, but near zero the fixed `atol` is the only
protection, while accumulated rounding error grows with reduction length
rather than with the element's own magnitude.

## 3. Method

**Valid corpus (registered).** Six (reference, candidate) pairs, each an
algebraic identity over the reals: split reduction, reassociation, scalar
multiplication moved past matmul, LayerNorm two-pass vs. fused one-pass
variance, naive vs. online softmax, matmul K-tiling. Identities hold
symbolically; the float64 check (18 cells within 6e-16 to 3.5e-15) validates
the implementations; the float64 oracle is validated against 50-digit mpmath
(worst drift 8.2e-16) at small shapes and assumed adequate at large ones. A
negative control establishes a nonzero instrument reading.

**Mutant corpus (amended, registered before execution).** Six bug classes,
parameterized into sixteen instances: dropped reduction elements (j in 1 to
32), dropped contraction columns (j in 1 to 32), Bessel-style divisors
(n minus j), eps added to the standard deviation instead of the variance
(eps in 1e-5 to 1e-3), a missing online-softmax rescale, and a dropped
K-tile. Float64 divergences span 4.1e-6 to 8.2e-1, overlapping the valid
corpus's floating-point disagreement range, which is what makes the
discrimination question nondegenerate.

**Instrument.** Oracle: the reference computed in float64 on the same rounded
inputs the test-precision run sees, excluding input quantization. Per cell:
floor, total, differential, the direction ratio total/floor with absolute
errors alongside (ratios are unstable as floor approaches zero), element
exceedance fractions, and the rule verdict. An earlier ratio
(differential/floor) was direction-blind and is retired; the errata records
the misreading it caused.

**Protocol labels.** Analyses are marked registered (in the original claim),
amended (registered by dated amendment before execution: the mutant arm, the
tolerance frontier, the K extension, the decomposition, the library
reference), or exploratory (single-draw sweeps that motivated later
registered runs). Predictions precede every run in the committed log.

## 4. Findings

**F1 (amended). Rejection of the valid tiled matmul reaches the contraction
widths of deployed models, and the mechanism is near-zero elements under a
fixed absolute floor.** Under the literal elementwise rule at M=N=64, unit
scale, 100 draws per point: 0/100 rejections at K=2048 [0-4%], 48/100 at
K=4096 [38-58%], 100/100 at K=11008 [96-100%]. K=4096 and 11008 are the
projection and MLP contraction widths of Llama-2-7B. In failing draws the
violation concentrates on elements far below the tensor median (750x in the
diagnosed case), where the relative allowance vanishes; tensor-scale
relative disagreement stays near 1e-6 throughout, so no tensor-scale
relative statistic predicts the verdict. The tensor-level rate is the
per-element exceedance aggregated by the rule's any-element quantifier:
measured per-element exceedance 3.42e-05 at K=512 and activation scale
predicts a 13.1% tensor rate under independence, inside the observed
[9-22%]. A correction from this revision: an earlier version reported
"fails at K=2048" by comparing the maximum absolute difference against
`atol` alone, ignoring the `rtol` term, the same quantity confusion this
paper criticizes; the corrected measurement moves the onset to K=4096 and
is recorded in the errata.

**F2 (amended). The tolerance frontier contains no separator, and the
optimum is the published constant.** Over all 64 (atol, rtol) grid points
from 1e-8 to 1e-1, with 50 draws per program: no point achieves zero
valid-rewrite rejection and zero mutant miss. The minimum total error is at
exactly (1e-4, 1e-4), Axon's published constant, with 0% valid rejection at
unit scale and 6.2% mutant miss, which is precisely the eps-1e-5 bug evading
all 50 of its draws (its worst violation profile, 6.8e-5, sits below every
threshold that admits the valid corpus). Gross mutants are caught on every
draw at every reasonable point. Two readings, both supported: the published
constant is not miscalibrated, it is the optimum available to its rule
family on this corpus; and the family is structurally unable to separate the
classes, because a real bug's disagreement signature lies inside the
disagreement range of valid rewrites. The bug's severity is input-dependent
(it grows as row variance approaches eps), so this is a statement at the
measured input statistics, and an oracle-referenced check with a
condition-aware budget could see it where the sampled rule cannot; whether a
principled budget at these statistics actually resolves 4.1e-6 is not
demonstrated here.

**F3. The verdict and the direction of the accuracy comparison depend on
the reference implementation.** Against three references for the same
online-softmax candidate at fp16 (absolute oracle errors in parentheses):
sequential reference, total/floor = 0.07 (6.0e-4 vs 8.9e-3), rule fails the
candidate; strided-tree reference, 0.62 (6.0e-4 vs 9.7e-4), rule passes;
`torch.sum` reference, 1.64 (6.0e-4 vs 3.7e-4), rule passes. At bf16 the
ratios are 0.02, 1.92, and 2.14. The same candidate is 15x closer, 1.6x
closer, or 1.6x farther from the oracle depending on an implementation
choice no paper specifies, and the verdict flips with it. `torch.sum`'s
floor beating our explicit tree is consistent with wider internal
accumulation, exactly the undocumented reference property FPRev measures in
real libraries, where tensor-core paths are (8+1)-term multiway trees on
A100. Reference-relative validation also penalizes differing rather than
being wrong: the cells where our two gate definitions disagree are the cells
where the candidate is closer to the oracle than its reference.

**F4. Detection is a rate, and the papers do not permit computing it.** The
valid K=512 tiling trips the rule on 14/100 draws [9-22%] at the measured
activation scale (sigma 2.67) and 0/100 at unit scale (upper bound 4%). Our
own exploratory single-draw sweep missed it; the registered repeated-draw
run caught it. Axon specifies random-input validation but not the draw count
for the numerical gate (its evaluation separately reports 100 timing
repetitions on random inputs); Prism states random equivalence testing
without count or tolerance; Mirage states filtering without threshold or
protocol. These absences were verified against the papers with a
positive-control search method after an earlier tooling failure produced a
false absence claim.

**F5. An FP32-class fixed tolerance does not transfer to reduced
precision.** An untransformed matmul deviates from exact arithmetic on its
own inputs by 3.95e-4 at fp16 and 3.10e-3 at bf16, at or above the 1e-4
scale, and all 36 reduced-precision cells fail a 1e-4-class rule. The
direction ratio shows the failures are dominated by reference and precision
error rather than by the rewrites: against the sequential reference the
reordering candidates are 8x to 50x closer to the oracle than the reference
they fail against. Axon scopes its constant to FP32 explicitly; Prism's
benchmarks are half-precision with no stated threshold; so this finding is
a constraint on any future stated rule, not evidence that a system's actual
criterion fails.

**F6. Real activations are benign in this model.** Every real-activation
cell passes at fp32 with at least 10x headroom, including post-LayerNorm row
sums at condition number 368,927, 400x beyond unit-gaussian reach
(condition rose 400x, error rose 11x). The fused-variance hazard does not
materialize because the captured residual stream is empirically near
zero-mean per row (cancellation statistic at most 0.016), an observation
about this 6-layer pre-norm model, not an architectural guarantee. The
evidence base is one small character-level model and one batch; broader
pretrained-model coverage is the highest-value extension.

**F7. Inputs that force valid-corpus failures are far outside the observed
activation distribution.** Constructed cancellation inputs trip the rule on
every draw, but their median row condition number is 4.5e6 against 42.8 for
real rows (10.4 sigma on a log scale), and the elementwise violations are
near 2x tolerance. Online softmax is immune under every strategy measured;
it is also the transformation outside Mirage's Lax fragment and rejected by
Axon's uninterpreted-function handling. Exceptional regimes (NaN, Inf,
subnormals, overflow) are not covered by our gaussian and activation inputs
and remain open.

## 5. Threats to validity

**Backend and reimplementation (F1-F5).** No emitted kernel from any system
is executed; PyTorch reimplementations on Ampere/M3 stand in, while FPRev
and FTTN document that real accumulation orders and extra-precision
behavior vary across libraries and accelerators. The frontier and K results
characterize the rule on our backend; magnitudes on Trainium or cuBLAS
paths may differ.

**Mutant severity is input-dependent (F2).** Detection rates are statements
at the stated input statistics; the evading bug grows as variance
approaches eps.

**Reference coverage (F3).** Three references (sequential, strided-tree,
`torch.sum`) bound the effect on one backend; production references are
undocumented multiway trees.

**One model, one batch (F6).** A pretrained-transformer activation corpus
across layers and batches is the single most valuable missing dataset.

**Oracle and ratios.** The float64 oracle is validated at small shapes;
direction ratios are reported with absolute errors because they are
unstable as floor approaches zero. Exploratory single-draw cells exist and
are labeled; headline claims rest on registered repeated-draw runs.

**Selection amplification (not measured).** Inside a search over many
candidates, per-candidate miss probabilities compound; under independence a
6.2% miss over dozens of adaptive candidates approaches certainty. This is
the natural next registered experiment, not a result of this paper.

## 6. Implications

**For Axon.** Its constant is, on our corpus, the optimum of its rule
family (F2), and the family still cannot separate the classes; its rule's
verdict becomes shape-dependent at K=4096 for a system whose proofs are
shape-generic (F1). What the measurements support reporting: the draw
count, the reference implementation and its accumulation order, and an
error budget that scales with accumulation length, of the kind derived for
mixed-precision GEMM by Blanchard et al. and Higham and Mary. An oracle
comparison could preserve accuracy-improving candidates (F3); a fixed
reference-relative rule cannot.

**For Prism.** Half-precision evaluation with an unstated threshold sits
exactly where F5 shows fixed FP32-class constants do not transfer; a
stated, floor-derived tolerance per shape and precision would make its
random testing interpretable.

**For Mirage.** The v3 numerical-stability filter is the right mechanism;
stating its threshold, reference, and draw count would make it the first
fully specified floating-point acceptance rule in this literature.

**Method integrity.** Four of our own results were retracted or corrected
during this study: a mischaracterization of Mirage's testing (stale paper
version), a direction-blind metric interpretation refuted by our own
records, an unsupported single-draw attribution, and, in this revision, a
K=2048 rejection verdict produced by comparing a maximum absolute
difference against `atol` alone, the same quantity confusion this paper
criticizes in tolerance reporting. Every correction moved against our
registered claim. The errata and dated run log are part of the artifact.

## 7. Related work

TASO began validation-adjacent practice in tensor superoptimization, using
random-tensor execution to identify candidate-equivalent graphs before
formally verifying substitutions; Mirage, Prism, Axon, and TensorRight
represent the move to stronger exact or idealized reasoning with a residual
empirical floating-point check. TTrace is the closest adjacent study: in
distributed-training validation it shows fixed `torch.allclose` tolerances
produce false positives, false negatives, or both, and replaces them with
perturbation-derived dynamic tolerances; our subject differs (superoptimizer
acceptance rules, real-equivalent rewrites and injected bugs, reference and
shape sensitivity), and our frontier result complements its diagnosis by
showing the best fixed point on our corpus is already the published constant.
FPRev infers the undocumented accumulation orders of real libraries and
accelerators (multiway trees on tensor cores), and FTTN shows accelerator
numerical features are under-documented across vendors; both make F3's
reference sensitivity a systems fact rather than a constructed example.
LifeJacket and Alive-FP verify LLVM floating-point optimizations under
actual FP semantics and found real incorrect optimizations, precedent for
evaluating both valid and invalid transformations; Minotaur brings formal
FP reasoning to SIMD superoptimization. NNSmith treats input generation as
central to exposing compiler bugs, as our scale and distribution findings
do for validation. Herbie rewrites for accuracy, the direction sensitivity
reference-relative rules lack. FPBench standardizes accuracy measures for
FP tools, the measurement-definition problem our oracle and metric choices
address. Rounding-error analyses (Goldberg; Higham; SATIRE; Blanchard et
al.; Higham and Mary) derive the accumulation-dependent budgets a specified
rule could use in place of a constant. Mytkowicz et al. showed performance
evaluation produces wrong conclusions from unexamined setup; this paper
applies the same scrutiny to correctness validation.

## 8. Conclusion

Measured on a representative rule: rejection of a valid rewrite begins
within the contraction widths of deployed 7B-class models; the best fixed
tolerance on the corpus is the one already published, and it still passes a
real bug on every draw; the verdict and the direction of the accuracy
comparison move with an unspecified reference implementation; and the
detection rate depends on scale and draw count that no paper reports. Real
activations pass everywhere we measured, and no evidence indicates any
system has shipped an incorrect kernel. The constructive reading is a
reporting checklist: reference implementation and accumulation order, draw
count, precision scope, and an accumulation-aware error budget. With those
stated, this stage becomes as interpretable as the verifiers above it; the
measurements here are what their absence currently costs.

## References

[1] Wu et al. Mirage: A Multi-Level Superoptimizer for Tensor Programs.
OSDI 2025. arXiv:2405.05751 (v3).
[2] Wu, Jiang, Padon, Jia. Prism: Symbolic Superoptimization of Tensor
Programs. arXiv:2604.15272.
[3] Kothari, Zhu, Kroening, Sung. Axon: A Synthesizing Superoptimizer for
Tensor Programs. arXiv:2606.26344.
[4] TensorRight: Automated Verification of Tensor Graph Rewrites. POPL 2025.
[5] Jia, Padon, Thomas, Warszawski, Zaharia, Aiken. TASO: Optimizing Deep
Learning Computation with Automatic Generation of Graph Substitutions.
SOSP 2019.
[6] Jiang et al. TTrace: Lightweight Error Checking and Diagnosis for
Distributed Training. arXiv:2506.09280.
[7] Xie et al. Revealing Floating-Point Accumulation Orders in
Software/Hardware Implementations (FPRev). USENIX ATC 2025.
arXiv:2411.00442.
[8] FTTN: Feature-Targeted Testing for Numerical Properties of NVIDIA and
AMD Matrix Accelerators. arXiv:2403.00232.
[9] Nandi, Willsey, et al. Rewrite Rule Inference Using Equality
Saturation. OOPSLA 2021.
[10] Panchekha, Sanchez-Stern, Wilcox, Tatlock. Automatically Improving
Accuracy for Floating Point Expressions. PLDI 2015.
[11] Menendez, Nagarakatte, Martin. Alive-FP: Automated Verification of
Floating Point Based Peephole Optimizations in LLVM. SAS 2016.
[12] Notzli, Brown. LifeJacket: Verifying Precise Floating-Point
Optimizations in LLVM. SOAP 2016.
[13] Liu, Mada, Regehr. Minotaur: A SIMD-Oriented Synthesizing
Superoptimizer. OOPSLA 2024.
[14] Liu et al. NNSmith: Generating Diverse and Valid Test Cases for Deep
Learning Compilers. ASPLOS 2023.
[15] Damouche, Martel, Panchekha, Qiu, Sanchez-Stern, Tatlock. Toward a
Standard Benchmark Format and Suite for Floating-Point Analysis. NSV 2016.
[16] Blanchard, Higham, Lopez, Mary, Pranesh. Mixed Precision Block Fused
Multiply-Add: Error Analysis and Application to GPU Tensor Cores. SIAM J.
Sci. Comput. 2020.
[17] Higham, Mary. A New Approach to Probabilistic Rounding Error
Analysis. SIAM J. Sci. Comput. 2019.
[18] Das et al. Scalable yet Rigorous Floating-Point Error Analysis
(SATIRE). SC 2020.
[19] Goldberg. What Every Computer Scientist Should Know About
Floating-Point Arithmetic. ACM Computing Surveys 1991.
[20] Higham. Accuracy and Stability of Numerical Algorithms. SIAM 2002.
[21] Mytkowicz, Diwan, Hauswirth, Sweeney. Producing Wrong Data Without
Doing Anything Obviously Wrong! ASPLOS 2009.
[22] Touvron et al. Llama 2: Open Foundation and Fine-Tuned Chat Models.
2023.

## Artifact

All measurements reproduce from this repository; each experiment runs from
one script, raw per-cell records are committed, and the pre-registration,
amendments, run log, and errata are public. AI tools (Claude) assisted with
code, measurements, and documentation; the developer directed, reviewed,
and verified all work, and no measured number comes from a model.
