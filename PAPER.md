# Characterizing Floating-Point Validation Rules for Tensor Superoptimization

Ayaan Faisal. Rev. 6 of 2026-08-16: structure and pacing pass against
venue conventions; content and numbers unchanged from rev. 5a.4.
Artifact: this repository.

## Abstract

Tensor superoptimizers verify candidate equivalence over exact or
idealized arithmetic, then accept the compiled floating-point kernel with
an empirical check: compare against a reference implementation on random
inputs, elementwise, within `|g_i - b_i| <= atol + rtol*|b_i|`. Only Axon
states its constants (`rtol = atol = 1e-4` on FP32); no system fully
specifies the reference, draw protocol, and precision scope together. We
measure what the unstated coordinates cost, on six rewrites that are
exact identities over the reals and eighteen injected bugs. The rule's
verdict moves with every coordinate we vary. In float32 PyTorch on an
Apple M3 (CPU), the valid tiled matmul is rejected on 48/100 draws at
K=4096 and 100/100 at K=11008, the contraction widths of Llama-2-7B, and
the onset moves fourfold with input scale. An exact envelope analysis
shows no (atol, rtol) pair separates real-equivalent rewrites from
injected bugs under the two extreme cross-draw semantics,
all-draws-must-pass and any-draw-may-pass, at any rtol >= 0; a real
eps-placement bug evades every recorded draw at the published constant. The choice of reference implementation flips
both the verdict and the direction of the accuracy comparison.
Real-activation FP32 cells pass everywhere measured. These results do
not demonstrate failures in the measured systems; they identify the
protocol coordinates that must be stated for empirical numerical
validation to be reproducible and interpretable.

## 1. Introduction

A kernel autotuner tunes a fixed computation against a template that pins
semantics; one reference validates every candidate. A superoptimizer
searches program structure across millions of candidates, and the search
selects for whatever passes the acceptance rule. Mirage, Prism, and Axon
verify candidate equivalence over exact or idealized arithmetic (Section 2)
and close the distance to the compiled floating-point kernel with a sampled
comparison against a reference implementation. Axon and Mirage state this
limitation; Prism reports random equivalence testing without a stated
criterion and does not otherwise discuss floating-point behavior. None
fully specifies the reference, threshold, draw protocol, and precision
scope together, so a reader cannot compute what the check accepts or
rejects.

The organizing observation of this study is that a floating-point
validation verdict is not determined by a tolerance pair. It is determined
by a protocol with several independent coordinates: the reference
implementation, the precision, the input distribution and scale, the
per-element comparison rule, the within-tensor aggregation (one violating
element rejects the tensor), the number of independent draws, and the
cross-draw aggregation rule. We vary the reference, precision, input
distribution, scale, tolerance constants, draw count, and cross-draw
semantics, and the verdict moves with each; the per-element functional
form and the within-tensor aggregation are held fixed, their roles
isolated analytically (F1). Each finding sits on one or two coordinates:
F1 (contraction length, output multiplicity, input scale), F2 (tolerance
constants, cross-draw semantics), F3 (reference implementation), F4 (draw
count, cross-draw aggregation), F5 (precision), F6-F7 (input distribution
beyond scale).

Two failure directions are measured on the same rule at the same
tolerance: rejection of rewrites that are algebraic identities over the
reals, and acceptance of injected bugs. The question is not whether any
system has shipped a wrong kernel; we find no evidence of that.

Scope, stated first: we characterize the rule, not the systems' outputs.
The rewrites are PyTorch reimplementations of transformations these
systems accept, with per-pair provenance recorded in the artifact (the
online-softmax rescale form is unclassified by all three source papers
and is in the corpus for that reason, F7). They run on an Ampere GPU and
an Apple M3 with TF32 disabled. No kernel emitted by Mirage, Prism, or
Axon is executed; Axon's deployment target (Trainium/NKI) differs from
our backend in accumulation and lowering, and FPRev and FTTN document
that accumulation order and extra-precision behavior vary across real
libraries and accelerators and are largely unspecified. Findings transfer
to the extent the rule, not the backend, drives them; Section 5 marks
which findings that caveat touches.

A note on labels: "valid" below means equivalent over the reals, and
"bug" means inequivalent over the reals. Real-equivalence is not
numerical acceptability: a real identity can be numerically unstable, and
a semantic bug can be numerically negligible on a given input domain. Our
corpus contains both cases, and Section 4 keeps the two axes separate.

Contributions:

1. A two-class corpus: six real-equivalent rewrites with recorded
   per-pair provenance (identities established symbolically;
   implementations validated to 3.5e-15 against a 50-digit-validated
   float64 oracle), and eighteen injected bugs across six classes whose
   float64 divergences span 4.1e-6 to 8.2e-1.
2. An instrument reporting, per cell, the reference's oracle error
   (floor), the candidate's oracle error (total), their disagreement
   (differential, what the rule tests), the direction ratio total/floor,
   and the rule's verdict under three references.
3. An exact corpus-level separability analysis under the extreme
   Boolean cross-draw semantics, characterized over the whole domain
   rtol >= 0 (Appendix B), in addition to a 64-point tolerance grid.
4. An audit trail: pre-registration with dated amendments, predictions
   committed before every run, and a complete errata of our own
   corrected results.

## 2. Background

Mirage restricts search to its Lax fragment and probabilistically
verifies candidate equivalence by evaluation over random finite-field
values. The theorem bounds the probability of accepting a non-equivalent
program; the implementation runs a single random test with primes stated
in the paper as p=227 and q=113, chosen so their product fits in 16-bit
integers, which the authors acknowledge (the released artifact's
constants differ; Appendix A). Separately, Mirage v3 (section 5.2)
states that floating-point tests filter muGraphs with "significant
numerical errors"; threshold, reference, and draw protocol are unstated.

Prism reasons over roughly 70 hand-written axioms in an e-graph. The
axioms are intended to be sound, a formal proof is stated to be beyond
scope, and every generated kernel is subjected to random equivalence
testing with no stated tolerance, in an evaluation that is entirely
half-precision; the paper does not otherwise discuss floating-point
behavior.

Axon proves equivalence over the reals with Z3 (1650x faster than Z3's
floating-point theory on its own example) and validates compiled NKI
kernels on Trainium against a reference implementation on random FP32
inputs at `rtol = atol = 1e-4`. Axon does not state how many independent
inputs the numerical gate uses; its evaluation separately reports 100
timing repetitions on random inputs, so the correctness draw count may
be one, one hundred, or something else. The three mechanisms occupy
different roles, a candidate-selection gate (Axon), a post-generation
validation check (Prism), and a numerical-stability filter (Mirage); we
measure the shared rule shape, not any one system's pipeline.

Two structural properties of the rule drive the measurements. It is
reference-relative: the target is another floating-point implementation,
not a higher-precision oracle, so the rule cannot observe which side of a
disagreement is more accurate. And it is mixed absolute-relative: `rtol`
scales with each output element, but near zero the fixed `atol` is the
only protection, while accumulated rounding error grows with reduction
length rather than with the element's own magnitude.

## 3. Method

**Valid corpus (registered).** Six (reference, candidate) pairs, each an
algebraic identity over the reals: split reduction, reassociation, scalar
multiplication moved past matmul, LayerNorm two-pass vs. fused one-pass
variance, naive vs. online softmax, matmul K-tiling. Identities hold
symbolically; the float64 check (18 cells within 6e-16 to 3.5e-15)
validates the implementations; the float64 oracle is validated against
50-digit mpmath (worst drift 8.2e-16) at small shapes and assumed
adequate at large ones. A negative control establishes a nonzero
instrument reading.

**Mutant corpus (amended, registered before execution).** Six bug
classes, eighteen instances. Four classes parameterize into the sixteen
frontier instances: dropped reduction elements (j in {1,2,4,8,16,32}),
dropped contraction columns (j in {1,2,8,32}), Bessel-style divisors
(n-j, j in {1,2,8}), and eps added to the standard deviation instead of
the variance (eps in 1e-5 to 1e-3). Two gross single-instance classes
run in the detection arm only. The implemented column grid is sparser
than registered; the deviation is recorded in the errata. Float64
divergences span 4.1e-6 to 8.2e-1, overlapping the valid corpus's
disagreement range, which makes the discrimination question
nondegenerate.

**Instrument.** Oracle: the reference computed in float64 on the same
rounded inputs the test-precision run sees, excluding input quantization.
Per cell: floor, total, differential, the direction ratio total/floor
with absolute errors alongside (ratios are unstable as floor approaches
zero), element exceedance fractions, and the rule verdict.

**Separability analysis (E5, E8).** F2's exact statements run on one
fixed recorded dataset: the six valid pairs at their mlp shapes plus the
K=2048 tiling cell (seven program cells) and the sixteen frontier
instances, fifty unit-scale draws per program on shared seeds. For each
recorded draw, the minimal accepting `atol` at each `rtol` is an
explicit piecewise-linear envelope of the recorded per-element
differences; class-level envelopes under each cross-draw semantics then
decide, exactly, whether any (atol >= 0, rtol) accepts every valid
program while rejecting every mutant. The construction, the four
class-level envelopes, and the certificates that extend the sampled
rtol grid to the whole domain rtol >= 0 are in Appendix B.

**Evidence classes.** Claims below signal their evidence type by verb.
Recorded facts about the measured corpus use "observed," "on all recorded
draws," "on the measured corpus," and may be stated exactly. Statistical
estimates carry rates with Wilson 95% intervals and "estimated" or
"consistent with." Generalizations beyond the backend, model, or corpus
use "on our backend," "suggests," or "may transfer," and carry scope
qualifiers.

**Protocol labels.** Analyses are marked registered (in the original
claim), amended (registered by dated amendment before execution: the
mutant arm, the tolerance frontier, the K extension, the decomposition,
the library reference, the C1-C3 pass, the separability computations E5,
E6, and E8, the grid completion E7, and the verification and diagnostics
V1, D1, and D2), or exploratory (single-draw sweeps that motivated later
registered runs). Predictions precede every run in the committed log.

## 4. Findings

**F1. Rejection of the valid tiled matmul reaches the contraction widths
of deployed models.** In float32 PyTorch on the Apple M3 (CPU), under
the literal elementwise rule at M=N=64 with 100 draws per cell:

| K | unit scale | sigma 2.67 |
|---|---|---|
| 512 | 0/100 [0-4%] | 11/100 [6-19%] |
| 1024 | 0/100 [0-4%] | 69/100 [59-77%] |
| 2048 | 0/100 [0-4%] | 100/100 [96-100%] |
| 4096 | 48/100 [38-58%] | 100/100 [96-100%] |
| 11008 | 100/100 [96-100%] | 100/100 [96-100%] |

K=4096 and 11008 are the projection and MLP contraction widths of
Llama-2-7B. The sigma column uses scale-matched gaussian inputs (sigma
2.67, the measured activation scale); the onset falls roughly fourfold
from its unit-scale position, to near K=1024. Rejection is
non-decreasing along both axes.

The mechanism is backend-agnostic in formulation: near-zero output
elements lose the relative allowance, accumulated rounding disagreement
grows with reduction length, and the any-element aggregation turns rare
element violations into tensor rejection. In diagnosed failing draws the
violation is one element in 4096, its magnitude 750x below the tensor
median. Tensor-scale relative disagreement (maximum absolute difference
over maximum oracle magnitude) stays near 1e-6 throughout, so the
tensor-scale statistic we report does not track the verdict. From the
measured per-element exceedance of 3.42e-05 at K=512, a simple
independence approximation over the rule's any-element quantifier gives
13.1%, consistent with the observed [9-22%]; dependence among output
elements is not established either way.

Two same-kernel observations tie the onset to this mechanism rather than
to a kernel change. At fixed K=512, input scale alone drives rejection
from 0 to 74 per 100 (sigma 0.5 to 4.0: 0, 0, 1, 16, 74), where no
K-correlated kernel change can act. And the recorded maximum absolute
disagreement grows smoothly with K across all five widths (median
6.1e-05 at K=512 to 1.3e-03 at K=11008, no isolated jump), with its
argmax at large, rtol-protected elements (D1): the binding violations
are the near-zero elements.

**F2. No fixed tolerance separates the recorded corpus under either
extreme cross-draw semantics; only a class-conditioned draw assignment
would.** In plain terms: however the constants are chosen, a validator
applying either extreme cross-draw rule cannot accept every valid
rewrite while catching all sixteen injected bugs on the recorded data.
The recorded errors themselves are separable, but only by an assignment
that already knows which programs are the bugs.

*Exact result.* On the separability corpus, with the envelopes of
Appendix B: under all-draws-must-pass (a candidate is accepted iff every
draw passes), no (atol >= 0, rtol >= 0) separates, with supremum
separation gap -4.0e-06. Under any-draw-may-pass (accepted if any draw
passes), the same, at -4.2e-06. The pessimal pairing, valid programs
held to every draw while mutants must fail every draw, reads -3.5e-05.
All 400 inter-sample intervals in each analysis carry the
single-witness certificate, and the mutant-side envelopes are
nonpositive at rtol = 100 (-5.5e-05 and -2.4e+02), which extends all
three results to the whole domain rtol >= 0.

*The pairing that separates.* The fourth pairing is a class-conditioned
information bound, not an operational validator: it judges each valid
program on its most favorable recorded draw and each mutant on its least
favorable one, an assignment that requires knowing the class in advance.
Under it, separators exist on a contiguous band rtol in [7.8e-03, 7.9]
with maximum margin +9.5e-06 (for example, atol = 1e-06 at rtol =
0.056). So the pointwise recorded errors are not themselves inseparable;
the nonseparability emerges under the two uniform extreme semantics.
This falsified our registered prediction that all four pairings would
fail to separate, and it makes F4 concrete: which draws a program is
judged on can decide separability outright. The pairing of per-class
semantics, which no source paper states, swings the margin from
-3.5e-05 across zero to +9.5e-06. Intermediate Boolean rules (k-of-n
acceptance) evaluate the k-th order statistic of the same per-draw
envelopes; they are coherent, unanalyzed here, and remain part of the
specification space.

*On the 64-point grid.* The minimum observed total error (an
equal-weight sum of the valid-rejection and mutant-miss fractions; the
weighting is an analytical choice, and the two components are reported
separately in the artifact) is a three-point plateau, (1e-4, 1e-4),
(1e-4, 1e-3), (1e-4, 1e-2), at 0% valid rejection (unit scale) plus a
6.25% mutant-instance miss fraction: 1 of the 16 frontier instances,
equivalently 50 of 800 mutant draws, all from the eps-1e-5 bug. This
fraction describes the constructed corpus, not any candidate population.
Axon's published constant lies on that plateau, and atol is the binding
coordinate, the near-zero-element mechanism restated.

*The evading bug.* The eps bug's severity is input-dependent (it grows
as row variance approaches eps), so detection statements hold at the
measured input statistics. Under scale-matched gaussian inputs at sigma
2.67 the picture is unchanged: the five other detection-arm mutants at
100/100 detection, the eps bug evading all 100 draws, its divergence
shrinking to 2.46e-6 as the registered mechanism predicts. An
oracle-referenced check with a condition-aware budget could see it where
the sampled rule cannot, though whether a principled budget resolves
4.1e-6 at these statistics is not demonstrated.

**F3. Reference choice changes the verdict and the direction of the
accuracy comparison.** Observed at fp16 on the registered cell at
(512, 1024), for the same online-softmax candidate (oracle error
6.0e-4 on the registered draw):

| reference | floor (draw) | vs. candidate (draw) | vs. candidate (100-draw mean) | rule verdict (100 draws) |
|---|---|---|---|---|
| sequential | 8.9e-3 | 15x closer | 11x closer | fails 100/100 [96-100%] |
| strided tree | 9.7e-4 | 1.6x closer | 1.7x farther | passes 98/100 [93-99%] |
| torch.sum | 3.7e-4 | 1.6x farther | 2.3x farther | passes 99/100 [95-100%] |

"Closer" and "farther" give the candidate's accuracy against the float64
oracle relative to the reference it is compared with. The verdict flips
with the reference, and the direction against the tree reference is
itself draw-dependent: closer on the registered draw, farther in the
100-draw mean. The mean floors order seq > tree > torch.sum, but the
per-draw ordering holds on only 80/100 (fp16) and 85/100 (bf16) draws,
so the ranking of references is also draw-dependent. At bf16 the
single-draw ratios are 1.92, 2.14, and 0.02, and the rule fails all
three on 100/100.

`torch.sum`'s smaller floor is consistent with a different internal
accumulation strategy, which we do not characterize here; FPRev
independently demonstrates that such accumulation orders are
implementation properties that are often undocumented. Reference-relative
validation observes disagreement but not its direction with respect to
oracle accuracy: the cells where our two gate definitions disagree are
exactly the cells where the candidate is closer to the oracle than its
reference.

**F4. Detection is a rate; the source papers do not fully specify the
draw count and cross-draw aggregation.** Observed: the valid K=512
tiling trips the rule on 14/100 draws [9-22%] under scale-matched
gaussian inputs (sigma 2.67) and 0/100 at unit scale (upper bound 4%);
an independent registered replication reads 16/100 [10-24%]. Our own
exploratory single-draw sweep missed the failure; the registered
repeated-draw run caught it.

Two aggregation questions govern what such rates mean. Within a tensor,
Axon's rule is elementwise and unambiguous: one violating element
rejects the tensor. Across random inputs, everything is unstated: the
number of draws, whether a candidate must pass all of them, and how
errors aggregate. As hypothetical consequences under IID draws and an
all-draws-must-pass rule, not measurements of any system: a valid
rewrite with a 14% per-draw rejection rate survives 20 draws with
probability near 0.86^20, about 4.9%; and a 0/100 observation is
consistent both with a true catch probability of zero (more draws never
help) and with a rate at its 3.7% upper bound, where 100 draws would
catch the bug with probability near 97.7%. A reported draw count without
its cross-draw aggregation rule is still not enough to reproduce a
validator.

**F5. An FP32-class fixed tolerance does not transfer to reduced
precision.** An untransformed matmul deviates from exact arithmetic on
its own inputs (scale-relative: maximum absolute difference over maximum
oracle magnitude) by 3.91e-4 at fp16 and 3.45e-3 at bf16 at the mlp
shape, and all 36 reduced-precision cells fail a 1e-4-class rule.

The direction ratio shows these failures are dominated by reference and
precision error rather than by the rewrites: against their sequential
references the three accumulation-reordering pairs (split reduction,
reassociation, online softmax) are 3x to 73x closer to the oracle than
the reference they fail against (8x to 62x at the mlp shapes). Axon
scopes its constant to FP32 explicitly; Prism's benchmarks are
half-precision with no stated threshold; so this characterizes what
would happen if an FP32-class threshold were transferred, not what any
system's actual criterion does.

**F6. Real activations are benign in this model.** Observed: every
real-activation FP32 cell passes with at least 10x headroom, including
post-LayerNorm row sums at condition number 368,927, 400x beyond
unit-gaussian reach (condition rose 400x, error rose 11x). The
fused-variance hazard does not materialize because the captured residual
stream is empirically near zero-mean in the bulk of its rows (per-row
cancellation, |mean| over mean|x|: 99th percentile at most 0.012, worst
row 0.16, across the two residual captures), an observation about this
6-layer pre-norm model, not an architectural guarantee.

Matching scale is not matching distribution: the scale-matched gaussian
draws that reject in F1 share only variance with these activations, and
the real tensors pass. The evidence base is one small character-level
model and one batch; broader pretrained-model coverage is the principal
remaining external-validity limitation.

**F7. Inputs that force valid-corpus failures are far outside the
observed activation distribution.** Constructed cancellation inputs trip
the rule on every draw, but their median row condition number is 4.5e6
against 42.8 for real rows (10.4 sigma on a log scale), and the
elementwise violations are near 2x tolerance.

Online softmax is immune under every strategy measured; it is also the
pair the verifiers handle worst, by our inference from their mechanisms:
its running max is not a Lax operator, so Mirage partitions around it,
and Axon's uninterpreted exp blocks the rescale identity; neither paper
classifies the transformation itself. Exceptional regimes (NaN, Inf,
subnormals, overflow) are not covered by our gaussian and activation
inputs and remain open.

## 5. Threats to validity

**Backend and reimplementation (F1-F5).** No emitted kernel from any
system is executed; PyTorch reimplementations stand in (headline K and
scale grids in float32 on the Apple M3 CPU; corpus equivalence and
hardware characterization verified on an Ampere A10), and measured
onsets (F1) and margins (F2) are backend-conditioned. FPRev and FTTN
document that real accumulation orders and extra-precision behavior vary
across libraries and accelerators.

**Mutant severity is input-dependent (F2).** Detection rates are
statements at the stated input statistics.

**Reference coverage (F3).** Three references demonstrate the effect on
one backend; production references may use blocked or multiway
accumulation orders that differ from the three measured here and are
often undocumented.

**One model, one batch (F6).** A pretrained-transformer activation
corpus across layers and batches is the single most valuable missing
dataset.

**Oracle and ratios.** The float64 oracle is validated at small shapes,
and the F3 headline cell is spot-checked against a 50-digit oracle on
three draws, worst relative deviation 1.7e-13 (D2); direction ratios are
reported with absolute errors. Exploratory single-draw cells exist and
are labeled; headline claims rest on registered repeated-draw runs.

**Selection amplification (not measured).** A validator inside a
superoptimizer is queried repeatedly by an adaptive candidate-generation
process, so rare misses may be amplified by selection. Quantifying that
requires a candidate distribution or an actual search loop; it is the
natural next registered experiment, and the mutant-instance miss
fraction above is not an estimate of any candidate-population miss
probability.

## 6. Implications

**For Axon.** Its constant is, on our corpus, on the minimum-error
plateau of the tested 64-point grid (F2), and no constant in the family
separates the recorded corpus under the two extreme cross-draw semantics
analyzed; its rule's verdict becomes shape-dependent within deployed
contraction widths on our backend, for a system whose proofs are
shape-generic (F1). What the measurements support reporting: the draw
count, the reference implementation and its accumulation order, and a
justified error budget for the operator, shape, and precision, of the
kind derived for mixed-precision GEMM by Blanchard et al. and Higham and
Mary. An oracle comparison could preserve accuracy-improving candidates
(F3); a reference-relative rule cannot determine whether a disagreement
improves or worsens oracle-relative accuracy.

**For Prism.** Half-precision evaluation with an unstated threshold sits
where F5 shows fixed FP32-class constants do not transfer; a stated and
justified tolerance for its half-precision regime would make its random
testing interpretable.

**For Mirage.** Mirage v3 already places a numerical-stability filter
after its algebraic verification; stating the filter's threshold,
reference, and draw protocol would make it reproducible and
interpretable from the paper specification. Appendix A records that the
filter is not locatable in the released artifact at the audited commits.

**Method integrity.** Our own results required correction repeatedly
during this study, and every correction moved against our own claims;
the errata records each with its cause. The most instructive: a K=2048
rejection verdict produced by comparing a maximum absolute difference
against `atol` alone, the same quantity confusion this paper studies.
The errata and dated run log are part of the artifact.

## 7. Related work

TASO began validation-adjacent practice in tensor superoptimization,
using random-tensor execution to identify candidate-equivalent graphs
before formally verifying substitutions. TensorRight develops stronger
idealized rewrite verification; Mirage, Prism, and Axon combine stronger
semantic reasoning with residual empirical validation of generated
implementations. TTrace is one close adjacent study: in
distributed-training validation it shows fixed `torch.allclose`
tolerances produce false positives, false negatives, or both, and
replaces them with perturbation-derived dynamic tolerances; our subject
differs (superoptimizer acceptance rules, real-equivalent rewrites and
injected bugs, reference and shape sensitivity), and our envelope
analysis complements its diagnosis by showing no fixed pair separates
our corpus at all. A concurrent line of 2026 measurements by one author
(Sarkar) reaches adjacent conclusions on neighboring workloads:
fixed-shape small-sample allclose oracles pass seeded-bug GPU kernels,
per-operator and per-dtype tolerance calibration trades detection
against false positives, and input-generation strategy materially
changes which kernel bugs surface; none studies the superoptimizer
acceptance stage or reference sensitivity.

FPRev infers the undocumented accumulation orders of real libraries and
accelerators, and FTTN shows accelerator numerical features are
under-documented across vendors; both make F3's reference sensitivity a
systems fact. LifeJacket and Alive-FP verify LLVM floating-point
optimizations under actual FP semantics and found real incorrect
optimizations; Minotaur brings formal FP reasoning to SIMD
superoptimization. NNSmith treats input generation as central to
exposing compiler bugs, as our scale and distribution findings do for
validation.

Herbie rewrites for accuracy, the direction sensitivity
reference-relative rules lack. FPBench standardizes accuracy measures
for FP tools, the measurement-definition problem our oracle and metric
choices address. Rounding-error analyses (Goldberg; Higham; SATIRE;
Blanchard et al.; Higham and Mary) derive the accumulation-dependent
budgets a specified rule could draw on. Mytkowicz et al. showed
performance evaluation produces wrong conclusions from unexamined setup;
this paper applies the same scrutiny to correctness validation.

## 8. Conclusion

A numerical validation result is interpretable only relative to a
specified validation protocol, and the measurements here show that
individual protocol coordinates materially change the verdict:
contraction length and output multiplicity (F1), tolerance constants and
the cross-draw semantics (F2), reference implementation (F3), draw count
(F4), precision (F5), and input scale and distribution (F1, F6, F7). On
the recorded corpus, no nonnegative mixed absolute-relative tolerance
separates real-equivalent rewrites from injected bugs under the two
extreme cross-draw semantics analyzed, at any rtol >= 0; only a
class-conditioned draw assignment, which no uniform rule realizes, would
separate it. Real-activation FP32 cells passed everywhere measured, and
no evidence indicates any system has shipped an incorrect kernel.

The constructive conclusion is a reporting standard, not a replacement
validator. A floating-point acceptance result should specify at least:
(1) the reference implementation and its relevant accumulation behavior;
(2) the precision; (3) the input distribution and scale; (4) the
per-element comparison function and constants; (5) the within-tensor
aggregation rule; (6) the number of independent draws; (7) the
cross-draw aggregation rule; and (8) a justification for the numerical
error budget in the operator, shape, and precision regime. This paper
does not prescribe one universal validator; it identifies the
information necessary to make a validator's result reproducible and
interpretable, and measures what remains undetermined when that
information is absent.

## References

[1] Wu et al. Mirage: A Multi-Level Superoptimizer for Tensor Programs.
OSDI 2025. arXiv:2405.05751 (v3).
[2] Wu, Jiang, Padon, Jia. Prism: Symbolic Superoptimization of Tensor
Programs. arXiv:2604.15272.
[3] Kothari, Zhu, Kroening, Sung. Axon: A Synthesizing Superoptimizer for
Tensor Programs. arXiv:2606.26344.
[4] Arora et al. TensorRight: Automated Verification of Tensor Graph
Rewrites. PACMPL 9 (POPL), 2025.
[5] Jia, Padon, Thomas, Warszawski, Zaharia, Aiken. TASO: Optimizing Deep
Learning Computation with Automatic Generation of Graph Substitutions.
SOSP 2019.
[6] Jiang et al. TTrace: Lightweight Error Checking and Diagnosis for
Distributed Training. arXiv:2506.09280.
[7] Xie et al. Revealing Floating-Point Accumulation Orders in
Software/Hardware Implementations (FPRev). USENIX ATC 2025.
arXiv:2411.00442.
[8] Li et al. FTTN: Feature-Targeted Testing for Numerical Properties of
NVIDIA and AMD Matrix Accelerators. arXiv:2403.00232.
[9] Nandi, Willsey, et al. Rewrite Rule Inference Using Equality
Saturation. OOPSLA 2021.
[10] Panchekha, Sanchez-Stern, Wilcox, Tatlock. Automatically Improving
Accuracy for Floating Point Expressions. PLDI 2015.
[11] Menendez, Nagarakatte, Gupta. Alive-FP: Automated Verification of
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
[23] Sarkar. The Correctness Illusion in LLM-Generated GPU Kernels.
arXiv:2606.20128.
[24] Sarkar. Operator-Aware Mixed-Precision Tolerance Calibration for
Tensor Kernels. arXiv:2607.16228.
[25] Sarkar. Test-Input Generation for Tensor Programs: What Actually
Finds Kernel Bugs. arXiv:2606.27396.

## Appendix A: implementation observations on the Mirage artifact

Exploratory; the checklist of Section 8 applied to a released artifact.
Audited: mirage-project/mirage at commit 5c28cc6, the tip of the default
mpk branch (2026-07-28), with the evaluation branch and the older main
branch (ffe38df, 2026-04-17) cross-checked; method, search terms, and
positive controls in audits/mirage-fp-filter.md. The released
superoptimize path is search, exact finite-field fingerprint
verification, transpilation, then performance-only selection over random
inputs; we found no floating-point comparison in the audited acceptance
path at these commits, which leaves the tolerance, precision, and
input-distribution coordinates unexercised there, while the per-element
rule (exact field equality), reference (input-graph fingerprints), draw
count (a single fingerprint evaluation per candidate, consistent with
the paper's acknowledged single test), and aggregation (all outputs must
match) are determined by code. The numerical-stability filter described
in the paper's v3 (section 5.2) is not locatable in the released search
path at these commits.

The closest thing to it is a CI test, `is_closed()` in
tests/python/test_tensor_program.py, which compares hand-constructed
graphs against torch references and counts an element mismatched only
when relative and absolute error both exceed 1e-1. We record it because
a reader will find it: it is not the filter, since it never calls
`superoptimize()`, never reaches the verifier, and rejects no candidate,
and its conjunctive criterion is looser than an `allclose` rule. Other
tolerances (demo scripts, transpiler tests) sit equally far from the
acceptance path. The audit also covered the published camera-ready
(identical sentence, no appendix), the artifact-evaluation instructions
that earned the Results Reproduced badge (performance and search time
only, no numerical check), the group's later papers, the repository
documentation, and the issue and discussion history; the protocol is
stated in none of them. Two recorded talks were not transcribed, so a
threshold named aloud would not have been seen. The finite-field constants are FP_P=167, FP_Q=83 on all three
audited branches, while the paper states p=227, q=113. Theorem 2 of the
paper bounds a single test's acceptance probability by
8dk^4/q + q^(-1/k^2); both terms grow as q falls, so the released q
loosens the stated bound at every fixed (d, k) by a factor that depends
on (d, k). An earlier 1.4x reading treated the bound as proportional to
1/q and is retracted in the errata. These are observations about the
released repository at pinned commits, not about any deployed system.

## Appendix B: envelope construction and certificates

The separability corpus (Method): seven valid program cells and sixteen
mutant instances, fifty unit-scale draws per program on shared seeds.
The two gross detection-arm instances are excluded; adding mutant
instances can only lower the mutant-side envelope, so non-separation
extends a fortiori to supersets.

For a recorded draw with per-element differences d_i = |g_i - b_i| and
reference magnitudes x_i = |b_i|, T(r) = max_i (d_i - r * x_i) is the
unclamped absolute slack the draw requires at relative tolerance r: any
real atol >= T(r) accepts, and since the rule restricts atol >= 0 the
smallest admissible accepting threshold is max(T(r), 0), the clamp the
separator test applies. Each T is a finite maximum of lines through
recorded points, evaluated exactly on the Pareto set of the (x_i, d_i);
it is convex, piecewise linear, and non-increasing.

Four class-level envelopes cover the extreme Boolean per-class
semantics: F_all(r), the maximum over all valid draws (a valid program
is accepted only if every draw passes); F_any(r), the maximum over valid
programs of the minimum over each program's draws (accepted if some draw
passes); G_every(r), the minimum over all mutant draws (a mutant is
caught only if every draw fails); and G_some(r), the minimum over mutant
instances of the maximum over each instance's draws (caught if some draw
fails). A separator exists at r for a pairing (F, G) iff
max(F(r), 0) < G(r).

The computation samples 401 rtol values (zero and 400 log-spaced in
[1e-9, 100]) and certifies each inter-sample interval with a
single-witness bound: a witness draw envelope, or for G_some a witness
instance's max-envelope (itself a maximum of convex functions), is
convex and so lies below the larger of its endpoint values on the
interval, while every F is non-increasing and so lies above its
right-endpoint value; when the witness bound does not exceed max(F, 0)
at the right endpoint, no r in the interval separates. Convexity is used
only for single witness envelopes, never for the min-envelopes G, which
need not be convex. Every envelope is non-increasing, so a nonpositive
mutant-side envelope at r = 100 extends no-separation to all r > 100,
closing the domain rtol >= 0.

## Artifact

All measurements reproduce from this repository; each experiment runs
from one script, raw per-cell records and the activation fixtures are
committed, and the pre-registration, amendments, run log, and errata are
public. One script, tools/check_paper_numbers.py, asserts every number
in this paper against the committed records. AI tools (Claude) assisted
with code, measurements, and documentation; the developer directed,
reviewed, and verified all work, and no measured number comes from a
model.
