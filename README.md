# The floating-point gap in verified tensor superoptimizers

Mirage, Prism, and Axon prove kernel equivalence over exact or idealized
arithmetic, then accept the compiled kernel with an empirical
floating-point check: compare against a reference implementation on
random inputs, elementwise, within `|g_i - b_i| <= atol + rtol*|b_i|`.
Only Axon states its constants. No system states the reference, draw
protocol, and precision scope together.

This repo measures what that check accepts and rejects.

**The paper:** [`PAPER.md`](PAPER.md) (readable) · [`paper.tex`](paper.tex)
(Overleaf-ready single file). As of 08/19, The Manuscript Draft has been added to the repository.

## What it finds

**A validator that draws its own inputs rejects a valid rewrite at
deployed widths.** A K-tiled matmul that is an exact identity over the
reals is rejected on 48/100 draws at K=4096 and 100/100 at K=11008, the
contraction widths of Llama-2-7B, under the published constant. The
rejected side is the more accurate one: at fp32 the tiled variant is 2.0x
to 6.0x closer to a float64 oracle than the reference it fails against.
Both operands are drawn i.i.d., which is what a standalone kernel
validator does and is not a claim about a deployed matmul.

**No fixed tolerance separates the corpus.** On the recorded corpus
(seven valid program cells, sixteen injected-bug instances), no
nonnegative `(atol, rtol)` pair separates the two classes under either
extreme cross-draw rule, at any `rtol >= 0`. This is exact rather than a grid sweep: per-draw envelopes,
interval certificates, and a monotonicity argument closing the tail
(Appendix B).

**Reference choice flips the verdict.** The same online-softmax candidate
at fp16 passes against two references and fails against a third, and the
direction of the accuracy comparison is itself draw-dependent.

**Real activations are benign.** Every real-activation FP32 cell passes.
We do not quantify the margin: the tensor-scale statistic that would
suggest a wide one is the statistic this paper shows does not track an
elementwise rule.

These results do not demonstrate failures in the measured systems. They
identify the protocol coordinates that must be stated for a numerical
acceptance result to be reproducible.

## Checking the numbers

Every number in the paper is asserted against the committed records by
one script. Six of them resolve to quoted lines of the run log rather
than to a data file, and the script labels which.

```bash
python tools/check_paper_numbers.py    # 109 claims against results/
python -m pytest -q                    # corpus exit criteria, envelope tests
```

## Reading order

| File | What it is |
|---|---|
| [`PAPER.md`](PAPER.md) | The paper |
| [`CLAIM.md`](CLAIM.md) | The registered claim and dated amendments. Frozen; verbose by design |
| [`BACKGROUND.md`](BACKGROUND.md) | Why the gap exists, per system, with the corpus provenance table |
| [`NOTEBOOK.md`](NOTEBOOK.md) | Dated run record. Prediction before each run, outcome after |
| [`ERRATA.md`](ERRATA.md) | Every error, retraction, and falsified prediction, by failure mode |
| [`audits/mirage-fp-filter.md`](audits/mirage-fp-filter.md) | Method and evidence for the paper's Appendix A |
| [`ROADMAP.md`](ROADMAP.md) · [`PROGRESS.md`](PROGRESS.md) · [`AUDIT.md`](AUDIT.md) | Historical: the original plan, phase status, and the 2026-08-04 method audit |

## The corpus

Six transformation pairs, each exactly equal over the reals, so any
measured difference is floating-point behavior alone. Provenance per pair
is in [`fpgap/corpus.py`](fpgap/corpus.py) and tabulated in
[`BACKGROUND.md`](BACKGROUND.md).

| Pair | Identity | Accepted by |
|---|---|---|
| `split_reduction` | `sum(x) == sum of chunk sums` | Prism, Mirage, MPK |
| `reassociation` | `(a+b)+c == a+(b+c)` | Prism, Mirage, Axon |
| `scalar_past_matmul` | `(aA)B == a(AB)` | Axon, Prism |
| `layernorm_variance` | `E[(x-u)^2] == E[x^2] - u^2` | Axon (in principle) |
| `softmax_online` | naive == online softmax | Prism (chunked form only) |
| `matmul_k_tiling` | full-K == tiled-K matmul | Prism, Axon, Mirage |

`softmax_online` is the one pair the three systems do not all accept, and
it is in the corpus for that reason. Prism verifies the chunked form
without the running max. Neither Mirage nor Axon classifies the rescale
form; our inferences about why are labeled as inferences in the paper.

Summation order is pinned in
[`fpgap/accumulate.py`](fpgap/accumulate.py), never `torch.sum`, whose
reduction order varies by backend and would contaminate the measurement.

## Code layout

Each tool's docstring holds its run command.

| Path | Role |
|---|---|
| `fpgap/` | Order-pinned summation, the six pairs, the per-cell harness, input generators |
| `tools/run_*.py` | One script per registered experiment; each prints its prediction before running |
| `tools/verify_separability.py` | Independent reimplementation of the envelope analysis |
| `tools/check_paper_numbers.py` | Asserts the paper's numbers against `results/` |
| `results/` | Raw per-cell records. Every table derives from these |
| `fixtures/` | Real activation tensors and their statistics |

## Corrections

Claims retracted during this work are recorded, not removed: a "100x
tolerance gap" built on a category error, a direction-blind metric
interpretation refuted by our own records, a Mirage misreading from a
stale paper version, a K=2048 verdict produced by comparing a maximum
absolute difference against `atol` alone, and a withdrawn margin claim.
Details and causes: [`ERRATA.md`](ERRATA.md).

## AI use

AI tools (Claude) assisted with code, measurements, and documentation.
The developer directed, reviewed, and verified all work. No measured
number comes from a model: every result comes from the instruments in
`fpgap/` and `tools/`, gated by pre-registered thresholds
([`CLAIM.md`](CLAIM.md)), a float64 reference validated against 50-digit
arithmetic, a negative control, and the raw per-cell records in
[`results/`](results/).

## License

MIT
