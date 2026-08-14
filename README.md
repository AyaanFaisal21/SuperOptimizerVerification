# The floating-point gap in verified tensor superoptimizers

Verified superoptimizers prove kernel equivalence over exact arithmetic.
The kernels they ship run in floating point.
Random testing closes the distance, with no error bound.
This repo measures that distance.

The claim under test is [`CLAIM.md`](CLAIM.md).
It was registered 2026-08-03, with thresholds fixed before any measurement code.

## AI use

AI tools (Claude) assisted with code, measurements, and documentation. The
developer directed, reviewed, and verified all work. No measured number comes
from a model: every result comes from the instruments in `fpgap/` and `tools/`,
gated by pre-registered thresholds ([`CLAIM.md`](CLAIM.md)), a float64 reference
validated against 50-digit arithmetic, a negative control, and raw per-cell
records committed in [`results/`](results/).

## Reading order

| File | What it is |
|---|---|
| [`CLAIM.md`](CLAIM.md) | The registered claim, thresholds, and dated verdict. Frozen: kept verbose by design |
| [`BACKGROUND.md`](BACKGROUND.md) | Why the gap exists, per system, with citations |
| [`ROADMAP.md`](ROADMAP.md) | The original plan and kill criteria. Frozen: the plan the work is measured against |
| [`PROGRESS.md`](PROGRESS.md) | Phase status against the roadmap's own exit criteria |
| [`NOTEBOOK.md`](NOTEBOOK.md) | Dated run record. Prediction before each run, outcome after |
| [`ERRATA.md`](ERRATA.md) | Every error, retraction, and falsified claim, by failure mode |
| [`AUDIT.md`](AUDIT.md) | Method audit of 2026-08-04: nine findings, remaining steps |
| [`papers/README.md`](papers/README.md) | The six-paper working set |

## The corpus

Six transformation pairs. Each pair is exactly equal over the reals.
Any measured difference is floating-point behavior alone.

| Pair | Identity | Accepted by | Source |
|---|---|---|---|
| `split_reduction` | `sum(x) == sum of chunk sums` | Prism, Mirage, MPK | Prism Table 1 |
| `reassociation` | `(a+b)+c == a+(b+c)` | Prism, Mirage, Axon | Prism §4; Mirage §5 |
| `scalar_past_matmul` | `(aA)B == a(AB)` | Axon, Prism | Axon §2/§5.1 example |
| `layernorm_variance` | `E[(x-u)^2] == E[x^2] - u^2` | Axon (in principle) | cornfield ln_kernel |
| `softmax_online` | naive == online softmax | Prism (chunked form); Axon, Mirage: no (inferred) | TransformerOp attn_ext.cu |
| `matmul_k_tiling` | full-K == tiled-K matmul | Prism, Axon, Mirage | Prism §3.4/§5 |

`softmax_online` is the one pair the three systems do not all accept.
Prism verifies the chunked form, without the running max (its Fig. 2).
The running max is not a Lax operator, so we infer Mirage partitions around this pair.
Axon leaves `exp` uninterpreted, so we infer it rejects the rescale identity.
Both inferences are ours. Neither paper classifies this transformation.

Summation order is pinned in [`fpgap/accumulate.py`](fpgap/accumulate.py), never `torch.sum`.
Torch's own reduction order differs by backend and would contaminate the measurement.

## Code layout

Each tool's docstring holds its run command.

| Path | Role |
|---|---|
| `fpgap/accumulate.py` | Order-pinned summation primitives |
| `fpgap/corpus.py` | The six pairs, with provenance and hazard notes |
| `fpgap/harness.py` | One cell: floor, total, differential, both gate readings |
| `fpgap/seeds.py` | Phase 5 input generators |
| `tools/probe_hardware.py` | Machine arithmetic characterisation |
| `tools/check_corpus_device.py` | Float64 equivalence gate, per device |
| `tools/validate_reference.py` | Float64 truth check against 50-digit mpmath |
| `tools/dump_activations.py` | Regenerates the activation fixture from a checkpoint |
| `tools/run_sweep.py` | Phase 4 synthetic arm, 54 cells |
| `tools/run_sweep_activations.py` | Phase 4 activation arm plus matched controls |
| `tools/run_seeded.py` | Phase 5 catch rates, 100 trials per cell |
| `tests/test_corpus.py` | Phase 1 exit criteria plus the negative control |
| `results/` | Raw per-cell records. The tables derive from these |
| `fixtures/` | Real activation tensors and their statistics |

## Status

Phases 0-5 of 6 are complete. Only the writeup remains.
Measured on an NVIDIA A10 (Ampere) and an Apple M3 Pro.

```bash
python -m pytest tests/ -q                # phase 1 exit criteria
python tools/validate_reference.py        # float64 vs 50-digit truth
python tools/run_sweep.py                 # 54-cell synthetic sweep
python tools/run_sweep_activations.py     # real activations + controls
python tools/run_seeded.py                # phase 5 catch rates
```

## Result

Neither C1 nor A1. The dated verdict is in [`CLAIM.md`](CLAIM.md).

The gate under test: pass if `|new - old| <= atol + rtol*|old|`, with `atol = rtol = 1e-4`.

**Main finding: the constant `atol` makes the verdict depend on output scale, not on correctness.**
The same valid rewrite, unchanged, at unit-scale inputs:

| K | max abs error | verdict |
|---|---|---|
| 128 | 1.34e-05 | pass |
| 512 | 7.63e-05 | pass, at 0.8x atol |
| 2048 | 2.37e-04 | fail |

Relative errors across all six pairs span a factor of 3.
Absolute errors span four orders of magnitude, because output magnitudes do.
Production LLM matmuls run K = 4096-16384, past the failure point.
The fix costs nothing: a relative-only gate, or an `atol` scaled to the output.

Supporting results:

- A failure on 14% of draws [CI 9-22%] at activation input scale (0/100 at unit scale) is missed 86% of the time by the single-draw validation this project itself ran in Phase 4.
- The gate rejects more accurate kernels. Online softmax at fp16 is 15x closer to float64 truth than the reference it is compared against, and fails.
- Real activations are benign. Every real site passes with 10x headroom, including post-LayerNorm summation at row condition number 368,927.
- At fp16 and bf16 the failures measure precision, not transformations: `d/floor` is near 1 for two-thirds of the corpus.

Two claims were retracted during the work and are recorded, not removed:
a "100x tolerance gap" built on a category error, and "C1 survives at fp32,"
whose seeded inputs sit 10.4 sigma outside the real distribution and whose
violations are near 2x tolerance, not the 1600x first reported.
Details: [`ERRATA.md`](ERRATA.md).

## License

MIT
