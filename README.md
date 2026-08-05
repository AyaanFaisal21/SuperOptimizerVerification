# The floating-point gap in verified tensor superoptimizers

Every verified superoptimizer proves equivalence over an exact arithmetic and
then ships a floating-point kernel. Mirage bounds its error probability over
finite fields; Prism reasons algebraically over ~70 hand-written axioms; Axon
proves over ℝ because the floating-point theory is 1650× slower. In all three,
the distance between what is proven and what executes is closed by random
testing — with no error bound, and no characterization of what the sampling
might miss.

That is a reasonable engineering choice. It is also an unmeasured one.

**This repo measures it.** The claim under test is [`CLAIM.md`](CLAIM.md),
registered 2026-08-03 with pre-registered thresholds, before any measurement code
was written. The result is informative either way: if transformation-induced
error stays well inside validation tolerance, the field's shortcut is justified
and that is worth establishing rather than assuming. If it does not, the gap is a
correctness concern rather than a precision footnote.

## Reading order

| File | What it is |
|---|---|
| [`CLAIM.md`](CLAIM.md) | The claim (C1), the alternative (A1), and the thresholds — fixed in advance |
| [`BACKGROUND.md`](BACKGROUND.md) | Why the gap exists, per system, with citations |
| [`ROADMAP.md`](ROADMAP.md) | Phases, exit criteria, and kill criteria |
| [`PROGRESS.md`](PROGRESS.md) | Where the work stands against those phases, and where the outline turned out wrong |
| [`NOTEBOOK.md`](NOTEBOOK.md) | Dated entries — prediction before each run, outcome after |
| [`ERRATA.md`](ERRATA.md) | Every error, retraction and falsified claim, by failure mode — including five that would have produced clean wrong numbers |
| [`papers/README.md`](papers/README.md) | The six-paper working set and what to read in each |

## The corpus

Six transformation pairs, each *exactly* equal over ℝ, each drawn from what the
three systems actually accept. Over ℝ the two sides of a pair are the same
function, so any difference the harness measures is entirely floating-point.

| Pair | Identity | Accepted by | Source |
|---|---|---|---|
| `split_reduction` | `sum(x) == Σⱼ sum(chunkⱼ)` | Prism, Mirage, MPK | Prism §4 Table 1 (`part`/`red`/`comb`) |
| `reassociation` | `(a+b)+c == a+(b+c)` | Prism, Mirage, Axon | Prism §4; Mirage §4.3 (`A_eq`) |
| `scalar_past_matmul` | `(αA)B == α(AB)` | Axon, Prism | Axon §4.2 worked example |
| `layernorm_variance` | `E[(x−μ)²] == E[x²] − μ²` | Axon, Prism | `cornfield/autotune_layernorm.py` |
| `softmax_online` | `softmax(x) == online_softmax(x)` | Prism; Axon partial; **Mirage cannot** | `TransformerOp/kernels/attn_ext.cu:81` |
| `matmul_k_tiling` | `AB == Σₜ A[:,t]B[t,:]` | Prism, Axon, Mirage | Prism §5 (instantiation) |

`softmax_online` is the one entry not uniformly accepted, which is why it is in
the corpus: more than one `exp` on an input→output path puts it outside Mirage's
Lax fragment, so Mirage partitions around it; Axon treats `exp` as uninterpreted
and conservatively rejects transformations through it; Prism accepts it
axiomatically. It measures what the systems give up, not only what they accept.

Accumulation order is controlled explicitly in [`fpgap/accumulate.py`](fpgap/accumulate.py)
rather than delegated to `torch.sum` — torch uses a blocked pairwise cascade on
CPU and a tree reduction on CUDA, and measuring torch against torch would measure
torch's internals invisibly.

## Code layout

The package is four modules; `tools/` holds the runnable experiments, and every
tool's docstring carries its own run command.

| path | role |
|---|---|
| [`fpgap/accumulate.py`](fpgap/accumulate.py) | order-pinned summation (`seq`/`chunked`/`tree`) — the primitive everything rests on |
| [`fpgap/corpus.py`](fpgap/corpus.py) | the six pairs, each with provenance: which system accepts it, which paper section, and a `hazard` field |
| [`fpgap/harness.py`](fpgap/harness.py) | one cell → floor / total / differential, both gate definitions, both T1 readings |
| [`fpgap/seeds.py`](fpgap/seeds.py) | Phase-5 generators: adversarial arrangements of values bounded inside the real activation range |
| [`tools/probe_hardware.py`](tools/probe_hardware.py) | characterises a machine's arithmetic (TF32, storage rounding, accumulate width, fp64) so the record travels with results |
| [`tools/check_corpus_device.py`](tools/check_corpus_device.py) | re-runs the Phase-1 float64 equivalence gate on a chosen device |
| [`tools/validate_reference.py`](tools/validate_reference.py) | float64-as-truth check against mpmath at 50 digits |
| [`tools/dump_activations.py`](tools/dump_activations.py) | regenerates `fixtures/activations.pt` from a TransformerOp checkpoint, with per-row cancellation stats |
| [`tools/run_sweep.py`](tools/run_sweep.py) | Phase 4, synthetic arm — 54 cells on `randn` |
| [`tools/run_sweep_activations.py`](tools/run_sweep_activations.py) | Phase 4, activation arm + shape-matched `randn` controls |
| [`tools/run_seeded.py`](tools/run_seeded.py) | Phase 5 — catch rates, uniform vs four seeded strategies, 100 trials each |
| [`tests/test_corpus.py`](tests/test_corpus.py) | Phase-1 exit criteria: ℝ-equivalence plus the negative control proving the instrument reads nonzero |
| [`results/`](results/) | raw per-cell records — the artifact the tables are derived from |
| [`fixtures/`](fixtures/) | real activation tensors (committed) + distribution stats |

## Status

**Phases 0–5 of 6 complete**; only the writeup remains. Measured on an NVIDIA A10
(Ampere, native bf16) and an M3 Pro.

```bash
python -m pytest tests/ -q                # phase 1: equivalence over ℝ
python tools/validate_reference.py        # float64 vs mpmath at 50 digits
python tools/run_sweep.py                 # 54-cell synthetic sweep
python tools/run_sweep_activations.py     # real activations + matched controls
python tools/run_seeded.py                # phase 5 catch rates
```

## Result

**Neither C1 nor A1.** The full verdict is in [`CLAIM.md`](CLAIM.md); the headline:

**`atol = 1e-4` is an absolute constant applied to tensors of arbitrary magnitude, so
whether a *correct* transformation passes depends on output scale rather than on its
soundness.** Same code, unchanged:

| K | abs err | |
|---|---|---|
| 128 | 1.34e-05 | pass |
| 512 | 7.63e-05 | pass (0.8× atol) |
| **2048** | **2.37e-04** | **fail** |

Across all six pairs the **relative** errors span a factor of 3 (3.3e-07 – 1.0e-06).
The **absolute** errors span four orders of magnitude, entirely because output
magnitudes do. Matmul outputs grow as σ²√K; row-reductions as σ√n. Production LLMs
run K = 4096–16384. A relative-only gate, or an `atol` scaled to output magnitude,
would make the check measure the transformation.

Supporting results:

- **Single-sample validation misses real failures.** `matmul_k_tiling` fails **14% of
  uniform draws** at fp32 — and this project's own single-draw sweep missed it.
- **The gate can reject a *more accurate* kernel.** Online softmax at fp16 is **15×
  closer to float64 truth** than the reference it is checked against, and fails —
  visible only because both gate definitions are reported.
- **Real activations are benign.** Every real site passes with ~10× headroom,
  including post-LayerNorm at row condition number **368,927**, 400× more
  ill-conditioned than uniform sampling produces.
- **At fp16/bf16 the failures are about precision, not transformations** — `d/floor ≈ 1`
  for two-thirds of the corpus.

Two claims were **retracted** during the work, both recorded rather than removed: a
"100× tolerance gap" that rested on a category error, and "C1 survives at fp32," whose
seeded inputs sit **+10.4σ** outside the real distribution. Every error and falsified
claim is in [`ERRATA.md`](ERRATA.md); phase-by-phase status in
[`PROGRESS.md`](PROGRESS.md).

## License

MIT
