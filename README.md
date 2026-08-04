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
| [`ERRATA.md`](ERRATA.md) | Every error and falsified claim, by failure mode — including four that would have produced clean wrong numbers |
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

## Status

**Phases 0–1 of 6 complete.** Claim registered before any code; corpus built and
both exit criteria met — all 18 cells (6 pairs × 3 shape classes) agree to 1e-15 in
float64, and a negative control confirms the accumulation orders genuinely diverge
in fp32, so a null result later would mean something.

```bash
python -m pytest tests/ -q
```

**No measurement has been made yet.** Nothing in this repo currently bears on
whether C1 or A1 is true.

Phase 2 is blocked on one decision — which hardware — because it determines whether
bf16 is measured or simulated. Phase 3 is blocked on a prerequisite the outline did
not anticipate: there is no trained TransformerOp checkpoint to draw activations
from. Full accounting, including three places the roadmap's own premises turned out
to be wrong, is in [`PROGRESS.md`](PROGRESS.md).

## License

MIT
