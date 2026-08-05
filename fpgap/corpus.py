"""the transformation corpus.

each entry is a pair (baseline, variant) that is *exactly* equal over R, together
with a record of which system would accept it and on what grounds. the pairing is
the experiment: over R the two sides are the same function, so any difference the
harness measures is entirely floating-point.

two rules the corpus holds itself to:

  - only real-arithmetic-exact pairs. a pair that disagrees in float64 is a bug in
    this file, not a finding. tests/test_corpus.py enforces that at 1e-12.
  - one variable at a time. where a transformation is algebraic (layernorm's
    variance identity) the accumulation order is held fixed across both sides, so
    the measured difference is attributable to the algebra and not to a reduction
    order that happened to change with it.

three variants read a knob out of the inputs dict with a default -- io.get("k")
(split chunk count), io.get("chunk") (online-softmax block), io.get("tile")
(K-tile width). a sweep can override them by injecting the key; nothing currently
does, so the defaults are what every committed result used.
"""

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor

from .accumulate import chunked_sum, seq_sum, tree_sum

Inputs = dict[str, Tensor]

# every make_inputs draws from this so a (transformation, shape, seed) triple is
# reproducible across processes and devices.
SEED = 20260803


@dataclass(frozen=True)
class Transformation:
    """a pair of implementations equal over R, plus provenance.

    accepted_by / citation / justification are not decoration: the claim is about
    the class of transformations these systems accept, so an entry that no system
    would accept does not belong in the corpus, and the record is what makes that
    checkable by a reader.
    """

    name: str
    baseline: Callable[[Inputs], Tensor]
    variant: Callable[[Inputs], Tensor]
    make_inputs: Callable[..., Inputs]
    shapes: dict[str, tuple]
    accepted_by: tuple[str, ...]
    citation: str
    justification: str
    identity: str
    notes: str = ""
    hazard: str = ""


def _gen(device: str, seed: int) -> torch.Generator:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return g


def _randn(shape, dtype=torch.float32, device="cpu", seed=SEED) -> Tensor:
    """draw at float64, then round down to the precision under test.

    two reasons, one of which cost a debugging cycle (NOTEBOOK 2026-08-03):

    - across precisions, every run then sees the *same underlying real values*
      rounded differently, so an fp32-vs-fp16 comparison is not confounded by the
      two having been handed different draws.
    - drawing at float32 and casting *up* to float64 leaves 29 spare mantissa bits,
      which makes float64 summation of a few thousand such values exact in every
      order. the phase-1 equivalence check then reads a perfect 0.000e+00 for the
      reduction pairs -- passing not because the pair is equivalent but because the
      reference had too much headroom to round at all.
    """
    x = torch.randn(shape, generator=_gen(device, seed), device=device, dtype=torch.float64)
    return x.to(dtype)


# ---------------------------------------------------------------------------
# 1. split reduction  --  red(x) = comb(map(red, part(x, k)))
# ---------------------------------------------------------------------------

def _split_inputs(shape, dtype=torch.float32, device="cpu", seed=SEED, **_) -> Inputs:
    return {"x": _randn(shape, dtype, device, seed)}


SPLIT_REDUCTION = Transformation(
    name="split_reduction",
    baseline=lambda io: seq_sum(io["x"], dim=-1),
    variant=lambda io: chunked_sum(io["x"], k=io.get("k", 32), dim=-1),
    make_inputs=_split_inputs,
    shapes={"small": (64, 256), "mlp": (512, 3072), "attention": (256, 1024)},
    accepted_by=("Prism", "Mirage", "megakernel/MPK"),
    citation="Prism §4 Table 1 (part/red/comb); Mirage §4.3 (A_eq); MPK SM-splitting",
    justification=(
        "associativity of +. Prism's part/red/comb axioms rewrite a reduction into a "
        "partitioned reduction plus a combine; splitting a reduction across SMs is the "
        "same rewrite applied for scheduling reasons."
    ),
    identity="sum(x) == sum_j sum_{i in chunk_j} x_i",
    notes="the single most common restructuring in the megakernel literature.",
    hazard="k partial sums each accumulate over n/k terms, so the variant is usually "
           "*more* accurate than sequential -- the sign of the effect is not assumed.",
)


# ---------------------------------------------------------------------------
# 2. reassociation  --  (a+b)+c  vs  a+(b+c), at scale
# ---------------------------------------------------------------------------

REASSOCIATION = Transformation(
    name="reassociation",
    baseline=lambda io: seq_sum(io["x"], dim=-1),
    variant=lambda io: tree_sum(io["x"], dim=-1),
    make_inputs=_split_inputs,
    shapes={"small": (64, 256), "mlp": (512, 3072), "attention": (256, 1024)},
    accepted_by=("Prism", "Mirage", "Axon"),
    citation="Prism §4 (associativity axioms); Mirage §4.3 (A_eq); Axon §4.2.1",
    justification=(
        "associativity of + over R. every system in scope treats this as free; it is "
        "the base case the others are built on."
    ),
    identity="((a+b)+c) == (a+(b+c))",
    notes="serial-loop accumulation vs a shared-memory tree reduction -- the two "
          "orders a real kernel actually picks between.",
    hazard="tree summation has O(log n) error growth vs O(n) sequential, so the "
           "*baseline* is the weaker side here.",
)


# ---------------------------------------------------------------------------
# 3. scalar multiply past matmul  --  (alpha A) B == alpha (A B)
# ---------------------------------------------------------------------------

def _scale_inputs(shape, dtype=torch.float32, device="cpu", seed=SEED, alpha=None, **_) -> Inputs:
    m, k, n = shape
    a = _randn((m, k), dtype, device, seed)
    b = _randn((k, n), dtype, device, seed + 1)
    # 1/sqrt(d) is the attention scale. non-dyadic on purpose: for d a power of
    # four (d=64 -> 0.125) the scaling is exact in binary FP and the transformation
    # is a no-op, which would silently contribute a row of zeros to the results.
    if alpha is None:
        alpha = float(1.0 / (k ** 0.5))
    return {"a": a, "b": b, "alpha": torch.tensor(alpha, dtype=a.dtype, device=device)}


SCALAR_PAST_MATMUL = Transformation(
    name="scalar_past_matmul",
    baseline=lambda io: (io["alpha"] * io["a"]) @ io["b"],
    variant=lambda io: io["alpha"] * (io["a"] @ io["b"]),
    make_inputs=_scale_inputs,
    shapes={"small": (32, 48, 32), "mlp": (256, 768, 3072), "attention": (256, 80, 256)},
    accepted_by=("Axon", "Prism"),
    citation="Axon §4.2 (worked example); Prism §4 (scalar propagation)",
    justification=(
        "distributivity of scalar multiplication over the matmul contraction. Axon's "
        "operator-propagation pass accepts this explicitly and uses it as its worked "
        "example of a transformation proved sound over R."
    ),
    identity="(alpha * A) @ B == alpha * (A @ B)",
    notes="the only pair whose difference is rounding *placement* rather than "
          "accumulation order -- both sides call the same matmul.",
    hazard="exact, and therefore a null result, whenever alpha is a power of two. K "
           "values here are chosen so 1/sqrt(K) is not dyadic.",
)


# ---------------------------------------------------------------------------
# 4. layernorm variance  --  two-pass  vs  fused one-pass (sum + sumsq)
# ---------------------------------------------------------------------------

def _ln_inputs(shape, dtype=torch.float32, device="cpu", seed=SEED, shift=0.0, **_) -> Inputs:
    rows, cols = shape
    x = _randn((rows, cols), dtype, device, seed)
    if shift:
        # a nonzero mean is what makes E[x^2] - E[x]^2 cancel. left at 0 by default
        # so the phase-1 equivalence check runs well-conditioned; phase 4/5 sweep it.
        x = x + torch.tensor(shift, dtype=x.dtype, device=device)
    return {
        "x": x,
        "gamma": _randn((cols,), dtype, device, seed + 1),
        "beta": _randn((cols,), dtype, device, seed + 2),
        "eps": torch.tensor(1e-5, dtype=x.dtype, device=device),
    }


def _ln_apply(x, mean, var, io):
    return (x - mean) * torch.rsqrt(var + io["eps"]) * io["gamma"] + io["beta"]


def _ln_two_pass(io: Inputs) -> Tensor:
    x = io["x"]
    n = x.shape[-1]
    mean = (seq_sum(x, dim=-1) / n).unsqueeze(-1)
    var = (seq_sum((x - mean) ** 2, dim=-1) / n).unsqueeze(-1)
    return _ln_apply(x, mean, var, io)


def _ln_one_pass(io: Inputs) -> Tensor:
    x = io["x"]
    n = x.shape[-1]
    # the fused form: one traversal accumulating s and q together, exactly as
    # cornfield/autotune_layernorm.py's ln_kernel does.
    mean = (seq_sum(x, dim=-1) / n).unsqueeze(-1)
    var = (seq_sum(x * x, dim=-1) / n).unsqueeze(-1) - mean * mean
    return _ln_apply(x, mean, var, io)


LAYERNORM_VARIANCE = Transformation(
    name="layernorm_variance",
    baseline=_ln_two_pass,
    variant=_ln_one_pass,
    make_inputs=_ln_inputs,
    shapes={"small": (64, 256), "mlp": (512, 3072), "attention": (256, 768)},
    accepted_by=("Axon", "Prism"),
    citation="Axon §4.2.1 (per-operator semantics); Prism §5 (fusion as instantiation)",
    justification=(
        "E[(x-mu)^2] == E[x^2] - mu^2, an identity over R. the fused form needs one "
        "traversal instead of two, which is why a fusing compiler picks it."
    ),
    identity="sum((x - mean)^2)/n == sum(x^2)/n - mean^2",
    notes="reimplemented from cornfield/autotune_layernorm.py, which ships the "
          "one-pass form and validates it at rtol=atol=1e-3.",
    hazard="accumulation order is held identical (seq_sum on both sides) so the "
           "measured difference is the algebra alone. cancellation is governed by "
           "mean^2 / E[x^2]; `shift` exposes it and is 0 by default.",
)


# ---------------------------------------------------------------------------
# 5. softmax  --  naive two-pass  vs  online/streaming recurrence
# ---------------------------------------------------------------------------

def _softmax_naive(io: Inputs) -> Tensor:
    x = io["x"]
    m = x.amax(dim=-1, keepdim=True)
    e = torch.exp(x - m)
    return e / seq_sum(e, dim=-1).unsqueeze(-1)


def _softmax_online(io: Inputs) -> Tensor:
    """flash-attention's identity: one pass, rescaling the running sum on each new max.

    mirrors the rescale in TransformerOp/kernels/attn_ext.cu:81 --
        float corr = expf(m - new_m);
    """
    x = io["x"]
    chunk = int(io.get("chunk", 64))
    n = x.shape[-1]
    m = torch.full(x.shape[:-1], float("-inf"), dtype=x.dtype, device=x.device)
    s = torch.zeros(x.shape[:-1], dtype=x.dtype, device=x.device)
    for start in range(0, n, chunk):
        blk = x[..., start : start + chunk]
        m_new = torch.maximum(m, blk.amax(dim=-1))
        corr = torch.exp(m - m_new)
        corr = torch.where(torch.isinf(m), torch.zeros_like(corr), corr)  # first block
        s = s * corr + seq_sum(torch.exp(blk - m_new.unsqueeze(-1)), dim=-1)
        m = m_new
    return torch.exp(x - m.unsqueeze(-1)) / s.unsqueeze(-1)


SOFTMAX_ONLINE = Transformation(
    name="softmax_online",
    baseline=_softmax_naive,
    variant=_softmax_online,
    make_inputs=_split_inputs,
    shapes={"small": (64, 256), "mlp": (512, 1024), "attention": (256, 1024)},
    accepted_by=("Prism", "Axon (partial)", "Mirage (non-Lax: partitioned around)"),
    citation="Prism §4 (streaming/scan axioms); Axon §5.2 (exp uninterpreted); "
             "Mirage §7 (non-Lax partitioning)",
    justification=(
        "exp(a-m)*exp(m-m') == exp(a-m'), so a running max can be corrected rather "
        "than recomputed. FlashAttention's core identity."
    ),
    identity="softmax(x) == online_softmax(x)",
    notes=(
        "the one entry that is *not* uniformly accepted, which is why it is in the "
        "corpus. Mirage cannot verify it -- more than one exp on an input->output "
        "path puts it outside Lax, so Mirage partitions around it. Axon treats exp as "
        "an uninterpreted function and conservatively rejects transformations through "
        "it. Prism accepts it axiomatically. it therefore measures what the systems "
        "give up, not only what they accept."
    ),
    hazard="the subtraction of a running max is a stabilization, so both sides are "
           "well-conditioned by construction; a null result here is meaningful.",
)


# ---------------------------------------------------------------------------
# 6. matmul K-loop tiling
# ---------------------------------------------------------------------------

def _ktiled_matmul(a: Tensor, b: Tensor, tile: int) -> Tensor:
    k = a.shape[-1]
    partials = [a[:, t : t + tile] @ b[t : t + tile, :] for t in range(0, k, tile)]
    return seq_sum(torch.stack(partials, dim=0), dim=0)


def _mm_inputs(shape, dtype=torch.float32, device="cpu", seed=SEED, **_) -> Inputs:
    m, k, n = shape
    return {
        "a": _randn((m, k), dtype, device, seed),
        "b": _randn((k, n), dtype, device, seed + 1),
    }


MATMUL_K_TILING = Transformation(
    name="matmul_k_tiling",
    baseline=lambda io: _ktiled_matmul(io["a"], io["b"], tile=io["a"].shape[-1]),
    variant=lambda io: _ktiled_matmul(io["a"], io["b"], tile=int(io.get("tile", 64))),
    make_inputs=_mm_inputs,
    shapes={"small": (32, 256, 32), "mlp": (256, 768, 3072), "attention": (256, 512, 256)},
    accepted_by=("Prism", "Axon", "Mirage"),
    citation="Prism §5 (instantiation = tiling choice); Axon §4.2.1; Mirage §4.3",
    justification=(
        "the contraction over K is a reduction, so any partition of the K range "
        "recombines to the same value over R. choosing the partition is scheduling, "
        "which is exactly what instantiation searches over."
    ),
    identity="A @ B == sum_t A[:, t] @ B[t, :]",
    notes="baseline is a single full-K tile (one matmul call); the variant splits K "
          "and accumulates partials, which is what a tiled kernel emits.",
    hazard="both sides route through the same underlying matmul, so what changes is "
           "the number and width of accumulation steps, not the inner kernel.",
)


CORPUS: tuple[Transformation, ...] = (
    SPLIT_REDUCTION,
    REASSOCIATION,
    SCALAR_PAST_MATMUL,
    LAYERNORM_VARIANCE,
    SOFTMAX_ONLINE,
    MATMUL_K_TILING,
)

BY_NAME = {t.name: t for t in CORPUS}
