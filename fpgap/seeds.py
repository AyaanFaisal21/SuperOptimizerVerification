"""phase 5: adversarial-but-realistic input generation.

Ruler §6.2's lesson transferred. Uniform random sampling missed edge cases in
bitvector domains -- "even if x > 0 it is possible to have x*x = 0" -- and seeding
with interesting constants found them. The floating-point analogue is that uniform
sampling of activations essentially never produces the arrangements that make
reassociation diverge.

The credibility problem with adversarial inputs is that anything can be broken by
inputs nobody would ever feed it. This module's answer:

    SEED THE ARRANGEMENT, NOT THE MAGNITUDES.

Every value produced here lies inside REAL_BOX -- the widest range observed across
the phase-3 activation fixtures. No denormals, no near-overflow, nothing a trained
model could not emit. What is adversarial is the *combination*: which magnitudes sit
next to which, and in what order they accumulate. A row of values a real model
produced, permuted into an order it happens never to produce, is still a row of
realistic values.

That keeps the result honest. A failure found here cannot be dismissed as an
artificial input, only as an unlikely one -- and how unlikely is exactly what the
catch-rate experiment measures.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

# widest [min, max] over every captured activation site (resid_pre_ln_L5).
# every strategy clamps into this box, so no generated value is outside what the
# trained model actually produced.
REAL_BOX = (-22.051, 21.328)
REAL_STD = 2.666            # std of that same site

__all__ = ["STRATEGIES", "generate", "REAL_BOX"]


def _g(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def uniform(shape, seed, box=REAL_BOX) -> Tensor:
    """the control: what everyone actually samples. gaussian at realistic scale."""
    x = torch.randn(shape, generator=_g(seed), dtype=torch.float64) * REAL_STD
    return x.clamp(*box)


def wide_range(shape, seed, box=REAL_BOX) -> Tensor:
    """log-uniform magnitudes across the realistic range, random signs.

    a sum over values spanning many orders of magnitude loses the small ones
    entirely once the running total is large -- and the loss depends on order,
    which is exactly what reassociation changes.
    """
    g = _g(seed)
    hi = math.log(max(abs(box[0]), abs(box[1])))
    lo = math.log(1e-6)                       # small, but nowhere near denormal
    e = torch.rand(shape, generator=g, dtype=torch.float64) * (hi - lo) + lo
    sign = torch.randint(0, 2, shape, generator=g, dtype=torch.float64) * 2 - 1
    return (e.exp() * sign).clamp(*box)


def cancellation(shape, seed, box=REAL_BOX) -> Tensor:
    """near-cancelling pairs: the row sums to almost nothing while its terms are large.

    the condition number of the summation is sum|x_i| / |sum x_i|, so driving the
    denominator toward zero amplifies every rounding error in the accumulation.
    """
    g = _g(seed)
    n = shape[-1]
    half = n // 2
    v = torch.rand((*shape[:-1], half), generator=g, dtype=torch.float64) * box[1] * 0.9
    jitter = torch.randn((*shape[:-1], half), generator=g, dtype=torch.float64) * 1e-4
    x = torch.cat([v, -v + jitter], dim=-1)
    if x.shape[-1] < n:                       # odd extent
        x = torch.cat([x, torch.zeros((*shape[:-1], n - x.shape[-1]), dtype=torch.float64)], -1)
    return x[..., torch.randperm(n, generator=g)].clamp(*box)


def dynamic_mix(shape, seed, box=REAL_BOX) -> Tensor:
    """one large value among many tiny ones -- classic sequential-sum stagnation.

    once the running total reaches the large value, each subsequent small addend is
    below half an ulp and vanishes. a chunked or tree order sums the small values
    together first and keeps them, so the two orders genuinely disagree.
    """
    g = _g(seed)
    n = shape[-1]
    x = torch.rand(shape, generator=g, dtype=torch.float64) * 1e-4
    x[..., 0] = box[1] * 0.9
    return x[..., torch.randperm(n, generator=g)].clamp(*box)


def shifted(shape, seed, box=REAL_BOX) -> Tensor:
    """large mean relative to spread -- aimed squarely at E[x^2] - mu^2.

    cancellation severity is mu^2/E[x^2]; this drives it toward 1, which is the
    regime the fused one-pass variance form is known to fail in and which zero-mean
    activations never reach.
    """
    base = box[1] * 0.8
    x = base + torch.randn(shape, generator=_g(seed), dtype=torch.float64) * 0.05
    return x.clamp(*box)


STRATEGIES = {
    "uniform": uniform,              # control
    "wide_range": wide_range,
    "cancellation": cancellation,
    "dynamic_mix": dynamic_mix,
    "shifted": shifted,
}


def generate(strategy: str, shape, seed: int, dtype=torch.float32) -> Tensor:
    """draw at float64, then round down -- same convention as corpus._randn."""
    return STRATEGIES[strategy](tuple(shape), seed).to(dtype)
