"""the detection arm: plausible wrong rewrites (mutants).

registered in CLAIM.md (2026-08-08 amendment) before any mutant ran. each mutant
is a bug a kernel author could actually write, produces the same output shape as
the correct baseline, and must differ from it in float64 (a real bug, not a
no-op). the complement to the valid corpus: the corpus measures false rejection,
the mutants measure detection.
"""

import torch

from .accumulate import chunked_sum, seq_sum

__all__ = ["MUTANTS"]


# each entry: name -> (correct_fn, mutant_fn, make_inputs_shape_kind, note)

def _mm_correct(io):
    a, b = io["a"], io["b"]
    k = a.shape[-1]
    parts = [a[:, t:t + 64] @ b[t:t + 64, :] for t in range(0, k, 64)]
    return seq_sum(torch.stack(parts, 0), 0)


def _mm_drop_tile(io):
    a, b = io["a"], io["b"]
    k = a.shape[-1]
    parts = [a[:, t:t + 64] @ b[t:t + 64, :] for t in range(0, k - 64, 64)]
    return seq_sum(torch.stack(parts, 0), 0)          # last K-tile never added


def _mm_drop_col(io):
    return io["a"][:, :-1] @ io["b"][:-1, :]          # one K element dropped


def _ln_correct(io):
    x, g, b, eps = io["x"], io["gamma"], io["beta"], io["eps"]
    n = x.shape[-1]
    mean = (seq_sum(x, -1) / n).unsqueeze(-1)
    var = (seq_sum((x - mean) ** 2, -1) / n).unsqueeze(-1)
    return (x - mean) * torch.rsqrt(var + eps) * g + b


def _ln_bessel(io):
    x, g, b, eps = io["x"], io["gamma"], io["beta"], io["eps"]
    n = x.shape[-1]
    mean = (seq_sum(x, -1) / n).unsqueeze(-1)
    var = (seq_sum((x - mean) ** 2, -1) / (n - 1)).unsqueeze(-1)   # n-1 divisor
    return (x - mean) * torch.rsqrt(var + eps) * g + b


def _ln_eps_to_std(io):
    x, g, b, eps = io["x"], io["gamma"], io["beta"], io["eps"]
    n = x.shape[-1]
    mean = (seq_sum(x, -1) / n).unsqueeze(-1)
    var = (seq_sum((x - mean) ** 2, -1) / n).unsqueeze(-1)
    return (x - mean) / (torch.sqrt(var) + eps) * g + b           # eps on std


def _sm_correct(io):
    x = io["x"]
    m = x.amax(-1, keepdim=True)
    e = torch.exp(x - m)
    return e / seq_sum(e, -1).unsqueeze(-1)


def _sm_no_rescale(io):
    x = io["x"]
    n = x.shape[-1]
    m = torch.full(x.shape[:-1], float("-inf"), dtype=x.dtype)
    s = torch.zeros(x.shape[:-1], dtype=x.dtype)
    for start in range(0, n, 64):
        blk = x[..., start:start + 64]
        m_new = torch.maximum(m, blk.amax(-1))
        # BUG: running sum never rescaled by exp(m - m_new)
        s = s + seq_sum(torch.exp(blk - m_new.unsqueeze(-1)), -1)
        m = m_new
    return torch.exp(x - m.unsqueeze(-1)) / s.unsqueeze(-1)


def _red_correct(io):
    return seq_sum(io["x"], -1)


def _red_drop_last(io):
    return seq_sum(io["x"][..., :-1], -1)             # sum over n-1 elements


MUTANTS = {
    "mm_drop_last_tile": (_mm_correct, _mm_drop_tile, "matmul",
                          "final K-tile omitted from the accumulation"),
    "mm_drop_last_col":  (_mm_correct, _mm_drop_col, "matmul",
                          "off-by-one on the contraction bound"),
    "ln_bessel":         (_ln_correct, _ln_bessel, "layernorm",
                          "variance divided by n-1 instead of n"),
    "ln_eps_to_std":     (_ln_correct, _ln_eps_to_std, "layernorm",
                          "eps added to std instead of variance"),
    "sm_no_rescale":     (_sm_correct, _sm_no_rescale, "softmax",
                          "online softmax missing the running-max correction"),
    "red_drop_last":     (_red_correct, _red_drop_last, "reduction",
                          "reduction over n-1 elements"),
}
