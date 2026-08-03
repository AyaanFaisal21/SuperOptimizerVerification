"""order-controlled summation.

every transformation in the corpus is, underneath, a statement about the order in
which a reduction accumulates. so the harness cannot call torch.sum: on CPU it is a
blocked pairwise cascade, on CUDA a tree reduction, and neither is the order the
kernels under study actually execute. delegating to it would measure torch's
internals rather than the transformation, and would do so invisibly.

each primitive here fixes its order explicitly. the reduction extent is walked by a
python loop, so the order is a property of this file and not of the backend, the
dtype, or the device. only the reduction dim is walked -- batch dims stay
vectorized -- so cost scales with row length (thousands of steps), not with batch.

these are deliberately slow and deliberately correct. nothing here is on a hot path.
"""

import torch
from torch import Tensor

__all__ = ["seq_sum", "chunked_sum", "tree_sum"]


def seq_sum(x: Tensor, dim: int = -1) -> Tensor:
    """strict left-to-right accumulation: ((x0 + x1) + x2) + ...

    the order a single thread produces when it walks a row with a serial loop --
    e.g. the per-lane `s += v` in cornfield's ln_kernel. the least accurate of the
    three and the one a naive kernel actually gives you.
    """
    x = x.movedim(dim, 0)
    acc = torch.zeros_like(x[0])
    for i in range(x.shape[0]):
        acc = acc + x[i]
    return acc


def chunked_sum(x: Tensor, k: int, dim: int = -1) -> Tensor:
    """split into k chunks, reduce each, then reduce the partials.

    Prism's part -> map(red) -> comb, and what SM-splitting a reduction across a
    megakernel does physically. identical to seq_sum over R by associativity; that
    identity is the transformation under test.
    """
    x = x.movedim(dim, -1)
    n = x.shape[-1]
    if n % k:
        raise ValueError(f"reduction extent {n} not divisible by k={k}")
    parts = x.reshape(*x.shape[:-1], k, n // k)
    partials = seq_sum(parts, dim=-1)      # sequential within each chunk
    return seq_sum(partials, dim=-1)       # then combine the k partials


def tree_sum(x: Tensor, dim: int = -1) -> Tensor:
    """strided pairwise halving: acc[i] += acc[i + stride], stride from n/2 down.

    not adjacent-pair summation -- strided, because that is what a shared-memory
    tree reduction does. cornfield's ln_kernel is literally this:

        for (int st = TPR / 2; st > 0; st >>= 1)
            if (lane < st) seg_s[lane] += seg_s[lane + st];

    odd extents leave a tail element that carries forward unpaired, matching the
    guarded `if (lane < st)`.
    """
    x = x.movedim(dim, -1)
    while x.shape[-1] > 1:
        n = x.shape[-1]
        half = n // 2
        paired = x[..., :half] + x[..., half : 2 * half]
        tail = x[..., 2 * half :]                      # unpaired when n is odd
        x = torch.cat([paired, tail], dim=-1) if tail.shape[-1] else paired
    return x[..., 0]
