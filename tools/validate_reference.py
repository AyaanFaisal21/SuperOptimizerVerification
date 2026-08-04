"""is float64 actually adequate as truth?

the whole project rests on float64 being a stand-in for exact arithmetic. that is
an assumption, and it is cheap to check: recompute the baselines at 50 decimal
digits with mpmath and see how far float64 drifts. if float64 truth is itself wrong
by an amount comparable to what we are trying to measure, every number downstream
is noise.

small shapes only -- mpmath is ~10^4x slower than torch, and the question is about
the reference's accuracy, not about scale. the four implementations below cover all
six corpus pairs: sums (split_reduction, reassociation), matmul (scalar_past_matmul,
matmul_k_tiling), layernorm, and softmax.

    python tools/validate_reference.py
"""

import sys
from pathlib import Path

import mpmath as mp
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fpgap.accumulate import seq_sum

mp.mp.dps = 50          # 50 decimal digits ~= 166 bits, vs float64's 53
TOL = 1e-14             # float64 eps is 2.2e-16; allow ~50x for accumulation


def _rel(a_f64: torch.Tensor, b_mp: list) -> float:
    """max relative difference between a float64 tensor and an mpmath result."""
    flat = a_f64.flatten().tolist()
    num = max(abs(mp.mpf(x) - y) for x, y in zip(flat, b_mp))
    den = max(abs(y) for y in b_mp)
    return float(num / den)


def check_sum(rows=4, cols=256, seed=0):
    x = torch.randn(rows, cols, generator=torch.Generator().manual_seed(seed),
                    dtype=torch.float64)
    exact = []
    for r in x.tolist():
        s = mp.mpf(0)
        for v in r:                    # same left-to-right order as seq_sum
            s += mp.mpf(v)
        exact.append(s)
    return _rel(seq_sum(x, dim=-1), exact)


def check_matmul(m=6, k=128, n=6, seed=1):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(m, k, generator=g, dtype=torch.float64)
    b = torch.randn(k, n, generator=g, dtype=torch.float64)
    al, bl = a.tolist(), b.tolist()
    exact = []
    for i in range(m):
        for j in range(n):
            s = mp.mpf(0)
            for t in range(k):
                s += mp.mpf(al[i][t]) * mp.mpf(bl[t][j])
            exact.append(s)
    return _rel(a @ b, exact)


def check_layernorm(rows=4, cols=256, seed=2, eps=1e-5):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, cols, generator=g, dtype=torch.float64)
    gamma = torch.randn(cols, generator=g, dtype=torch.float64)
    beta = torch.randn(cols, generator=g, dtype=torch.float64)

    n = cols
    mean = (seq_sum(x, dim=-1) / n).unsqueeze(-1)
    var = (seq_sum((x - mean) ** 2, dim=-1) / n).unsqueeze(-1)
    got = (x - mean) * torch.rsqrt(var + eps) * gamma + beta

    exact = []
    gl, bl = gamma.tolist(), beta.tolist()
    for r in x.tolist():
        mu = mp.fsum([mp.mpf(v) for v in r]) / n
        va = mp.fsum([(mp.mpf(v) - mu) ** 2 for v in r]) / n
        inv = 1 / mp.sqrt(va + mp.mpf(eps))
        exact += [(mp.mpf(v) - mu) * inv * mp.mpf(gv) + mp.mpf(bv)
                  for v, gv, bv in zip(r, gl, bl)]
    return _rel(got, exact)


def check_softmax(rows=4, cols=256, seed=3):
    x = torch.randn(rows, cols, generator=torch.Generator().manual_seed(seed),
                    dtype=torch.float64)
    m = x.amax(dim=-1, keepdim=True)
    e = torch.exp(x - m)
    got = e / seq_sum(e, dim=-1).unsqueeze(-1)

    exact = []
    for r in x.tolist():
        mx = max(mp.mpf(v) for v in r)
        ex = [mp.e ** (mp.mpf(v) - mx) for v in r]
        s = mp.fsum(ex)
        exact += [v / s for v in ex]
    return _rel(got, exact)


CHECKS = {
    "sum (split_reduction, reassociation)": check_sum,
    "matmul (scalar_past_matmul, matmul_k_tiling)": check_matmul,
    "layernorm (layernorm_variance)": check_layernorm,
    "softmax (softmax_online)": check_softmax,
}


def main():
    print(f"float64 vs mpmath at {mp.mp.dps} digits   (gate: rel < {TOL:.0e})")
    print(f"{'operation':<46}{'max rel':>12}{'':>7}")
    worst, failed = 0.0, 0
    for label, fn in CHECKS.items():
        r = fn()
        worst = max(worst, r)
        ok = r < TOL
        failed += not ok
        print(f"{label:<46}{r:>12.3e}{'  ok' if ok else '  FAIL':>7}")
    print(f"\nworst {worst:.3e}  ({worst / 2.22e-16:.0f}x float64 eps)")
    if failed:
        print("float64 is NOT adequate as truth for at least one operation.")
    else:
        print("float64 is adequate as truth: drift is orders below the 1e-4 gate")
        print("this project measures against.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
