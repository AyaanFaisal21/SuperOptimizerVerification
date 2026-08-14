"""E1: the tolerance frontier. Registered in CLAIM.md before this ran.

per program-draw, the whole (atol, rtol) grid reduces to one function:
pass iff atol >= f(rtol) where f(rtol) = max_i(|d_i| - rtol*|b_i|).
so each draw is evaluated once and the full grid is derived from f.

    python tools/run_frontier.py
"""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fpgap.accumulate import seq_sum
from fpgap.corpus import BY_NAME
from fpgap.mutants import MUTANTS
from fpgap.mutants import _ln_correct, _mm_correct, _sm_correct  # noqa

GRID = [10.0 ** e for e in range(-8, 0)]          # 1e-8 .. 1e-1
DRAWS = 50


def f_of_rtol(diff, ref):
    """f(rtol) for each grid rtol: max_i(|d_i| - rtol*|b_i|)."""
    d, b = diff.abs().flatten(), ref.abs().flatten()
    return [float((d - r * b).max()) for r in GRID]


def gen(shape, seed, sigma=1.0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(shape, generator=g, dtype=torch.float64) * sigma).float()


# ---- valid corpus: six pairs at sweep shapes + the K=2048 tiling cell -------

def valid_programs():
    progs = []
    for name in ("split_reduction", "reassociation", "scalar_past_matmul",
                 "layernorm_variance", "softmax_online", "matmul_k_tiling"):
        t = BY_NAME[name]
        shape = t.shapes["mlp"]

        def run(seed, t=t, shape=shape):
            io = t.make_inputs(shape, dtype=torch.float32, seed=seed)
            return t.variant(io).double(), t.baseline(io).double()
        progs.append((name, run))

    def k2048(seed):
        a, b = gen((64, 2048), seed), gen((2048, 64), seed + 1)
        io = {"a": a, "b": b, "tile": 64}
        t = BY_NAME["matmul_k_tiling"]
        return t.variant(io).double(), t.baseline(io).double()
    progs.append(("k2048_tiling", k2048))
    return progs


# ---- mutant continuum -------------------------------------------------------

def mutant_programs():
    progs = []

    def red(j):
        def run(seed, j=j):
            x = gen((64, 1024), seed)
            return seq_sum(x[..., :-j], -1).double(), seq_sum(x, -1).double()
        return run

    def mm(j):
        def run(seed, j=j):
            io = {"a": gen((64, 512), seed), "b": gen((512, 64), seed + 1)}
            mut = io["a"][:, :-j] @ io["b"][:-j, :]
            return mut.double(), _mm_correct(io).double()
        return run

    def ln_div(j):
        def run(seed, j=j):
            io = {"x": gen((64, 1024), seed), "gamma": gen((1024,), seed + 1),
                  "beta": gen((1024,), seed + 2), "eps": torch.tensor(1e-5)}
            x, g_, b_, eps = io["x"], io["gamma"], io["beta"], io["eps"]
            n = x.shape[-1]
            mean = (seq_sum(x, -1) / n).unsqueeze(-1)
            var = (seq_sum((x - mean) ** 2, -1) / (n - j)).unsqueeze(-1)
            mut = (x - mean) * torch.rsqrt(var + eps) * g_ + b_
            return mut.double(), _ln_correct(io).double()
        return run

    def ln_eps(e):
        def run(seed, e=e):
            io = {"x": gen((64, 1024), seed), "gamma": gen((1024,), seed + 1),
                  "beta": gen((1024,), seed + 2), "eps": torch.tensor(1e-5)}
            x, g_, b_ = io["x"], io["gamma"], io["beta"]
            n = x.shape[-1]
            mean = (seq_sum(x, -1) / n).unsqueeze(-1)
            var = (seq_sum((x - mean) ** 2, -1) / n).unsqueeze(-1)
            mut = (x - mean) / (torch.sqrt(var) + e) * g_ + b_
            return mut.double(), _ln_correct(io).double()
        return run

    for j in (1, 2, 4, 8, 16, 32):
        progs.append((f"red_drop{j}", red(j)))
    for j in (1, 2, 8, 32):
        progs.append((f"mm_drop{j}col", mm(j)))
    for j in (1, 2, 8):
        progs.append((f"ln_div_n-{j}", ln_div(j)))
    for e in (1e-5, 1e-4, 1e-3):
        progs.append((f"ln_eps{e:g}", ln_eps(e)))
    return progs


def main():
    print("PREDICTION (CLAIM E1): no grid point reaches 0 valid-rejection AND")
    print("0 mutant-miss; any point passing k2048_tiling misses ln_eps1e-05.")
    print()

    data = {}
    for cls, progs in (("valid", valid_programs()), ("mutant", mutant_programs())):
        for name, run in progs:
            fs = []
            for i in range(DRAWS):
                var, ref = run(9000 + i * 31)
                fs.append(f_of_rtol(var - ref, ref))
            data[name] = {"class": cls, "f": fs}
            worst = max(f[0] for f in fs)
            print(f"  {cls:<7}{name:<18} worst f(rtol=1e-8) {worst:.2e}")

    print()
    best = None
    rows = []
    for ai, atol in enumerate(GRID):
        for ri, rtol in enumerate(GRID):
            vr = mm_ = vt = mt = 0
            for name, rec in data.items():
                for f in rec["f"]:
                    fails = f[ri] > atol
                    if rec["class"] == "valid":
                        vt += 1
                        vr += fails
                    else:
                        mt += 1
                        mm_ += not fails
            rows.append({"atol": atol, "rtol": rtol,
                         "valid_reject": vr / vt, "mutant_miss": mm_ / mt})
            tot = vr / vt + mm_ / mt
            if best is None or tot < best[0]:
                best = (tot, atol, rtol, vr / vt, mm_ / mt)

    sep = [r for r in rows if r["valid_reject"] == 0 and r["mutant_miss"] == 0]
    print(f"grid points: {len(rows)}   perfect separators: {len(sep)}")
    t, a, r, v, m = best
    print(f"best point: atol={a:g} rtol={r:g}  valid-reject {v:.1%}  "
          f"mutant-miss {m:.1%}  (sum {t:.1%})")

    Path("results").mkdir(exist_ok=True)
    with open("results/tolerance_frontier.json", "w") as fh:
        json.dump({"grid": GRID, "draws": DRAWS, "rows": rows}, fh)
    print("raw -> results/tolerance_frontier.json")


if __name__ == "__main__":
    sys.exit(main())
