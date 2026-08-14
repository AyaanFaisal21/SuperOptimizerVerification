"""E2-E4: K extension, per-element decomposition, library reference.

registered in CLAIM.md before this ran.

    python tools/run_k_extension.py
"""

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fpgap.accumulate import seq_sum, tree_sum
from fpgap.corpus import BY_NAME, _randn
from fpgap.harness import DTYPES, GATE, ErrorRecord
from fpgap.seeds import generate


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def gen(shape, seed, sigma=1.0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(shape, generator=g, dtype=torch.float64) * sigma).float()


def main():
    t = BY_NAME["matmul_k_tiling"]

    print("PREDICTIONS (CLAIM E2-E4): K=4096 and K=11008 fail on >=95% of")
    print("draws at unit scale; iid 1-(1-p)^n predicts the tensor rate at")
    print("K=512 sigma=2.67; torch.sum reference within 2x of tree, same verdicts.")

    print("\n--- E2: K extension, M=N=64, unit scale, 100 draws ---")
    for K in (2048, 4096, 11008):
        fails = 0
        for i in range(100):
            io = {"a": gen((64, K), 7000 + i * 17), "b": gen((K, 64), 7001 + i * 17),
                  "tile": 64}
            ref, var = t.baseline(io).double(), t.variant(io).double()
            fails += not bool(((var - ref).abs() <= GATE + GATE * ref.abs()).all())
        lo, hi = wilson(fails, 100)
        print(f"  K={K:<6} fail rate {fails}/100  [{lo:.0%}, {hi:.0%}]")

    print("\n--- E3: per-element decomposition, K=512, sigma=2.67, 100 draws ---")
    elem_fail = elem_total = tensor_fail = 0
    for i in range(100):
        io = {"a": generate("uniform", (64, 512), 1000 + i * 17, torch.float32),
              "b": generate("uniform", (512, 64), 1001 + i * 17, torch.float32)}
        ref, var = t.baseline(io).double(), t.variant(io).double()
        viol = (var - ref).abs() > GATE + GATE * ref.abs()
        elem_fail += int(viol.sum())
        elem_total += viol.numel()
        tensor_fail += bool(viol.any())
    p_elem = elem_fail / elem_total
    pred = 1 - (1 - p_elem) ** 4096
    lo, hi = wilson(tensor_fail, 100)
    print(f"  per-element exceedance p = {p_elem:.2e}")
    print(f"  observed tensor rate {tensor_fail}/100  [{lo:.0%}, {hi:.0%}]")
    print(f"  iid prediction 1-(1-p)^4096 = {pred:.1%}"
          f"   {'INSIDE CI' if lo <= pred <= hi else 'OUTSIDE CI'}")

    print("\n--- E4: torch.sum as the softmax reference ---")
    for prec in ("fp16", "bf16"):
        dt = DTYPES[prec]
        x = _randn((512, 1024), dt)
        x64 = x.double()
        m64 = x64.amax(-1, keepdim=True)
        e64 = torch.exp(x64 - m64)
        oracle = e64 / tree_sum(e64, -1).unsqueeze(-1)

        def naive(sumfn):
            m = x.amax(-1, keepdim=True)
            e = torch.exp(x - m)
            return (e / sumfn(e).unsqueeze(-1)).double()

        from tools.run_tree_baseline import softmax_online
        online = softmax_online(x).double()
        for label, fn in (("tree", lambda e: tree_sum(e, -1)),
                          ("torch.sum", lambda e: torch.sum(e, -1))):
            ref = naive(fn)
            fl = ErrorRecord.of(ref, oracle)
            to = ErrorRecord.of(online, oracle)
            df = ErrorRecord.of(online, ref)
            print(f"  {prec} {label:<10} floor {fl.scale_rel:.2e}  "
                  f"online total {to.scale_rel:.2e}  t/f {to.scale_rel/fl.scale_rel:.2f}  "
                  f"gate {'pass' if df.gate_pass else 'FAIL'}")


if __name__ == "__main__":
    sys.exit(main())
