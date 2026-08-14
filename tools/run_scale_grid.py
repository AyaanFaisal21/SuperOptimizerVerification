"""C3: rejection of the valid K-tiled matmul across K and input scale.

registered in CLAIM.md before running (see measure.md for the amendment
block). two axes, one anchor cell shared between them:

    axis 1: K in {512, 1024, 2048, 4096} at sigma = 2.67, the measured
            activation scale (fixtures/activations.json, resid_pre_ln_L5)
    axis 2: sigma in {0.5, 1.0, 2.0, 2.67, 4.0} at K = 512  (AUDIT step 5)

100 draws per cell, the literal elementwise rule, M = N = 64, fp32.
why: F1's onset (K=4096) is a unit-scale statement and F4's 14% is a
K=512 statement at activation scale. this grid joins them: how the
rejection onset moves with scale, and how the rate moves with scale at
fixed K. gaussian scaling only -- no clamp box -- so the two axes stay
clean; the clamped-box variant is Phase 5's `uniform` strategy.

    python tools/run_scale_grid.py
"""

import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fpgap.corpus import BY_NAME
from fpgap.harness import GATE

K_AXIS = (512, 1024, 2048, 4096)          # at SIGMA_STAR
SIGMA_AXIS = (0.5, 1.0, 2.0, 2.67, 4.0)   # at K = 512
SIGMA_STAR = 2.67
DRAWS = 100


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def gen(shape, seed, sigma):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(shape, generator=g, dtype=torch.float64) * sigma).float()


def rate(t, K, sigma, base_seed):
    fails = 0
    for i in range(DRAWS):
        io = {"a": gen((64, K), base_seed + i * 17, sigma),
              "b": gen((K, 64), base_seed + 1 + i * 17, sigma),
              "tile": 64}
        ref, var = t.baseline(io).double(), t.variant(io).double()
        fails += not bool(((var - ref).abs() <= GATE + GATE * ref.abs()).all())
    return fails


def main():
    t = BY_NAME["matmul_k_tiling"]
    print("PREDICTION (CLAIM C3): rejection non-decreasing in K and in sigma;")
    print(f"K=2048 at sigma={SIGMA_STAR} rejected on >=95% of draws; the sigma")
    print("sweep at K=512 reads 0/100 at sigma <= 1.0 and contains the")
    print("committed 14% at sigma = 2.67 inside its CI.")

    out = {"k_axis": {}, "sigma_axis": {}}

    print(f"\n--- axis 1: K sweep at sigma = {SIGMA_STAR}, M=N=64, {DRAWS} draws ---")
    for j, K in enumerate(K_AXIS):
        fails = rate(t, K, SIGMA_STAR, 21000 + j * 10007)
        lo, hi = wilson(fails, DRAWS)
        out["k_axis"][str(K)] = {"fails": fails, "draws": DRAWS, "ci": [lo, hi]}
        print(f"  K={K:<5} rejected {fails}/{DRAWS}  [{lo:.0%}, {hi:.0%}]")

    print(f"\n--- axis 2: sigma sweep at K = 512, M=N=64, {DRAWS} draws ---")
    for j, s in enumerate(SIGMA_AXIS):
        fails = rate(t, 512, s, 22000 + j * 10007)
        lo, hi = wilson(fails, DRAWS)
        out["sigma_axis"][f"{s:g}"] = {"fails": fails, "draws": DRAWS, "ci": [lo, hi]}
        print(f"  sigma={s:<5g} rejected {fails}/{DRAWS}  [{lo:.0%}, {hi:.0%}]")

    Path("results").mkdir(exist_ok=True)
    with open("results/scale_grid.json", "w") as f:
        json.dump({"gate": GATE, "draws": DRAWS, "shape": [64, 64],
                   "sigma_star": SIGMA_STAR, **out}, f, indent=2)
    print("\nraw -> results/scale_grid.json")


if __name__ == "__main__":
    sys.exit(main())
