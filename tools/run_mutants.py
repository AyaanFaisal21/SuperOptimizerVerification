"""the detection arm: does the gate catch plausible bugs?

registered in CLAIM.md (2026-08-08) before this ran. per mutant: 100 draws at
fp32; per draw, the Axon gate (rtol = atol = 1e-4, elementwise, vs the correct
baseline). detection rate = fraction of draws on which the gate FAILS the
mutant. each mutant must first differ from the correct baseline in float64.

    python tools/run_mutants.py
"""

import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fpgap.harness import GATE
from fpgap.mutants import MUTANTS

SHAPES = {"matmul": None, "layernorm": (64, 1024),
          "softmax": (64, 1024), "reduction": (64, 1024)}


def make_io(kind, seed):
    g = torch.Generator().manual_seed(seed)

    def r(shape):
        return torch.randn(shape, generator=g, dtype=torch.float64).float()

    if kind == "matmul":
        return {"a": r((64, 512)), "b": r((512, 64))}
    io = {"x": r(SHAPES[kind])}
    if kind == "layernorm":
        io |= {"gamma": r((1024,)), "beta": r((1024,)),
               "eps": torch.tensor(1e-5)}
    return io


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def main():
    print("PREDICTIONS (registered in CLAIM.md before this run):")
    print("  gross mutants detected on >=95% of draws; ln_eps_to_std evades")
    print("  detection on >=95% of draws (shift ~1e-5 relative, under the gate).")
    print()
    hdr = (f"{'mutant':<20}{'f64 sanity':>11}{'detect':>8}{'95% CI':>13}"
           f"{'  note'}")
    print(hdr)
    print("-" * 86)

    out = {}
    for name, (correct, mutant, kind, note) in MUTANTS.items():
        io64 = {k: (v.double() if torch.is_floating_point(v) else v)
                for k, v in make_io(kind, 0).items()}
        c64, m64 = correct(io64), mutant(io64)
        sanity = ((m64 - c64).abs().max() / c64.abs().max()).item()
        assert sanity > 1e-9, f"{name} is a no-op, not a bug"

        detected = 0
        for i in range(100):
            io = make_io(kind, 5000 + i * 13)
            ref, mut = correct(io).double(), mutant(io).double()
            gate_pass = bool(((mut - ref).abs() <= GATE + GATE * ref.abs()).all())
            detected += not gate_pass
        lo, hi = wilson(detected, 100)
        out[name] = {"f64_divergence": sanity, "detection_rate": detected / 100,
                     "ci": [lo, hi], "note": note}
        print(f"{name:<20}{sanity:>11.2e}{detected/100:>8.0%}"
              f"{f'[{lo:.0%},{hi:.0%}]':>13}  {note}")

    Path("results").mkdir(exist_ok=True)
    with open("results/mutant_detection.json", "w") as f:
        json.dump({"gate": GATE, "draws": 100, "results": out}, f, indent=2)
    print("\nraw -> results/mutant_detection.json")


if __name__ == "__main__":
    sys.exit(main())
