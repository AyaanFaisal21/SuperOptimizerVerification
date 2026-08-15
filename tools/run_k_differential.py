"""D1 diagnostic (CLAIM 2026-08-15): differential growth across K.

does max |tiled - reference| grow broadly with K at the pinned unit-scale
geometry, or jump at one width? 20 draws per K, seeds shared with E2's
first twenty. records per-draw max abs diff, the |reference| magnitude at
the argmax element, and the tensor median |reference|.

    python tools/run_k_differential.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fpgap.corpus import BY_NAME
from tools.run_k_extension import gen


def main():
    t = BY_NAME["matmul_k_tiling"]
    print("PREDICTION (D1): median max-diff grows broadly with K, no isolated")
    print("jump at one width; argmax elements sit well below the tensor median.")
    out = {}
    for K in (512, 1024, 2048, 4096, 11008):
        rows = []
        for i in range(20):
            io = {"a": gen((64, K), 7000 + i * 17),
                  "b": gen((K, 64), 7001 + i * 17), "tile": 64}
            ref, var = t.baseline(io).double(), t.variant(io).double()
            d = (var - ref).abs()
            j = int(d.argmax())
            rows.append({"max_diff": float(d.max()),
                         "ref_at_argmax": float(ref.flatten()[j].abs()),
                         "ref_median": float(ref.abs().median())})
        med = sorted(r["max_diff"] for r in rows)[10]
        med_arg = sorted(r["ref_at_argmax"] for r in rows)[10]
        med_med = sorted(r["ref_median"] for r in rows)[10]
        out[str(K)] = rows
        print(f"  K={K:<6} median max|d| {med:.3e}   |ref|@argmax {med_arg:.3e}"
              f"   tensor median {med_med:.3e}")
    Path("results").mkdir(exist_ok=True)
    with open("results/k_differential.json", "w") as f:
        json.dump({"draws": 20, "shape": [64, 64], "tile": 64, "per_k": out},
                  f, indent=2)
    print("raw -> results/k_differential.json")


if __name__ == "__main__":
    sys.exit(main())
