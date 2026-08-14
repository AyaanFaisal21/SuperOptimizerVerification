"""C1: F3 with draws. three references for the same online-softmax candidate.

registered in CLAIM.md before running (see measure.md for the amendment
block). F3's numbers -- the verdict and even the direction of the accuracy
comparison depend on the reference implementation -- rest on one draw per
reference. this runs 100 draws per precision at the mlp shape (512, 1024)
and reports, per reference (sequential, strided-tree, torch.sum): the mean
and range of its oracle floor, the candidate's total, the gate verdict rate
with a Wilson CI, and how often the floor ordering seq > tree > torch.sum
holds. the oracle is the float64 naive softmax with a tree-order normalizer;
float64 order differences are ~1e-15 and negligible at these precisions.

    python tools/run_reference_sensitivity.py
"""

import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fpgap.accumulate import seq_sum, tree_sum
from fpgap.corpus import _randn
from fpgap.harness import DTYPES, ErrorRecord
from tools.run_tree_baseline import softmax_online, softmax_ref

SHAPE = (512, 1024)
DRAWS = 100
REFS = (("seq", seq_sum), ("tree", tree_sum),
        ("torch.sum", lambda e, d=-1: torch.sum(e, d)))


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def main():
    print("PREDICTION (CLAIM C1): at fp16 the candidate FAILs vs seq and")
    print("passes vs tree and torch.sum on >=95% of draws each; floor ordering")
    print("seq > tree > torch.sum on >=95% of draws; at bf16 the candidate")
    print("FAILs vs all three references on >=95% of draws.")

    out = {}
    for prec in ("fp16", "bf16"):
        dt = DTYPES[prec]
        acc = {name: {"floors": [], "totals": [], "fails": 0} for name, _ in REFS}
        ordering_holds = 0
        for i in range(DRAWS):
            x = _randn(SHAPE, dt, seed=3000 + i * 17)
            oracle = softmax_ref(x.double(), tree_sum)
            online = softmax_online(x).double()
            floors = {}
            for name, fn in REFS:
                ref = softmax_ref(x, fn).double()
                fl = ErrorRecord.of(ref, oracle)
                to = ErrorRecord.of(online, oracle)
                df = ErrorRecord.of(online, ref)
                acc[name]["floors"].append(fl.scale_rel)
                acc[name]["totals"].append(to.scale_rel)
                acc[name]["fails"] += not df.gate_pass
                floors[name] = fl.scale_rel
            ordering_holds += floors["seq"] > floors["tree"] > floors["torch.sum"]

        out[prec] = {"ordering_seq_gt_tree_gt_torchsum":
                     {"holds": ordering_holds, "draws": DRAWS,
                      "ci": list(wilson(ordering_holds, DRAWS))}}
        print(f"\n--- {prec}, shape {SHAPE}, {DRAWS} draws ---")
        print(f"{'reference':<11}{'floor mean':>12}{'floor min':>12}{'floor max':>12}"
              f"{'t/f mean':>10}{'FAIL rate':>10}{'95% CI':>14}")
        for name, _ in REFS:
            a = acc[name]
            fm = sum(a["floors"]) / DRAWS
            tf = sum(t / f for t, f in zip(a["totals"], a["floors"])) / DRAWS
            lo, hi = wilson(a["fails"], DRAWS)
            out[prec][name] = {
                "floor_mean": fm, "floor_min": min(a["floors"]),
                "floor_max": max(a["floors"]),
                "total_mean": sum(a["totals"]) / DRAWS,
                "t_over_f_mean": tf,
                "gate_fail": a["fails"], "draws": DRAWS, "ci": [lo, hi]}
            print(f"{name:<11}{fm:>12.2e}{min(a['floors']):>12.2e}"
                  f"{max(a['floors']):>12.2e}{tf:>10.2f}"
                  f"{a['fails']/DRAWS:>10.0%}{f'[{lo:.0%},{hi:.0%}]':>14}")
        oh = out[prec]["ordering_seq_gt_tree_gt_torchsum"]
        print(f"floor ordering seq > tree > torch.sum: {oh['holds']}/{DRAWS}")

    Path("results").mkdir(exist_ok=True)
    with open("results/reference_sensitivity.json", "w") as f:
        json.dump({"shape": list(SHAPE), "draws": DRAWS, "results": out}, f, indent=2)
    print("\nraw -> results/reference_sensitivity.json")


if __name__ == "__main__":
    sys.exit(main())
