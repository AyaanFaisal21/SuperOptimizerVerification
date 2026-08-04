"""the phase 4 cross product on synthetic inputs.

    python tools/run_sweep.py [--device cuda] [--out results/sweep_randn.json]

raw per-cell records go to disk -- those are the artifact a reader needs to check
the claim. the printed table is a convenience, not the result.
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fpgap.harness import dump, sweep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/sweep_randn.json")
    args = ap.parse_args()

    if args.device == "cuda":
        # a TF32 fp32 baseline would be a ~10-bit-mantissa baseline, inflating every
        # matmul cell in the direction that makes C1 look true. never inherited.
        torch.backends.cuda.matmul.allow_tf32 = False

    cells = sweep(device=args.device)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    dump(cells, args.out)

    hdr = (f"{'pair':<21}{'shape':<11}{'prec':<6}{'floor':>10}{'total':>10}"
           f"{'diff':>10}{'d/flr':>7}{'axon':>6}{'reg':>5}{'T1':>4}")
    print(hdr)
    print("-" * len(hdr))
    last = None
    for c in cells:
        if last and last != c.transformation:
            print()
        last = c.transformation
        v = c.verdict()
        print(f"{c.transformation:<21}{c.shape_class:<11}{c.precision:<6}"
              f"{c.floor.scale_rel:>10.2e}{c.total.scale_rel:>10.2e}"
              f"{c.differential.scale_rel:>10.2e}{c.d_over_floor:>7.2f}"
              f"{'pass' if v['gate_axon'] else 'FAIL':>6}"
              f"{'P' if v['gate_as_registered'] else 'F':>5}"
              f"{'X' if v['t1_fail_differential'] else '.':>4}")

    n = len(cells)
    ax = sum(c.verdict()["gate_axon"] for c in cells)
    dis = [c for c in cells if not c.verdict()["gates_agree"]]
    t1 = sum(c.verdict()["t1_fail_differential"] for c in cells)
    print(f"\n{n} cells   axon gate: {ax} pass / {n - ax} fail   "
          f"T1 failures: {t1}   gate definitions disagree: {len(dis)}")
    for c in dis:
        print(f"  disagree: {c.transformation}/{c.shape_class}/{c.precision} "
              f"axon={c.verdict()['gate_axon']} registered={c.verdict()['gate_as_registered']}")
    print(f"\nraw cells -> {args.out}")


if __name__ == "__main__":
    sys.exit(main())
