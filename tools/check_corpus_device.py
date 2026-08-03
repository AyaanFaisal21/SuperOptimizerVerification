"""run the phase-1 float64 equivalence check on a chosen device.

the corpus has to be equivalent over R wherever it runs. CPU and CUDA take
different code paths -- different reduction kernels, different matmul libraries --
so "equivalent on the Mac" is not evidence of "equivalent on the A10".

    python tools/check_corpus_device.py --device cuda
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fpgap import CORPUS

TOL = 1e-12
SHAPES = ("small", "mlp", "attention")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.device == "cuda":
        # a TF32 fp32 baseline would not affect this float64 check, but pin it
        # anyway so no script in this repo ever runs with it left to chance.
        torch.backends.cuda.matmul.allow_tf32 = False

    print(f"device: {args.device}   gate: scale-rel < {TOL:.0e}")
    print(f"{'pair / shape':<34} {'scale-rel':>11} {'per-elt rel':>12}  {'':>4}")
    failing = 0
    for t in CORPUS:
        for cls in SHAPES:
            io = t.make_inputs(t.shapes[cls], dtype=torch.float64, device=args.device)
            io = {k: (v.double() if torch.is_floating_point(v) else v)
                  for k, v in io.items()}
            ref, got = t.baseline(io), t.variant(io)
            diff = (got - ref).abs()
            scaled = (diff.max() / ref.abs().max()).item()
            per_elt = (diff / (ref.abs() + 1e-30)).max().item()
            ok = scaled < TOL
            failing += not ok
            print(f"{t.name + '/' + cls:<34} {scaled:>11.3e} {per_elt:>12.3e}  "
                  f"{'ok' if ok else 'FAIL':>4}")

    n = len(CORPUS) * len(SHAPES)
    print(f"\n{n} cells, {failing} failing on {args.device}")
    return 1 if failing else 0


if __name__ == "__main__":
    sys.exit(main())
