"""D2 (CLAIM 2026-08-15): 50-digit oracle spot-check of the F3 headline cell.

recomputes the fp16 softmax cell's floor and total statistics at
(512, 1024) against an mpmath dps=50 oracle, on E4's draw plus two more
seeds, for the tree, torch.sum, and sequential references. prediction:
the float64-oracle statistics agree to better than 1e-6 relative, so
every F3 ratio is unchanged at reported precision.

    python tools/run_oracle_spotcheck.py
"""

import json
import sys
from pathlib import Path

import torch
from mpmath import mp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fpgap.accumulate import seq_sum, tree_sum
from fpgap.corpus import _randn
from fpgap.harness import DTYPES
from tools.run_k_extension import gen
from tools.run_tree_baseline import softmax_online

mp.dps = 50
SHAPE = (512, 1024)


def mp_softmax_rows(x64):
    """exact-to-50-digits softmax per row; returns nested lists of mpf."""
    out = []
    for row in x64.tolist():
        m = max(row)
        e = [mp.exp(mp.mpf(v) - m) for v in row]
        s = mp.fsum(e)
        out.append([v / s for v in e])
    return out


def stats_vs(oracle_rows, t):
    """floor/total style scale-relative statistic against the mp oracle."""
    md = mp.mpf(0)
    mo = mp.mpf(0)
    for orow, trow in zip(oracle_rows, t.tolist()):
        for o, v in zip(orow, trow):
            md = max(md, abs(mp.mpf(v) - o))
            mo = max(mo, abs(o))
    return float(md / mo)


def main():
    dt = DTYPES["fp16"]
    print("PREDICTION (D2): float64-oracle statistics within 1e-6 relative of")
    print("the 50-digit oracle; F3 ratios unchanged at reported precision.")
    out = []
    for tag, x in (("e4-draw", _randn(SHAPE, dt)),
                   ("seed-4242", gen(SHAPE, 4242).to(dt)),
                   ("seed-9099", gen(SHAPE, 9099).to(dt))):
        x64 = x.double()
        m64 = x64.amax(-1, keepdim=True)
        e64 = torch.exp(x64 - m64)
        oracle64 = e64 / tree_sum(e64, -1).unsqueeze(-1)
        oracle_mp = mp_softmax_rows(x64)

        def naive(sumfn):
            m = x.amax(-1, keepdim=True)
            e = torch.exp(x - m)
            return (e / sumfn(e).unsqueeze(-1)).double()

        cands = {"online": softmax_online(x).double(),
                 "tree": naive(lambda e: tree_sum(e, -1)),
                 "torch.sum": naive(lambda e: torch.sum(e, -1)),
                 "seq": naive(lambda e: seq_sum(e, -1))}
        row = {"tag": tag}
        worst = 0.0
        for name, t in cands.items():
            s64 = float((t - oracle64).abs().max() / oracle64.abs().max())
            smp = stats_vs(oracle_mp, t)
            rel = abs(s64 - smp) / smp
            worst = max(worst, rel)
            row[name] = {"stat_f64_oracle": s64, "stat_mp_oracle": smp,
                         "rel_deviation": rel}
            print(f"  {tag:<10} {name:<10} f64-oracle {s64:.6e}  "
                  f"mp-oracle {smp:.6e}  rel dev {rel:.2e}")
        row["worst_rel_deviation"] = worst
        out.append(row)
    worst_all = max(r["worst_rel_deviation"] for r in out)
    print(f"worst relative deviation across draws/references: {worst_all:.2e}")
    print("VERDICT:", "float64 oracle adequate for F3 at reported precision"
          if worst_all < 1e-6 else "PREDICTION FALSIFIED - report before use")
    Path("results").mkdir(exist_ok=True)
    with open("results/oracle_spotcheck.json", "w") as f:
        json.dump({"dps": 50, "shape": list(SHAPE), "rows": out}, f, indent=2)
    print("raw -> results/oracle_spotcheck.json")


if __name__ == "__main__":
    sys.exit(main())
