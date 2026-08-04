"""phase 5: does adversarial-but-realistic seeding find what uniform sampling misses?

CLAIM.md T3 pre-registers the headline as a *rate*, not an existence claim: "report
the rate at which uniform random sampling catches failures, not merely whether
failures exist." A failure uniform sampling finds 1 time in 10^4 is a different
finding from one it finds every time.

So: N independent trials per (transformation, strategy, precision). Each trial draws
fresh inputs and asks whether the pair trips the gate. Catch rate = fraction of
trials that detect a failure. `uniform` is the control -- it is what the field
actually samples.

    python tools/run_seeded.py [--trials 100] [--precision fp32]
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fpgap.corpus import BY_NAME
from fpgap.harness import DTYPES, GATE, T1_FRAC, measure
from fpgap.seeds import STRATEGIES, generate

SHAPES = {                      # (rows, cols) or (m, k, n)
    "split_reduction": (64, 1024),
    "reassociation": (64, 1024),
    "layernorm_variance": (64, 1024),
    "softmax_online": (64, 1024),
    "scalar_past_matmul": (64, 512, 64),
    "matmul_k_tiling": (64, 512, 64),
}


def build(name, strategy, shape, dt, seed):
    """inputs dict with every float operand drawn from `strategy`."""
    if name in ("scalar_past_matmul", "matmul_k_tiling"):
        m, k, n = shape
        io = {"a": generate(strategy, (m, k), seed, dt),
              "b": generate(strategy, (k, n), seed + 1, dt)}
        if name == "scalar_past_matmul":
            io["alpha"] = torch.tensor(float(k ** -0.5), dtype=dt)
        return io
    x = generate(strategy, shape, seed, dt)
    if name == "layernorm_variance":
        cols = shape[-1]
        return {"x": x,
                "gamma": generate("uniform", (cols,), seed + 1, dt),
                "beta": generate("uniform", (cols,), seed + 2, dt),
                "eps": torch.tensor(1e-5, dtype=dt)}
    return {"x": x}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--precision", default="fp32")
    ap.add_argument("--out", default="results/seeded_catch_rates.json")
    args = ap.parse_args()
    dt = DTYPES[args.precision]

    print(f"precision {args.precision}   {args.trials} trials per cell   "
          f"gate rtol=atol={GATE:.0e}   T1 = >{GATE:.0e} on >={T1_FRAC:.0%} of elements")
    hdr = f"{'pair':<21}" + "".join(f"{s:>14}" for s in STRATEGIES)
    print("\ncatch rate: fraction of trials where the Axon gate FAILS")
    print(hdr); print("-" * len(hdr))

    out = {}
    for name in SHAPES:
        row, rec = f"{name:<21}", {}
        for strat in STRATEGIES:
            fails = t1s = 0
            worst = 0.0
            for i in range(args.trials):
                io = build(name, strat, SHAPES[name], dt, seed=1000 + i * 17)
                c = measure(name, precision=args.precision, inputs=io,
                            input_kind=strat, shape=SHAPES[name],
                            shape_class="seeded")
                v = c.verdict()
                fails += not v["gate_axon"]
                t1s += v["t1_fail_differential"]
                worst = max(worst, c.differential.scale_rel)
            rate = fails / args.trials
            rec[strat] = {"gate_fail_rate": rate,
                          "t1_fail_rate": t1s / args.trials,
                          "worst_differential": worst}
            row += f"{rate:>13.0%} "
        out[name] = rec
        print(row)

    print("\nworst differential seen (scale-relative), any trial")
    print(hdr); print("-" * len(hdr))
    for name, rec in out.items():
        print(f"{name:<21}" + "".join(f"{rec[s]['worst_differential']:>14.2e}"
                                      for s in STRATEGIES))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"precision": args.precision, "trials": args.trials,
                   "results": out}, f, indent=2)

    u = {n: r["uniform"]["gate_fail_rate"] for n, r in out.items()}
    best = {n: max(r[s]["gate_fail_rate"] for s in STRATEGIES if s != "uniform")
            for n, r in out.items()}
    print(f"\n{'pair':<21}{'uniform':>10}{'best seeded':>14}   verdict")
    for n in out:
        gap = ("seeded finds what uniform misses" if best[n] > u[n]
               else "no advantage" if best[n] == u[n] else "uniform stricter")
        print(f"{n:<21}{u[n]:>10.0%}{best[n]:>14.0%}   {gap}")
    print(f"\nraw -> {args.out}")


if __name__ == "__main__":
    sys.exit(main())
