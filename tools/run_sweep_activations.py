"""phase 4, activation arm: the same cross product on real model activations.

CLAIM.md T2 pre-registers that "realistic" means activations sampled from a trained
model. this runs the corpus against the phase-3 fixtures and is directly comparable
to tools/run_sweep.py, which runs it against randn.

the fixture-to-transformation mapping is deliberate, not convenient -- each pair is
fed the tensor the corresponding real kernel would actually see:

  layernorm_variance  <- resid_pre_ln, the residual stream ENTERING LayerNorm. not
                         post_ln (normalised by construction, so feeding it back
                         into a variance computation tests nothing) and not
                         post_gelu (more biased, but in a pre-norm transformer
                         LayerNorm never sees it).
  softmax_online      <- attn_scores, pre-softmax. exactly the real input.
  split_reduction,    <- run on both resid_pre_ln (centred) and post_gelu (biased,
  reassociation          row-max cancellation 0.36) to separate distribution
                         effects from shape effects.
  matmul pairs        <- activation @ activation, which is what Q@K^T is. using two
                         real activation tensors keeps both operands realistic
                         without needing the weight matrices.

one deliberate difference from the randn path: fixtures are float32, so float32 is
the "underlying real value" here rather than float64. real activations *are* fp32 in
the model, so this is the honest reference -- but it means the fp32 column measures
a slightly different thing than in the synthetic sweep, and is reported as such.

    python tools/run_sweep_activations.py
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fpgap.harness import DTYPES, dump, measure

FIX = Path(__file__).resolve().parent.parent / "fixtures" / "activations.pt"


def build(fx, name, dt):
    """inputs dict for `name` at dtype `dt`, drawn from the fixture set."""
    g = torch.Generator().manual_seed(7)

    def site(k):
        return fx[k].to(dt)

    if name in ("split_reduction", "reassociation"):
        return {"resid": {"x": site("resid_pre_ln_L5")},
                "post_gelu": {"x": site("post_gelu_L5")}}
    if name == "softmax_online":
        return {"attn_scores": {"x": site("attn_scores")}}
    if name == "layernorm_variance":
        x = site("resid_pre_ln_L5")
        cols = x.shape[-1]
        # gamma/beta are applied *after* the variance computation under test, so
        # they cannot affect the transformation's conditioning. randn is fine and
        # avoids implying the real weights were used.
        return {"resid": {
            "x": x,
            "gamma": torch.randn(cols, generator=g, dtype=torch.float32).to(dt),
            "beta": torch.randn(cols, generator=g, dtype=torch.float32).to(dt),
            "eps": torch.tensor(1e-5, dtype=dt),
        }}
    if name in ("scalar_past_matmul", "matmul_k_tiling"):
        a = site("resid_pre_ln_L5")                  # (512, 384)
        b = site("post_ln_L5").transpose(0, 1).contiguous().to(dt)   # (384, 512)
        io = {"a": a, "b": b}
        if name == "scalar_past_matmul":
            k = a.shape[-1]
            io["alpha"] = torch.tensor(float(k ** -0.5), dtype=dt)
        return {"act_at_act": io}
    raise KeyError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/sweep_activations.json")
    args = ap.parse_args()

    fx = torch.load(FIX)
    names = ("split_reduction", "reassociation", "scalar_past_matmul",
             "layernorm_variance", "softmax_online", "matmul_k_tiling")

    cells = []
    for name in names:
        for prec in ("fp32", "fp16", "bf16"):
            for site, io in build(fx, name, DTYPES[prec]).items():
                shape = tuple(next(iter(io.values())).shape)
                cells.append(measure(name, shape_class=site, precision=prec,
                                     inputs=io, input_kind="activation",
                                     shape=shape))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    dump(cells, args.out)

    hdr = (f"{'pair':<21}{'site':<12}{'prec':<6}{'floor':>10}{'total':>10}"
           f"{'diff':>10}{'d/flr':>7}{'axon':>6}{'T1':>4}")
    print(hdr)
    print("-" * len(hdr))
    last = None
    for c in cells:
        if last and last != c.transformation:
            print()
        last = c.transformation
        v = c.verdict()
        print(f"{c.transformation:<21}{c.shape_class:<12}{c.precision:<6}"
              f"{c.floor.scale_rel:>10.2e}{c.total.scale_rel:>10.2e}"
              f"{c.differential.scale_rel:>10.2e}{c.d_over_floor:>7.2f}"
              f"{'pass' if v['gate_axon'] else 'FAIL':>6}"
              f"{'X' if v['t1_fail_differential'] else '.':>4}")

    n = len(cells)
    ax = sum(c.verdict()["gate_axon"] for c in cells)
    print(f"\n{n} cells   axon gate: {ax} pass / {n - ax} fail")
    print(f"raw cells -> {args.out}")

    # ---- matched control: randn at the SAME shapes -------------------------
    # the roadmap asks to "compare against randn on the same transformations".
    # comparing the activation cells against the synthetic sweep directly would
    # confound distribution with shape, since the fixtures have different column
    # counts than the synthetic shape classes. this pins shape and varies only the
    # distribution, which is the only way the comparison means anything.
    print("\n\ndistribution effect, SHAPE HELD FIXED (real vs randn at same shape)")
    hdr = (f"{'pair':<20}{'site':<12}{'prec':<6}{'floor real':>12}"
           f"{'floor randn':>13}{'ratio':>8}{'d/flr real':>12}{'d/flr randn':>13}")
    print(hdr)
    print("-" * len(hdr))
    control = []
    for c in cells:
        if c.precision == "fp32":
            continue
        shp = c.shape
        shape = (shp if c.transformation not in
                 ("scalar_past_matmul", "matmul_k_tiling") else (shp[0], shp[1], shp[0]))
        r = measure(c.transformation, shape_class="matched",
                    precision=c.precision, shape=shape)
        control.append(r)
        fr, fs = c.floor.scale_rel, r.floor.scale_rel
        print(f"{c.transformation:<20}{c.shape_class:<12}{c.precision:<6}"
              f"{fr:>12.2e}{fs:>13.2e}{fr / fs:>8.2f}"
              f"{c.d_over_floor:>12.2f}{r.d_over_floor:>13.2f}")
    dump(control, args.out.replace(".json", "_control.json"))
    print(f"\ncontrol cells -> {args.out.replace('.json', '_control.json')}")


if __name__ == "__main__":
    sys.exit(main())
