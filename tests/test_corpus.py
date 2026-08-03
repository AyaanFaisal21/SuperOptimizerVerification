"""phase 1 exit criteria.

three checks, and the second matters as much as the first:

  1. every pair agrees to 1e-12 in float64. a *bug check*, not a finding -- it
     establishes that baseline and variant really are the same function over R, so
     that any fp32/fp16 divergence measured later is attributable to arithmetic
     rather than to a typo.

  2. the accumulation primitives actually diverge in float32. without this the
     corpus could pass check 1 while measuring nothing: if the three orders
     collapsed to one implementation every result would be a clean zero and would
     read as strong evidence for A1. a null result only means something once the
     instrument is shown to have a nonzero reading somewhere.

  3. the float64 reference is genuinely rounding. same failure mode as (2) but
     subtler -- see the scale-relative note below.

run: python -m pytest tests/ -q     (or: python tests/test_corpus.py)


on the metric
-------------
agreement is gated on *scale-relative* error,

    max_i |g_i - T_i|  /  max_i |T_i|

not on max per-element relative error. per-element relative error is not well posed
here: an output that passes through cancellation can be arbitrarily close to zero,
and dividing by it reports an unbounded number for a bit-correct implementation.
measured directly -- scalar_past_matmul at the mlp shape has an element of magnitude
1.3e-05 in a tensor whose max is 4.88, and an absolute difference there of 9.8e-16,
which is float64 rounding noise at the scale of the terms being summed. as a
per-element relative error that is 7.5e-11; as a fact about the implementation it is
nothing.

this is the same reasoning behind the atol term in CLAIM.md's registered gate
(|g-T| <= atol + rtol|T|), applied to the reference check. both numbers are printed
below so the choice is visible rather than asserted, and none of the C1 thresholds
are affected -- those govern fp32/fp16 measurement, not this bug check.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fpgap import CORPUS, chunked_sum, seq_sum, tree_sum

REAL_EQUIV_TOL = 1e-12          # CLAIM.md phase-1 exit criterion
SHAPE_CLASSES = ("small", "mlp", "attention")


def _f64_inputs(t, cls):
    io = t.make_inputs(t.shapes[cls], dtype=torch.float64)
    return {k: (v.double() if torch.is_floating_point(v) else v) for k, v in io.items()}


def _errors(g, ref):
    """(scale-relative, max per-element relative) -- gate on the first, report both."""
    diff = (g - ref).abs()
    scaled = (diff.max() / ref.abs().max()).item()
    per_elt = (diff / (ref.abs() + 1e-30)).max().item()
    return scaled, per_elt


def measure_pairs():
    """{pair/shape: (scale-rel, per-element-rel)} in float64. shared by test and main."""
    out = {}
    for t in CORPUS:
        for cls in SHAPE_CLASSES:
            io = _f64_inputs(t, cls)
            out[f"{t.name}/{cls}"] = _errors(t.variant(io), t.baseline(io))
    return out


def measure_orders():
    """scale-relative spread of the three accumulation orders in float32."""
    x = torch.randn(8, 4096, generator=torch.Generator().manual_seed(0))
    s = seq_sum(x)
    return {
        "seq-vs-tree": _errors(tree_sum(x), s)[0],
        "seq-vs-chunked": _errors(chunked_sum(x, k=32), s)[0],
    }


def test_pairs_agree_in_float64():
    """each corpus pair is exactly equivalent over R, to within float64 noise."""
    for key, (scaled, _) in measure_pairs().items():
        assert scaled < REAL_EQUIV_TOL, (
            f"{key} differs by {scaled:.3e} of output scale in float64 "
            f"(> {REAL_EQUIV_TOL:.0e}). the pair is not equivalent over R -- "
            f"a bug in corpus.py, not a result."
        )


def test_accumulation_orders_differ_in_float32():
    """negative control: the primitives must not silently be the same computation."""
    for label, rel in measure_orders().items():
        assert rel > 0.0, (
            f"{label} is bit-identical in float32 -- the instrument reads zero by "
            f"construction and no result from it would mean anything."
        )


def test_float64_reference_actually_rounds():
    """the reference must not be exact, or check 1 passes for the wrong reason.

    inputs drawn at float32 and cast *up* to float64 carry only 24 mantissa bits,
    leaving 29 spare; summing a few thousand of them in float64 is then exact in
    every order, and the reduction pairs report a perfect 0.000e+00. that is the
    reference having too much headroom to round -- not evidence of anything.
    corpus._randn draws at float64 and rounds down to avoid it; this pins that.
    """
    x = torch.randn(4, 3072, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    assert _errors(tree_sum(x), seq_sum(x))[0] > 0.0, (
        "float64 summation is order-independent on these inputs -- the reference "
        "is exact and the equivalence check is vacuous."
    )
    narrowed = x.float().double()          # the failure mode, reproduced deliberately
    assert _errors(tree_sum(narrowed), seq_sum(narrowed))[0] == 0.0, (
        "expected float32-precision values to sum exactly in float64; if this fires, "
        "the headroom argument in corpus._randn needs revisiting."
    )


def test_primitives_agree_in_float64():
    """the three orders are the same reduction over R."""
    x = torch.randn(8, 4096, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    s = seq_sum(x)
    assert _errors(tree_sum(x), s)[0] < REAL_EQUIV_TOL
    assert _errors(chunked_sum(x, k=32), s)[0] < REAL_EQUIV_TOL


if __name__ == "__main__":
    test_accumulation_orders_differ_in_float32()
    print("negative control (float32, 8x4096) -- orders must differ:")
    for label, rel in measure_orders().items():
        print(f"  {label:16s} scale-rel {rel:.3e}")

    test_float64_reference_actually_rounds()
    test_primitives_agree_in_float64()
    print("\nfloat64 reference rounds, and the three orders agree over R: ok")

    test_pairs_agree_in_float64()
    out = measure_pairs()
    print(f"\nreal-arithmetic equivalence (float64, gate: scale-rel < {REAL_EQUIV_TOL:.0e}):")
    print(f"  {'pair / shape':<34} {'scale-rel':>11} {'per-elt rel':>12}")
    for key, (scaled, per_elt) in out.items():
        flag = "  <- cancellation" if per_elt > 1e-12 else ""
        print(f"  {key:<34} {scaled:>11.3e} {per_elt:>12.3e}{flag}")
    print(f"\n{len(CORPUS)} pairs x {len(SHAPE_CLASSES)} shape classes -- all pass.")
    print("per-elt outliers are near-zero outputs, not disagreements; see module docstring.")
