"""E5/E8: exact tolerance separability on the recorded corpus.

registered in CLAIM.md (2026-08-14, second and fourth amendments) before the
runs. E5 analyzed two (F, G) pairings; E8 completes the 2x2 box and closes
the rtol tail.

per draw, acceptance at (atol, rtol) is atol + rtol*|b|_i >= |g-b|_i for every
element, so the draw's minimum accepting atol at rtol = r is the envelope
T(r) = max_i(d_i - r*x_i) with d = |g-b|, x = |b|. T is a maximum of lines:
convex, piecewise linear, non-increasing (every slope -x_i <= 0). T is
evaluated exactly from the Pareto set of the draw's (x, d) points.

class-level envelopes, one per Boolean per-class semantics:
  F_all(r)   = max over valid draws of T     (valid accepted iff every draw passes)
  F_any(r)   = max over valid programs of the min over that program's draws
                                             (valid accepted iff some draw passes)
  G_every(r) = min over all mutant draws     (mutant caught iff every draw fails)
  G_some(r)  = min over mutants of the max over that mutant's draws
                                             (mutant caught iff some draw fails)
a separator exists at r for corner (F, G) iff max(F(r), 0) < G(r).

between sampled rtols, no-separation on [r1, r2] is certified from two facts.
(1) a single-witness chord bound: each per-draw envelope e is convex, so
e(r) <= max(e(r1), e(r2)) on the interval; G_every lies below its witness
draw's chord bound, and each per-mutant max-envelope is itself convex (a max
of convex functions), so G_some lies below its witness mutant's chord bound.
(2) every envelope here is non-increasing, so F(r) >= F(r2) on the interval.
if the witness bound <= max(F(r2), 0), no r in [r1, r2] separates. nothing is
assumed about G itself: a min of convex functions need not be convex, and the
certificate never uses convexity of G.

tail closure: envelopes are non-increasing, so if G(100) <= 0 then
G(r) <= 0 <= max(F(r), 0) for every r >= 100 and every F. no-separation on
[0, 100] plus a nonpositive G at 100 covers the whole domain rtol >= 0.

    python tools/run_separability.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.run_frontier import DRAWS, mutant_programs, valid_programs

R_GRID = np.concatenate([[0.0], np.logspace(-9, 2, 400)])


def hull_lines(var, ref):
    """(x, d) Pareto vertices; enough to evaluate T exactly."""
    x = ref.abs().flatten().numpy().astype(np.float64)
    d = (var - ref).abs().flatten().numpy().astype(np.float64)
    # pareto prefilter for directions (-r, +1), r >= 0: scanning x ascending,
    # keep strict d records. exact for this direction family.
    order = np.argsort(x, kind="stable")
    x, d = x[order], d[order]
    keep = d > np.concatenate([[-np.inf], np.maximum.accumulate(d)[:-1]])
    return x[keep], d[keep]


def env_on_grid(x, d):
    """T(r) = max_i(d_i - r*x_i) at every grid r. pareto set is small."""
    return (d[None, :] - R_GRID[:, None] * x[None, :]).max(axis=1)


def main():
    print("PREDICTION (CLAIM E5/E8): no separator at any of the four corners;")
    print("every interval certified; (F_any, G_some) least negative sup gap;")
    print("G_every(100) <= 0 and G_some(100) <= 0, closing the tail.")
    print()

    by_valid = {}
    for name, run in valid_programs():
        for i in range(DRAWS):
            var, ref = run(9000 + i * 31)
            by_valid.setdefault(name, []).append(env_on_grid(*hull_lines(var, ref)))
    V = {n: np.stack(es) for n, es in by_valid.items()}
    F_all = np.max(np.concatenate(list(V.values())), axis=0)
    F_any = np.max(np.stack([np.min(es, axis=0) for es in V.values()]), axis=0)
    n_valid = sum(len(es) for es in V.values())
    print(f"valid: {len(V)} programs x {DRAWS} draws = {n_valid} envelopes")
    print(f"  F_all(0)={F_all[0]:.3e}  F_any(0)={F_any[0]:.3e}")

    by_mut = {}
    for name, run in mutant_programs():
        for i in range(DRAWS):
            var, ref = run(9000 + i * 31)
            by_mut.setdefault(name, []).append(env_on_grid(*hull_lines(var, ref)))
    M = {n: np.stack(es) for n, es in by_mut.items()}
    E = np.concatenate(list(M.values()))
    G_every = np.min(E, axis=0)
    G_some = np.min(np.stack([np.max(es, axis=0) for es in M.values()]), axis=0)
    n_mut = len(E)
    print(f"mutants: {len(M)} instances x {DRAWS} draws = {n_mut} envelopes")
    print(f"  G_every(0)={G_every[0]:.3e}  G_some(0)={G_some[0]:.3e}")
    print()

    # G-side interval witness bounds, shared across corners.
    bound_every = np.maximum(E[:, :-1], E[:, 1:]).min(axis=0)
    bound_some = np.min(np.stack([np.maximum(es[:, :-1], es[:, 1:]).max(axis=0)
                                  for es in M.values()]), axis=0)

    corners = {
        "all_every": (F_all, G_every, bound_every),
        "all_some": (F_all, G_some, bound_some),
        "any_every": (F_any, G_every, bound_every),
        "any_some": (F_any, G_some, bound_some),
    }
    n_int = len(R_GRID) - 1
    out = {}
    ok = True
    for key, (F, G, bound) in corners.items():
        Fpos = np.maximum(F, 0.0)
        gap = G - Fpos
        sup_gap = float(gap.max())
        sep = int((gap > 0).sum())
        cert = int((bound <= np.maximum(F[1:], 0.0)).sum())
        out[key] = {"sup_gap": sup_gap, "separators": sep, "certified": cert}
        ok = ok and sep == 0 and cert == n_int
        print(f"{key:>10}: separators {sep}/{len(R_GRID)}  sup gap {sup_gap:+.3e}"
              f"  certified {cert}/{n_int}")

    tail_closed = bool(G_every[-1] <= 0.0 and G_some[-1] <= 0.0)
    print()
    print(f"tail: G_every(100)={G_every[-1]:.3e}  G_some(100)={G_some[-1]:.3e}"
          f"  closed={tail_closed}")
    verdict = ok and tail_closed
    print("VERDICT: " + ("no (atol>=0, rtol>=0) tolerance separates the recorded"
                         " corpus at any corner" if verdict
                         else "PREDICTION FALSIFIED OR INCOMPLETE - report before use"))

    Path("results").mkdir(exist_ok=True)
    with open("results/separability.json", "w") as f:
        json.dump({"r_grid": R_GRID.tolist(),
                   "F": F_all.tolist(), "F_any": F_any.tolist(),
                   "G": G_every.tolist(), "G_some": G_some.tolist(),
                   "every": out["all_every"], "some": out["all_some"],
                   "corners": out,
                   "tail": {"G_every_at_100": float(G_every[-1]),
                            "G_some_at_100": float(G_some[-1]),
                            "closed": tail_closed},
                   "corpus": {"valid_programs": len(V), "mutant_instances": len(M),
                              "draws_per_program": DRAWS},
                   "intervals_total": n_int}, f)
    print("raw -> results/separability.json")


if __name__ == "__main__":
    sys.exit(main())
