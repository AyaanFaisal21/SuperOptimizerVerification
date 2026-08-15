"""E5: exact tolerance separability on the recorded corpus.

registered in CLAIM.md (2026-08-14, second amendment) before this ran.

per draw, acceptance at (atol, rtol) is atol + rtol*|b|_i >= |g-b|_i for every
element, so the draw's boundary is the envelope env(r) = max_i(d_i - r*x_i)
with d = |g-b|, x = |b|. env is a maximum of lines, so it is determined exactly
by the convex hull of the draw's (x, d) points. a separator exists iff some
rtol has max(F(rtol), 0) < G(rtol), where F = pooled valid envelope (accept all
valid draws needs atol >= F) and G = min over mutant draws (reject all mutant
draws needs atol < G; atol >= 0 forces G > 0).

between sampled rtols, no-separation is certified from convexity: each mutant
envelope lies below its chord, so e(r) <= max(e(r1), e(r2)) on [r1, r2], and F
is non-increasing, so F(r) >= F(r2). if min over mutant draws of
max(e(r1), e(r2)) <= max(F(r2), 0), no r in [r1, r2] separates.

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
    """(x, d) convex-hull vertices; enough to evaluate env exactly."""
    x = ref.abs().flatten().numpy().astype(np.float64)
    d = (var - ref).abs().flatten().numpy().astype(np.float64)
    # pareto prefilter for directions (-r, +1), r >= 0: scanning x ascending,
    # keep strict d records. exact for this direction family.
    order = np.argsort(x, kind="stable")
    x, d = x[order], d[order]
    keep = d > np.concatenate([[-np.inf], np.maximum.accumulate(d)[:-1]])
    return x[keep], d[keep]


def env_on_grid(x, d):
    """env(r) = max_i(d_i - r*x_i) at every grid r. pareto set is small."""
    return (d[None, :] - R_GRID[:, None] * x[None, :]).max(axis=1)


def main():
    print("PREDICTION (CLAIM E5): no separator under EITHER cross-draw criterion")
    print("(E6): every interval certified; both sup gaps < 0; SOME gap less negative.")
    print()

    F = np.full_like(R_GRID, -np.inf)
    valid_n = 0
    for name, run in valid_programs():
        for i in range(DRAWS):
            var, ref = run(9000 + i * 31)
            F = np.maximum(F, env_on_grid(*hull_lines(var, ref)))
            valid_n += 1
    print(f"valid envelopes pooled: {valid_n} draws   F(0)={F[0]:.3e}  F(100)={F[-1]:.3e}")

    mut_envs = []
    for name, run in mutant_programs():
        for i in range(DRAWS):
            var, ref = run(9000 + i * 31)
            mut_envs.append((name, env_on_grid(*hull_lines(var, ref))))
    G = np.min(np.stack([e for _, e in mut_envs]), axis=0)
    print(f"mutant envelopes: {len(mut_envs)} draws   G(0)={G[0]:.3e}  G(100)={G[-1]:.3e}")

    # E6, criterion SOME: each mutant caught on at least one draw. per mutant,
    # the binding quantity is the max over its draws (its worst draw must still
    # be rejected... no: SOME needs at least one draw rejected, i.e. atol below
    # that mutant's MAX draw envelope; all mutants simultaneously -> min over
    # mutants of max over draws.
    by_mut = {}
    for name, e in mut_envs:
        by_mut.setdefault(name, []).append(e)
    G_some = np.min(np.stack([np.max(np.stack(es), axis=0)
                              for es in by_mut.values()]), axis=0)
    print(f"criterion SOME: G_some(0)={G_some[0]:.3e}")

    Fpos = np.maximum(F, 0.0)
    gap = G - Fpos
    sup_gap = float(gap.max())
    sep_at_sample = int((gap > 0).sum())
    gap_some = G_some - Fpos
    sup_gap_some = float(gap_some.max())
    sep_some = int((gap_some > 0).sum())

    # interval certificates, both criteria. chord bound per draw; for SOME the
    # per-mutant bound is max over its draws of the chord bound (max of convex
    # functions lies below the max of their chords), then min over mutants.
    E = np.stack([e for _, e in mut_envs])            # (n_draws, n_r)
    M = [np.stack(es) for es in by_mut.values()]      # per mutant (draws, n_r)
    cert = cert_some = 0
    uncert = uncert_some = 0
    for i in range(len(R_GRID) - 1):
        if np.maximum(E[:, i], E[:, i + 1]).min() <= max(F[i + 1], 0.0):
            cert += 1
        else:
            uncert += 1
        bound_some = min(np.maximum(m[:, i], m[:, i + 1]).max() for m in M)
        if bound_some <= max(F[i + 1], 0.0):
            cert_some += 1
        else:
            uncert_some += 1
    print()
    print(f"EVERY: separators at samples {sep_at_sample}/{len(R_GRID)}  "
          f"sup gap {sup_gap:.3e}  certified {cert}/{len(R_GRID) - 1}")
    print(f"SOME:  separators at samples {sep_some}/{len(R_GRID)}  "
          f"sup gap {sup_gap_some:.3e}  certified {cert_some}/{len(R_GRID) - 1}")
    verdict = (sep_at_sample == 0 and cert == len(R_GRID) - 1
               and sep_some == 0 and cert_some == len(R_GRID) - 1)
    print(f"VERDICT: {'no (atol>=0, rtol<=100) separates the recorded corpus'
                     if verdict else 'PREDICTION FALSIFIED OR INCOMPLETE - report before use'}")

    Path("results").mkdir(exist_ok=True)
    with open("results/separability.json", "w") as f:
        json.dump({"r_grid": R_GRID.tolist(), "F": F.tolist(), "G": G.tolist(),
                   "G_some": G_some.tolist(),
                   "every": {"sup_gap": sup_gap, "separators": sep_at_sample,
                             "certified": cert},
                   "some": {"sup_gap": sup_gap_some, "separators": sep_some,
                            "certified": cert_some},
                   "intervals_total": len(R_GRID) - 1,
                   "draws_per_program": DRAWS}, f)
    print("raw -> results/separability.json")


if __name__ == "__main__":
    sys.exit(main())
