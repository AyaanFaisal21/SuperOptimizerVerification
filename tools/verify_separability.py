"""V1: independent recomputation of E5/E8, straight from the definitions.

registered in CLAIM.md (2026-08-15, fifth amendment) with predictions in
NOTEBOOK.md before the first run. no Pareto prefilter and no shared
envelope code with tools/run_separability.py: every envelope value is a
direct max over every recorded element at every grid point, and the class
envelopes, sup gaps, separator counts, and interval certificates are
rebuilt from the Method-section definitions, then compared against
results/separability.json.

    python tools/verify_separability.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

R_GRID = np.concatenate([[0.0], np.logspace(-9, 2, 400)])


def env_direct(x, d, grid=None, chunk=64):
    """T(r) = max_i(d_i - r*x_i) by direct evaluation over every element."""
    grid = R_GRID if grid is None else np.asarray(grid, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)
    out = np.empty(len(grid))
    for i in range(0, len(grid), chunk):
        r = grid[i : i + chunk]
        out[i : i + chunk] = (d[None, :] - r[:, None] * x[None, :]).max(axis=1)
    return out


def valid_envelopes(per_program):
    """{name: (draws, n_r)} -> F_all (max over every draw), F_any (max over
    programs of the min over each program's draws)."""
    F_all = np.max(np.concatenate(list(per_program.values())), axis=0)
    F_any = np.max(np.stack([np.min(es, axis=0) for es in per_program.values()]),
                   axis=0)
    return F_all, F_any


def mutant_envelopes(per_program):
    """{name: (draws, n_r)} -> G_every (min over every draw), G_some (min over
    instances of the max over each instance's draws)."""
    G_every = np.min(np.concatenate(list(per_program.values())), axis=0)
    G_some = np.min(np.stack([np.max(es, axis=0) for es in per_program.values()]),
                    axis=0)
    return G_every, G_some


def mutant_bounds(per_mut):
    """interval witness bounds: per-interval chord bound of the best witness
    draw (every) or witness instance max-envelope (some)."""
    E = np.concatenate(list(per_mut.values()))
    bound_every = np.maximum(E[:, :-1], E[:, 1:]).min(axis=0)
    bound_some = np.min(np.stack([np.maximum(es[:, :-1], es[:, 1:]).max(axis=0)
                                  for es in per_mut.values()]), axis=0)
    return bound_every, bound_some


def corner(F, G, bound):
    """separator and certificate counts for one (F, G) pairing."""
    Fpos = np.maximum(F, 0.0)
    gap = G - Fpos
    return {"sup_gap": float(gap.max()),
            "separators": int((gap > 0).sum()),
            "certified": int((bound <= np.maximum(F[1:], 0.0)).sum())}


def main():
    from tools.run_frontier import DRAWS, mutant_programs, valid_programs

    print("PREDICTION (V1): every envelope value matches results/separability.json")
    print("to 1e-12 absolute; identical separator, certificate, and sup-gap")
    print("numbers at all four corners.")

    def collect(programs):
        per = {}
        for name, run in programs():
            for i in range(DRAWS):
                var, ref = run(9000 + i * 31)
                x = ref.abs().flatten().numpy()
                d = (var - ref).abs().flatten().numpy()
                per.setdefault(name, []).append(env_direct(x, d))
        return {n: np.stack(es) for n, es in per.items()}

    V = collect(valid_programs)
    M = collect(mutant_programs)
    F_all, F_any = valid_envelopes(V)
    G_every, G_some = mutant_envelopes(M)
    bound_every, bound_some = mutant_bounds(M)

    stored = json.load(open("results/separability.json"))
    dev = {"F": np.abs(F_all - np.array(stored["F"])).max(),
           "F_any": np.abs(F_any - np.array(stored["F_any"])).max(),
           "G": np.abs(G_every - np.array(stored["G"])).max(),
           "G_some": np.abs(G_some - np.array(stored["G_some"])).max()}
    print("max abs envelope deviation vs stored:",
          {k: f"{v:.2e}" for k, v in dev.items()})

    corners = {"all_every": corner(F_all, G_every, bound_every),
               "all_some": corner(F_all, G_some, bound_some),
               "any_every": corner(F_any, G_every, bound_every),
               "any_some": corner(F_any, G_some, bound_some)}
    ok = all(v < 1e-12 for v in dev.values())
    for key, c in corners.items():
        s = stored["corners"][key]
        match = (c["separators"] == s["separators"]
                 and c["certified"] == s["certified"]
                 and abs(c["sup_gap"] - s["sup_gap"]) < 1e-12)
        ok = ok and match
        print(f"{key:>10}: sep {c['separators']} cert {c['certified']} "
              f"sup {c['sup_gap']:+.3e}  {'MATCH' if match else 'MISMATCH'}")
    print("VERDICT:", "independent recomputation agrees; E5/E8 verified"
          if ok else "DISAGREEMENT - report before use")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
