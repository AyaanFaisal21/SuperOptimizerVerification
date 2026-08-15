"""separability machinery: Pareto exactness and known-answer cases (V1).

registered in NOTEBOOK.md 2026-08-15 before the first run. the pipeline
functions under test come from tools/verify_separability.py, the
independent implementation; test_pareto_filter_is_lossless additionally
pins tools/run_separability.py's Pareto path to the direct evaluation.
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.run_separability import env_on_grid, hull_lines
from tools.verify_separability import (corner, env_direct, mutant_bounds,
                                       mutant_envelopes, valid_envelopes)


def _pair(x_signed, d):
    ref = torch.tensor(x_signed, dtype=torch.float64)
    var = ref + torch.tensor(d, dtype=torch.float64)
    return var, ref


def test_pareto_filter_is_lossless():
    rng = np.random.default_rng(0)
    for _ in range(25):
        n = int(rng.integers(2, 400))
        signs = rng.choice([-1.0, 1.0], size=n)
        x = np.abs(rng.normal(size=n)) * rng.choice([0.0, 1.0], size=n, p=[0.1, 0.9])
        d = np.abs(rng.normal(size=n)) * 1e-4
        var, ref = _pair(x * signs, d)
        x_eff = ref.abs().flatten().numpy()
        d_eff = (var - ref).abs().flatten().numpy()
        assert np.array_equal(env_on_grid(*hull_lines(var, ref)),
                              env_direct(x_eff, d_eff))


def _stats(V, M):
    F_all, F_any = valid_envelopes(V)
    G_every, G_some = mutant_envelopes(M)
    be, bs = mutant_bounds(M)
    return {"all_every": corner(F_all, G_every, be),
            "all_some": corner(F_all, G_some, bs),
            "any_every": corner(F_any, G_every, be),
            "any_some": corner(F_any, G_some, bs)}


def test_known_separator_found_everywhere():
    # valid needs shrinking slack; mutant needs 1e-3 at x=0 forever.
    V = {"v": np.stack([env_direct([1.0], [1e-5])])}
    M = {"m": np.stack([env_direct([0.0], [1e-3])])}
    for c in _stats(V, M).values():
        assert c["separators"] == 401 and c["certified"] == 0
        assert c["sup_gap"] > 9e-4


def test_known_nonseparator_fully_certified():
    V = {"v": np.stack([env_direct([0.0], [1e-3])])}
    M = {"m": np.stack([env_direct([0.0], [1e-5])])}
    for c in _stats(V, M).values():
        assert c["separators"] == 0 and c["certified"] == 400
        assert c["sup_gap"] < 0


def test_hidden_separator_lands_uncertified():
    # on the coarse grid {0, 1} a separator exists only strictly inside the
    # interval; the machinery must refuse to certify rather than miss it.
    grid = np.array([0.0, 1.0])
    V = {"v": np.stack([env_direct([1.0], [5e-4], grid=grid)])}
    M = {"m": np.stack([env_direct([3e-4], [3e-4], grid=grid)])}
    c = _stats(V, M)["all_every"]
    assert c["separators"] == 0
    assert c["certified"] == 0


def test_order_statistics_hand_computed():
    grid = np.array([0.0, 1.0, 2.0])
    def const(v):
        return env_direct([0.0], [v], grid=grid)
    V = {"A": np.stack([const(1e-3), const(2e-3)]),
         "B": np.stack([const(5e-4), const(3e-3)])}
    M = {"C": np.stack([const(1e-4), const(4e-3)]),
         "D": np.stack([const(2e-4), const(5e-5)])}
    F_all, F_any = valid_envelopes(V)
    G_every, G_some = mutant_envelopes(M)
    assert np.allclose(F_all, 3e-3) and np.allclose(F_any, 1e-3)
    assert np.allclose(G_every, 5e-5) and np.allclose(G_some, 2e-4)
