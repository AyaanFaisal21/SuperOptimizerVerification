"""the measurement harness.

takes (transformation, shape, precision, inputs) and returns the error record for
that cell. everything downstream -- the phase 4 sweep, the phase 5 seeded
experiment, the headline table -- is this function called in a loop.

three quantities per cell, and the distinction between them is the whole result:

    floor        = err(baseline @ precision  vs  float64 truth)
    total        = err(variant  @ precision  vs  float64 truth)
    differential = err(variant  @ precision  vs  baseline @ precision)

`differential` is the gate the systems actually apply -- Axon §4.6 checks
|emitted_code - reference| <= atol + rtol|reference| against a *reference
implementation*, not against a high-precision oracle, and Mirage and Prism compare
candidate against input program the same way. `total` is CLAIM.md's as-registered
form, retained and co-reported so the amendment is auditable rather than trusted.

`floor` is what makes the reduced-precision cells readable at all. at bf16 a plain
matmul deviates from truth by ~3e-3 before anything is transformed, so a
differential of the same order says nothing about the transformation. d/floor is
the quantity that separates "this precision is inaccurate" from "this
transformation is inaccurate".

truth is computed on the *same rounded inputs* the low-precision run sees, so input
quantisation is excluded and only arithmetic error is measured.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import torch
from torch import Tensor

from .corpus import BY_NAME, SEED, Transformation

GATE = 1e-4          # rtol = atol, the tolerance the literature states
T1_FRAC = 0.01       # CLAIM.md T1: fails if >1e-4 relative on >= 1% of elements
DELTA = 1e-30        # divide-by-zero guard only; 20+ orders below any activation

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


# ---------------------------------------------------------------------------

@dataclass
class ErrorRecord:
    """one comparison, summarised. all arithmetic done in float64."""

    scale_rel: float        # max|a-b| / max|b| -- well posed under cancellation
    max_rel: float          # max per-element relative -- unbounded near zeros
    frac_over_gate: float   # fraction of elements with relative error > GATE
    p50: float
    p99: float
    gate_pass: bool         # the literal allclose predicate at rtol=atol=GATE

    @staticmethod
    def of(got: Tensor, ref: Tensor) -> "ErrorRecord":
        got, ref = got.double(), ref.double()
        diff = (got - ref).abs()
        rel = diff / (ref.abs() + DELTA)
        q = torch.quantile(rel.flatten().float(), torch.tensor([0.50, 0.99])).tolist()
        return ErrorRecord(
            scale_rel=(diff.max() / ref.abs().max()).item(),
            max_rel=rel.max().item(),
            frac_over_gate=(rel > GATE).double().mean().item(),
            p50=q[0],
            p99=q[1],
            gate_pass=bool((diff <= GATE + GATE * ref.abs()).all()),
        )


@dataclass
class Cell:
    """the record for one (transformation, shape, precision, inputs) point."""

    transformation: str
    shape_class: str
    shape: tuple
    precision: str
    input_kind: str
    seed: int
    device: str

    floor: ErrorRecord          # baseline @ precision vs truth
    total: ErrorRecord          # variant  @ precision vs truth   (as-registered)
    differential: ErrorRecord   # variant  vs baseline @ precision (the real gate)

    d_over_floor: float

    def verdict(self) -> dict:
        """the two gate readings and the two T1 readings, side by side.

        dual reporting is not decoration -- CLAIM.md's gate was amended after the
        claim was registered, so both definitions travel with every number.
        """
        return {
            "gate_axon": self.differential.gate_pass,          # corrected
            "gate_as_registered": self.total.gate_pass,        # original
            "gates_agree": self.differential.gate_pass == self.total.gate_pass,
            "t1_fail_differential": self.differential.frac_over_gate >= T1_FRAC,
            "t1_fail_total": self.total.frac_over_gate >= T1_FRAC,
        }

    def to_dict(self) -> dict:
        d = asdict(self)
        d["shape"] = list(self.shape)
        d["verdict"] = self.verdict()
        return d


# ---------------------------------------------------------------------------

def _widen(io: dict) -> dict:
    """same values, float64 container -- the reference sees exactly what the
    low-precision run saw, so input quantisation is not counted as error."""
    return {k: (v.double() if torch.is_floating_point(v) else v) for k, v in io.items()}


def measure(
    t: Transformation | str,
    shape_class: str = "mlp",
    precision: str = "fp32",
    inputs: dict[str, Tensor] | None = None,
    input_kind: str = "randn",
    seed: int = SEED,
    device: str = "cpu",
    shape: tuple | None = None,
) -> Cell:
    """one cell. `inputs` overrides generation -- that is the hook phase 3
    (activation fixtures) and phase 5 (seeded inputs) plug into."""
    if isinstance(t, str):
        t = BY_NAME[t]
    dt = DTYPES[precision]
    shape = shape or t.shapes[shape_class]

    io_p = inputs if inputs is not None else t.make_inputs(
        shape, dtype=dt, device=device, seed=seed
    )
    io_64 = _widen(io_p)

    truth = t.baseline(io_64)          # float64, on the rounded inputs
    base_p = t.baseline(io_p)          # at test precision
    var_p = t.variant(io_p)            # at test precision

    floor = ErrorRecord.of(base_p, truth)
    total = ErrorRecord.of(var_p, truth)
    differential = ErrorRecord.of(var_p, base_p)

    return Cell(
        transformation=t.name,
        shape_class=shape_class,
        shape=tuple(shape),
        precision=precision,
        input_kind=input_kind,
        seed=seed,
        device=device,
        floor=floor,
        total=total,
        differential=differential,
        d_over_floor=(differential.scale_rel / floor.scale_rel
                      if floor.scale_rel > 0 else float("inf")),
    )


def sweep(
    transformations=None,
    shape_classes=("small", "mlp", "attention"),
    precisions=("fp32", "fp16", "bf16"),
    device: str = "cpu",
    seed: int = SEED,
    progress: bool = False,
) -> list[Cell]:
    """the phase 4 cross product. returns raw cells; formatting is elsewhere."""
    from .corpus import CORPUS

    picked = [BY_NAME[x] if isinstance(x, str) else x
              for x in (transformations or CORPUS)]
    cells = []
    for t in picked:
        for cls in shape_classes:
            for p in precisions:
                if progress:
                    print(f"  {t.name}/{cls}/{p}", flush=True)
                cells.append(measure(t, cls, p, device=device, seed=seed))
    return cells


def dump(cells: list[Cell], path: str) -> None:
    """raw per-cell results to disk -- these are the artifact, not the table."""
    with open(path, "w") as f:
        json.dump([c.to_dict() for c in cells], f, indent=2)
