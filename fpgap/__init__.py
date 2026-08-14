"""fpgap -- measuring the distance between what verified superoptimizers prove
and what their kernels execute.

four modules, in dependency order:

    accumulate.py   summation with the order pinned, never torch.sum -- the
                    primitive everything above it rests on
    corpus.py       six transformation pairs, each exactly equal over R, each
                    carrying which system accepts it and on what grounds
    harness.py      one (transformation, shape, precision, inputs) cell ->
                    floor / total / differential, plus both gate readings
    seeds.py        phase-5 input generators: adversarial arrangements of
                    values bounded inside the observed real activation range
    mutants.py      the detection arm: plausible wrong rewrites, each a real
                    bug in float64, complement to the valid corpus

tools/ holds the runnable experiments; each one's docstring carries its command.
CLAIM.md holds the registered claim, thresholds, and the dated verdict.
"""

from .accumulate import chunked_sum, seq_sum, tree_sum
from .corpus import BY_NAME, CORPUS, SEED, Transformation
from .harness import DTYPES, GATE, Cell, ErrorRecord, measure, sweep
from .seeds import REAL_BOX, STRATEGIES, generate

__all__ = [
    "seq_sum", "chunked_sum", "tree_sum",
    "CORPUS", "BY_NAME", "SEED", "Transformation",
    "measure", "sweep", "Cell", "ErrorRecord", "GATE", "DTYPES",
    "STRATEGIES", "generate", "REAL_BOX",
]
