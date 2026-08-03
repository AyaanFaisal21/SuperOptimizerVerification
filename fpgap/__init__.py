"""fpgap -- measuring the distance between what verified superoptimizers prove
and what their kernels execute.

see CLAIM.md for the registered claim and thresholds.
"""

from .accumulate import chunked_sum, seq_sum, tree_sum
from .corpus import BY_NAME, CORPUS, Transformation

__all__ = ["CORPUS", "BY_NAME", "Transformation", "seq_sum", "chunked_sum", "tree_sum"]
