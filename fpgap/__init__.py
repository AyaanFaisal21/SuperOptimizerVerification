"""fpgap -- measuring the distance between what verified superoptimizers prove
and what their kernels execute.

see CLAIM.md for the registered claim and thresholds.
"""

from .accumulate import chunked_sum, seq_sum, tree_sum

__all__ = ["seq_sum", "chunked_sum", "tree_sum"]
