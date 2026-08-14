"""AUDIT step 3: re-measure the reduction pairs against a tree-order reference.

the corpus baselines use sequential summation. production reference kernels
reduce in blocked or tree order, so F4's "variant is 15x closer to the oracle
than its reference" may partly measure the weakness of a sequential reference.
this pins the reference to strided-tree order and re-measures.

    python tools/run_tree_baseline.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fpgap.accumulate import chunked_sum, seq_sum, tree_sum
from fpgap.harness import DTYPES, ErrorRecord
from fpgap.corpus import _randn


def softmax_ref(x, sumfn):
    m = x.amax(dim=-1, keepdim=True)
    e = torch.exp(x - m)
    return e / sumfn(e, -1).unsqueeze(-1)


def softmax_online(x, chunk=64):
    n = x.shape[-1]
    m = torch.full(x.shape[:-1], float("-inf"), dtype=x.dtype)
    s = torch.zeros(x.shape[:-1], dtype=x.dtype)
    for start in range(0, n, chunk):
        blk = x[..., start:start + chunk]
        m_new = torch.maximum(m, blk.amax(dim=-1))
        corr = torch.exp(m - m_new)
        corr = torch.where(torch.isinf(m), torch.zeros_like(corr), corr)
        s = s * corr + seq_sum(torch.exp(blk - m_new.unsqueeze(-1)), dim=-1)
        m = m_new
    return torch.exp(x - m.unsqueeze(-1)) / s.unsqueeze(-1)


def cellrow(label, ref_p, var_p, oracle):
    floor = ErrorRecord.of(ref_p, oracle)
    total = ErrorRecord.of(var_p, oracle)
    diff = ErrorRecord.of(var_p, ref_p)
    print(f"{label:<34}{floor.scale_rel:>10.2e}{total.scale_rel:>10.2e}"
          f"{total.scale_rel/floor.scale_rel:>8.2f}{diff.scale_rel:>10.2e}"
          f"{'pass' if diff.gate_pass else 'FAIL':>6}")


def main():
    print("PREDICTIONS (before run): tree reference shrinks softmax's 15x")
    print("  advantage to <=2x, possibly parity; reduction differentials vs a")
    print("  tree reference shrink several-fold vs the seq reference; gate")
    print("  verdicts at fp16/bf16 stay FAIL.")
    print()
    hdr = f"{'case':<34}{'floor':>10}{'total':>10}{'t/f':>8}{'diff':>10}{'gate':>6}"

    for prec in ("fp16", "bf16"):
        dt = DTYPES[prec]
        print(f"--- softmax, mlp shape (512,1024), {prec} ---")
        print(hdr)
        x = _randn((512, 1024), dt)
        x64 = x.double()
        oracle = softmax_ref(x64, tree_sum)
        for label, ref in (("seq reference (as in paper)", seq_sum),
                           ("tree reference (this run)", tree_sum)):
            cellrow(f"online vs {label}",
                    softmax_ref(x, ref).double(),
                    softmax_online(x).double(), oracle)
        print()

    for prec in ("fp16", "bf16"):
        dt = DTYPES[prec]
        print(f"--- split_reduction (chunked k=32), mlp shape (512,3072), {prec} ---")
        print(hdr)
        x = _randn((512, 3072), dt)
        oracle = seq_sum(x.double(), -1)
        var = chunked_sum(x, 32, -1).double()
        cellrow("chunked vs seq reference", seq_sum(x, -1).double(), var, oracle)
        cellrow("chunked vs tree reference", tree_sum(x, -1).double(), var, oracle)
        print()


if __name__ == "__main__":
    sys.exit(main())
