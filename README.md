# The floating-point gap in verified tensor superoptimizers

Every verified superoptimizer proves equivalence over an exact arithmetic and
then ships a floating-point kernel. Mirage bounds its error probability over
finite fields; Prism reasons algebraically over ~70 hand-written axioms; Axon
proves over ℝ because the floating-point theory is 1650× slower. In all three,
the distance between what is proven and what executes is closed by random
testing — with no error bound, and no characterization of what the sampling
might miss.

That is a reasonable engineering choice. It is also an unmeasured one.

**This repo measures it.** The claim under test is [`CLAIM.md`](CLAIM.md),
registered 2026-08-03 with pre-registered thresholds, before any measurement code
was written. The result is informative either way: if transformation-induced
error stays well inside validation tolerance, the field's shortcut is justified
and that is worth establishing rather than assuming. If it does not, the gap is a
correctness concern rather than a precision footnote.

## Reading order

| File | What it is |
|---|---|
| [`CLAIM.md`](CLAIM.md) | The claim (C1), the alternative (A1), and the thresholds — fixed in advance |
| [`BACKGROUND.md`](BACKGROUND.md) | Why the gap exists, per system, with citations |
| [`ROADMAP.md`](ROADMAP.md) | Phases, exit criteria, and kill criteria |
| [`NOTEBOOK.md`](NOTEBOOK.md) | Dated entries — prediction before each run, outcome after |
| [`papers/README.md`](papers/README.md) | The six-paper working set and what to read in each |

## Status

Phase 0 complete — claim registered. Phase 1 (transformation corpus) next.

## License

MIT
