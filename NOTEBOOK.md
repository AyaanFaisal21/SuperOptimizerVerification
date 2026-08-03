# Notebook

Dated entries. **Prediction before each run, outcome after.** The prediction is
written and committed before the run executes — that ordering is the whole point
of the file, and it is visible in git history whether or not it was honored.

Format:

```
## YYYY-MM-DD — short title

**Setup.** What is being run, on what.
**Prediction.** What I expect, with a number where possible. Written first.
**Outcome.** What happened.
**Read.** What it means, including "nothing" if that is the answer.
```

---

## 2026-08-03 — Phase 0, project registered

**Setup.** `cornfieldV2` as the repo root; it already held the six-paper working
set (`papers/README.md` is the committed index, PDFs gitignored). Consolidated
`BACKGROUND.md` and `ROADMAP.md` in from `~/Documents/fp-verification-gap/`, which
was sitting inside the home-directory git repo and would never have been
separately publishable. `CLAIM.md` written and dated before any measurement code.

Toolchain on this machine: Python 3.13.1, torch 2.8.0, numpy 2.3.1, mpmath 1.3.0.
**No CUDA** — `torch.cuda.is_available()` is False; MPS only. The 2060 (Turing
sm_75) named in the roadmap is a different box, the Windows one that
`cornfield/winbuild.bat` targets.

**Prediction.** None — this is setup, nothing is being measured. Recording it so
the first real entry has a baseline to point at.

**Outcome.** Phase 0 complete. Repo has a dated claim in history.

**Read.** One thing that surfaced early and matters later: Phases 0–1 are pure
float64 CPU work and are unaffected by the missing GPU, but **Phase 2's fp16
native-vs-simulated validation gate assumes CUDA**. On CPU, torch's float16 ops
frequently upcast to fp32 internally, which would make "native fp16" here itself a
kind of simulation — and comparing a simulation against a simulation proves
nothing. That gate has to run on the 2060, or the bf16 claim gets dropped per the
roadmap's own kill criterion. Not a blocker now; flagged at the Phase 1→2 boundary
so it is decided deliberately rather than discovered.

---
