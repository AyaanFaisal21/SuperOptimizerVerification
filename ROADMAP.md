# Roadmap

**Definition of done:** a public repo containing reproducible code, raw results, a figure set, and a written report that answers C1 with numbers — regardless of which way the answer goes.

Scope discipline: one claim, one table, one figure set. Everything below that doesn't serve C1 is cut.

---

## The claim (write this into `CLAIM.md` and date it before writing any code)

> **C1.** Tensor-program transformations that are valid under real arithmetic — the class accepted by Axon's operator propagation and Prism's parallelization axioms — admit floating-point error exceeding the `rtol = atol = 1e-4` FP32 threshold used to validate them, and the gap widens under the reduced precisions (bf16/fp16) that production inference runs in.

> **A1.** No such transformation exists at realistic ML kernel shapes and activation distributions. Real-arithmetic proof plus 1e-4 FP32 sampling is empirically adequate, and reduced precision does not change the verdict.

**Pre-registered thresholds — fix these now, do not move them after seeing data:**
- A transformation *fails* if relative error exceeds `1e-4` on **≥1%** of output elements.
- "Realistic" inputs means activations sampled from a trained model, not `torch.randn`.
- Report the *rate* at which uniform random sampling catches failures, not merely whether failures exist.
- Soundness of any individual finding is checked against a float64 reference, not against the other variant.

---

## Phase 0 — Setup (½ day)

- `git init`, MIT license, `.gitignore`
- `CLAIM.md` with C1/A1 and today's date
- `NOTEBOOK.md` — dated entries, prediction before each run, outcome after. This is the artifact that proves you did science rather than curve-fitting.

**Done when:** the claim is committed with a date you cannot edit without it showing in history.

---

## Phase 1 — Transformation corpus (3–5 days)

Build a set of transformation pairs, each mathematically identical over ℝ, each drawn from what the three systems actually accept.

| Transformation | Source | Why it's in scope |
|---|---|---|
| Split reduction into *k* chunks, then combine | megakernel SM-splitting; Prism's `part`/`red`/`comb` | the single most common restructuring |
| Reassociation: `(a+b)+c` vs `a+(b+c)` at scale | Prism axioms; Mirage's A_eq | the base case |
| Scalar/broadcast multiply past matmul | Axon §4.2 worked example | Axon explicitly accepts this |
| LayerNorm: two-pass vs fused one-pass (sum + sumsq) | your own `autotune_layernorm.py` | you already wrote the numerically weaker variant |
| Softmax: naive vs online/streaming recurrence | your own `attn_ext.cu` | FlashAttention's core identity |
| Matmul with different K-loop tiling | Prism instantiation | changes accumulation order |

Two of these you have already implemented. That's not a coincidence — it's why this project fits you.

Each entry is a pair of Python functions plus a metadata record: which system accepts it, which paper's section, and the claimed justification.

**Done when:** ≥6 pairs, each verified identical under float64 within 1e-12.

---

## Phase 2 — Reference and precision harness (3–4 days)

- **Ground truth:** float64. For small cases, cross-check against `mpmath` at 50 digits to confirm float64 is itself adequate as truth.
- **Precisions under test:** fp32 and fp16 natively (both work on Turing sm_75). **bf16 is not native on your card** — simulate by rounding through `torch.bfloat16` and computing in fp32.
- **Validate the simulation:** run fp16 both natively *and* simulated, and show the two track each other. That justifies the bf16 simulation and it's the methodological move a reviewer will look for. If they diverge, say so and restrict claims to fp32/fp16.

**Limitation to state plainly in the writeup:** simulated bf16 rounds inputs and outputs but does not reproduce native bf16 tensor-core accumulation. Claims about bf16 are therefore about rounding-induced divergence, not about full hardware behavior.

**Done when:** the harness takes (transformation, shape, precision, input tensor) and returns per-element relative error against float64 truth.

---

## Phase 3 — Realistic inputs (2–3 days)

Dump real activations from your TransformerOp checkpoint — post-LayerNorm, post-GELU, attention scores, MLP intermediates. Store as a fixture set with the shapes and distributions recorded.

Then compare against `randn` on the same transformations. If the two give materially different error profiles, that is itself a finding: **the field validates on synthetic inputs and ships on real ones.**

**Done when:** a fixture file of real activation tensors, committed, with a script that regenerates it.

---

## Phase 4 — The main sweep (4–6 days, mostly compute)

Cross product: 6+ transformations × shapes (small / MLP-sized / attention-sized) × {fp32, fp16, bf16-sim} × {real activations, randn}.

For each cell record: max relative error, fraction of elements over 1e-4, error distribution, and whether the pair would pass Axon's `rtol=atol=1e-4` gate.

Runs on your 2060, or CPU. No rental, no clock.

**Done when:** raw per-cell results on disk, and the headline table exists.

---

## Phase 5 — The seeded-input experiment (3–4 days)

Ruler §6.2's lesson transferred: uniform random sampling missed edge cases in bitvector domains (*"even if x > 0 it is possible to have x·x = 0"*), and seeding with interesting constants found them.

The FP analogue: seed inputs with denormals, values near overflow, catastrophic-cancellation pairs, and wide dynamic ranges — while staying inside distributions a real model could produce.

**The question:** does adversarial-but-realistic seeding find failures that uniform sampling misses, and at what rate?

This is the sharper half of the paper. Phase 4 asks whether the gap exists; Phase 5 asks whether the *sampling method everyone uses* would find it.

**Done when:** a catch-rate comparison — uniform vs seeded — per transformation.

---

## Phase 6 — Writeup (5–7 days)

1. **Background** — already drafted in `BACKGROUND.md`
2. **Method** — corpus, reference, precisions, inputs, the bf16 simulation and its validation
3. **Results** — one headline table, error-distribution figures, the seeded-vs-uniform catch rate
4. **Threats to validity** — non-negotiable. Simulated bf16; one hardware platform; transformations reimplemented in torch rather than run through the actual systems; activation fixtures from one small model; float64-as-truth assumption
5. **What it means** — for each of the three systems specifically, not in general
6. **Related work** — the citation spine already exists

**Done when:** a reader who has not read Axon or Prism understands what was measured, why, and what the number means.

---

## Total: 4–6 weeks part-time, $0

## Kill criteria

Stop and reassess if:
- **Phase 1 stalls** — you cannot construct transformations both valid over ℝ and plausibly accepted by these systems. Then the corpus is the problem and the project needs different scope.
- **Phase 2's fp16 simulation doesn't track native** — restrict to fp32/fp16 and drop the bf16 claim rather than hand-wave it.
- **Phase 4 shows error 3+ orders of magnitude inside tolerance everywhere** — A1 wins decisively. Write it up short, publish it, move on. That outcome takes three weeks instead of six and is still worth having.

## What this is not

Not a claim that any of these systems is wrong. All three are explicit about operating over idealized arithmetic. The contribution is measuring what that costs, which nobody has done, and which each paper's own limitation section implicitly invites.
