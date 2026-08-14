# measure.md — measurement queue for the M3

This file instructs the measurement pass on the M3.
The Windows machine has no Python. All items below run on the M3.
Delete this file in the final commit of the pass, or mark each item DONE.

## Ground rules

1. Pull `main` first. The tools this file names were patched on 2026-08-14.
2. Record the environment before anything else:
   `python tools/probe_hardware.py` and `python -c "import torch; print(torch.__version__, torch.get_num_threads())"`.
   Put both in the NOTEBOOK entry for the pass.
3. Error measurements do not fluctuate with system load. They are deterministic
   arithmetic. The real risks are torch version, thread count, and device.
   The committed records came from torch 2.8.0 on this machine, default threads.
   If the version now differs, say so in the NOTEBOOK before running.
4. P0 re-runs are deterministic (fixed seeds). Diff them against the committed
   records. If any number drifts, report the drift in the NOTEBOOK. If any
   VERDICT changes, stop and report before committing anything.
5. P1 runs are new measurements. Their predictions are written below.
   Copy the amendment block into CLAIM.md with the run date, commit it,
   then run. Never the other way around.
6. One commit per completed item. Terse messages. No co-author trailer.
7. No threshold moves. `rtol = atol = 1e-4` and T1-T4 are frozen.

## P0 — artifact completeness (no predictions needed; registered runs re-emitting records)

**P0.1 — regenerate the sweeps with the current harness.**
Why: the committed sweep JSONs predate the `acc_ratio` field (schema drift,
review #3). The paper's F5 ratios should be first-class in the records.
Run:
    python tools/run_sweep.py
    python tools/run_sweep_activations.py
Expect: identical numbers to the committed records plus the new field.
Commit: `results: sweeps regenerated with acc_ratio`.

**P0.2 — commit raw records for E2-E4.**
Why: F1's K sweep and F3's three-reference numbers currently exist only as
printed output. The paper says raw records are committed; make that true.
Run:
    python tools/run_k_extension.py
Expect: E2 reproduces 0/100 (K=2048), 48/100 (K=4096), 100/100 (K=11008).
E3 reproduces p_elem near 3.42e-05 and the iid prediction inside the CI.
Writes `results/k_extension.json`.
Commit: `results: E2-E4 raw records`.

**P0.3 — commit the tree-baseline records.**
Why: same reason as P0.2, for AUDIT step 3's re-measurement.
Run:
    python tools/run_tree_baseline.py
Writes `results/tree_baseline.json`.
Commit: `results: tree-baseline raw records`.

## P1 — registered extensions (copy the amendment below into CLAIM.md first)

Amendment block for CLAIM.md. Insert the run date. Commit before running.

    ### 2026-08-XX — C1-C3 registered (before any run)

    **C1 — reference sensitivity with draws.** 100 draws per precision
    (fp16, bf16) at the mlp shape (512, 1024). Prediction: at fp16 the
    online-softmax candidate fails against the sequential reference and
    passes against the tree and torch.sum references on >=95% of draws
    each, and the floor ordering seq > tree > torch.sum holds on >=95% of
    draws. At bf16 the candidate fails against all three references on
    >=95% of draws: the precision floor alone exceeds a 1e-4-class gate.

    **C2 — detection at activation scale.** The six-mutant arm at
    sigma = 2.67. Prediction: the five gross mutants stay at >=95%
    detection; ln_eps_to_std still evades on >=95% of draws. LayerNorm
    normalizes scale away, and larger row variance shrinks the eps shift
    relative to std.

    **C3 — rejection across K and scale.** The valid K-tiled matmul,
    M=N=64, fp32, 100 draws per cell. Axis 1: K in {512, 1024, 2048, 4096}
    at sigma = 2.67. Axis 2: sigma in {0.5, 1.0, 2.0, 2.67, 4.0} at K=512
    (AUDIT step 5). Prediction: rejection is non-decreasing in K and in
    sigma; at sigma = 2.67, K = 2048 is rejected on >=95% of draws; the
    sigma sweep at K = 512 reads 0/100 at sigma <= 1.0 and contains the
    committed 14% at sigma = 2.67 inside its CI.

    No prior threshold moves.

**P1.1 — C1.** Why: F3 (verdict and accuracy direction depend on the
reference) rests on one draw per reference. Reviewers will ask for CIs.
Run:
    python tools/run_reference_sensitivity.py
Writes `results/reference_sensitivity.json`.
Commit: `results: reference sensitivity, 100 draws (C1)`.

**P1.2 — C2.** Why: the detection arm ran at unit scale only, and the eps
bug's severity is input-dependent. This closes the "did you check detection
at realistic scale" question.
Run:
    python tools/run_mutants.py --sigma 2.67 --out results/mutant_detection_sigma267.json
Do not overwrite `results/mutant_detection.json`. That file is the
registered unit-scale record.
Commit: `results: mutant detection at activation scale (C2)`.

**P1.3 — C3.** Why: F1's onset (K=4096) is a unit-scale statement and F4's
14% is a K=512 statement at activation scale. This one grid joins them and
closes AUDIT step 5.
Run:
    python tools/run_scale_grid.py
Writes `results/scale_grid.json`.
Commit: `results: K-by-scale rejection grid (C3)`.

## P2 — open anomalies and stretch items (optional this pass)

**P2.1 — AUDIT step 6.** d/floor does not order by K (4.01 at K=512 vs 3.48
at K=768, confounded shapes; ERRATA §4). Controlled sweep: d/floor for
matmul_k_tiling at K in {128, 256, 512, 768, 1024, 2048}, M=N pinned at 64,
fp16, 10 draws per K, report mean and spread. Write the small runner
following harness conventions; register a prediction line in the NOTEBOOK
first (any prediction; the point is the discipline).

**P2.2 — pretrained activations.** F6's evidence base is one 6-layer
char-level model. One forward pass of GPT-2-small (huggingface
`transformers`, one batch of text, sites mirroring tools/dump_activations.py)
would upgrade F6 more than any other day of work. Only if time permits;
requires network for the weights. Keep the fixture small (rows x d_model
slices, `.clone()` before save; see ERRATA §3 on `.contiguous()`).

**Out of scope for the M3:** the GPU half of AUDIT step 4 (bitwise
cross-platform matmul) needs a CUDA machine. Do not attempt here.

## Report back

For each item: the command, the one-line outcome, drift vs committed records
(P0) or prediction held/failed (P1), and the commit hash. Put the pass
summary in one NOTEBOOK entry dated on the run day. If anything surprises,
stop at the surprise and report rather than absorbing it.
