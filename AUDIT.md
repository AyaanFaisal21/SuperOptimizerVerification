# Method audit: 2026-08-04

This document uses the rules of ASD-STE100 (Simplified Technical English).
Sentences are short. Each sentence gives one idea. Each instruction is one step.

We examined the full method for errors of logic, approach, and calculation.
We measured three suspected errors before we wrote this document.
The measurements are in `NOTEBOOK.md` (entry 2026-08-04).

---

## Part 1: Findings

**Finding 1: the catch rate changes with the input scale.**
The Phase 5 "uniform" inputs had a scale of 2.7.
The Phase 4 inputs had a scale of 1.0.
We measured the catch rate at the two scales.
At scale 2.7, the rate is 14 of 100 tests.
At scale 1.0, the rate is 0 of 100 tests.
Our reports did not give the scale.
That was an error.
The scale of 2.7 is the measured scale of real activations.
Thus the 14% result is correct for that scale only.
This finding agrees with the main result: output magnitude controls the gate.

**Finding 2: the severity numbers were too large.**
We reported gate violations of 1600 times the tolerance.
That number was a ratio with a very small denominator.
We measured the true gate violation for a cancellation test.
The true violation is approximately 2 times the tolerance.
The 1600 number does not show severity.
It shows a small denominator.

**Finding 3: our base kernel is not the field's reference kernel.**
Our base kernel adds values one by one.
Library kernels add values in a tree order.
The field compares a new kernel with a library kernel.
Our comparison can show larger differences than the field's comparison.
We did not measure this effect.
The reduction pairs have this risk.
The matmul pairs do not have this risk.

**Finding 4: the two-computer claim was about a statistic, not about bits.**
We wrote that the two computers gave identical results.
The equal values were summary statistics with 5 digits.
We did not compare the output tensors bit by bit.
Also, the CPU bf16 matmul does not agree with our two simple models of it.
It is not equal to fp32 accumulation with one final rounding.
It is not equal to bf16 accumulation at each step.
The cause of the cross-platform agreement is not known.

**Finding 5: the softmax explanation was wrong.**
We wrote that the tolerance is larger than the usual softmax output.
The usual output is 0.00059 at row length 1024.
The tolerance is 0.0001.
The output is approximately 6 times larger than the tolerance.
The correct cause of the immunity is different.
The relative error (1e-6) is 100 times less than rtol (1e-4).
Softmax outputs are not more than 1.
Thus the gate cannot fail for softmax at fp32.

**Finding 6: the rates had no confidence limits.**
The rate 14 of 100 has 95% limits of 9% to 22%.
We did not give limits for any rate.
Also, we wrote "five times in six" for 86%.
The correct words are "six times in seven".

**Finding 7: one sentence about the systems was not verified.**
Some text says that the systems test one time.
We did not verify the number of test inputs in Axon or in Prism.
We must remove that text or change it.

**Finding 8: some evidence is narrow.**
The "3 times" spread of relative error came from one seed and one shape for each pair.
The 50-digit truth check used small shapes only.
The activation data came from one small model and one batch.
The seed range came from one activation site.

**Finding 9: "d/floor tracks the floor" says too much.**
The value d/floor is a ratio of two maximum values.
The two maximum values can come from different elements.
The ratio compares magnitudes.
It does not show element-level agreement.

---

## Part 2: Remaining steps

**Step 1.** Add the input scale to each catch-rate sentence. Status: **DONE** (this commit)

**Step 2.** Add the 95% confidence limits to each rate. Use the Wilson method. Status: **DONE** for the headline rate; **OPEN** for the Phase 5 table

**Step 3.** Change the base kernel to a tree kernel for the two reduction pairs.
Measure the differentials again. Record the difference. Status: **OPEN**

**Step 4.** Measure the matmul accumulator on the two computers.
Compare the output tensors bit by bit. Then correct the platform text. Status: **PART DONE**
The CPU half is measured (2026-08-04, `NOTEBOOK.md`).
The CPU matmul agrees with fp32 accumulation for 99.4% (fp16) and 99.96% (bf16) of elements.
Thus the CPU does not accumulate in the narrow type for matmul.
The earlier platform text said the opposite. It is corrected.
The GPU half needs a CUDA computer. The instance does not answer.

**Step 5.** Measure the catch rate at scales 0.5, 1.0, 2.0, 2.7, and 4.0.
Make a plot of rate against scale. Status: **OPEN**

**Step 6.** Do the K sweep for d/floor with M and N constant.
Find the cause of the 4.01 value. Status: **OPEN** (ERRATA §4)

**Step 7.** Correct the severity numbers and the softmax sentences in the
current documents. Point to this audit. Status: **DONE** (this commit)

**Step 8.** Remove the unverified sentence about one-time tests from the
current documents. Status: **DONE** (this commit)

**Step 9.** Write the report (Phase 6). Use the corrected numbers only.
Include this audit as a section. Status: **OPEN**

**Step 10.** Get an arXiv endorsement. Then submit the report. Status: **OPEN**

**Step 11.** The model checkpoint was only on the rented computer.
That computer does not answer.
If a GPU computer is rented again, copy the checkpoint first.
If the checkpoint is lost, train the model again with the same script.
Then record that the new fixtures are equivalent, but not bit-equal. Status: **OPEN**

Corrections go forward, never by edits to old entries. Full verbose versions of
all documents stay in git history. This is the rule of the repo.
