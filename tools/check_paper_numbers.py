"""every number in PAPER.md, asserted against the committed records.

three verdicts per claim. PASS: recomputed or located in a committed
results file. PROSE: the source is a quoted line of the committed run log
(NOTEBOOK.md), named so a reader can find it. FAIL: the assertion does
not hold; fix the paper or the record before use.

    python tools/check_paper_numbers.py
"""

import json
import math
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
R = lambda p: json.load(open(ROOT / "results" / p))
NB = (ROOT / "NOTEBOOK.md").read_text()

rows = []
def check(name, cond, detail=""):
    rows.append((name, "PASS" if cond else "FAIL", detail))
def prose(name, needle):
    hit = needle in NB
    rows.append((name, "PROSE" if hit else "FAIL", f"NOTEBOOK {'has' if hit else 'MISSING'}: {needle!r}"))

def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)
def pct(k, n):
    lo, hi = wilson(k, n)
    return [round(lo * 100), round(hi * 100)]

# ---------------- F1 table and mechanism ----------------
e = R("k_extension.json"); sg = R("scale_grid.json"); gf = R("grid_fill.json")
sw = R("sweep_randn.json")
ka, sa = sg["k_axis"], sg["sigma_axis"]
check("F1 unit 512=0", sa["1"]["fails"] == 0)
check("F1 unit 1024=0", gf["unit_K1024"]["fails"] == 0)
check("F1 unit 2048/4096/11008 = 0/48/100",
      [e["e2"][k]["fails"] for k in ("2048", "4096", "11008")] == [0, 48, 100])
check("F1 s267 col 11/69/100/100 + grid-fill 100",
      [ka[k]["fails"] for k in ("512", "1024", "2048", "4096")]
      == [11, 69, 100, 100] and gf["s267_K11008"]["fails"] == 100)
check("F1 CI [38-58] for 48/100", pct(48, 100) == [38, 58])
check("F1 CI [96-100] for 100/100", pct(100, 100) == [96, 100])
check("F1 CI [0-4] for 0/100", pct(0, 100) == [0, 4])
check("F1 CI [6-19] for 11/100", pct(11, 100) == [6, 19])
check("F1 CI [59-77] for 69/100", pct(69, 100) == [59, 77])
check("F1 sigma axis 0,0,1,16,74",
      [sa[k]["fails"] for k in ("0.5", "1", "2", "2.67", "4")] == [0, 0, 1, 16, 74])
check("F1 onset near 1024 at s267", ka["512"]["fails"] < 50 <= ka["1024"]["fails"])
e3 = e["e3"]
check("F1 p_elem 3.42e-05", abs(e3["p_elem"] - 3.42e-5) < 5e-8)
check("F1 iid 13.1%", abs(e3["iid_prediction"] - 0.131) < 1e-3)
check("F1 iid inside observed CI", e3["ci"][0] <= e3["iid_prediction"] <= e3["ci"][1])
check("F1/F4 14/100 with CI [9-22]",
      e3["tensor_fail"] == 14 and pct(14, 100) == [9, 22])
prose("F1 one element in 4096, 750x below median",
      "1 bad element of 4096. Its magnitude is 750x below the tensor median")
kd = R("k_differential.json")
med = {k: sorted(r["max_diff"] for r in v)[10] for k, v in kd["per_k"].items()}
seq = [med[k] for k in ("512", "1024", "2048", "4096", "11008")]
check("F1/D1 median max|d| 6.1e-05 to 1.3e-03",
      abs(seq[0] - 6.1e-5) < 5e-7 and abs(seq[-1] - 1.34e-3) < 2e-5)
check("F1/D1 growth monotone, steps < 3.2x",
      all(1 < b / a < 3.2 for a, b in zip(seq, seq[1:])))
argm = {k: sorted(r["ref_at_argmax"] for r in v)[10] for k, v in kd["per_k"].items()}
medm = {k: sorted(r["ref_median"] for r in v)[10] for k, v in kd["per_k"].items()}
check("F1/D1 argmax-diff sits at large elements",
      all(argm[k] > medm[k] for k in argm))
# D1's registered third clause said the opposite and was logged as held;
# ERRATA 2.14. The assertion above is the measured direction.
check("F1/D1 argmax 2.9x-3.9x above the median",
      2.8 < min(argm[k] / medm[k] for k in argm)
      and max(argm[k] / medm[k] for k in argm) < 4.0,
      f"range [{min(argm[k]/medm[k] for k in argm):.2f}, "
      f"{max(argm[k]/medm[k] for k in argm):.2f}]")
# rev. 6: F1 now states the rejected variant is the more accurate side.
mmf32 = {r["shape_class"]: r for r in sw
         if r["transformation"] == "matmul_k_tiling" and r["precision"] == "fp32"}
tf = {k: v["total"]["scale_rel"] / v["floor"]["scale_rel"] for k, v in mmf32.items()}
check("F1 variant 2.0x-6.0x closer to oracle than its reference (fp32)",
      abs(tf["small"] - 0.500) < 5e-3 and abs(tf["mlp"] - 0.167) < 5e-3
      and abs(tf["attention"] - 0.262) < 5e-3,
      f"total/floor {tf['small']:.3f}, {tf['mlp']:.3f}, {tf['attention']:.3f}")

# ---------------- F2 ----------------
sep = R("separability.json")
c = sep["corners"]
check("F2 all-must-pass sup -4.0e-06", abs(c["all_some"]["sup_gap"] + 4.018e-6) < 1e-9)
check("F2 any-may-pass sup -4.2e-06", abs(c["any_every"]["sup_gap"] + 4.236e-6) < 1e-9)
check("F2 pessimal sup -3.5e-05", abs(c["all_every"]["sup_gap"] + 3.487e-5) < 1e-8)
check("F2 three corners 0 sep, 400/400 cert",
      all(c[k]["separators"] == 0 and c[k]["certified"] == 400
          for k in ("all_some", "any_every", "all_every")))
check("F2 fourth corner +9.5e-06, 110 samples",
      abs(c["any_some"]["sup_gap"] - 9.470e-6) < 1e-9
      and c["any_some"]["separators"] == 110)
check("F2 tails -5.5e-05 and -2.4e+02",
      abs(sep["tail"]["G_some_at_100"] + 5.469e-5) < 1e-8
      and abs(sep["tail"]["G_every_at_100"] + 243.1) < 0.1
      and sep["tail"]["closed"])
r_g = np.array(sep["r_grid"]); F_any = np.array(sep["F_any"]); G_s = np.array(sep["G_some"])
gap = G_s - np.maximum(F_any, 0.0)
idx = np.where(gap > 0)[0]
check("F2 band [7.8e-03, 7.9] contiguous",
      abs(r_g[idx[0]] - 7.8e-3) < 1e-4 and abs(r_g[idx[-1]] - 7.89) < 0.05
      and np.all(np.diff(idx) == 1))
j = int(np.argmin(np.abs(r_g - 0.056)))
check("F2 witness atol=1e-6 at rtol=0.056",
      max(F_any[j], 0.0) <= 1e-6 < G_s[j])
check("F2 corpus 7 programs, 16 instances, 50 draws",
      sep["corpus"] == {"valid_programs": 7, "mutant_instances": 16,
                        "draws_per_program": 50})
# rev. 6 additions. Method's non-degeneracy argument (ERRATA 2.17) and F2's
# plateau-margin sentence both quote envelope values; assert them directly.
F0, G0 = sep["F"][0], sep["G"][0]
check("Method mildest bug 4.1e-5 below valid corpus 5.0e-4 at rtol=0",
      abs(G0 - 4.101e-5) < 1e-8 and abs(F0 - 5.035e-4) < 1e-7 and G0 < F0,
      f"G(0)={G0:.3e} F(0)={F0:.3e}")
j1e4 = min(range(len(r_g)), key=lambda i: abs(r_g[i] - 1e-4))
check("F2 valid side needs atol 9.4e-5 at the grid rtol nearest 1e-4",
      abs(sep["F"][j1e4] - 9.394e-5) < 1e-7 and sep["F"][j1e4] < 1e-4,
      f"F={sep['F'][j1e4]:.3e} at rtol={r_g[j1e4]:.3e}")
tf = R("tolerance_frontier.json")
tot = {(r["atol"], r["rtol"]): r["valid_reject"] + r["mutant_miss"] for r in tf["rows"]}
plateau = {(1e-4, 1e-4), (1e-4, 1e-3), (1e-4, 1e-2)}
mn = min(tot.values())
check("F2 grid min 0.0625 exactly on the three plateau points",
      abs(mn - 0.0625) < 1e-12 and {k for k, v in tot.items() if v == mn} == plateau)
check("F2 plateau: 0% valid reject",
      all(r["valid_reject"] == 0.0 for r in tf["rows"]
          if (r["atol"], r["rtol"]) in plateau))
check("F2 6.25% = 1 of 16 = 50 of 800", 0.0625 * 16 == 1 and 50 / 800 == 0.0625)
m267 = R("mutant_detection_sigma267.json")["results"]
check("F2 s267: five detected at 100/100, eps at 0",
      sum(v["detection_rate"] == 1.0 for v in m267.values()) == 5
      and m267["ln_eps_to_std"]["detection_rate"] == 0.0)
check("F2 eps divergence 2.46e-6 at s267",
      abs(m267["ln_eps_to_std"]["f64_divergence"] - 2.46e-6) < 1e-8)
md = R("mutant_detection.json")["results"]
check("F2/abstract eps evades at published constant (unit)",
      md["ln_eps_to_std"]["detection_rate"] == 0.0)
check("Method mutant divergences span 4.1e-6 to 8.2e-1",
      abs(min(v["f64_divergence"] for v in md.values()) - 4.1e-6) < 1e-7
      and abs(max(v["f64_divergence"] for v in md.values()) - 0.82) < 0.02)

# ---------------- F3 ----------------
smx = {r["precision"]: r for r in sw
       if r["transformation"] == "softmax_online" and r["shape_class"] == "mlp"}
fl16 = smx["fp16"]["floor"]["scale_rel"]; to16 = smx["fp16"]["total"]["scale_rel"]
check("F3 seq floor 8.9e-3 (sweep cell)", abs(fl16 - 8.938e-3) < 1e-5)
check("F3 online total 6.0e-4", abs(to16 - 6.025e-4) < 1e-6)
e4 = {(r["reference"], r["precision"]): r for r in e["e4"]}
check("F3 tree floor 9.7e-4", abs(e4[("tree", "fp16")]["floor"] - 9.693e-4) < 1e-6)
check("F3 torch.sum floor 3.7e-4",
      abs(e4[("torch.sum", "fp16")]["floor"] - 3.675e-4) < 1e-6)
check("F3 1.6x closer vs tree (single draw)",
      abs(e4[("tree", "fp16")]["floor"] / to16 - 1.61) < 0.02)
check("F3 1.6x farther vs torch.sum",
      abs(to16 / e4[("torch.sum", "fp16")]["floor"] - 1.64) < 0.02)
check("F3 15x closer vs seq", abs(fl16 / to16 - 14.8) < 0.2)
check("F3 bf16 ratios 1.92, 2.14",
      abs(e4[("tree", "bf16")]["t_over_f"] - 1.92) < 0.01
      and abs(e4[("torch.sum", "bf16")]["t_over_f"] - 2.14) < 0.01)
bf = smx["bf16"]
check("F3 bf16 seq ratio 0.02",
      0.015 < bf["total"]["scale_rel"] / bf["floor"]["scale_rel"] < 0.025)
rs = R("reference_sensitivity.json")["results"]
check("F3 pass rates 98, 99, fail 100 (fp16)",
      rs["fp16"]["tree"]["gate_fail"] == 2 and rs["fp16"]["torch.sum"]["gate_fail"] == 1
      and rs["fp16"]["seq"]["gate_fail"] == 100)
check("F3 CIs [93-99], [95-100], [96-100]",
      pct(98, 100) == [93, 99] and pct(99, 100) == [95, 100]
      and pct(100, 100) == [96, 100])
check("F3 bf16 fails all three 100/100",
      all(rs["bf16"][k]["gate_fail"] == 100 for k in ("seq", "tree", "torch.sum")))
check("F3 ordering 80/100 fp16, 85/100 bf16",
      rs["fp16"]["ordering_seq_gt_tree_gt_torchsum"]["holds"] == 80
      and rs["bf16"]["ordering_seq_gt_tree_gt_torchsum"]["holds"] == 85)
for prec in ("fp16", "bf16"):
    m = rs[prec]
    check(f"F3 mean floors seq>tree>torch.sum ({prec})",
          m["seq"]["floor_mean"] > m["tree"]["floor_mean"] > m["torch.sum"]["floor_mean"])
check("F3 100-draw mean ratios: tree 1.7x farther, torch.sum 2.3x, seq 11x closer",
      abs(rs["fp16"]["tree"]["t_over_f_mean"] - 1.73) < 0.01
      and abs(rs["fp16"]["torch.sum"]["t_over_f_mean"] - 2.35) < 0.01
      and abs(1 / rs["fp16"]["seq"]["t_over_f_mean"] - 10.7) < 0.1)
osc = R("oracle_spotcheck.json")
check("F3/D2 oracle deviation < 1e-6 (worst 1.7e-13)",
      max(r["worst_rel_deviation"] for r in osc["rows"]) < 1e-6)

# ---------------- F4 ----------------
check("F4 replication 16/100 [10-24]",
      sa["2.67"]["fails"] == 16 and pct(16, 100) == [10, 24])
check("F4 0.86^20 = 4.9%", abs(0.86 ** 20 - 0.049) < 0.001)
up = wilson(0, 100)[1]
check("F4 3.7% upper bound; 97.7% at 100 draws",
      abs(up - 0.037) < 0.0005 and abs(1 - (1 - up) ** 100 - 0.977) < 0.005)

# ---------------- F5 ----------------
red = [r for r in sw if r["precision"] in ("fp16", "bf16")]
check("F5 36 reduced-precision cells", len(red) == 36)
check("F5 all 36 fail the 1e-4 rule",
      all(not r["differential"]["gate_pass"] for r in red))
cand = {}
for r in red:
    if r["transformation"] in ("split_reduction", "reassociation", "softmax_online"):
        cand[(r["transformation"], r["shape_class"], r["precision"])] = (
            r["floor"]["scale_rel"] / r["total"]["scale_rel"])
lo, hi = min(cand.values()), max(cand.values())
check("F5 three reordering pairs 3x to 73x closer", 3.0 < lo < 3.3 and 72 < hi < 73.5,
      f"actual [{lo:.1f}, {hi:.1f}]")
mlp = [v for k, v in cand.items() if k[1] == "mlp"]
check("F5 8x to 62x at mlp shapes", 8.2 < min(mlp) < 8.5 and 61 < max(mlp) < 62.5,
      f"actual [{min(mlp):.1f}, {max(mlp):.1f}]")
mm = {r["precision"]: r for r in sw
      if r["transformation"] == "matmul_k_tiling" and r["shape_class"] == "mlp"}
check("F5 untransformed matmul floor 3.91e-4 fp16, 3.45e-3 bf16 (mlp)",
      abs(mm["fp16"]["floor"]["scale_rel"] - 3.913e-4) < 2e-6
      and abs(mm["bf16"]["floor"]["scale_rel"] - 3.447e-3) < 2e-5,
      f"actual {mm['fp16']['floor']['scale_rel']:.3e}, {mm['bf16']['floor']['scale_rel']:.3e}")

# ---------------- F6 ----------------
act = R("sweep_activations.json")
f32 = [r for r in act if r["precision"] == "fp32"]
check("F6 8 fp32 activation cells, all pass",
      len(f32) == 8 and all(r["differential"]["gate_pass"] for r in f32))
# rev. 6: the "10x headroom" claim is withdrawn (ERRATA 2.15). it was
# 1e-4/scale_rel, a tensor-scale ratio, on a rule F1 shows the tensor-scale
# statistic does not track. what the paper now claims is the verdict, plus
# the two cells where elements pass on atol alone.
elem = sorted(r["differential"]["max_rel"] for r in f32)[-2:]
check("F6 two cells reach 1.4e-1 and 2.4e-1 per-element relative",
      abs(elem[0] - 1.374e-1) < 1e-3 and abs(elem[1] - 2.351e-1) < 1e-3,
      f"top two {elem[0]:.3e}, {elem[1]:.3e}")
check("F6 those cells still pass (atol carries them)",
      all(r["differential"]["gate_pass"] for r in f32
          if r["differential"]["max_rel"] > 1e-2))
meta = json.load(open(ROOT / "fixtures" / "activations.json"))
p99 = max(meta[k]["cancel_row_p99"] for k in ("resid_pre_ln_L0", "resid_pre_ln_L5"))
check("F6 residual cancellation p99 <= 0.012", p99 <= 0.0122, f"p99 {p99:.4f}")
import torch
fx = torch.load(ROOT / "fixtures" / "activations.pt")
rmax = max(float((fx[k].double().mean(-1).abs()
                  / fx[k].double().abs().mean(-1)).max())
           for k in ("resid_pre_ln_L0", "resid_pre_ln_L5"))
check("F6 residual worst-row cancellation 0.16", 0.15 < rmax < 0.17,
      f"row max {rmax:.4f}")
prose("F6 condition 368,927; 400x; 11x", "row condition 368,927")


# ---------------- F7 ----------------
sc = R("seeded_catch_rates.json")["results"]
check("F7 softmax immune under every strategy",
      all(v["gate_fail_rate"] == 0.0 for v in sc["softmax_online"].values()))
canc = {k: v["cancellation"] for k, v in sc.items() if k != "softmax_online"}
check("F7 cancellation trips susceptible pairs on every draw",
      any(v["gate_fail_rate"] == 1.0 for v in canc.values()),
      str({k: v["gate_fail_rate"] for k, v in canc.items()}))
prose("F7 true gate exceedance 2.0x on cancellation inputs",
      "Cancellation gate exceedance: 2.0x true")
prose("F7 condition 4.5e6 vs 42.8, 10.4 sigma", "10.4")

# ---------------- Method / oracle chain ----------------
prose("Method 18 cells at 6e-16..3.5e-15", "18/18 cells at 6e-16 to 3.5e-15")
prose("Method mpmath worst drift 8.2e-16", "worst drift vs 50-digit mpmath 8.2e-16")

# ---------------- Threats: corpus composition ----------------
# the element counts in the threats section are derived from the corpus
# shapes, not produced by a run, so they are recomputed here rather than
# read from a record. valid side: the six pairs at mlp shape plus k2048.
sys.path.insert(0, str(ROOT))
from fpgap.corpus import BY_NAME

def _out_elems(name):
    s = BY_NAME[name].shapes["mlp"]
    if name in ("split_reduction", "reassociation"):
        return s[0]                       # one output per row
    if name in ("scalar_past_matmul", "matmul_k_tiling"):
        return s[0] * s[2]                # (m, n)
    return s[0] * s[1]                    # elementwise-shaped output

valid_elems = sum(_out_elems(n) for n in
                  ("split_reduction", "reassociation", "scalar_past_matmul",
                   "layernorm_variance", "softmax_online", "matmul_k_tiling"))
valid_elems += 64 * 64                    # the k2048 tiling cell
mutant_elems = (6 * 64                    # red_drop{1,2,4,8,16,32}
                + 4 * 64 * 64             # mm_drop{1,2,8,32}col
                + 3 * 64 * 1024           # ln_div_n-{1,2,8}
                + 3 * 64 * 1024)          # ln_eps{1e-5,1e-4,1e-3}
check("Threats valid side ~3.7M elements per draw",
      valid_elems == 3_675_136, f"{valid_elems}")
check("Threats mutant side ~0.4M elements per draw",
      mutant_elems == 409_984, f"{mutant_elems}")

# ---------------- paper text ----------------
# Everything above asserts record -> literal: the numbers this script
# names still match the committed data. That direction cannot catch an
# edit that changes a number in the paper and not here. This direction
# asserts literal -> paper text, so the two can no longer drift apart
# silently. Whitespace-normalized, because line wrapping is not content.
PAPER = re.sub(r"\s+", " ", (ROOT / "PAPER.md").read_text())

def present(name, s):
    rows.append((f"paper text: {name}", "PASS" if s in PAPER else "FAIL",
                 "" if s in PAPER else f"not in PAPER.md: {s!r}"))

for _name, _s in [
    ("F1 K=4096 rate", "48/100"), ("F1 K=11008 rate", "100/100"),
    ("F1 sigma axis", "0, 0, 1, 16, 74"), ("F1 accuracy ratio", "2.0x"),
    ("F1 median max|d| low", "6.1e-05"), ("F1 median max|d| high", "1.3e-03"),
    ("F2 all-draws-must-pass gap", "-4.0e-06"),
    ("F2 any-draw-may-pass gap", "-4.2e-06"),
    ("F2 pessimal gap", "-3.5e-05"), ("F2 fourth-corner margin", "+9.5e-06"),
    ("F2 separating band low end", "7.8e-03"), ("F2 miss fraction", "6.25%"),
    ("F2 valid side near rtol 1e-4", "9.4e-5"),
    ("Method mildest bug slack", "4.1e-5"),
    ("Method valid corpus slack", "5.0e-4"),
    ("F3 sequential floor", "8.9e-3"), ("F3 tree floor", "9.7e-4"),
    ("F4 per-draw rate", "14/100"), ("F4 replication", "16/100"),
    ("F4 third reading", "11/100"),
    ("F5 fp16 control floor", "3.91e-4"),
    ("F5 bf16 control floor", "3.45e-3"),
    ("F6 per-element relative a", "1.4e-1"),
    ("F6 per-element relative b", "2.4e-1"),
    ("F6 cancellation p99", "0.012"), ("F6 worst row", "0.16"),
    ("Threats valid elements/draw", "3.7M"),
    ("Threats mutant elements/draw", "0.4M"),
    ("Appendix A released FP_P", "167"), ("Appendix A released FP_Q", "83"),
    ("Appendix B rtol samples", "401"),
]:
    present(_name, _s)

# ---------------- report ----------------
w = max(len(n) for n, _, _ in rows)
fails = 0
for name, verdict, detail in rows:
    fails += verdict == "FAIL"
    print(f"{name:<{w}}  {verdict}" + (f"  {detail}" if detail and verdict != "PROSE" else ""))
prose_n = sum(1 for _, v, _ in rows if v == "PROSE")
print(f"\n{len(rows)} claims checked: {len(rows)-fails-prose_n} PASS, "
      f"{prose_n} PROSE (committed log), {fails} FAIL")
sys.exit(1 if fails else 0)
