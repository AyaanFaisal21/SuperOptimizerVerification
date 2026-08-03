"""characterise the arithmetic of whatever machine this runs on.

every result in this project is a statement about floating-point behaviour, so the
machine's numerics are part of the measurement and have to be recorded alongside it.
this script produces that record. it is deliberately device-agnostic so the Mac and
the A10 emit comparable output.

nothing here trusts a documented default. the flags are read *and* the behaviour is
measured, because the failure mode that matters -- a silently reduced-precision
baseline -- looks exactly like a real result.

    python tools/probe_hardware.py [--device cuda]
"""

import argparse
import json
import platform
import sys

import torch

SEP = "-" * 68


def _rel(got, ref):
    """max |got - ref| / max |ref|, both compared at float64."""
    got, ref = got.double(), ref.double()
    return ((got - ref).abs().max() / ref.abs().max()).item()


def environment(device):
    env = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "device": device,
    }
    if device == "cuda":
        p = torch.cuda.get_device_properties(0)
        env |= {
            "gpu": p.name,
            "compute_capability": f"{p.major}.{p.minor}",
            "memory_GiB": round(p.total_memory / 2**30, 1),
            "cuda": torch.version.cuda,
        }
    else:
        env["cpu"] = platform.processor() or platform.machine()
    return env


def precision_flags():
    """documented defaults are not evidence; record what this build actually says."""
    f = {}
    b = torch.backends
    for path in (
        "cuda.matmul.allow_tf32",
        "cudnn.allow_tf32",
        "cuda.matmul.allow_fp16_reduced_precision_reduction",
        "cuda.matmul.allow_bf16_reduced_precision_reduction",
    ):
        obj = b
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            f[path] = bool(obj)
        except AttributeError:
            f[path] = None      # flag absent in this torch build
    return f


def tf32_check(device):
    """THE critical one on Ampere.

    TF32 carries a 10-bit mantissa. If it is on for matmul, the fp32 *baseline* is
    not fp32 and every matmul-shaped cell shows inflated fp32 error for a reason
    unrelated to the transformation -- which makes C1 look more true. Measured, not
    assumed: an fp32 matmul is compared against float64 on the same inputs.

    true fp32 lands near 1e-7; TF32 near 1e-3, four orders worse.
    """
    if device != "cuda":
        return None
    g = torch.Generator(device=device).manual_seed(0)
    a = torch.randn(1024, 1024, generator=g, device=device, dtype=torch.float32)
    b = torch.randn(1024, 1024, generator=g, device=device, dtype=torch.float32)
    ref = a.double() @ b.double()
    out = {}
    for setting in (False, True):
        torch.backends.cuda.matmul.allow_tf32 = setting
        out[f"allow_tf32={setting}"] = _rel(a @ b, ref)
    torch.backends.cuda.matmul.allow_tf32 = False        # leave it pinned off
    return out


def storage_rounding(device):
    """does elementwise accumulation round to the storage type at every step?

    the stagnation probe: 1 + n*(eps/4). each addend is below half an ulp of the
    running sum, so true in-type accumulation returns exactly 1.0 while any wider
    internal accumulator returns something larger. if this does NOT stagnate, the
    corpus is not measuring the precision it claims to.
    """
    out = {}
    for dt, eps in ((torch.float16, 9.77e-4), (torch.bfloat16, 7.81e-3)):
        n = 4096
        v = torch.cat([
            torch.ones(1, dtype=dt, device=device),
            torch.full((n,), float(eps / 4), dtype=dt, device=device),
        ])
        acc = torch.zeros((), dtype=dt, device=device)
        for i in range(v.shape[0]):
            acc = acc + v[i]
        out[str(dt).split(".")[-1]] = {
            "got": acc.item(),
            "exact": 1.0 + n * (eps / 4),
            "stagnates": acc.item() == 1.0,
        }
    return out


def matmul_accumulate(device):
    """narrow-dtype matmul error against exact arithmetic on the SAME rounded inputs.

    inputs are rounded to the narrow type first and the reference uses those exact
    values, so what is measured is the matmul's own arithmetic -- products,
    accumulation, and output rounding -- and not input quantisation.

    on Ampere the tensor-core MMA accumulates fp16/bf16 products in fp32 regardless;
    the reduced_precision_reduction flags govern the split-k reduction across
    partial tiles. both settings are measured rather than reasoned about.
    """
    out = {}
    g = torch.Generator(device=device).manual_seed(0)
    a32 = torch.randn(512, 4096, generator=g, device=device, dtype=torch.float32)
    b32 = torch.randn(4096, 512, generator=g, device=device, dtype=torch.float32)
    for dt, flag in ((torch.float16, "allow_fp16_reduced_precision_reduction"),
                     (torch.bfloat16, "allow_bf16_reduced_precision_reduction")):
        a, b = a32.to(dt), b32.to(dt)
        ref = a.double() @ b.double()          # exact, on the same rounded values
        entry = {}
        settings = (None,) if device != "cuda" else (True, False)
        for s in settings:
            if s is not None and hasattr(torch.backends.cuda.matmul, flag):
                setattr(torch.backends.cuda.matmul, flag, s)
            entry[f"{flag}={s}" if s is not None else "native"] = _rel(a @ b, ref)
        out[str(dt).split(".")[-1]] = entry
    return out


def determinism(device):
    """same op twice, bitwise. split-k schedules can vary run to run."""
    g = torch.Generator(device=device).manual_seed(0)
    a = torch.randn(512, 2048, generator=g, device=device, dtype=torch.float32)
    b = torch.randn(2048, 512, generator=g, device=device, dtype=torch.float32)
    return {"matmul_bitwise_repeatable": bool(torch.equal(a @ b, a @ b))}


def float64_support(device):
    """the reference lives in float64; confirm it exists and time it against fp32.

    A10 is 1:32 for fp64, so a slow ratio here is expected and is an argument for
    computing the reference host-side, not evidence of a problem.
    """
    import time
    out = {}
    try:
        g = torch.Generator(device=device).manual_seed(0)
        for dt in (torch.float32, torch.float64):
            x = torch.randn(1024, 1024, generator=g, device=device, dtype=dt)
            for _ in range(3):
                _ = x @ x
            if device == "cuda":
                torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(10):
                _ = x @ x
            if device == "cuda":
                torch.cuda.synchronize()
            out[str(dt).split(".")[-1] + "_ms"] = round((time.perf_counter() - t) * 100, 3)
        out["fp64_slowdown"] = round(out["float64_ms"] / out["float32_ms"], 1)
    except Exception as e:
        out["error"] = str(e)[:120]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--json", action="store_true", help="emit machine-readable only")
    args = ap.parse_args()
    dev = args.device

    report = {
        "environment": environment(dev),
        "flags_as_found": precision_flags(),
        "tf32": tf32_check(dev),
        "storage_rounding": storage_rounding(dev),
        "matmul_accumulate": matmul_accumulate(dev),
        "determinism": determinism(dev),
        "float64": float64_support(dev),
        "flags_as_left": precision_flags(),
    }

    if args.json:
        print(json.dumps(report, indent=2))
        return

    for section, body in report.items():
        print(f"\n{SEP}\n{section}\n{SEP}")
        print(json.dumps(body, indent=2))

    print(f"\n{SEP}\nverdict\n{SEP}")
    tf32 = report["tf32"]
    if tf32:
        off, on = tf32["allow_tf32=False"], tf32["allow_tf32=True"]
        print(f"  fp32 matmul vs float64: {off:.3e} with TF32 off, {on:.3e} with it on")
        print(f"  -> TF32 costs {on / off:.0f}x accuracy; it is left OFF. Any fp32")
        print("     number recorded with it ON would be a ~10-bit-mantissa baseline.")
    stag = report["storage_rounding"]
    ok = all(v["stagnates"] for v in stag.values())
    print(f"  narrow-dtype accumulation rounds in-type: {ok}"
          f"{'' if ok else '  <-- corpus would not measure the claimed precision'}")


if __name__ == "__main__":
    sys.exit(main())
