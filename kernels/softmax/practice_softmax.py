# -*- coding: utf-8 -*-
"""
practice_softmax.py — Practice-use softmax benchmark
=====================================================
Use this for repeated practice after working through my_softmax.{cu,py}:
  - One best kernel per (algorithm, dtype) combo:
      * softmax_f32                   — naive, FP32 vec4
      * safe_softmax_f32              — safe, FP32 vec4
      * safe_softmax_f16_f32          — safe, FP16 in/out + FP32 acc, 128-bit pack
      * online_safe_softmax_f32       — online safe, FP32 vec4 single-pass
  - Two features only: check_correctness and benchmark
  - Plain kernel names (no x4 / _pack hints), so you re-derive the optimal form
  - Shape grid mirrors softmax.py (S=4096 across H sweep + (8192,8192))
"""

import argparse
import os
import time
from functools import partial
from typing import Optional

import torch
from torch.utils.cpp_extension import load

torch.set_grad_enabled(False)

_HERE = os.path.dirname(os.path.abspath(__file__))
_BUILD_DIR = os.path.join(_HERE, "build", "practice_softmax_lib")
os.makedirs(_BUILD_DIR, exist_ok=True)

lib = load(
    name="practice_softmax_lib",
    sources=[os.path.join(_HERE, "practice_softmax.cu")],
    extra_cuda_cflags=[
        "-O3",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
    ],
    extra_cflags=["-std=c++17"],
    build_directory=_BUILD_DIR,
)


# ---------------------------------------------------------------------------
# Best-kernel max-H limits driven by practice_softmax.cu's DISPATCH_PRACTICE:
#   NT = H / n_elements must land in {32, 64, 128, 256, 512, 1024}.
#   - fp32 vec4 (n_elements=4) -> max H = 4096
#   - fp16 pack8 (n_elements=8) -> max H = 8192
# Forms that overshoot the table fall through `default` and raise — guard
# them out per-shape so the H=8192 row still runs the fp16 kernel.
# ---------------------------------------------------------------------------
_MAX_H = {
    "softmax_f32_per_token": 4096,
    "safe_softmax_f32_per_token": 4096,
    "online_safe_softmax_f32_per_token": 4096,
    "safe_softmax_f16_f32_per_token": 8192,
}


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------
def check_correctness(
    perf_func: callable,
    x: torch.Tensor,
    tag: str,
    out: Optional[torch.Tensor] = None,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> bool:
    ref = torch.softmax(x, dim=1)
    if out is not None:
        out.fill_(0)
        perf_func(x, out)
        got = out
    else:
        got = perf_func(x)
    torch.cuda.synchronize()
    ok = torch.allclose(got, ref, atol=atol, rtol=rtol)
    status = "PASS" if ok else "FAIL"
    print(f"[correctness] {tag}: {status}")
    if not ok:
        diff = (got.float() - ref.float()).abs()
        print(
            f"             max_abs_diff={diff.max().item()}, "
            f"mean_abs_diff={diff.mean().item()}"
        )
    return ok


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
def run_benchmark(
    perf_func: callable,
    x: torch.Tensor,
    tag: str,
    out: Optional[torch.Tensor] = None,
    warmup: int = 10,
    iters: int = 100,
) -> float:
    if out is not None:
        out.fill_(0)
        for _ in range(warmup):
            perf_func(x, out)
    else:
        for _ in range(warmup):
            _ = perf_func(x)
    torch.cuda.synchronize()

    start = time.time()
    if out is not None:
        for _ in range(iters):
            perf_func(x, out)
    else:
        for _ in range(iters):
            out = perf_func(x)
    torch.cuda.synchronize()
    end = time.time()

    mean_ms = (end - start) * 1000.0 / iters
    out_val = out.flatten().detach().cpu().tolist()[:3]
    out_val = [f"{round(v, 8):<12}" for v in out_val]
    print(f"{'out_' + tag:>24}: {out_val}, time:{mean_ms:.8f}ms")
    return mean_ms


def _bench(name, perf_func, x, tag, out, check, atol, rtol):
    """Skip kernels that would blow past the dispatch table for this H."""
    if name is not None and x.size(-1) > _MAX_H[name]:
        return True  # treat skipped as ok
    ok = True
    if check:
        ok = check_correctness(perf_func, x, tag, out, atol=atol, rtol=rtol)
    run_benchmark(perf_func, x, tag, out)
    return ok


# ---------------------------------------------------------------------------
# Main driver — shape grid mirrors softmax.py exactly.
# ---------------------------------------------------------------------------
def run(check: bool = True):
    all_ok = True

    SHs = [(4096, 256), (4096, 512), (4096, 1024), (4096, 2048),
           (4096, 4096), (4096, 8192), (8192, 8192)]

    for S, H in SHs:
        print("-" * 100)
        print(" " * 45 + f"S={S}, H={H}")
        print("-" * 100)

        # ---- FP32 ----
        x = torch.randn((S, H), device="cuda").cuda().float().contiguous()
        y = torch.zeros_like(x).cuda().float().contiguous()
        all_ok &= _bench(
            "softmax_f32_per_token",
            lib.softmax_f32_per_token, x, "f32(naive)", y, check, 1e-5, 1e-5,
        )
        all_ok &= _bench(
            "safe_softmax_f32_per_token",
            lib.safe_softmax_f32_per_token, x, "f32(safe)", y, check, 1e-5, 1e-5,
        )
        all_ok &= _bench(
            "online_safe_softmax_f32_per_token",
            lib.online_safe_softmax_f32_per_token, x, "f32(safe+online)", y,
            check, 1e-5, 1e-5,
        )
        all_ok &= _bench(
            None,
            partial(torch.softmax, dim=1, out=y), x, "f32_th", None, check,
            1e-5, 1e-5,
        )

        # ---- FP16 ----
        print("-" * 100)
        x_f16 = x.half().contiguous()
        y_f16 = y.half().contiguous()
        all_ok &= _bench(
            "safe_softmax_f16_f32_per_token",
            lib.safe_softmax_f16_f32_per_token, x_f16, "f16f32(safe)", y_f16,
            check, 1e-3, 1e-3,
        )
        all_ok &= _bench(
            None,
            partial(torch.softmax, dim=1, out=y_f16), x_f16, "f16_th", None,
            check, 1e-3, 1e-3,
        )
        print("-" * 100)

    if check:
        print("\n[summary] ALL PASS" if all_ok else "\n[summary] SOME FAIL")
    return all_ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-check", action="store_true", help="Skip correctness checks"
    )
    args = parser.parse_args()
    ok = run(check=not args.no_check)
    exit(0 if ok else 1)
