import argparse
import os
import time
from functools import partial
from typing import Optional

import torch
from torch.utils.cpp_extension import load

torch.set_grad_enabled(False)

_HERE = os.path.dirname(os.path.abspath(__file__))
_BUILD_DIR = os.path.join(_HERE, "build", "softmax_lib")
os.makedirs(_BUILD_DIR, exist_ok=True)

lib = load(
    name="softmax_lib",
    sources=[os.path.join(_HERE, "my_softmax.cu")],
    extra_cuda_cflags=[
        "-O3",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        "-lineinfo",
    ],
    extra_cflags=["-std=c++17"],
    build_directory=_BUILD_DIR,
)


# ---------------------------------------------------------------------------
# Per-kernel maximum supported head size (H). Driven by the dispatch macros
# in my_softmax.cu — scalar variants cap at NUM_THREADS=1024 (so H=1024),
# vec2 at H=2048, vec4 at H=4096, vec8 (f16x8_pack) at H=8192.
# ---------------------------------------------------------------------------
_MAX_H = {
    "softmax_f32_per_token": 1024,
    "softmax_f32x4_per_token": 4096,
    "safe_softmax_f32_per_token": 1024,
    "safe_softmax_f32x4_per_token": 4096,
    "safe_softmax_f16_f32_per_token": 1024,
    "safe_softmax_f16x2_f32_per_token": 2048,
    "safe_softmax_f16x8_pack_f32_per_token": 8192,
    "online_safe_softmax_f32_per_token": 1024,
    "online_safe_softmax_f32x4_pack_per_token": 4096,
}


def run_benchmark(
    perf_func: callable,
    x: torch.Tensor,
    tag: str,
    out: Optional[torch.Tensor] = None,
    warmup: int = 10,
    iters: int = 1000,
    show_all: bool = False,
):
    if out is not None:
        out.fill_(0)
    # warmup
    if out is not None:
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

    total_time = (end - start) * 1000  # ms
    mean_time = total_time / iters
    out_info = f"out_{tag}"
    out_val = out.flatten().detach().cpu().numpy().tolist()[:2]
    out_val = [round(v, 8) for v in out_val]
    out_val = [f"{v:<12}" for v in out_val]
    print(f"{out_info:>32}: {out_val}, time:{mean_time:.8f}ms")
    if show_all:
        print(out)
    return out, mean_time


def run_profiling(
    perf_func: callable,
    x: torch.Tensor,
    tag: str,
    out: Optional[torch.Tensor] = None,
    warmup: int = 10,
):
    if out is not None:
        out.fill_(0)
    if out is not None:
        for _ in range(warmup):
            perf_func(x, out)
    else:
        for _ in range(warmup):
            _ = perf_func(x)
    torch.cuda.synchronize()

    torch.cuda.nvtx.range_push("profiling")
    if out is not None:
        perf_func(x, out)
    else:
        _ = perf_func(x)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()


def check_correctness(
    perf_func: callable,
    x: torch.Tensor,
    tag: str,
    out: Optional[torch.Tensor] = None,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    safe: bool = True,
) -> bool:
    """Verify perf_func(x[, out]) matches torch.softmax(x, dim=-1).

    `safe=False` -> compare against the unsafe naive form sum(exp(x))/exp(x).
    Naive softmax overflows for input magnitudes ~> 80 in fp32; we generate
    inputs from N(0, 1) so the unsafe path is still finite, and torch.softmax
    is mathematically equivalent (it does max-sub internally), so we can use
    torch.softmax as the reference for both safe and naive variants.
    """
    ref = torch.softmax(x, dim=-1)
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


def _maybe_run(name, kernel_fn, x, out, tag, check, atol, rtol):
    """Run check + benchmark only when H is supported by this kernel."""
    H = x.size(-1)
    if H > _MAX_H[name]:
        return
    if check:
        check_correctness(kernel_fn, x, tag, out, atol=atol, rtol=rtol)
    run_benchmark(kernel_fn, x, tag, out)


def run_benchmark_for_all_test(check: bool = True):
    Ss = [1024, 2048, 4096]
    Ks = [1024, 2048, 4096]
    SKs = [(S, K) for S in Ss for K in Ks]

    for S, K in SKs:
        print("-" * 100)
        print(" " * 45 + f"S={S}, K={K}")
        print("-" * 100)

        # ---- FP32 path ----
        x = torch.randn((S, K)).cuda().float().contiguous()
        y = torch.zeros_like(x).cuda().float().contiguous()
        _maybe_run(
            "softmax_f32_per_token",
            lib.softmax_f32_per_token, x, y, "f32(naive)", check, 1e-5, 1e-5,
        )
        _maybe_run(
            "softmax_f32x4_per_token",
            lib.softmax_f32x4_per_token, x, y, "f32x4(naive)", check, 1e-5, 1e-5,
        )
        _maybe_run(
            "safe_softmax_f32_per_token",
            lib.safe_softmax_f32_per_token, x, y, "f32(safe)", check, 1e-5, 1e-5,
        )
        _maybe_run(
            "safe_softmax_f32x4_per_token",
            lib.safe_softmax_f32x4_per_token, x, y, "f32x4(safe)", check,
            1e-5, 1e-5,
        )
        _maybe_run(
            "online_safe_softmax_f32_per_token",
            lib.online_safe_softmax_f32_per_token, x, y, "f32(safe+online)",
            check, 1e-5, 1e-5,
        )
        _maybe_run(
            "online_safe_softmax_f32x4_pack_per_token",
            lib.online_safe_softmax_f32x4_pack_per_token, x, y,
            "f32x4(safe+online)", check, 1e-5, 1e-5,
        )
        if check:
            check_correctness(partial(torch.softmax, dim=-1), x, "f32_th")
        run_benchmark(partial(torch.softmax, dim=-1, out=y), x, "f32_th", y)

        # ---- FP16 path ----
        print("-" * 100)
        x_f16 = x.half().contiguous()
        y_f16 = y.half().contiguous()
        _maybe_run(
            "safe_softmax_f16_f32_per_token",
            lib.safe_softmax_f16_f32_per_token, x_f16, y_f16, "f16f32(safe)",
            check, 1e-3, 1e-3,
        )
        _maybe_run(
            "safe_softmax_f16x2_f32_per_token",
            lib.safe_softmax_f16x2_f32_per_token, x_f16, y_f16,
            "f16x2f32(safe)", check, 1e-3, 1e-3,
        )
        _maybe_run(
            "safe_softmax_f16x8_pack_f32_per_token",
            lib.safe_softmax_f16x8_pack_f32_per_token, x_f16, y_f16,
            "f16x8packf32(safe)", check, 1e-3, 1e-3,
        )
        if check:
            check_correctness(
                partial(torch.softmax, dim=-1), x_f16, "f16_th",
                atol=1e-3, rtol=1e-3,
            )
        run_benchmark(
            partial(torch.softmax, dim=-1, out=y_f16), x_f16, "f16_th", y_f16,
        )
        print("-" * 100)


def run_profiling_for_test(
    kernel_name: str,
    dtype: torch.dtype,
    S: int = 4096,
    K: int = 4096,
):
    x = torch.randn((S, K)).cuda().float().contiguous()
    y = torch.zeros_like(x).cuda().float().contiguous()
    x_half = x.half().contiguous()
    y_half = y.half().contiguous()

    if dtype == torch.float32:
        if kernel_name == "softmax_f32_per_token":
            run_profiling(lib.softmax_f32_per_token, x, "profiling", y)
        elif kernel_name == "softmax_f32x4_per_token":
            run_profiling(lib.softmax_f32x4_per_token, x, "profiling", y)
        elif kernel_name == "safe_softmax_f32_per_token":
            run_profiling(lib.safe_softmax_f32_per_token, x, "profiling", y)
        elif kernel_name == "safe_softmax_f32x4_per_token":
            run_profiling(lib.safe_softmax_f32x4_per_token, x, "profiling", y)
        elif kernel_name == "online_safe_softmax_f32_per_token":
            run_profiling(
                lib.online_safe_softmax_f32_per_token, x, "profiling", y
            )
        elif kernel_name == "online_safe_softmax_f32x4_pack_per_token":
            run_profiling(
                lib.online_safe_softmax_f32x4_pack_per_token, x, "profiling", y
            )
        elif kernel_name == "softmax_th":
            run_profiling(partial(torch.softmax, dim=-1), x, "profiling", y)
        else:
            raise ValueError(f"Unsupported FP32 kernel name: {kernel_name}")
    elif dtype == torch.float16:
        if kernel_name == "safe_softmax_f16_f32_per_token":
            run_profiling(
                lib.safe_softmax_f16_f32_per_token, x_half, "profiling", y_half
            )
        elif kernel_name == "safe_softmax_f16x2_f32_per_token":
            run_profiling(
                lib.safe_softmax_f16x2_f32_per_token, x_half, "profiling",
                y_half,
            )
        elif kernel_name == "safe_softmax_f16x8_pack_f32_per_token":
            run_profiling(
                lib.safe_softmax_f16x8_pack_f32_per_token, x_half, "profiling",
                y_half,
            )
        elif kernel_name == "softmax_th":
            run_profiling(
                partial(torch.softmax, dim=-1), x_half, "profiling", y_half
            )
        else:
            raise ValueError(f"Unsupported FP16 kernel name: {kernel_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark", action="store_true", help="Run benchmark"
    )
    parser.add_argument(
        "--profiling", type=str, default=None,
        help="Run profiling for the given kernel name",
    )
    parser.add_argument(
        "--dtype", type=str, default="float32",
        help="Data type for profiling (float32 or float16)",
    )
    parser.add_argument(
        "--S", type=int, default=4096, help="Row size (seqlens) for profiling"
    )
    parser.add_argument(
        "--K", type=int, default=4096, help="Column size (head size) for profiling"
    )
    parser.add_argument(
        "--no-check", action="store_true",
        help="Skip correctness checks before benchmarking",
    )

    args = parser.parse_args()
    if args.benchmark:
        run_benchmark_for_all_test(check=not args.no_check)
    if args.profiling is not None:
        dtype = torch.float32 if args.dtype == "float32" else torch.float16
        run_profiling_for_test(args.profiling, dtype, S=args.S, K=args.K)
