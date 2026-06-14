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


def run_benchmark(
    perf_func: callable,
    x: torch.Tensor,
    tag: str,
    out: Optional[torch.Tensor] = None,
    warmup: int = 10,
    iters: int = 100,
    show_all: bool = False,
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
    out_val = out.flatten().detach().cpu().numpy().tolist()[:3]
    out_val = [round(v, 8) for v in out_val]
    out_val = [f"{v:<12}" for v in out_val]
    print(f"{out_info:>24}: {out_val}, time:{mean_time:.8f}ms")
    if show_all:
        print(out)
    return out, mean_time

def run_profiling(
    perf_func: callable,
    src_shape: tuple,
    *,
    src_dtype: torch.dtype,
    dst_dtype: torch.dtype,
    warmup: int = 10,
):
    x = torch.randn(src_shape, device="cuda", dtype=src_dtype).contiguous()
    out = torch.zeros_like(x, device="cuda", dtype=dst_dtype).contiguous()

    for _ in range(warmup):
        perf_func(x, out)
    torch.cuda.synchronize()

    torch.cuda.nvtx.range_push("profiling")
    perf_func(x, out)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

def check_correctness(
    perf_func: callable,
    x: torch.Tensor,
    tag: str,
    out: Optional[torch.Tensor] = None,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> bool:
    """Verify perf_func(x[, out]) matches torch.softmax(x, dim=1).

    Inputs are 2D (S, H) drawn from N(0, 1); torch.softmax internally does
    max-subtract so it serves as the reference for both safe and naive
    variants (naive can still match because |x| stays ~O(1) at this scale).
    """
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


def _bench(perf_func, x, tag, out, check, atol, rtol):
    """Tiny helper: optional correctness check + benchmark, in one call."""
    if check:
        check_correctness(perf_func, x, tag, out, atol=atol, rtol=rtol)
    run_benchmark(perf_func, x, tag, out)


def run_benchmark_for_all_test(check: bool = True):
    # ------------------------------------------------------------------
    # Mirror softmax.py: shapes, kernel dispatch sets, and dim=1 are kept
    # 1:1 with the reference. Only addition is an optional correctness
    # check before each run_benchmark.
    # ------------------------------------------------------------------

    # per token softmax
    print("-" * 100)
    S, H = 4096, 256
    print(" " * 45 + f"S={S}, H={H}")
    print("-" * 100)
    x = torch.randn((S, H), device="cuda").cuda().float().contiguous()
    out = torch.zeros_like(x).cuda().float().contiguous()
    _bench(lib.softmax_f32_per_token, x, "f32(per)", out, check, 1e-5, 1e-5)
    _bench(lib.softmax_f32x4_per_token, x, "f32x4(per)", out, check, 1e-5, 1e-5)
    _bench(lib.safe_softmax_f32_per_token, x, "f32(safe)", out, check, 1e-5, 1e-5)
    _bench(
        lib.online_safe_softmax_f32_per_token, x, "f32(safe+online)", out,
        check, 1e-5, 1e-5,
    )
    _bench(
        lib.online_safe_softmax_f32x4_pack_per_token, x, "f32x4(safe+online)",
        out, check, 1e-5, 1e-5,
    )
    _bench(
        lib.safe_softmax_f32x4_per_token, x, "f32x4(safe)", out, check, 1e-5,
        1e-5,
    )
    _bench(
        partial(torch.softmax, dim=1, out=out), x, "f32_th(per)", None, check,
        1e-5, 1e-5,
    )

    print("-" * 100)
    x_f16 = x.half().contiguous()
    out_f16 = out.half().contiguous()
    _bench(
        lib.safe_softmax_f16_f32_per_token, x_f16, "f16f32(safe)", out_f16,
        check, 1e-3, 1e-3,
    )
    _bench(
        lib.safe_softmax_f16x2_f32_per_token, x_f16, "f16x2f32(safe)", out_f16,
        check, 1e-3, 1e-3,
    )
    _bench(
        lib.safe_softmax_f16x8_pack_f32_per_token, x_f16, "f16x8packf32(safe)",
        out_f16, check, 1e-3, 1e-3,
    )
    _bench(
        partial(torch.softmax, dim=1, out=out_f16), x_f16, "f16_th(per)", None,
        check, 1e-3, 1e-3,
    )
    print("-" * 100)

    # per token softmax
    print("-" * 100)
    S, H = 4096, 512
    print(" " * 45 + f"S={S}, H={H}")
    print("-" * 100)
    x = torch.randn((S, H), device="cuda").cuda().float().contiguous()
    out = torch.zeros_like(x).cuda().float().contiguous()
    _bench(lib.softmax_f32_per_token, x, "f32(per)", out, check, 1e-5, 1e-5)
    _bench(lib.softmax_f32x4_per_token, x, "f32x4(per)", out, check, 1e-5, 1e-5)
    _bench(lib.safe_softmax_f32_per_token, x, "f32(safe)", out, check, 1e-5, 1e-5)
    _bench(
        lib.online_safe_softmax_f32_per_token, x, "f32(safe+online)", out,
        check, 1e-5, 1e-5,
    )
    _bench(
        lib.online_safe_softmax_f32x4_pack_per_token, x, "f32x4(safe+online)",
        out, check, 1e-5, 1e-5,
    )
    _bench(
        lib.safe_softmax_f32x4_per_token, x, "f32x4(safe)", out, check, 1e-5,
        1e-5,
    )
    _bench(
        partial(torch.softmax, dim=1, out=out), x, "f32_th(per)", None, check,
        1e-5, 1e-5,
    )

    print("-" * 100)
    x_f16 = x.half().contiguous()
    out_f16 = out.half().contiguous()
    _bench(
        lib.safe_softmax_f16_f32_per_token, x_f16, "f16f32(safe)", out_f16,
        check, 1e-3, 1e-3,
    )
    _bench(
        lib.safe_softmax_f16x2_f32_per_token, x_f16, "f16x2f32(safe)", out_f16,
        check, 1e-3, 1e-3,
    )
    _bench(
        lib.safe_softmax_f16x8_pack_f32_per_token, x_f16, "f16x8packf32(safe)",
        out_f16, check, 1e-3, 1e-3,
    )
    _bench(
        partial(torch.softmax, dim=1, out=out_f16), x_f16, "f16_th(per)", None,
        check, 1e-3, 1e-3,
    )
    print("-" * 100)

    # per token softmax
    print("-" * 100)
    S, H = 4096, 1024
    print(" " * 45 + f"S={S}, H={H}")
    print("-" * 100)
    x = torch.randn((S, H), device="cuda").cuda().float().contiguous()
    out = torch.zeros_like(x).cuda().float().contiguous()
    _bench(lib.softmax_f32_per_token, x, "f32(per)", out, check, 1e-5, 1e-5)
    _bench(lib.softmax_f32x4_per_token, x, "f32x4(per)", out, check, 1e-5, 1e-5)
    _bench(lib.safe_softmax_f32_per_token, x, "f32(safe)", out, check, 1e-5, 1e-5)
    _bench(
        lib.online_safe_softmax_f32_per_token, x, "f32(safe+online)", out,
        check, 1e-5, 1e-5,
    )
    _bench(
        lib.online_safe_softmax_f32x4_pack_per_token, x, "f32x4(safe+online)",
        out, check, 1e-5, 1e-5,
    )
    _bench(
        lib.safe_softmax_f32x4_per_token, x, "f32x4(safe)", out, check, 1e-5,
        1e-5,
    )
    _bench(
        partial(torch.softmax, dim=1, out=out), x, "f32_th(per)", None, check,
        1e-5, 1e-5,
    )

    print("-" * 100)
    x_f16 = x.half().contiguous()
    out_f16 = out.half().contiguous()
    _bench(
        lib.safe_softmax_f16_f32_per_token, x_f16, "f16f32(safe)", out_f16,
        check, 1e-3, 1e-3,
    )
    _bench(
        lib.safe_softmax_f16x2_f32_per_token, x_f16, "f16x2f32(safe)", out_f16,
        check, 1e-3, 1e-3,
    )
    _bench(
        lib.safe_softmax_f16x8_pack_f32_per_token, x_f16, "f16x8packf32(safe)",
        out_f16, check, 1e-3, 1e-3,
    )
    _bench(
        partial(torch.softmax, dim=1, out=out_f16), x_f16, "f16_th(per)", None,
        check, 1e-3, 1e-3,
    )
    print("-" * 100)

    # per token softmax
    print("-" * 100)
    S, H = 4096, 2048
    print(" " * 45 + f"S={S}, H={H}")
    print("-" * 100)
    x = torch.randn((S, H), device="cuda").cuda().float().contiguous()
    out = torch.zeros_like(x).cuda().float().contiguous()
    _bench(lib.softmax_f32x4_per_token, x, "f32x4(per)", out, check, 1e-5, 1e-5)
    _bench(
        lib.safe_softmax_f32x4_per_token, x, "f32x4(safe)", out, check, 1e-5,
        1e-5,
    )
    _bench(
        lib.online_safe_softmax_f32x4_pack_per_token, x, "f32x4(safe+online)",
        out, check, 1e-5, 1e-5,
    )
    _bench(
        partial(torch.softmax, dim=1, out=out), x, "f32_th(per)", None, check,
        1e-5, 1e-5,
    )

    print("-" * 100)
    x_f16 = x.half().contiguous()
    out_f16 = out.half().contiguous()
    _bench(
        lib.safe_softmax_f16x2_f32_per_token, x_f16, "f16x2f32(safe)", out_f16,
        check, 1e-3, 1e-3,
    )
    _bench(
        lib.safe_softmax_f16x8_pack_f32_per_token, x_f16, "f16x8packf32(safe)",
        out_f16, check, 1e-3, 1e-3,
    )
    _bench(
        partial(torch.softmax, dim=1, out=out_f16), x_f16, "f16_th(per)", None,
        check, 1e-3, 1e-3,
    )
    print("-" * 100)

    # per token softmax
    print("-" * 100)
    S, H = 4096, 4096
    print(" " * 45 + f"S={S}, H={H}")
    print("-" * 100)
    x = torch.randn((S, H), device="cuda").cuda().float().contiguous()
    out = torch.zeros_like(x).cuda().float().contiguous()
    _bench(lib.softmax_f32x4_per_token, x, "f32x4(per)", out, check, 1e-5, 1e-5)
    _bench(
        lib.safe_softmax_f32x4_per_token, x, "f32x4(safe)", out, check, 1e-5,
        1e-5,
    )
    _bench(
        lib.online_safe_softmax_f32x4_pack_per_token, x, "f32x4(safe+online)",
        out, check, 1e-5, 1e-5,
    )
    _bench(
        partial(torch.softmax, dim=1, out=out), x, "f32_th(per)", None, check,
        1e-5, 1e-5,
    )

    print("-" * 100)
    x_f16 = x.half().contiguous()
    out_f16 = out.half().contiguous()
    _bench(
        lib.safe_softmax_f16x8_pack_f32_per_token, x_f16, "f16x8packf32(safe)",
        out_f16, check, 1e-3, 1e-3,
    )
    _bench(
        partial(torch.softmax, dim=1, out=out_f16), x_f16, "f16_th(per)", None,
        check, 1e-3, 1e-3,
    )
    print("-" * 100)

    # per token softmax
    print("-" * 100)
    S, H = 4096, 8192
    print(" " * 45 + f"S={S}, H={H}")
    print("-" * 100)
    x = torch.randn((S, H), device="cuda").cuda().float().contiguous()
    out = torch.zeros_like(x).cuda().float().contiguous()
    x_f16 = x.half().contiguous()
    out_f16 = out.half().contiguous()
    _bench(
        lib.safe_softmax_f16x8_pack_f32_per_token, x_f16, "f16x8packf32(safe)",
        out_f16, check, 1e-3, 1e-3,
    )
    _bench(
        partial(torch.softmax, dim=1, out=out_f16), x_f16, "f16_th(per)", None,
        check, 1e-3, 1e-3,
    )

    # per token softmax
    print("-" * 100)
    S, H = 8192, 8192
    print(" " * 45 + f"S={S}, H={H}")
    print("-" * 100)
    x = torch.randn((S, H), device="cuda").cuda().float().contiguous()
    out = torch.zeros_like(x).cuda().float().contiguous()
    x_f16 = x.half().contiguous()
    out_f16 = out.half().contiguous()
    _bench(
        lib.safe_softmax_f16x8_pack_f32_per_token, x_f16, "f16x8packf32(safe)",
        out_f16, check, 1e-3, 1e-3,
    )
    _bench(
        partial(torch.softmax, dim=1, out=out_f16), x_f16, "f16_th(per)", None,
        check, 1e-3, 1e-3,
    )
    print("-" * 100)


def run_profiling_for_test(
    kernel_name: str,
    src_shape
):
    match kernel_name:
        case "softmax_f32_per_token":
            run_profiling(lib.softmax_f32_per_token, src_shape, src_dtype=torch.float32, dst_dtype=torch.float32)
        case "softmax_f32x4_per_token":
            run_profiling(lib.softmax_f32x4_per_token, src_shape, src_dtype=torch.float32, dst_dtype=torch.float32)
        case "safe_softmax_f32_per_token":
            run_profiling(lib.safe_softmax_f32_per_token, src_shape, src_dtype=torch.float32, dst_dtype=torch.float32)
        case "safe_softmax_f32x4_per_token":
            run_profiling(lib.safe_softmax_f32x4_per_token, src_shape, src_dtype=torch.float32, dst_dtype=torch.float32)
        case "online_safe_softmax_f32_per_token":
            run_profiling(lib.online_safe_softmax_f32_per_token, src_shape, src_dtype=torch.float32, dst_dtype=torch.float32)
        case "online_safe_softmax_f32x4_pack_per_token":
            run_profiling(lib.online_safe_softmax_f32x4_pack_per_token, src_shape, src_dtype=torch.float32, dst_dtype=torch.float32)
        case "safe_softmax_f16_f32_per_token":
            run_profiling(lib.safe_softmax_f16_f32_per_token, src_shape, src_dtype=torch.float16, dst_dtype=torch.float16)
        case "safe_softmax_f16x2_f32_per_token":
            run_profiling(lib.safe_softmax_f16x2_f32_per_token, src_shape, src_dtype=torch.float16, dst_dtype=torch.float16)
        case "safe_softmax_f16x8_pack_f32_per_token":
            run_profiling(lib.safe_softmax_f16x8_pack_f32_per_token, src_shape, src_dtype=torch.float16, dst_dtype=torch.float16)
        case "softmax_th":
            # torch.softmax accepts an `out=` kwarg, but we need to bind it at call time
            # (partial would freeze it before the tensor exists). Wrap in a lambda.
            run_profiling(
                lambda x, out: torch.softmax(x, dim=1, out=out),
                src_shape, src_dtype=torch.float16, dst_dtype=torch.float16,
            )
        case _:
            raise ValueError(f"Unsupported kernel name: {kernel_name}")

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
        "--S", type=int, default=4096, help="Row size (seqlens) for profiling"
    )
    parser.add_argument(
        "--K", type=int, default=1024,
        help="Column size (head size) for profiling. "
             "Default = largest H all kernels support.",
    )
    parser.add_argument(
        "--no-check", action="store_true",
        help="Skip correctness checks before benchmarking",
    )

    args = parser.parse_args()
    if args.benchmark:
        run_benchmark_for_all_test(check=not args.no_check)
    if args.profiling is not None:
        run_profiling_for_test(args.profiling, src_shape=(args.S, args.K))