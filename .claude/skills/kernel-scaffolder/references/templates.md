# Kernel scaffold templates

生成 my_/practice_ 文件时不确定结构，对照这里的模板。两种主要风格：**element-wise**（`sigmoid` / `relu` / `gelu` 这类一进一出 same-shape）和 **reduce**（`all_reduce_sum` 这类一进一出 scalar）。

## Table of contents
- [element-wise: my_$op.cu](#element-wise-my_opcu)
- [element-wise: practice_$op.cu](#element-wise-practice_opcu)
- [element-wise: my_$op.py](#element-wise-my_oppy)
- [element-wise: practice_$op.py](#element-wise-practice_oppy)
- [element-wise: my_profiling.py](#element-wise-my_profilingpy)
- [reduce: my_$op.cu](#reduce-my_opcu)
- [reduce: practice_$op.cu](#reduce-practice_opcu)
- [reduce: my_$op.py](#reduce-my_oppy)
- [reduce: practice_$op.py](#reduce-practice_oppy)
- [reduce: my_profiling.py](#reduce-my_profilingpy)

---

## element-wise: my_$op.cu

参考 `kernels/sigmoid/my_sigmoid.cu` / `kernels/relu/my_relu.cu`。骨架：

```cpp
#include <algorithm>
#include <cuda_fp16.h>     // 只在用到 half 时才 include
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>

// 数值常量保留（如果 op 需要它来钳制 dtype 范围）
#define MAX_EXP_F32  88.3762626647949f
#define MIN_EXP_F32 -88.3762626647949f
#define MAX_EXP_F16 __float2half(11.089866488461016f)
#define MIN_EXP_F16 __float2half(-9.704060527839234f)

// !!! 注意：参考实现里的 cast 宏（FLOAT4 / HALF2 / LDST128BITS / INT4 / BFLOAT2）
// 和 __device__ helper（warp_reduce_*、block_reduce_*、MD struct 之类）一律删掉，
// 不要在这里以注释形式保留——它们是练习对象。

__global__ void <op>_f32_kernel(float *x, float *y, int N) {
  // TODO: implement scalar fp32 <op>
}

__global__ void <op>_f32x4_kernel(float *x, float *y, int N) {
  // TODO: implement fp32 vec4 <op> (FLOAT4 load/store)
}

__global__ void <op>_f16_kernel(half *x, half *y, int N) {
  // TODO: implement scalar fp16 <op>
}

__global__ void <op>_f16x2_kernel(half *x, half *y, int N) {
  // TODO: implement fp16 half2 <op> (HALF2 load/store)
}

__global__ void <op>_f16x8_kernel(half *x, half *y, int N) {
  // TODO: implement fp16x8 <op> via 4x half2 (HALF2 x 4)
}

__global__ void <op>_f16x8_pack_kernel(half *x, half *y, int N) {
  // TODO: implement fp16x8 pack <op> (LDST128BITS load/store)
}

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

// 这部分宏定义和参考 cu 一字不差，照搬过来
#define TORCH_BINDING_<OP>(packed_type, th_type, element_type, n_elements)     \
  void <op>_##packed_type(torch::Tensor x, torch::Tensor y) {                  \
    /* ... 和参考一致 ... */                                                    \
  }

TORCH_BINDING_<OP>(f32, torch::kFloat32, float, 1)
TORCH_BINDING_<OP>(f32x4, torch::kFloat32, float, 4)
TORCH_BINDING_<OP>(f16, torch::kHalf, half, 1)
TORCH_BINDING_<OP>(f16x2, torch::kHalf, half, 2)
TORCH_BINDING_<OP>(f16x8, torch::kHalf, half, 8)
TORCH_BINDING_<OP>(f16x8_pack, torch::kHalf, half, 8)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(<op>_f32)
  TORCH_BINDING_COMMON_EXTENSION(<op>_f32x4)
  TORCH_BINDING_COMMON_EXTENSION(<op>_f16)
  TORCH_BINDING_COMMON_EXTENSION(<op>_f16x2)
  TORCH_BINDING_COMMON_EXTENSION(<op>_f16x8)
  TORCH_BINDING_COMMON_EXTENSION(<op>_f16x8_pack)
}
```

---

## element-wise: practice_$op.cu

参考 `kernels/sigmoid/practice_sigmoid.cu` / `kernels/relu/practice_relu.cu`：

```cpp
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <torch/types.h>

// ============================================================
// Practice kernels — best-performing variant per data type.
// Names are intentionally plain (no vector-width hints) so the
// exercise focuses on re-implementing the optimal kernel itself.
//   FP32  -> best: float4 vectorized (128-bit LD/ST)
//   FP16  -> best: 128-bit pack + <op-specific best>
// ============================================================

__global__ void <op>_f32_kernel(float *x, float *y, int N) {
  // TODO(practice): best FP32 <op> — 4-element vectorized
}

__global__ void <op>_f16_kernel(half *x, half *y, int N) {
  // TODO(practice): best FP16 <op> — 8-element 128-bit pack + <key intrinsic>
}

// ============================================================
// Torch bindings
// ============================================================
#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    throw std::runtime_error("Tensor dtype mismatch, expected " #th_type);     \
  }

// n_elements: how many elements each thread processes in the fast path
#define TORCH_BINDING_<OP>(packed_type, th_type, element_type, n_elements)     \
  void <op>_##packed_type(torch::Tensor x, torch::Tensor y) {                  \
    CHECK_TORCH_TENSOR_DTYPE(x, (th_type))                                     \
    CHECK_TORCH_TENSOR_DTYPE(y, (th_type))                                     \
    const int ndim = x.dim();                                                  \
    const int S = x.size(0);                                                   \
    const int K = (ndim >= 2) ? x.size(1) : 1;                                 \
    const int N = x.numel();                                                   \
    dim3 block, grid;                                                          \
    if (ndim == 2 && (K / (n_elements)) <= 1024) {                             \
      block = dim3(K / (n_elements));                                          \
      grid  = dim3(S);                                                         \
    } else {                                                                   \
      block = dim3(256 / (n_elements));                                        \
      grid  = dim3((N + 256 - 1) / 256);                                       \
    }                                                                          \
    <op>_##packed_type##_kernel<<<grid, block>>>(                              \
        reinterpret_cast<element_type *>(x.data_ptr()),                        \
        reinterpret_cast<element_type *>(y.data_ptr()), N);                    \
  }

TORCH_BINDING_<OP>(f32, torch::kFloat32, float, 4)
TORCH_BINDING_<OP>(f16, torch::kHalf,    half,  8)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(<op>_f32)
  TORCH_BINDING_COMMON_EXTENSION(<op>_f16)
}
```

`n_elements` 取最佳 kernel 的向量宽度：标量=1、x4=4、x8 / x8_pack=8。

---

## element-wise: my_$op.py

参考 `kernels/sigmoid/my_sigmoid.py` / `kernels/relu/my_relu.py`。

**核心方针：把参考 `$op.py` 当骨架照搬（形状网格 / `dim` 参数 / `iters` / 调用顺序 / tag 一律不改），只在外面加 argparse + check + profiling + build_dir + `-lineinfo`。** 下面只给"自用功能"骨架；中间 `run_benchmark_for_all_test` 的具体形状循环要从参考 `$op.py` 里**原样搬过来**，每个 `run_benchmark(...)` 之前加一行 `if check: check_correctness(...)`，**不要**写死成 `[1024,2048,4096]²`。

```python
import os
import time
from functools import partial
from typing import Optional
import argparse

import torch
from torch.utils.cpp_extension import load
from torch.cuda import nvtx

torch.set_grad_enabled(False)

_HERE = os.path.dirname(os.path.abspath(__file__))
_BUILD_DIR = os.path.join(_HERE, "build", "<op>_lib")
os.makedirs(_BUILD_DIR, exist_ok=True)

lib = load(
    name="<op>_lib",
    sources=[os.path.join(_HERE, "my_<op>.cu")],
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


def run_benchmark(perf_func, x, tag, out: Optional[torch.Tensor] = None,
                  warmup: int = 10, iters: int = 1000, show_all: bool = False):
    # 注意：iters 默认值与参考 $op.py 保持一致；如果参考用的是 100 就改成 100。
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
    mean_time = (end - start) * 1000 / iters
    out_val = out.flatten().detach().cpu().numpy().tolist()[:2]
    out_val = [round(v, 8) for v in out_val]
    out_val = [f"{v:<12}" for v in out_val]
    print(f"{'out_'+tag:>18}: {out_val}, time:{mean_time:.8f}ms")
    if show_all:
        print(out)
    return out, mean_time


def run_profiling(perf_func, src_shape, *, src_dtype, dst_dtype, warmup: int = 10):
    # NOTE: src_shape 是单个 tuple，不要写成 *src_shape——后者会让调用方
    # `run_profiling(fn, src_shape, ...)` 把 (S, K) 包成 ((S, K),)，torch.randn
    # 在某些 PyTorch 版本上侥幸能跑但语义错。
    # 统一走 out 路径——torch element-wise / softmax / relu 的 op 都接受 out= kwarg。
    x = torch.randn(src_shape, device="cuda", dtype=src_dtype).contiguous()
    out = torch.zeros_like(x, device="cuda", dtype=dst_dtype).contiguous()

    for _ in range(warmup):
        perf_func(x, out)
    torch.cuda.synchronize()

    torch.cuda.nvtx.range_push("profiling")
    perf_func(x, out)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()


def check_correctness(perf_func, x, tag, out: Optional[torch.Tensor] = None,
                      atol: float = 1e-5, rtol: float = 1e-5) -> bool:
    # ref 的写法**必须**和参考 $op.py 里 torch baseline 调用形式完全一致：
    #   - sigmoid:        ref = torch.sigmoid(x)
    #   - softmax dim=1:  ref = torch.softmax(x, dim=1)        ← 不要改成 dim=-1
    #   - reduce sum:     ref = torch.sum(x, dtype=torch.float32)
    ref = torch.<op>(x)
    if out is not None:
        out.fill_(0)
        perf_func(x, out)
        got = out
    else:
        got = perf_func(x)
    torch.cuda.synchronize()
    ok = torch.allclose(got, ref, atol=atol, rtol=rtol)
    print(f"[correctness] {tag}: {'PASS' if ok else 'FAIL'}")
    if not ok:
        diff = (got.float() - ref.float()).abs()
        print(f"             max_abs_diff={diff.max().item()}, "
              f"mean_abs_diff={diff.mean().item()}")
    return ok


def run_benchmark_for_all_test(check: bool = True):
    # ============================================================
    # 把参考 $op.py 的 benchmark 主体（顶层 for 循环 / 形状常量 /
    # `run_benchmark(lib.xxx, x, "...", out)` 调用序列）**原样**搬到这里。
    # 唯一允许的改动：每个 run_benchmark 之前加 `if check: check_correctness(...)`。
    # 不要把所有 op 都套成 [1024,2048,4096]² 网格，这会丢掉参考实现的 shape 选择意图。
    # ============================================================
    ...  # ← 从参考 $op.py 复制 benchmark 主体过来


def run_profiling_for_test(kernel_name: str, src_shape=(4096, 1024)):
    # 用 match/case 分支覆盖参考实现里 expose 的所有 kernel + torch baseline。
    # 每个分支硬编码自己的 src_dtype / dst_dtype——不再走 CLI --dtype。
    # 默认 src_shape 取"所有 kernel 都能跑的最大公共形状"（看每个 DISPATCH 宏的
    # H case 上限取最严格的那个）。
    match kernel_name:
        case "<op>_f32":
            run_profiling(lib.<op>_f32, src_shape,
                          src_dtype=torch.float32, dst_dtype=torch.float32)
        case "<op>_f32x4":
            run_profiling(lib.<op>_f32x4, src_shape,
                          src_dtype=torch.float32, dst_dtype=torch.float32)
        case "<op>_f16":
            run_profiling(lib.<op>_f16, src_shape,
                          src_dtype=torch.float16, dst_dtype=torch.float16)
        case "<op>_f16x2":
            run_profiling(lib.<op>_f16x2, src_shape,
                          src_dtype=torch.float16, dst_dtype=torch.float16)
        case "<op>_f16x8_pack":
            run_profiling(lib.<op>_f16x8_pack, src_shape,
                          src_dtype=torch.float16, dst_dtype=torch.float16)
        case "<op>_th":
            # torch 的 element-wise / softmax 也接受 out= kwarg，但要在 call 时绑定
            # （partial 会在创建时 freeze，那时 out tensor 还不存在）。用 lambda 包一下。
            run_profiling(
                lambda x, out: torch.<op>(x, out=out),
                src_shape, src_dtype=torch.float16, dst_dtype=torch.float16,
            )
        case _:
            raise ValueError(f"Unsupported kernel name: {kernel_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--profiling", type=str, default=None)
    parser.add_argument("--S", type=int, default=4096)
    parser.add_argument("--K", type=int, default=1024,
                        help="Default = the largest H all kernels support.")
    parser.add_argument("--no-check", action="store_true")
    args = parser.parse_args()

    if args.benchmark:
        run_benchmark_for_all_test(check=not args.no_check)
    if args.profiling is not None:
        run_profiling_for_test(args.profiling, src_shape=(args.S, args.K))
```

---

## element-wise: practice_$op.py

**核心方针：和 my_$op.py 同样照搬参考的形状网格 / `dim` / `iters`**，但每个形状只跑每种 (algorithm, dtype) 组合的最佳 kernel + torch baseline。如果某个形状下最佳 kernel 装不下（例如 H 超过 vec 宽度允许的最大 NUM_THREADS），跳过这个形状的该 kernel。

```python
# -*- coding: utf-8 -*-
"""
practice_<op>.py - Practice-use <Op> benchmark
==============================================
After learning the <op> kernel, use this for repeated practice:
  - Only the best-performing kernel per (algorithm, dtype) combo
  - Two features only: check_correctness and benchmark
  - Plain kernel names (<op>_f32 / <op>_f16), no optimization hints
  - Shape grid mirrors $op.py (do NOT replace with [1024,2048,4096]²).
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
_BUILD_DIR = os.path.join(_HERE, "build", "practice_<op>_lib")
os.makedirs(_BUILD_DIR, exist_ok=True)

lib = load(
    name="practice_<op>_lib",
    sources=[os.path.join(_HERE, "practice_<op>.cu")],
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


def check_correctness(perf_func, x, tag, out: Optional[torch.Tensor] = None,
                      atol: float = 1e-5, rtol: float = 1e-5) -> bool:
    # ref 必须照搬参考 $op.py 的 torch baseline 调用形式（dim 参数等不要改）。
    ref = torch.<op>(x)
    if out is not None:
        out.fill_(0)
        perf_func(x, out)
        got = out
    else:
        got = perf_func(x)
    torch.cuda.synchronize()
    ok = torch.allclose(got, ref, atol=atol, rtol=rtol)
    print(f"[correctness] {tag}: {'PASS' if ok else 'FAIL'}")
    if not ok:
        diff = (got.float() - ref.float()).abs()
        print(f"             max_abs_diff={diff.max().item()}, "
              f"mean_abs_diff={diff.mean().item()}")
    return ok


def run_benchmark(perf_func, x, tag, out: Optional[torch.Tensor] = None,
                  warmup: int = 10, iters: int = 1000) -> float:
    # iters 与参考 $op.py 保持一致。
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
    out_val = out.flatten().detach().cpu().tolist()[:2]
    out_val = [f"{round(v, 8):<12}" for v in out_val]
    print(f"{'out_'+tag:>18}: {out_val}, time:{mean_ms:.8f}ms")
    return mean_ms


def run(check: bool = True):
    # ============================================================
    # 形状循环照搬参考 $op.py，每个形状只跑：
    #   - 每个 (algorithm, dtype) 的最佳 kernel
    #   - 对应 dtype 的 torch baseline
    # 如果某形状下最佳 kernel 装不下，用 if 守卫跳过。
    # ============================================================
    all_ok = True
    ...  # ← 参照参考 $op.py 的 (S, H/K) 循环写在这里

    if check:
        print("\n[summary] ALL PASS" if all_ok else "\n[summary] SOME FAIL")
    return all_ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-check", action="store_true")
    args = parser.parse_args()
    ok = run(check=not args.no_check)
    exit(0 if ok else 1)
```

---

## element-wise: my_profiling.py

ncu 的薄 Python 封装。用一个**全局** `PROFILING_SHAPE` 常量——取"所有 kernel 都能跑的最大公共形状"（看参考 cu 里**每个** `DISPATCH_*_KERNEL` 宏的 H case 上限，取最严格的那个）。`--all` 跑全套，`--kernel <name>` 跑一个。

```python
import os
import argparse

dump_path = os.path.join(os.path.dirname(__file__), "profiling")
os.makedirs(dump_path, exist_ok=True)

# 统一形状：所有 kernel 都能跑的最大公共形状。
# 比 dispatch 宏 H 上限最严格的那个再小一点也行，但通常直接取那个上限。
PROFILING_SHAPE = (<S>, <largest_H_all_kernels_support>)
S, K = PROFILING_SHAPE

# kernel 名 -> dtype（仅记录用，my_<op>.py 的 match 分支已经硬编码了 dtype）。
# torch baseline 不进 dict —— ncu 跑 torch 没意义。
kernels_dict = {
    "<op>_f32":          "float32",
    "<op>_f32x4":        "float32",
    "<op>_f16":          "float16",
    "<op>_f16x2":        "float16",
    "<op>_f16x8_pack":   "float16",
}

cmds = {
    k: f'ncu --nvtx \
        --nvtx-include "profiling/" \
        --set full \
        --import-source yes \
        -f \
        -o {os.path.join(dump_path, k)} \
        -- python3 my_<op>.py --profiling {k} --S {S} --K {K}'
    for k in kernels_dict
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile <op> kernels")
    parser.add_argument("--all", "-a", action="store_true", help="Profile all kernels")
    parser.add_argument("--kernel", "-k", type=str, help="Profile a specific kernel")
    args = parser.parse_args()

    if args.all:
        for k, cmd in cmds.items():
            print(f"Profiling {k}...")
            os.system(cmd)
    elif args.kernel:
        if args.kernel in cmds:
            print(f"Profiling {args.kernel}...")
            os.system(cmds[args.kernel])
        else:
            print(f"Kernel {args.kernel} not found. Available: {list(cmds.keys())}")
```

**`-f` 是必须的**——重跑时 ncu 默认会因为 `<name>.ncu-rep` 已存在而报错退出。
**`--dtype` 不再传**——profiling 的 dtype 由 kernel 名唯一决定，写在 `my_<op>.py` 的 `match` 分支里就够了。

---

## reduce: my_$op.cu

参考 `kernels/reduce/my_all_reduce.cu`。reduce 风格的 kernel 是 templated `<int NUM_THREADS>`，TORCH_BINDING 用 dispatch macro。骨架：

```cpp
#include <algorithm>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>

// !!! reduce 风格的 kernel 强烈依赖 warp_reduce / block_reduce / MD struct 这类
// __device__ helper——但这些**就是练习对象**：写 reduce 的核心难度就在这里。
// 一律删掉，不要在这里以注释形式保留。WARP_SIZE / cast 宏（LDST128BITS 等）也删掉。
// 用户填 kernel 时会重新写出 warp shuffle 循环、shared memory 缩并、atomicAdd 收尾。

template <const int NUM_THREADS = 1024>
__global__ void <op>_f16x8_pack_kernel(half *x, float *y, int N) {
  // TODO: fp16x8 pack <op> — block reduce + atomicAdd into scalar y.
  //   load shape: 128-bit pack of 8 halves per thread.
  //   accumulate into fp32 register (kernel returns fp32 scalar).
  //   write final block result via atomicAdd.
}

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

// dispatch macros + TORCH_BINDING_REDUCE 完全照搬参考实现
// （拷贝过来即可，不需要修改 — 这部分是 binding 层，不是练习对象）
```

完整的 dispatch / binding macros 看 `kernels/reduce/my_all_reduce.cu` 第 73-154 行。

---

## reduce: practice_$op.cu

只保留最佳 kernel（通常是 `<op>_f16x8_pack_kernel`），名字保持原样（reduce 这条链已经是简洁名了），函数体掏空：

```cpp
// 同 my_$op.cu — 不要把 warp/block reduce helper、cast 宏、MD struct 之类
// 复制进 practice。它们是练习对象。

template <const int NUM_THREADS = 1024>
__global__ void <op>_f16_kernel(half *x, float *y, int N) {
  // TODO(practice): best FP16 <op> — 128-bit pack load + warp reduction + atomicAdd
}
```

注：reduce 链上原本只有 `_f16x8_pack`，practice 版的命名可以选择
- 保持 `<op>_f16x8_pack`（如果你觉得简洁名歧义太大），或者
- 简化成 `<op>_f16`（更符合 CLAUDE.md 描述的"去掉优化标签"）

参考 `kernels/reduce/practice_all_reduce.cu` 的写法。

---

## reduce: my_$op.py

参考 `kernels/reduce/my_all_reduce.py`。和 element-wise 版差异：
- **彻底删除 `out:` 参数**——`run_benchmark`、`run_profiling`、`check_correctness` 三个函数的签名里都不要保留 `out: Optional[torch.Tensor] = None`。reduce kernel 自己内部 `atomicAdd` 到一个新分配的 scalar tensor，调用形式是 `out = perf_func(x)`（或 `(a, b)`），不是 `perf_func(x, out)`。要么完全删掉 `out` 参数，要么至少把所有 `if out is not None:` 分支去掉。
- `check_correctness` 的 ref 是 `torch.sum(x, dtype=torch.float32)` 之类（dot-product 是 `(a.float() * b.float()).sum()` 或 `torch.dot`）；
- argparse 是否需要 `--dtype` 看具体 op：单 dtype（all_reduce f16-only）就去掉；多 dtype（dot-product 有 f32/f16）就保留。
- `run_profiling` 同样不传 `out`，在内部只分配 `x`，调用形式 `perf_func(x)`。

---

## reduce: practice_$op.py

参考 `kernels/reduce/practice_all_reduce.py`。比 element-wise practice 还要简单：
- 一个 dtype × 一个 kernel，调用接口是 `lib.<best_kernel_name>(x)` 返回 scalar tensor。

---

## reduce: my_profiling.py

和 element-wise 版几乎一样，差别有两处：
- ncu 命令多一个 `-k regex:"<name>_kernel"`（限定只采样目标 kernel，避免 reduce 链上初始化 / 收尾 kernel 也被采进来）；
- reduce 通常没有 H 上限不一致的问题，`PROFILING_SHAPE` 直接取参考 `$op.py` 形状网格里的某个代表形状即可。

```python
import os
import argparse

dump_path = os.path.join(os.path.dirname(__file__), "profiling")
os.makedirs(dump_path, exist_ok=True)

PROFILING_SHAPE = (4096, 4096)
S, K = PROFILING_SHAPE

kernels_dict = {
    "<op>_f16x8_pack": "float16",
    # ... 其他 reduce kernel
}

cmds = {
    k: f'ncu --nvtx \
        --nvtx-include "profiling/" \
        -k regex:"{k}_kernel" \
        --set full \
        --import-source yes \
        -f \
        -o {os.path.join(dump_path, k)} \
        -- python3 my_<op>.py --profiling {k} --S {S} --K {K}'
    for k in kernels_dict
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile <op> kernels")
    parser.add_argument("--all", "-a", action="store_true")
    parser.add_argument("--kernel", "-k", type=str)
    args = parser.parse_args()
    if args.all:
        for k, cmd in cmds.items():
            print(f"Profiling {k}..."); os.system(cmd)
    elif args.kernel:
        if args.kernel in cmds:
            print(f"Profiling {args.kernel}..."); os.system(cmds[args.kernel])
        else:
            print(f"Kernel {args.kernel} not found. Available: {list(cmds.keys())}")
```
