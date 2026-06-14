---
name: kernel-scaffolder
description: 在 LeetCUDA 仓库的某个 kernel 目录里，把上游参考实现 ($kernel.cu / $kernel.py) 转成两套学习用文件——my_$kernel.{cu,py} + my_profiling.py（保留所有变体，掏空 __global__ 函数体作为 TODO）和 practice_$kernel.{cu,py}（每种 dtype 只留最佳 kernel）。在用户说"为 X 创建 my/practice 文件"、"把这套学习工作流推广到 X kernel"、"为 X 生成 my_X 和 practice_X"、"按 CLAUDE.md 的三层约定接入新 kernel"、或者类似的"按已有约定 scaffold 一个新 kernel 目录"诉求时使用。也覆盖只想生成其中一套（只 my 或只 practice）的场景。
---

# kernel-scaffolder

把 LeetCUDA 仓库里某个 kernel 目录下的上游参考实现 `$kernel.cu` / `$kernel.py`，按仓库 CLAUDE.md 描述的约定生成两套学习用文件：

- **`my_$kernel.{cu,py}` + `my_profiling.py`** — 保留所有渐进版本（原 cu 里的所有 `__global__` kernel 全部留下），但把每个 kernel 函数体掏空换成 `// TODO: implement <这个变体在做什么>`。helper macros / `__device__` 辅助 / `TORCH_BINDING_*` 宏 / PyBind 注册全部原样保留，让 build 链能直接跑。Python driver 加上 `argparse` 暴露 `--benchmark` / `--no-check` / `--profiling <name>` / `--S` / `--K`，build 目录搬到 `build/<lib_name>/`，nvcc 加 `-lineinfo`。`my_profiling.py` 是 ncu 的薄封装：维护一份 `kernels_dict`，`--all` 一次跑全套，`--kernel <name>` 跑单个。
- **`practice_$kernel.{cu,py}`** — 每种 dtype 只留**最佳** kernel，函数名去掉优化后缀（如 `sigmoid_f32x4_kernel` → `sigmoid_f32_kernel`），函数体同样掏空标 TODO。Python driver 极简，只有 correctness + benchmark。

为什么要分两层：`my_*` 是渐进式学习用的——所有变体都在，对照原版一个一个填；`practice_*` 是反复练手用的——只有"最优"那一种，名字也去掉提示，逼自己回忆出最佳实现长什么样。

## 何时使用这个 skill

适用：用户在 LeetCUDA（或其 fork）里某个 `kernels/<op>/` 目录下，已经有上游 `$kernel.cu` + `$kernel.py`，希望按仓库 CLAUDE.md 的三层约定（reference / my / practice）生成 my 和/或 practice 文件。

不适用：
- 当前 kernel 目录里压根没有 `$kernel.cu` 或 `$kernel.py`（不是从空白起手 scaffold，而是从已有参考实现派生）；
- 用户想要生成的是参考实现本身（写一个新 op 的原版 cu/py，那是另一类工作）；
- 目录结构和 CLAUDE.md 描述不一致（比如 `kernels/hgemm/` 这种带自己的 `setup.py` 的子项目）。

如果是其中一种情况，先和用户确认意图再继续。

## 总体工作流

1. **定位输入**。从用户的话里抽取目标 kernel 目录。两种合法表达都接受：
   - **目录路径**：`kernels/gelu/` 或 `/code/LeetCUDA/kernels/gelu/`，从该目录下找 `*.cu` / `*.py` 推断 kernel 名。
   - **kernel 名**：`gelu`，默认对应 `kernels/gelu/`。
   只要其中一种确定，就够用。如果两者都没给清楚，问用户一句确认。

2. **找上游参考实现**。在目标目录里找 `<kernel>.cu` 和 `<kernel>.py`。如果只找到一个，问用户怎么处理；如果都找不到，停下来问用户参考实现在哪。注意 `kernels/reduce/` 是个特例——上游有 `block_all_reduce.{cu,py}`（完整版）和 `all_reduce.{cu,py}`（精简的 f16 链），用户说"reduce" 默认指的是 `all_reduce`。

3. **解析参考 cu**。提取以下信息（用 `Read` 看完整文件，不要 `head/grep` 局部看）：
   - 每个 `__global__ void <name>_kernel(...)` 的签名和它在做什么（函数体）。从函数体里推断"这个变体在做什么"——比如标量 / float4 vec / half2 / 4×half2 / 128-bit pack 等——后面要写进 TODO 注释。
   - 文件顶部的 `#define`（`FLOAT4`、`HALF2`、`LDST128BITS`、`MAX_EXP_*` 等）和任何 `__device__` helper。
   - `TORCH_BINDING_*` 宏的定义、参数（`packed_type, th_type, element_type, n_elements`）和实际调用。
   - `PYBIND11_MODULE` 里 expose 的所有名字。
   - 输入/输出张量布局（一进一出 vs 一进一出标量 vs 多进多出，是否要 atomicAdd 之类）。

4. **生成 `my_$kernel.cu`**。按 §my_kernel.cu 规则生成。

5. **生成 `practice_$kernel.cu`**。按 §practice_kernel.cu 规则生成；最佳 kernel 用 §选最佳 kernel 的启发式规则挑。

6. **生成 `my_$kernel.py`**。从参考 `$kernel.py` 起手，加 argparse 入口、check_correctness、run_profiling，build dir 改到 `build/<lib_name>/`，nvcc flags 加 `-lineinfo`。详见 §my_kernel.py。

7. **生成 `practice_$kernel.py`**。极简版，只跑最佳 kernel 的 correctness + benchmark。详见 §practice_kernel.py。

8. **生成 `my_profiling.py`**。维护 `kernels_dict` 列出每个 kernel 的 dtype + 安全形状，`--all` 一次跑全套 ncu profiling，`--kernel <name>` 单跑一个。详见 §my_profiling.py。

9. **检查产物**。每写完一组，做一遍机械检查：
   - `my_*.cu` 里所有原 kernel 都还在吗？所有 `TORCH_BINDING_*(...)` 调用、`PYBIND11_MODULE` 里的 `m.def`、helper macro 都在吗？
   - `practice_*.cu` 里 dtype × 1 个 kernel，名字是不是都去掉了优化后缀？
   - Python driver 的 `lib.<func>` 引用都和 cu 里 expose 的名字对得上吗？
   - 把扩展加载需要的 include / 宏都保留了吗？

10. **告诉用户接下来怎么用**。简短总结：生成了哪几个文件、推荐先跑什么命令验证（`python3 my_$kernel.py --benchmark --no-check` 或 `--no-check` 跳过校验，因为 `__global__` 还是空的，跑完 build 但 correctness 会 FAIL，是预期）。

整个过程不要直接编译——目标是产出**结构正确的脚手架**，让用户接下来填 kernel 函数体。如果用户问"能不能也帮我把函数体填上"，那是另一个任务，可以另起一轮做。

## 选最佳 kernel 的启发式规则

CLAUDE.md 描述的命名约定是 `<op>_<elem-dtype>[x<vec-width>][_pack]_kernel`。在仓库已有的 sigmoid / relu 中，"最佳"的实际选择对应：

| dtype | 最佳变体 | 选择理由 |
| --- | --- | --- |
| f32 | `<op>_f32x4_kernel` | float4 一次搬 128 bit，element-wise op 的最优形态 |
| f16 | `<op>_f16x8_pack_kernel` | 128-bit pack（`LDST128BITS`）+ 8 元素 SIMD，比 unpack 的 f16x8 寄存器压力更低 |
| bf16 | `<op>_bf16x8_pack_kernel` | 同 f16，如果有的话 |
| f8 / 其他低精度 | 走 pack 版 | 同样优先 128-bit pack |

挑选算法：

1. **先按"算法语义"分组**。如果 kernel 名里编码了不同的算法变体（如 `softmax` / `safe_softmax` / `online_safe_softmax`，三种都是 softmax 但数值稳定性 / pass 数不同），把每个算法当独立的 op 处理 — practice 版**每种算法**都要保留各自的最佳 dtype 实现。理由：这些是不同的练习对象，naive softmax 练基本 reduce、safe 练 max-subtract、online 练 single-pass。把它们合并成一个就丢了练习意义。
2. 在每个算法分组里，把同一个 dtype 的所有 kernel 列出来。
3. 在该 dtype 的候选里，按 **`_pack` > `x<最大向量宽度>` > 标量** 的顺序挑出排名最高的一个。
4. 如果某个 dtype 没有 `_pack` 也没有 vec 版（只有标量，比如某些原型）——就保留那个标量版。
5. 如果出现拿不准的情况（比如 `_pack` 和某个 `x<更大宽度>` 同时存在），把候选给用户看一下，问一句"f16 我选 f16x8_pack 行吗？"

reduce 这条链的"最佳"在仓库里直接是 `all_reduce_sum_f16x8_pack_kernel`（只有这一个），照搬即可。

## §my_kernel.cu

完整保留参考实现的**框架**结构（include / TORCH_BINDING / PYBIND），但 kernel 实现区域只留 `__global__` 签名 + TODO。规则：

- **include / namespace 全部保留**——build 链需要它们。
- **顶部纯数值常量 / 类型 typedef 保留**：例如 `#define MAX_EXP_F32 88.3762626647949f`、`#define MIN_EXP_F16 ...` 这种数值上下限是**签名层面**的（决定 dtype 的合法范围），不是练习对象，保留。
- **kernel 实现技巧相关的 `#define` 和 `__device__` 一律删除**，**不要**注释保留——这是这个 skill 的核心教学决策：
  - **cast 宏**（`FLOAT4` / `HALF2` / `BFLOAT2` / `LDST128BITS` / `INT4`）：删掉，不留注释。让用户自己想到"这个变体要 128-bit 搬运 → 我要写 cast macro"。
  - **`__device__` helper**（`warp_reduce_sum_f32` / `block_reduce_max_f32` / online softmax 的 `MD` struct + `warp_reduce_md_op` 等）：删掉。这些 helper 本身就是练习的一部分，复制过来用户就跳过了"怎么写 warp/block reduce""怎么定义 online merge 数据结构"这些核心练习。
  - **`WARP_SIZE` 这类基础常量**：删掉。学生写 reduce 时自然要重新定义。
  - 唯一例外：参考实现的某个 `__device__` helper **不**是练习对象（比如纯类型转换 utility，或者 op 数学定义本身——不是优化技巧），可以保留。但默认从严：**有疑问就删**。
- **不要**自己补"明显有用的 `__device__` helper 作为提示"——这是反向的，给提示就削弱了练习效果。
- **对每个 `__global__ void <name>_kernel(...)`**：保留签名（包括 templated `<int NUM_THREADS>`），函数体替换成
  ```cpp
  __global__ void sigmoid_f32x4_kernel(float *x, float *y, int N) {
    // TODO: implement fp32 vec4 sigmoid (FLOAT4 load/store)
  }
  ```
  TODO 注释要简短描述这个变体的特征（向量宽度 / 是否 pack / 用什么宏 / 关键操作）让人有方向，但**不要**把答案写得太具体（比如别在 TODO 里写"调用 `block_reduce_sum_f32`"——helper 都被删了，这就是练习要重新想的东西）。
- **所有 binding-side 代码全保留，原封不动**：`#define STRINGFY` / `TORCH_BINDING_COMMON_EXTENSION` / `CHECK_TORCH_TENSOR_DTYPE` / `TORCH_BINDING_<OP>` 宏定义、所有 `TORCH_BINDING_<OP>(...)` 实例化、`PYBIND11_MODULE` 里的所有 `m.def`。这部分是用户**填完 kernel 后立即能 build** 的保证，不能动。
- 如果 reduce 风格的参考实现把 dispatch 逻辑写在 `LANUCH_*` / `DISPATCH_*` 宏里——这些**也**保留，它们是 binding 层。

如果需要，可以参考 `references/templates.md` 里的"my_kernel.cu 骨架"部分对照修订。

## §practice_kernel.cu

按 §选最佳 kernel 的启发式 挑出每个 dtype 的最佳 kernel，然后写一个精简版：

- include 只保留必需的（`<cuda_fp16.h>` / `<cuda_runtime.h>` / `<torch/extension.h>` / `<torch/types.h>` 等）。
- 顶部加一段 banner 注释，说明这是 practice 版、每种 dtype 一个 kernel、名字是简洁版（去掉优化后缀），并简要列出每个 dtype 的最佳变体是什么以及为什么。
- 每个最佳 kernel 改名：去掉 `x<width>` 和 `_pack` 后缀。
  - `sigmoid_f32x4_kernel` → `sigmoid_f32_kernel`
  - `sigmoid_f16x8_pack_kernel` → `sigmoid_f16_kernel`
  - `all_reduce_sum_f16x8_pack_kernel` → `all_reduce_sum_f16_kernel`
- **当心 acc-type 后缀 ≠ 优化标签**。有些 kernel 名字编码了 accumulator dtype（典型例子：`dot_prod_f16x8_pack_f32_kernel`，末尾的 `_f32` 是"FP32 accumulator"，不是优化变体）。约定：识别"优化标签"只看 `x<num>` / `_pack` 这两类后缀；中间或末尾的 `_f32` / `_f16` 段如果**不是**这两类，就当成参与签名语义的一部分保留下来。比如 `dot_prod_f16x8_pack_f32` 简化成 `dot_prod_f16_f32`（保留 acc 后缀），不是 `dot_prod_f16`。如果不确定看一眼参考 cu 里的 kernel 是不是用相同 elem-dtype 但不同 acc-dtype 出现了多次——出现过就肯定是 acc 后缀。
- **同 my_*.cu，kernel 实现区域只留 `__global__` 签名 + TODO**。不要把参考里的 `__device__` helper、`MD` struct、cast macros、`WARP_SIZE` 这些复制过来。practice 是"逼自己回忆出最佳实现长什么样"，把脚手架搭好就等于剧透。
- 函数体替换成 `// TODO(practice): best <DTYPE> <op> — <最优形态简述>`。
- TORCH_BINDING 宏可以**精简**（不需要 `if (ndim != 2)` 全分支，但要保留 `n_elements` 参数和 dtype 检查）——参照已有的 `practice_sigmoid.cu` / `practice_relu.cu`。把 `n_elements` 设成最佳 kernel 的向量宽度（标量=1，f32x4=4，f16x8_pack=8）。
- 只 bind 简洁名（`sigmoid_f32` / `sigmoid_f16`），dtype × 1 个。
- 如果参考实现是 reduce 风格（kernel 是 templated `<int NUM_THREADS>`，TORCH_BINDING 用 dispatch macro），保留这个结构，参照 `kernels/reduce/practice_all_reduce.cu`。

## §my_kernel.py

**核心方针：完全对齐参考 `$kernel.py`，只加自用功能。**

`my_$kernel.py` 不是另起一套 benchmark，而是把参考 `$kernel.py` **当骨架**，在外面套上 argparse 入口、`check_correctness`、`run_profiling`、独立 build dir、`-lineinfo`。所有"参考实现里已经定下来的东西"——形状网格、`dim` 参数、`iters`、用到哪些 kernel、什么 dtype 走什么路径、打印格式——一律 **照搬不改**。

参考 `kernels/sigmoid/my_sigmoid.py` 和 `kernels/relu/my_relu.py`。规则：

**1. 必须照搬参考的部分（不要改、不要"优化"、不要"统一"）**：

- 形状扫描：参考用 `Ss = Ks = [1024, 2048, 4096]` 的对角矩阵就照样；参考用 `S=4096, H ∈ {256, 512, 1024, ...}` 加几条特殊形状（softmax 风格）就照样。**禁止把所有 op 都套成 `{1024,2048,4096}²` 网格**——element-wise 和 reduce/row-wise 风格的 op 选形状的目的不一样，改了网格就和参考性能数据不可比。
- `dim` 参数：参考写 `dim=1` 就是 `dim=1`，写 `dim=0` 就是 `dim=0`，写 `dim=-1` 就是 `dim=-1`。**`dim=1` 和 `dim=-1` 在 2D 张量上数学等价不构成改动理由**——参考的写法就是 spec。
- `iters`、`warmup`、tag 字符串、`run_benchmark` 的输出格式、调用顺序、是否传 `out`——和参考保持字节级一致。
- 跑哪些 kernel、跑哪些 dtype：参考的 `(S, K)` 块里在 fp32/fp16 路径下分别调了哪些 `lib.xxx`，照样列出来；参考某个形状跳过了某个 kernel（比如太大装不下），那就跳过。

**2. 在参考之上只加这些"自用功能"**：

- 顶部 `_HERE` / `_BUILD_DIR = build/<kernel>_lib/`，`os.makedirs(_BUILD_DIR, exist_ok=True)`。
- `torch.utils.cpp_extension.load` 的 `name=` 用 `<kernel>_lib`、`sources=[my_<kernel>.cu]` 走绝对路径、`build_directory=_BUILD_DIR`、`extra_cuda_cflags` 在原版基础上加 `-lineinfo`。
- `check_correctness(perf_func, x, tag, out=None, atol=1e-5, rtol=1e-5)`：ref 用参考实现里 torch baseline 的同一个调用形式（softmax 参考用 `torch.softmax(x, dim=1)`，那 ref 就是 `torch.softmax(x, dim=1)`，不是 `dim=-1`）。
- `run_profiling(perf_func, src_shape, *, src_dtype, dst_dtype, warmup=10)`：**`src_shape` 是单个 tuple**（不是 `*src_shape` 变长——后者在调用方 `run_profiling(fn, src_shape, ...)` 时会变成嵌套 tuple `((S, K),)`，`torch.randn` 在某些 PyTorch 版本上侥幸能跑但语义错），内部包 `nvtx.range_push("profiling")` / `range_pop()`。**统一走 out 路径**——torch 的 element-wise / softmax / relu 这些 op 都接受 `out=` kwarg，不需要分 has_out 分支；torch baseline 用 `lambda x, out: torch.<op>(x, ..., out=out)` 包一下传进去（partial 不行，`out=` 必须在 call 时绑定）。
- `run_profiling_for_test(kernel_name, src_shape=PROFILING_SHAPE)`：用 `match` / `case` 分支调 `run_profiling`，分支覆盖参考实现里 expose 的所有 kernel + torch baseline。每个分支硬编码自己的 `src_dtype` / `dst_dtype`（不再走 CLI `--dtype`）。
- 把参考那种"把 benchmark 主体写在模块顶层、import 时立即跑"的写法搬进 `run_benchmark_for_all_test(check=True)` 函数体（结构 1:1 复制，只在每个 `run_benchmark(...)` 之前加 `if check: check_correctness(...)`）。
- `if __name__ == "__main__":` argparse 加 `--benchmark` / `--profiling` / `--S` / `--K` / `--no-check`。`--K` 默认值取**所有 kernel 都能跑的最大公共形状**——参考 cu 里 dispatch 宏对 H 的最严格上限（softmax 这组里 `softmax_f32_per_token` 等几个只支持到 H=1024，所以默认给 1024）。**不要**加 `--dtype`——profiling 的 dtype 由 kernel 名唯一决定，写在 `match` 分支里就够了。

**3. 当心的特殊形状**：

- 如果参考实现里某些形状只跑部分 kernel（比如 H=8192 时只跑 `f16x8_pack`，因为 vec1/vec2 不够装），照搬这个选择就行——不要为了"统一"补齐。
- 如果是 reduce 风格（输出标量、无 `out` 参数），照搬 `kernels/reduce/my_all_reduce.py` 的 reduce 写法（调用形式是 `out = perf_func(x)` 而非 `perf_func(x, out)`）。

## §practice_kernel.py

**核心方针：和 my_$kernel.py 一样照搬参考的形状网格 / dim / iters，但只跑每种 (algorithm, dtype) 组合的最佳 kernel + torch baseline。**

参考 `kernels/sigmoid/practice_sigmoid.py` / `kernels/relu/practice_relu.py` / `kernels/reduce/practice_all_reduce.py`。规则：

**1. 必须照搬参考的部分**：

- 形状网格：和 §my_kernel.py 同一个原则——参考 `$kernel.py` 用什么形状网格就用什么，不要写死 `{1024,2048,4096}²`。
- `dim` 参数、`iters`、tag 字符串、`out` 是否传——同 my_*.py，照搬参考。
- 如果某个形状下参考根本没跑最佳 kernel（比如最佳 kernel 的向量宽度装不下 H），那个形状就**跳过**这个 kernel（continue 或 if 守卫），不要硬跑会越界的形状。

**2. 与 my_*.py 的差别**：

- 顶部 docstring 一句话说明这是 practice 版。
- `_BUILD_DIR = build/practice_<kernel>_lib/`；`lib = load(name="practice_<kernel>_lib", sources=[practice_<kernel>.cu], ...)`。
- 只有两个函数：`check_correctness` 和 `run_benchmark`，签名和 my_*.py 里一致。
- `run(check=True)`：循环参考的形状网格，每个形状只跑 dtype × 最佳 kernel + torch baseline，累加 `all_ok`。
- 末尾打印 `ALL PASS` / `SOME FAIL`。
- argparse 只有 `--no-check`，`exit(0 if ok else 1)`。

## §my_profiling.py

`my_profiling.py` 是 ncu 的薄 Python 封装——**不再生成 `my_$kernel.sh`**。维护一份 `kernels_dict` 列出所有要 profile 的 kernel，外层 `--all` 跑全套、`--kernel <name>` 跑一个。形状用一个**全局常量** `PROFILING_SHAPE`，取"所有 kernel 都能跑的最大公共形状"——这样：

- 一份形状所有 kernel 都能用，不需要 per-kernel 元组绕来绕去；
- 一次 `python3 my_profiling.py --all` 把所有 `.ncu-rep` 都生成到 `profiling/` 子目录；
- 用 Python 而不是 shell，跨 kernel 复制粘贴时少踩 `set -e` / 路径转义之类的坑。

模板：

```python
import os
import argparse

dump_path = os.path.join(os.path.dirname(__file__), "profiling")
os.makedirs(dump_path, exist_ok=True)

# 统一形状：所有 kernel 都能跑的最大公共形状。
# 看参考 cu 里所有 DISPATCH_*_KERNEL 宏的 H case 上限，取最严格的那个。
# softmax 这一组里某些 kernel 只到 H=1024，所以最大公共形状 = (4096, 1024)。
PROFILING_SHAPE = (4096, 1024)
S, K = PROFILING_SHAPE

# kernel 名 -> dtype（仅作记录用——my_<op>.py 的 match 分支已经硬编码了 dtype）。
# torch baseline 不进 dict —— ncu 跑 torch 没意义。
kernels_dict = {
    "<kernel_name_1>": "float32",
    "<kernel_name_2>": "float32",
    ...
    "<best_f16_kernel>": "float16",
}

cmds = {
    k: f'ncu --nvtx \
        --nvtx-include "profiling/" \
        --set full \
        --import-source yes \
        -f \
        -o {os.path.join(dump_path, k)} \
        -- python3 my_<kernel>.py --profiling {k} --S {S} --K {K}'
    for k in kernels_dict
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile <kernel> kernels")
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

要点：
- **`-f` 是必须的**——重跑时 ncu 默认会因为 `<name>.ncu-rep` 已存在而报错退出。
- `PROFILING_SHAPE` 取"所有 kernel 都能跑的最大公共形状"。具体看参考 cu 里**每个** `DISPATCH_*_KERNEL` 宏的 `case` 列表的 H 上限，取最严格的一个——例如 softmax 这组：

  | kernel | 最大支持 H |
  | --- | --- |
  | `softmax_f32_per_token` / `safe_softmax_f32_per_token` / `online_safe_softmax_f32_per_token` / `safe_softmax_f16_f32_per_token` | 1024 |
  | `safe_softmax_f16x2_f32_per_token` | 2048 |
  | `softmax_f32x4_per_token` / `safe_softmax_f32x4_per_token` / `online_safe_softmax_f32x4_pack_per_token` | 4096 |
  | `safe_softmax_f16x8_pack_f32_per_token` | 8192 |

  所以 K=1024 是最大公共形状（往上加任何一个就有 kernel 不支持）。如果用户**只想 profile 部分 kernel**，可以拆出第二份 dict + 更大的形状；默认只生成"全 kernel × 公共形状"这一份。
- reduce 风格只多一个 `-k regex:"<name>_kernel"` 过滤（限定 ncu 只采样目标 kernel，避免 reduce 链上初始化 / 收尾 kernel 也被采进来），其他完全一样。

## 对照模板

`references/templates.md` 里有完整的几个模板片段（element-wise op 风格、reduce 风格），生成时不确定的话先看那里再下笔。

## 边界情况

- **kernel 里有 templated `__global__`**（如 `template <const int NUM_THREADS> __global__ void ...`）：保留模板参数和默认值，函数体掏空。
- **多输入 / 多输出**：维持原签名，TODO 注释里说清楚每个参数代表什么。
- **kernel 函数体里有内嵌的 `#define`**（如 sigmoid 参考里把 `FLOAT4` 内嵌在 kernel 里）：删掉，不要保留也不要放到 TODO 上面。这些内嵌 cast 宏属于"实现技巧"，不是"签名"——和顶部的 cast 宏一视同仁删掉。
- **CLAUDE.md 没在仓库里**：照样按这个 skill 描述的约定走，但提醒用户检查仓库根的 `CLAUDE.md` 看是否有更新的约定。
- **目录里已经有 `my_*` 或 `practice_*` 文件**：先 `Read` 一下，如果内容像是用户写过的填充版（不只是脚手架），停下来问用户要不要覆盖；如果只是空脚手架可以直接覆盖，但还是先告诉用户一声。
