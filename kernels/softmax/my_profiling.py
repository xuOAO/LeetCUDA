import argparse
import os

dump_path = os.path.join(os.path.dirname(__file__), "profiling")
os.makedirs(dump_path, exist_ok=True)

# 统一形状：所有 kernel 都能跑的最大公共形状。
# softmax 这一组里 softmax_f32_per_token / safe_softmax_f32_per_token /
# online_safe_softmax_f32_per_token / safe_softmax_f16_f32_per_token 的
# DISPATCH 宏只支持到 H=1024，所以 (4096, 1024) 是最大公共形状。
PROFILING_SHAPE = (4096, 1024)
S, K = PROFILING_SHAPE

# kernel 名 -> dtype。dtype 由 kernel 名编码，仅作记录用——my_softmax.py 的
# match 分支已经硬编码了每个 kernel 的 src/dst dtype。
# torch baseline 不进 dict —— ncu 跑 torch 没意义，用 nsys 看更直接。
kernels_dict = {
    "softmax_f32_per_token":                    "float32",
    "softmax_f32x4_per_token":                  "float32",
    "safe_softmax_f32_per_token":               "float32",
    "safe_softmax_f32x4_per_token":             "float32",
    "online_safe_softmax_f32_per_token":        "float32",
    "online_safe_softmax_f32x4_pack_per_token": "float32",
    "safe_softmax_f16_f32_per_token":           "float16",
    "safe_softmax_f16x2_f32_per_token":         "float16",
    "safe_softmax_f16x8_pack_f32_per_token":    "float16",
}

cmds = {
    k: f'ncu --nvtx \
        --nvtx-include "profiling/" \
        --set full \
        --import-source yes \
        -f \
        -o {os.path.join(dump_path, k)} \
        -- python3 my_softmax.py --profiling {k} --S {S} --K {K}'
    for k in kernels_dict
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile softmax kernels")
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
            print(f"Kernel {args.kernel} not found. Available kernels: {list(cmds.keys())}")
