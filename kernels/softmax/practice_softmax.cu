#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <torch/extension.h>
#include <torch/types.h>

// ============================================================
// Practice kernels — best-performing variant per (algorithm, dtype).
// Names are intentionally plain (no vector-width hints) so the exercise
// focuses on re-implementing the optimal kernel itself.
//
// Three algorithm "families" — softmax has more practice surface than a
// pure element-wise op, so each gets its own slot:
//   naive softmax              -> best FP32: 4-element vec (FLOAT4)
//   safe softmax (max-sub)     -> best FP32: 4-element vec (FLOAT4)
//                                 best FP16 (fp32 acc): 8-element 128-bit pack
//   online safe softmax        -> best FP32: 4-element vec (FLOAT4) single-pass
//                                 (Milakov & Gimelshein, 1805.02867)
//
// Note: `_f32_` after the elem-dtype here is the *accumulator* dtype, not an
// optimization tag — we keep it in the practice name (e.g. safe_softmax_f16_f32)
// so the signature still encodes that fp16 in/out compute through fp32 acc.
// ============================================================

// === naive softmax: best fp32 ================================================
template <const int NUM_THREADS = 256 / 4>
__global__ void softmax_f32_per_token_kernel(float *x, float *y, int N) {
  // TODO(practice): best FP32 naive softmax — one token per block, each
  //   thread owns 4 elements via 128-bit load/store + a block-wide sum
  //   reduction. no max-subtract (numerically unsafe; that's the point).
}

// === safe softmax: best fp32 =================================================
template <const int NUM_THREADS = 256 / 4>
__global__ void safe_softmax_f32_per_token_kernel(float *x, float *y, int N) {
  // TODO(practice): best FP32 safe softmax — one token per block,
  //   128-bit load/store, two block-wide reductions: max -> exp(x-max)
  //   sum -> divide.
}

// === safe softmax: best fp16 (fp32 accumulator) ==============================
template <const int NUM_THREADS = 256>
__global__ void safe_softmax_f16_f32_per_token_kernel(half *x, half *y, int N) {
  // TODO(practice): best FP16 safe softmax — one token per block,
  //   each thread owns 8 elements via 128-bit pack. reductions and exp()
  //   in fp32 accumulator, store __float2half_rn(exp/sum) back via pack.
}

// === online safe softmax: best fp32 ==========================================
template <const int NUM_THREADS = 256 / 4>
__global__ void online_safe_softmax_f32_per_token_kernel(float *x, float *y,
                                                         int N) {
  // TODO(practice): best FP32 online safe softmax — single pass,
  //   128-bit load/store. running (m, d) state merged across threads via
  //   warp/block online-softmax reduction (lower-m side scales by
  //   exp(small.m - big.m)); y = expf(x - final.m) / final.d.
}

// ============================================================================
// Torch bindings
// ============================================================================
#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    throw std::runtime_error("Tensor dtype mismatch, expected " #th_type);     \
  }

#define CHECK_TORCH_TENSOR_SHAPE(T1, T2)                                       \
  for (int i = 0; i < (T1).dim(); ++i) {                                       \
    if ((T2).size(i) != (T1).size(i))                                          \
      throw std::runtime_error("Tensor size mismatch!");                       \
  }

// Per-token softmax dispatch: one block per token, NUM_THREADS = H/n_elements.
// Supports H in {32, 64, 128, 256, 512, 1024} for n_elements=1, scaled
// accordingly when the kernel processes more than one element per thread.
#define LAUNCH_PRACTICE(kernel, NT, elem_t)                                    \
  kernel<(NT)><<<grid, block>>>(reinterpret_cast<elem_t *>(x.data_ptr()),      \
                                reinterpret_cast<elem_t *>(y.data_ptr()), N);

#define DISPATCH_PRACTICE(kernel, elem_t, n_elements)                          \
  const int NT = H / (n_elements);                                             \
  dim3 block(NT);                                                              \
  dim3 grid(S);                                                                \
  switch (NT) {                                                                \
  case 32:                                                                     \
    LAUNCH_PRACTICE(kernel, 32, elem_t) break;                                 \
  case 64:                                                                     \
    LAUNCH_PRACTICE(kernel, 64, elem_t) break;                                 \
  case 128:                                                                    \
    LAUNCH_PRACTICE(kernel, 128, elem_t) break;                                \
  case 256:                                                                    \
    LAUNCH_PRACTICE(kernel, 256, elem_t) break;                                \
  case 512:                                                                    \
    LAUNCH_PRACTICE(kernel, 512, elem_t) break;                                \
  case 1024:                                                                   \
    LAUNCH_PRACTICE(kernel, 1024, elem_t) break;                               \
  default:                                                                     \
    throw std::runtime_error("only support H/n_elements in {32..1024}");       \
  }

void softmax_f32_per_token(torch::Tensor x, torch::Tensor y) {
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat32)
  CHECK_TORCH_TENSOR_SHAPE(x, y)
  const int S = x.size(0);
  const int H = x.size(1);
  const int N = S * H;
  DISPATCH_PRACTICE(softmax_f32_per_token_kernel, float, 4)
}

void safe_softmax_f32_per_token(torch::Tensor x, torch::Tensor y) {
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat32)
  CHECK_TORCH_TENSOR_SHAPE(x, y)
  const int S = x.size(0);
  const int H = x.size(1);
  const int N = S * H;
  DISPATCH_PRACTICE(safe_softmax_f32_per_token_kernel, float, 4)
}

void safe_softmax_f16_f32_per_token(torch::Tensor x, torch::Tensor y) {
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kHalf)
  CHECK_TORCH_TENSOR_SHAPE(x, y)
  const int S = x.size(0);
  const int H = x.size(1);
  const int N = S * H;
  DISPATCH_PRACTICE(safe_softmax_f16_f32_per_token_kernel, half, 8)
}

void online_safe_softmax_f32_per_token(torch::Tensor x, torch::Tensor y) {
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat32)
  CHECK_TORCH_TENSOR_SHAPE(x, y)
  const int S = x.size(0);
  const int H = x.size(1);
  const int N = S * H;
  DISPATCH_PRACTICE(online_safe_softmax_f32_per_token_kernel, float, 4)
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(softmax_f32_per_token)
  TORCH_BINDING_COMMON_EXTENSION(safe_softmax_f32_per_token)
  TORCH_BINDING_COMMON_EXTENSION(safe_softmax_f16_f32_per_token)
  TORCH_BINDING_COMMON_EXTENSION(online_safe_softmax_f32_per_token)
}
