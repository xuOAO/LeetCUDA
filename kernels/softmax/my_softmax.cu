#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>

// NOTE: softmax per-token
// Softmax x: (S,h), y: (S,h)
// grid(S*h/h), block(h), assume h<=1024
// one token per thread block, only support 64<=h<=1024 and 2^n
// HEAD_SIZE/KV_LEN=NUM_THREADS
template <const int NUM_THREADS = 256>
__global__ void softmax_f32_per_token_kernel(float *x, float *y, int N) {
  // TODO: naive (non-safe) per-token softmax in fp32, NUM_THREADS == H.
  //   one token per block, scalar load. expf each element, then a
  //   block-wide sum reduction; y[idx] = exp_val / exp_sum.
  //   no max-subtract -> may overflow for large x; that's intentional.
}

template <const int NUM_THREADS = 256 / 4>
__global__ void softmax_f32x4_per_token_kernel(float *x, float *y, int N) {
  // TODO: naive per-token softmax fp32 vec4, NUM_THREADS == H/4.
  //   each thread owns 4 elements (128-bit load/store), 4 expf per
  //   thread, then a block-wide sum reduction over per-thread partials.
  //   no max-subtract.
}

// safe_softmax per token
template <const int NUM_THREADS = 256>
__global__ void safe_softmax_f32_per_token_kernel(float *x, float *y, int N) {
  // TODO: safe per-token softmax in fp32, NUM_THREADS == H.
  //   pass1: block-wide max reduction -> max_val.
  //   pass2: expf(x[idx] - max_val); block-wide sum reduction -> exp_sum.
  //   y[idx] = exp_val / exp_sum.
}

template <const int NUM_THREADS = 256 / 4>
__global__ void safe_softmax_f32x4_per_token_kernel(float *x, float *y, int N) {
  // TODO: safe per-token softmax fp32 vec4, NUM_THREADS == H/4.
  //   each thread owns 4 elements via 128-bit load/store; do per-thread
  //   local max over the 4 lanes (clamp OOB to -FLT_MAX) then a block-wide
  //   max reduction; same shape for the sum-of-exp reduction.
}

template <const int NUM_THREADS = 256>
__global__ void safe_softmax_f16_f32_per_token_kernel(half *x, half *y, int N) {
  // TODO: safe per-token softmax fp16 in/out, fp32 accumulator,
  //   NUM_THREADS == H. convert half->float, do max-subtract + sum in
  //   fp32, write back __float2half_rn. scalar load.
}

template <const int NUM_THREADS = 256>
__global__ void safe_softmax_f16x2_f32_per_token_kernel(half *x, half *y,
                                                        int N) {
  // TODO: safe per-token softmax fp16 half2 (2-wide), fp32 acc.
  //   each thread owns 2 elements (NUM_THREADS == H/2), 2-wide load,
  //   convert to fp32 pair, max-subtract + sum in fp32, store as
  //   half2 result.
}

template <const int NUM_THREADS = 256>
__global__ void safe_softmax_f16x8_pack_f32_per_token_kernel(half *x, half *y,
                                                             int N) {
  // TODO: safe per-token softmax fp16x8 pack (128-bit LD/ST),
  //   fp32 accumulator, NUM_THREADS == H/8. each thread owns 8 elements
  //   in a register array (one 128-bit load); reductions in fp32 (mind
  //   OOB lanes); store __float2half_rn(exp_val/exp_sum) into the pack
  //   then one 128-bit store back.
}

template <const int NUM_THREADS = 256>
__global__ void online_safe_softmax_f32_per_token_kernel(const float *x,
                                                         float *y, int N) {
  // TODO: online safe softmax (single-pass), scalar fp32, NUM_THREADS == H.
  //   reference: https://arxiv.org/pdf/1805.02867
  //   one pass over the row, maintaining a running (m, d) pair where
  //   m is the running max and d is the running sum of exp(x - m).
  //   each thread starts with (m=x[gid], d=1.0f) (OOB: (-FLT_MAX, 0));
  //   pairwise merge across threads with the standard online-softmax
  //   update rule (the lower-m side scales by exp(small.m - big.m)),
  //   first across the warp, then across warps via shared memory.
  //   final m,d are broadcast back; y[gid] = expf(x[gid]-m) / d.
}

template <const int NUM_THREADS = 256 / 4>
__global__ void
online_safe_softmax_f32x4_pack_per_token_kernel(float *x, float *y, int N) {
  // TODO: online safe softmax fp32 vec4 (single-pass, 128-bit LD/ST),
  //   NUM_THREADS == H/4. each thread starts by computing a local (m, d)
  //   over its 4 owned elements (m = fmax across the 4, d = sum of
  //   exp(val - local_m)), then merges with other threads via the
  //   warp/block online-softmax reduction. write back 4 outputs via a
  //   single 128-bit store.
}

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

#define CHECK_TORCH_TENSOR_SHAPE(T1, T2)                                       \
  assert((T1).dim() == (T2).dim());                                            \
  for (int i = 0; i < (T1).dim(); ++i) {                                       \
    if ((T2).size(i) != (T1).size(i)) {                                        \
      throw std::runtime_error("Tensor size mismatch!");                       \
    }                                                                          \
  }

// softmax per token
#define LANUCH_SOFTMAX_F32_PER_TOKEN_KERNEL(H)                                 \
  softmax_f32_per_token_kernel<(H)>                                            \
      <<<grid, block>>>(reinterpret_cast<float *>(x.data_ptr()),               \
                        reinterpret_cast<float *>(y.data_ptr()), N);

#define DISPATCH_SOFTMAX_F32_PER_TOKEN_KERNEL(S, H)                            \
  dim3 block((H));                                                             \
  dim3 grid((S));                                                              \
  switch ((H)) {                                                               \
  case 32:                                                                     \
    LANUCH_SOFTMAX_F32_PER_TOKEN_KERNEL(32)                                    \
    break;                                                                     \
  case 64:                                                                     \
    LANUCH_SOFTMAX_F32_PER_TOKEN_KERNEL(64)                                    \
    break;                                                                     \
  case 128:                                                                    \
    LANUCH_SOFTMAX_F32_PER_TOKEN_KERNEL(128)                                   \
    break;                                                                     \
  case 256:                                                                    \
    LANUCH_SOFTMAX_F32_PER_TOKEN_KERNEL(256)                                   \
    break;                                                                     \
  case 512:                                                                    \
    LANUCH_SOFTMAX_F32_PER_TOKEN_KERNEL(512)                                   \
    break;                                                                     \
  case 1024:                                                                   \
    LANUCH_SOFTMAX_F32_PER_TOKEN_KERNEL(1024)                                  \
    break;                                                                     \
  default:                                                                     \
    throw std::runtime_error("only support H: 64/128/256/512/1024");           \
    break;                                                                     \
  }

#define LANUCH_SOFTMAX_F32x4_PER_TOKEN_KERNEL(H)                               \
  softmax_f32x4_per_token_kernel<(H) / 4>                                      \
      <<<grid, block>>>(reinterpret_cast<float *>(x.data_ptr()),               \
                        reinterpret_cast<float *>(y.data_ptr()), N);

#define DISPATCH_SOFTMAX_F32x4_PER_TOKEN_KERNEL(S, H)                          \
  const int NT = (H) / 4;                                                      \
  dim3 block(NT);                                                              \
  dim3 grid((S));                                                              \
  switch (H) {                                                                 \
  case 32:                                                                     \
    LANUCH_SOFTMAX_F32x4_PER_TOKEN_KERNEL(32) break;                           \
  case 64:                                                                     \
    LANUCH_SOFTMAX_F32x4_PER_TOKEN_KERNEL(64) break;                           \
  case 128:                                                                    \
    LANUCH_SOFTMAX_F32x4_PER_TOKEN_KERNEL(128) break;                          \
  case 256:                                                                    \
    LANUCH_SOFTMAX_F32x4_PER_TOKEN_KERNEL(256) break;                          \
  case 512:                                                                    \
    LANUCH_SOFTMAX_F32x4_PER_TOKEN_KERNEL(512) break;                          \
  case 1024:                                                                   \
    LANUCH_SOFTMAX_F32x4_PER_TOKEN_KERNEL(1024) break;                         \
  case 2048:                                                                   \
    LANUCH_SOFTMAX_F32x4_PER_TOKEN_KERNEL(2048) break;                         \
  case 4096:                                                                   \
    LANUCH_SOFTMAX_F32x4_PER_TOKEN_KERNEL(4096) break;                         \
  default:                                                                     \
    throw std::runtime_error("only support H: 64/128/.../1024*4");             \
    break;                                                                     \
  }

// safe softmax per token
#define LANUCH_SAFE_SOFTMAX_F32_PER_TOKEN_KERNEL(H)                            \
  safe_softmax_f32_per_token_kernel<(H)>                                       \
      <<<grid, block>>>(reinterpret_cast<float *>(x.data_ptr()),               \
                        reinterpret_cast<float *>(y.data_ptr()), N);

#define DISPATCH_SATE_SOFTMAX_F32_PER_TOKEN_KERNEL(S, H)                       \
  dim3 block((H));                                                             \
  dim3 grid((S));                                                              \
  switch ((H)) {                                                               \
  case 32:                                                                     \
    LANUCH_SAFE_SOFTMAX_F32_PER_TOKEN_KERNEL(32)                               \
    break;                                                                     \
  case 64:                                                                     \
    LANUCH_SAFE_SOFTMAX_F32_PER_TOKEN_KERNEL(64)                               \
    break;                                                                     \
  case 128:                                                                    \
    LANUCH_SAFE_SOFTMAX_F32_PER_TOKEN_KERNEL(128)                              \
    break;                                                                     \
  case 256:                                                                    \
    LANUCH_SAFE_SOFTMAX_F32_PER_TOKEN_KERNEL(256)                              \
    break;                                                                     \
  case 512:                                                                    \
    LANUCH_SAFE_SOFTMAX_F32_PER_TOKEN_KERNEL(512)                              \
    break;                                                                     \
  case 1024:                                                                   \
    LANUCH_SAFE_SOFTMAX_F32_PER_TOKEN_KERNEL(1024)                             \
    break;                                                                     \
  default:                                                                     \
    throw std::runtime_error("only support H: 64/128/256/512/1024");           \
    break;                                                                     \
  }

// online softmax per token
#define LANUCH_ONLINE_SOFTMAX_F32_PER_TOKEN_KERNEL(H)                          \
  online_safe_softmax_f32_per_token_kernel<(H)>                                \
      <<<grid, block>>>(reinterpret_cast<float *>(x.data_ptr()),               \
                        reinterpret_cast<float *>(y.data_ptr()), N);

#define DISPATCH_ONLINE_SOFTMAX_F32_PER_TOKEN_KERNEL(S, H)                     \
  dim3 block((H));                                                             \
  dim3 grid((S));                                                              \
  switch ((H)) {                                                               \
  case 32:                                                                     \
    LANUCH_ONLINE_SOFTMAX_F32_PER_TOKEN_KERNEL(32)                             \
    break;                                                                     \
  case 64:                                                                     \
    LANUCH_ONLINE_SOFTMAX_F32_PER_TOKEN_KERNEL(64)                             \
    break;                                                                     \
  case 128:                                                                    \
    LANUCH_ONLINE_SOFTMAX_F32_PER_TOKEN_KERNEL(128)                            \
    break;                                                                     \
  case 256:                                                                    \
    LANUCH_ONLINE_SOFTMAX_F32_PER_TOKEN_KERNEL(256)                            \
    break;                                                                     \
  case 512:                                                                    \
    LANUCH_ONLINE_SOFTMAX_F32_PER_TOKEN_KERNEL(512)                            \
    break;                                                                     \
  case 1024:                                                                   \
    LANUCH_ONLINE_SOFTMAX_F32_PER_TOKEN_KERNEL(1024)                           \
    break;                                                                     \
  default:                                                                     \
    throw std::runtime_error("only support H: 64/128/256/512/1024");           \
    break;                                                                     \
  }

// online softmax per token (vec4 pack)
#define LANUCH_ONLINE_SOFTMAX_F32X4_PACK_PER_TOKEN_KERNEL(H)                   \
  online_safe_softmax_f32x4_pack_per_token_kernel<(H / 4)>                     \
      <<<grid, block>>>(reinterpret_cast<float *>(x.data_ptr()),               \
                        reinterpret_cast<float *>(y.data_ptr()), N);

#define DISPATCH_ONLINE_SOFTMAX_F32X4_PACK_PER_TOKEN_KERNEL(S, H)              \
  dim3 block((H / 4));                                                         \
  dim3 grid((S));                                                              \
  switch ((H)) {                                                               \
  case 128:                                                                    \
    LANUCH_ONLINE_SOFTMAX_F32X4_PACK_PER_TOKEN_KERNEL(128)                     \
    break;                                                                     \
  case 256:                                                                    \
    LANUCH_ONLINE_SOFTMAX_F32X4_PACK_PER_TOKEN_KERNEL(256)                     \
    break;                                                                     \
  case 512:                                                                    \
    LANUCH_ONLINE_SOFTMAX_F32X4_PACK_PER_TOKEN_KERNEL(512)                     \
    break;                                                                     \
  case 1024:                                                                   \
    LANUCH_ONLINE_SOFTMAX_F32X4_PACK_PER_TOKEN_KERNEL(1024)                    \
    break;                                                                     \
  case 2048:                                                                   \
    LANUCH_ONLINE_SOFTMAX_F32X4_PACK_PER_TOKEN_KERNEL(2048)                    \
    break;                                                                     \
  case 4096:                                                                   \
    LANUCH_ONLINE_SOFTMAX_F32X4_PACK_PER_TOKEN_KERNEL(4096)                    \
    break;                                                                     \
  default:                                                                     \
    throw std::runtime_error("only support H: 128/256/.../4096;");             \
    break;                                                                     \
  }

#define LANUCH_SAFE_SOFTMAX_F32x4_PER_TOKEN_KERNEL(H)                          \
  safe_softmax_f32x4_per_token_kernel<(H) / 4>                                 \
      <<<grid, block>>>(reinterpret_cast<float *>(x.data_ptr()),               \
                        reinterpret_cast<float *>(y.data_ptr()), N);

#define DISPATCH_SATE_SOFTMAX_F32x4_PER_TOKEN_KERNEL(S, H)                     \
  const int NT = (H) / 4;                                                      \
  dim3 block(NT);                                                              \
  dim3 grid((S));                                                              \
  switch (H) {                                                                 \
  case 32:                                                                     \
    LANUCH_SAFE_SOFTMAX_F32x4_PER_TOKEN_KERNEL(32) break;                      \
  case 64:                                                                     \
    LANUCH_SAFE_SOFTMAX_F32x4_PER_TOKEN_KERNEL(64) break;                      \
  case 128:                                                                    \
    LANUCH_SAFE_SOFTMAX_F32x4_PER_TOKEN_KERNEL(128) break;                     \
  case 256:                                                                    \
    LANUCH_SAFE_SOFTMAX_F32x4_PER_TOKEN_KERNEL(256) break;                     \
  case 512:                                                                    \
    LANUCH_SAFE_SOFTMAX_F32x4_PER_TOKEN_KERNEL(512) break;                     \
  case 1024:                                                                   \
    LANUCH_SAFE_SOFTMAX_F32x4_PER_TOKEN_KERNEL(1024) break;                    \
  case 2048:                                                                   \
    LANUCH_SAFE_SOFTMAX_F32x4_PER_TOKEN_KERNEL(2048) break;                    \
  case 4096:                                                                   \
    LANUCH_SAFE_SOFTMAX_F32x4_PER_TOKEN_KERNEL(4096) break;                    \
  default:                                                                     \
    throw std::runtime_error("only support H: 64/128/.../1024*4");             \
    break;                                                                     \
  }

#define LANUCH_SAFE_SOFTMAX_F16_F32_PER_TOKEN_KERNEL(H)                        \
  safe_softmax_f16_f32_per_token_kernel<(H)>                                   \
      <<<grid, block>>>(reinterpret_cast<half *>(x.data_ptr()),                \
                        reinterpret_cast<half *>(y.data_ptr()), N);

#define DISPATCH_SATE_SOFTMAX_F16_F32_PER_TOKEN_KERNEL(S, H)                   \
  dim3 block((H));                                                             \
  dim3 grid((S));                                                              \
  switch ((H)) {                                                               \
  case 32:                                                                     \
    LANUCH_SAFE_SOFTMAX_F16_F32_PER_TOKEN_KERNEL(32)                           \
    break;                                                                     \
  case 64:                                                                     \
    LANUCH_SAFE_SOFTMAX_F16_F32_PER_TOKEN_KERNEL(64)                           \
    break;                                                                     \
  case 128:                                                                    \
    LANUCH_SAFE_SOFTMAX_F16_F32_PER_TOKEN_KERNEL(128)                          \
    break;                                                                     \
  case 256:                                                                    \
    LANUCH_SAFE_SOFTMAX_F16_F32_PER_TOKEN_KERNEL(256)                          \
    break;                                                                     \
  case 512:                                                                    \
    LANUCH_SAFE_SOFTMAX_F16_F32_PER_TOKEN_KERNEL(512)                          \
    break;                                                                     \
  case 1024:                                                                   \
    LANUCH_SAFE_SOFTMAX_F16_F32_PER_TOKEN_KERNEL(1024)                         \
    break;                                                                     \
  default:                                                                     \
    throw std::runtime_error("only support H: 64/128/256/512/1024");           \
    break;                                                                     \
  }

#define LANUCH_SAFE_SOFTMAX_F16x2_F32_PER_TOKEN_KERNEL(H)                      \
  safe_softmax_f16x2_f32_per_token_kernel<(H) / 2>                             \
      <<<grid, block>>>(reinterpret_cast<half *>(x.data_ptr()),                \
                        reinterpret_cast<half *>(y.data_ptr()), N);

#define DISPATCH_SATE_SOFTMAX_F16x2_F32_PER_TOKEN_KERNEL(S, H)                 \
  const int NT = (H) / 2;                                                      \
  dim3 block(NT);                                                              \
  dim3 grid((S));                                                              \
  switch (H) {                                                                 \
  case 32:                                                                     \
    LANUCH_SAFE_SOFTMAX_F16x2_F32_PER_TOKEN_KERNEL(32) break;                  \
  case 64:                                                                     \
    LANUCH_SAFE_SOFTMAX_F16x2_F32_PER_TOKEN_KERNEL(64) break;                  \
  case 128:                                                                    \
    LANUCH_SAFE_SOFTMAX_F16x2_F32_PER_TOKEN_KERNEL(128) break;                 \
  case 256:                                                                    \
    LANUCH_SAFE_SOFTMAX_F16x2_F32_PER_TOKEN_KERNEL(256) break;                 \
  case 512:                                                                    \
    LANUCH_SAFE_SOFTMAX_F16x2_F32_PER_TOKEN_KERNEL(512) break;                 \
  case 1024:                                                                   \
    LANUCH_SAFE_SOFTMAX_F16x2_F32_PER_TOKEN_KERNEL(1024) break;                \
  case 2048:                                                                   \
    LANUCH_SAFE_SOFTMAX_F16x2_F32_PER_TOKEN_KERNEL(2048) break;                \
  default:                                                                     \
    throw std::runtime_error("only support H: 64/128/.../1024*2");             \
    break;                                                                     \
  }

#define LANUCH_SAFE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(H)                 \
  safe_softmax_f16x8_pack_f32_per_token_kernel<(H) / 8>                        \
      <<<grid, block>>>(reinterpret_cast<half *>(x.data_ptr()),                \
                        reinterpret_cast<half *>(y.data_ptr()), N);

#define DISPATCH_SATE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(S, H)            \
  const int NT = (H) / 8;                                                      \
  dim3 block(NT);                                                              \
  dim3 grid((S));                                                              \
  switch (H) {                                                                 \
  case 32:                                                                     \
    LANUCH_SAFE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(32) break;             \
  case 64:                                                                     \
    LANUCH_SAFE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(64) break;             \
  case 128:                                                                    \
    LANUCH_SAFE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(128) break;            \
  case 256:                                                                    \
    LANUCH_SAFE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(256) break;            \
  case 512:                                                                    \
    LANUCH_SAFE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(512) break;            \
  case 1024:                                                                   \
    LANUCH_SAFE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(1024) break;           \
  case 2048:                                                                   \
    LANUCH_SAFE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(2048) break;           \
  case 4096:                                                                   \
    LANUCH_SAFE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(4096) break;           \
  case 8192:                                                                   \
    LANUCH_SAFE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(8192) break;           \
  default:                                                                     \
    throw std::runtime_error("only support H: 64/128/.../1024*8");             \
    break;                                                                     \
  }

// per token fp32
void softmax_f32_per_token(torch::Tensor x, torch::Tensor y) {
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat32)
  CHECK_TORCH_TENSOR_SHAPE(x, y)
  const int S = x.size(0); // seqlens
  const int H = x.size(1); // head size/kv_len
  const int N = S * H;
  DISPATCH_SOFTMAX_F32_PER_TOKEN_KERNEL(S, H)
}

void softmax_f32x4_per_token(torch::Tensor x, torch::Tensor y) {
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat32)
  CHECK_TORCH_TENSOR_SHAPE(x, y)
  const int S = x.size(0); // seqlens
  const int H = x.size(1); // head size/kv_len
  const int N = S * H;
  DISPATCH_SOFTMAX_F32x4_PER_TOKEN_KERNEL(S, H)
}

void safe_softmax_f32_per_token(torch::Tensor x, torch::Tensor y) {
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat32)
  CHECK_TORCH_TENSOR_SHAPE(x, y)
  const int S = x.size(0); // seqlens
  const int H = x.size(1); // head size/kv_len
  const int N = S * H;
  DISPATCH_SATE_SOFTMAX_F32_PER_TOKEN_KERNEL(S, H)
}

void safe_softmax_f32x4_per_token(torch::Tensor x, torch::Tensor y) {
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat32)
  CHECK_TORCH_TENSOR_SHAPE(x, y)
  const int S = x.size(0); // seqlens
  const int H = x.size(1); // head size/kv_len
  const int N = S * H;
  DISPATCH_SATE_SOFTMAX_F32x4_PER_TOKEN_KERNEL(S, H)
}

// per token fp16
void safe_softmax_f16_f32_per_token(torch::Tensor x, torch::Tensor y) {
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kHalf)
  CHECK_TORCH_TENSOR_SHAPE(x, y)
  const int S = x.size(0); // seqlens
  const int H = x.size(1); // head size/kv_len
  const int N = S * H;
  DISPATCH_SATE_SOFTMAX_F16_F32_PER_TOKEN_KERNEL(S, H)
}

void safe_softmax_f16x2_f32_per_token(torch::Tensor x, torch::Tensor y) {
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kHalf)
  CHECK_TORCH_TENSOR_SHAPE(x, y)
  const int S = x.size(0); // seqlens
  const int H = x.size(1); // head size/kv_len
  const int N = S * H;
  DISPATCH_SATE_SOFTMAX_F16x2_F32_PER_TOKEN_KERNEL(S, H)
}

void safe_softmax_f16x8_pack_f32_per_token(torch::Tensor x, torch::Tensor y) {
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kHalf)
  CHECK_TORCH_TENSOR_SHAPE(x, y)
  const int S = x.size(0); // seqlens
  const int H = x.size(1); // head size/kv_len
  const int N = S * H;
  DISPATCH_SATE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(S, H)
}

void online_safe_softmax_f32_per_token(torch::Tensor x, torch::Tensor y) {
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat32)
  CHECK_TORCH_TENSOR_SHAPE(x, y)
  const int S = x.size(0); // seqlens
  const int H = x.size(1); // head size/kv_len
  const int N = S * H;
  DISPATCH_ONLINE_SOFTMAX_F32_PER_TOKEN_KERNEL(S, H)
}

void online_safe_softmax_f32x4_pack_per_token(torch::Tensor x,
                                              torch::Tensor y) {
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat32)
  CHECK_TORCH_TENSOR_SHAPE(x, y)
  const int S = x.size(0);
  const int H = x.size(1);
  const int N = S * H;
  DISPATCH_ONLINE_SOFTMAX_F32X4_PACK_PER_TOKEN_KERNEL(S, H)
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(softmax_f32_per_token)
  TORCH_BINDING_COMMON_EXTENSION(softmax_f32x4_per_token)
  TORCH_BINDING_COMMON_EXTENSION(safe_softmax_f32_per_token)
  TORCH_BINDING_COMMON_EXTENSION(safe_softmax_f32x4_per_token)
  TORCH_BINDING_COMMON_EXTENSION(safe_softmax_f16_f32_per_token)
  TORCH_BINDING_COMMON_EXTENSION(safe_softmax_f16x2_f32_per_token)
  TORCH_BINDING_COMMON_EXTENSION(safe_softmax_f16x8_pack_f32_per_token)
  TORCH_BINDING_COMMON_EXTENSION(online_safe_softmax_f32_per_token)
  TORCH_BINDING_COMMON_EXTENSION(online_safe_softmax_f32x4_pack_per_token)
}
