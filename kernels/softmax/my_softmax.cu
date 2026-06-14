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

#define WARP_SIZE 32
#define FLOAT4(val) (reinterpret_cast<float4*>(&(val)))[0]
#define HALF2(val) (reinterpret_cast<half2*>(&(val)))[0]
#define LDST128BITS(val) (reinterpret_cast<float4*>(&(val)))[0]

template<const int reduce_size>
__device__ __forceinline__ float warp_reduce_sum(float exp_sum) {
#pragma unroll
  for (int mask = reduce_size / 2; mask >= 1; mask >>= 1) {
    exp_sum += __shfl_xor_sync(0xffffffff, exp_sum, mask);
  }
  return exp_sum;
}

template<const int NUM_THREADS>
__device__ __forceinline__ float block_reduce_sum(float exp_sum) {
  const int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ float shmem[NUM_WARPS];
  int tid = threadIdx.x;
  int lane_id = tid % WARP_SIZE;
  int warp_id = tid / WARP_SIZE;

  exp_sum = warp_reduce_sum<WARP_SIZE>(exp_sum);

  if (lane_id == 0) {
    shmem[warp_id] = exp_sum;
  }
  __syncthreads();

  exp_sum = (lane_id < NUM_WARPS) ? shmem[lane_id] : 0.0f;
  exp_sum = warp_reduce_sum<NUM_WARPS>(exp_sum);
  exp_sum = __shfl_sync(0xffffffff, exp_sum, 0);

  return exp_sum;
}

template<const int reduce_size>
__device__ __forceinline__ float warp_reduce_max(float x_max) {
#pragma unroll
  for (int mask = reduce_size / 2; mask >= 1; mask >>= 1) {
    x_max = fmaxf(x_max, __shfl_xor_sync(0xffffffff, x_max, mask));
  }
  return x_max;
}

template<const int NUM_THREADS>
__device__ __forceinline__ float block_reduce_max(float x_max) {
  const int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ float shmem[NUM_WARPS];
  int tid = threadIdx.x;
  int lane_id = tid % WARP_SIZE;
  int warp_id = tid / WARP_SIZE;

  x_max = warp_reduce_max<WARP_SIZE>(x_max);
  if (lane_id == 0) {
    shmem[warp_id] = x_max;
  }
  __syncthreads();

  x_max = (lane_id < NUM_WARPS) ? shmem[lane_id] : -FLT_MAX;
  x_max = warp_reduce_max<NUM_WARPS>(x_max);
  x_max = __shfl_sync(0xffffffff, x_max, 0);

  return x_max;
}

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
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  float exp_sum, exp_x;
  exp_x = (idx < N) ? expf(x[idx]) : 0.0f;
  exp_sum = block_reduce_sum<NUM_THREADS>(exp_x);
  
  float factor_exp_sum = __fdividef(1.0f, exp_sum);
  if (idx < N) {
    y[idx] = factor_exp_sum * exp_x;
  }
}

template <const int NUM_THREADS = 256 / 4>
__global__ void softmax_f32x4_per_token_kernel(float *x, float *y, int N) {
  // TODO: naive per-token softmax fp32 vec4, NUM_THREADS == H/4.
  //   each thread owns 4 elements (128-bit load/store), 4 expf per
  //   thread, then a block-wide sum reduction over per-thread partials.
  //   no max-subtract.
  int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
  float x_pack[4], exp_x_pack[4], y_pack[4], exp_sum = 0.0f;
  if (idx < N) {
    FLOAT4(x_pack[0]) = FLOAT4(x[idx]);
#pragma unroll
    for (int i = 0; i < 4; i++) {
      exp_x_pack[i] = expf(x_pack[i]);
      exp_sum += exp_x_pack[i];
    }
  }

  exp_sum = block_reduce_sum<NUM_THREADS>(exp_sum);

  float factor_exp_sum = __fdividef(1.0f, exp_sum);
  if (idx < N) {
#pragma unroll
    for(int i = 0; i < 4; i++) {
      y_pack[i] = factor_exp_sum * exp_x_pack[i];
    }
    FLOAT4(y[idx]) = FLOAT4(y_pack[0]);
  }
}

// safe_softmax per token
template <const int NUM_THREADS = 256>
__global__ void safe_softmax_f32_per_token_kernel(float *x, float *y, int N) {
  // TODO: safe per-token softmax in fp32, NUM_THREADS == H.
  //   pass1: block-wide max reduction -> max_val.
  //   pass2: expf(x[idx] - max_val); block-wide sum reduction -> exp_sum.
  //   y[idx] = exp_val / exp_sum.
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  float reg_x = (idx < N) ? x[idx] : -FLT_MAX;
  float x_max = block_reduce_max<NUM_THREADS>(reg_x);
  float exp_x = expf(reg_x - x_max);
  float exp_sum = block_reduce_sum<NUM_THREADS>(exp_x);
  float factor_exp_sum = __fdividef(1.0f, exp_sum);
  if (idx < N) {
    y[idx] = factor_exp_sum * exp_x;
  }
}

template <const int NUM_THREADS = 256 / 4>
__global__ void safe_softmax_f32x4_per_token_kernel(float *x, float *y, int N) {
  // TODO: safe per-token softmax fp32 vec4, NUM_THREADS == H/4.
  //   each thread owns 4 elements via 128-bit load/store; do per-thread
  //   local max over the 4 lanes (clamp OOB to -FLT_MAX) then a block-wide
  //   max reduction; same shape for the sum-of-exp reduction.
  int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
  float x_pack[4], exp_x[4], y_pack[4];
  FLOAT4(x_pack[0]) = FLOAT4(x[idx]);
#pragma unroll
  for (int i = 0; i < 4; i++) {
    x_pack[i] = (idx + i < N) ? x_pack[i] : -FLT_MAX;
  }
  float x_max = fmaxf(fmaxf(x_pack[0], x_pack[1]), fmaxf(x_pack[2], x_pack[3]));
  x_max = block_reduce_max<NUM_THREADS>(x_max);

  float exp_sum = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; i++) {
    exp_x[i] = (idx + i < N) ? expf(x_pack[i] - x_max) : 0.0f;
    exp_sum += exp_x[i];
  }
  exp_sum = block_reduce_sum<NUM_THREADS>(exp_sum);

  float factor_exp_sum = __fdividef(1.0f, exp_sum);
#pragma unroll
  for (int i = 0; i < 4; i++) {
    y_pack[i] = factor_exp_sum * exp_x[i];    
  }
  FLOAT4(y[idx]) = FLOAT4(y_pack[0]);
}

template<const int reduce_size>
__device__ __forceinline__ half warp_reduce_max(half x_max) {
#pragma unroll
  for (int mask = reduce_size >> 1; mask >= 1; mask >>= 1) {
    x_max = __hmax(x_max , __shfl_xor_sync(0xffffffff, x_max, mask));
  }
  return x_max;
}

template<const int NUM_THREADS>
__device__ __forceinline__ half block_reduce_max(half x_max) {
  const int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ half shmem[NUM_WARPS];
  int tid = threadIdx.x;
  int lane_id = tid % WARP_SIZE;
  int warp_id = tid / WARP_SIZE;
  x_max = warp_reduce_max<WARP_SIZE>(x_max);
  if (lane_id == 0) {
    shmem[warp_id] = x_max;
  }
  __syncthreads();
  x_max = (lane_id < NUM_WARPS) ? shmem[lane_id] : -CUDART_MAX_NORMAL_FP16;
  x_max = warp_reduce_max<NUM_WARPS>(x_max);
  x_max = __shfl_sync(0xffffffff, x_max, 0);
  return x_max;
}

template <const int NUM_THREADS = 256>
__global__ void safe_softmax_f16_f32_per_token_kernel(half *x, half *y, int N) {
  // TODO: safe per-token softmax fp16 in/out, fp32 accumulator,
  //   NUM_THREADS == H. convert half->float, do max-subtract + sum in
  //   fp32, write back __float2half_rn. scalar load.
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  half reg_x = (idx < N) ? x[idx] : -CUDART_MAX_NORMAL_FP16;
  half x_max = block_reduce_max<NUM_THREADS>(reg_x);
  float exp_x = (idx < N) ? expf(__half2float(reg_x) - __half2float(x_max)) : 0.0f;
  float exp_sum = block_reduce_sum<NUM_THREADS>(exp_x);
  float factor_exp_sum = __fdividef(1.0f, exp_sum);
  if (idx < N) {
    y[idx] = __float2half(factor_exp_sum * exp_x);
  }
}

template <const int NUM_THREADS = 256>
__global__ void safe_softmax_f16x2_f32_per_token_kernel(half *x, half *y,
                                                        int N) {
  // TODO: safe per-token softmax fp16 half2 (2-wide), fp32 acc.
  //   each thread owns 2 elements (NUM_THREADS == H/2), 2-wide load,
  //   convert to fp32 pair, max-subtract + sum in fp32, store as
  //   half2 result.
  int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
  float2 reg_x = __half22float2(HALF2(x[idx]));
  reg_x.x = (idx + 0 < N) ? reg_x.x : -FLT_MAX;
  reg_x.y = (idx + 1 < N) ? reg_x.y : -FLT_MAX;
  float x_max = fmaxf(reg_x.x, reg_x.y);
  x_max = block_reduce_max<NUM_THREADS>(x_max);
  float exp_x_x, exp_x_y;
  exp_x_x = (idx + 0 < N) ? expf(reg_x.x - x_max) : 0.0f;
  exp_x_y = (idx + 1 < N) ? expf(reg_x.y - x_max) : 0.0f;
  float exp_sum = exp_x_x + exp_x_y;
  exp_sum = block_reduce_sum<NUM_THREADS>(exp_sum);
  float factor_exp_sum = __fdividef(1.0f, exp_sum);
  float2 reg_y;
  reg_y.x = (idx + 0 < N) ? factor_exp_sum * exp_x_x : 0.0f;
  reg_y.y = (idx + 1 < N) ? factor_exp_sum * exp_x_y : 0.0f;
  HALF2(y[idx]) = __float22half2_rn(reg_y);
}

template <const int NUM_THREADS = 256>
__global__ void safe_softmax_f16x8_pack_f32_per_token_kernel(half *x, half *y,
                                                             int N) {
  // TODO: safe per-token softmax fp16x8 pack (128-bit LD/ST),
  //   fp32 accumulator, NUM_THREADS == H/8. each thread owns 8 elements
  //   in a register array (one 128-bit load); reductions in fp32 (mind
  //   OOB lanes); store __float2half_rn(exp_val/exp_sum) into the pack
  //   then one 128-bit store back.
  int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
  half x_pack[8];
  LDST128BITS(x_pack[0]) = LDST128BITS(x[idx]);
  float x_fp32[8];
#pragma unroll
  for (int i = 0; i < 8; i++) {
    x_fp32[i] = (idx + i < N) ? __half2float(x_pack[i]) : -FLT_MAX;
  }
  float x_max = -FLT_MAX;
#pragma unroll
  for (int i = 0; i < 8; i++) {
    x_max = fmaxf(x_max, x_fp32[i]);
  }
  x_max = block_reduce_max<NUM_THREADS>(x_max);

  float exp_x[8];
#pragma unroll
  for (int i = 0; i < 8; i++) {
    exp_x[i] = (idx + i < N)? expf(x_fp32[i] - x_max) : 0.0f;
  }
  float exp_sum = 0.0;
#pragma unroll
  for (int i = 0; i < 8; i++) {
    exp_sum += exp_x[i];
  }
  exp_sum = block_reduce_sum<NUM_THREADS>(exp_sum);

  float factor_exp_sum = __fdividef(1.0f, exp_sum);
  half y_pack[8];
#pragma unroll
  for (int i = 0; i < 8; i++) {
    y_pack[i] = __float2half_rn(factor_exp_sum * exp_x[i]);
  }
  LDST128BITS(y[idx]) = LDST128BITS(y_pack[0]);
}

struct __align__(8) MD {
  float m;
  float d;
};

template<const int reduce_size>
__device__ __forceinline__ MD warp_reduce_update_md(MD val) {
  MD other;
  for (int mask = reduce_size >> 1; mask >= 1; mask >>= 1) {
    other.m = __shfl_xor_sync(0xffffffff, val.m, mask);
    other.d = __shfl_xor_sync(0xffffffff, val.d, mask);
    MD greater, smaller;
    bool is_greater = other.m > val.m;
    greater = is_greater ? other : val;
    smaller = is_greater ? val : other;
    val.m = greater.m;
    val.d = greater.d + smaller.d * expf(smaller.m - greater.m);
  }
  return val;
}

template<const int NUM_THREADS>
__device__ __forceinline__ MD block_reduce_update_md(MD val) {
  const int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ MD shmem[NUM_WARPS];
  int tid = threadIdx.x;
  int lane_id = tid % WARP_SIZE;
  int warp_id = tid / WARP_SIZE;
  val = warp_reduce_update_md<WARP_SIZE>(val);

  if (lane_id == 0) {
    shmem[warp_id] = val;
  }
  __syncthreads();

  val = shmem[lane_id];
  val = warp_reduce_update_md<NUM_WARPS>(val);
  val.m = __shfl_sync(0xffffffff, val.m, 0);
  val.d = __shfl_sync(0xffffffff, val.d, 0);
  return val;
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
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  float reg_x = (idx < N) ? x[idx] : -FLT_MAX;
  MD val;
  val.m = (idx < N) ? reg_x : -FLT_MAX;
  val.d = (idx < N) ? 1.0f : 0.0f;
  val = block_reduce_update_md<NUM_THREADS>(val);
  float factor_exp_sum = __fdividef(1.0f, val.d);
  if (idx < N) {
    y[idx] = factor_exp_sum * expf(reg_x - val.m);
  }
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
  int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
  float4 reg_x = FLOAT4(x[idx]);
  float local_m = fmaxf(fmaxf(reg_x.x, reg_x.y), fmaxf(reg_x.z, reg_x.w));
  float local_d = expf(reg_x.x - local_m) + expf(reg_x.y - local_m)
                + expf(reg_x.z - local_m) + expf(reg_x.w - local_m);
  MD val = {local_m, local_d};
  val = block_reduce_update_md<NUM_THREADS>(val);
  float4 reg_y;
  float factor_exp_sum = __fdividef(1.0f, val.d);
  reg_y.x = factor_exp_sum * expf(reg_x.x -  val.m);
  reg_y.y = factor_exp_sum * expf(reg_x.y -  val.m);
  reg_y.z = factor_exp_sum * expf(reg_x.z -  val.m);
  reg_y.w = factor_exp_sum * expf(reg_x.w -  val.m);
  FLOAT4(y[idx]) = reg_y;
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
