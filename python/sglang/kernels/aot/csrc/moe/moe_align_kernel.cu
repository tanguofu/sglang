/* Copyright 2025 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <THC/THCAtomics.cuh>

#include "utils.h"

#define VEC_SIZE 4
using Vec = int4;

template <typename scalar_t>
__global__ void count_and_sort_expert_tokens_kernel(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ cumsum_buffer,
    size_t numel,
    bool ignore_invalid_expert) {
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = blockDim.x * gridDim.x;

  for (size_t i = tid; i < numel; i += stride) {
    if (ignore_invalid_expert && topk_ids[i] < 0) continue;
    int32_t expert_id = topk_ids[i] + 1;
    int32_t rank_post_pad = atomicAdd(&cumsum_buffer[expert_id], 1);
    sorted_token_ids[rank_post_pad] = i;
  }
}

#ifdef __CUDA_ARCH__
__device__ __forceinline__ int warp_exclusive_scan(int v, unsigned mask = 0xffffffffu) {
  int original = v;
#pragma unroll
  for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
    int n = __shfl_up_sync(mask, v, offset);
    if ((threadIdx.x & (WARP_SIZE - 1)) >= offset) v += n;
  }
  return v - original;
}
#endif

template <typename scalar_t>
__global__ void moe_align_block_size_kernel(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad,
    int32_t num_experts,
    int32_t block_size,
    size_t numel,
    int32_t* __restrict__ cumsum,
    bool pad_sorted_token_ids,
    bool ignore_invalid_expert,
    const int32_t scan_size,
    int32_t max_num_tokens_padded) {
  // Use a separate thread block to populate sorted_token_ids
  if (blockIdx.x == 1) {
    if (pad_sorted_token_ids) {
      Vec fill_vec;
      fill_vec.x = fill_vec.y = fill_vec.z = fill_vec.w = numel;
      // Whole vectors only; a partial last vector would write past the buffer.
      int32_t total_vecs = max_num_tokens_padded / VEC_SIZE;
      Vec* out_ptr = reinterpret_cast<Vec*>(sorted_token_ids);
      for (int32_t i = threadIdx.x; i < total_vecs; i += blockDim.x) {
        out_ptr[i] = fill_vec;
      }
      for (int32_t i = total_vecs * VEC_SIZE + threadIdx.x; i < max_num_tokens_padded; i += blockDim.x) {
        sorted_token_ids[i] = numel;
      }
    }
    return;
  }

  extern __shared__ int32_t smem[];
  int32_t* shared_counts = smem;                  // [num_experts]
  int32_t* prefix = shared_counts + num_experts;  // [num_experts + 1]
  int32_t* scan_buf = prefix + num_experts + 1;   // [scan_size]
  __shared__ int32_t s_total_tokens_post_pad;

  const size_t tid = threadIdx.x;
  const size_t stride = blockDim.x;

  if (tid < num_experts) {
    shared_counts[tid] = 0;
  }

  __syncthreads();

  for (size_t i = tid; i < numel; i += stride) {
    if (ignore_invalid_expert && topk_ids[i] < 0) continue;
    int expert_id = topk_ids[i] + 1;
    atomicAdd(&shared_counts[expert_id], 1);
  }

  __syncthreads();

  int32_t padded_count = 0;
  if (tid < num_experts) {
    int32_t count = shared_counts[tid];
    padded_count = (count + block_size - 1) / block_size * block_size;
    scan_buf[tid] = padded_count;
  }

#ifndef __CUDA_ARCH__  // HIP

  if (tid >= num_experts && tid < scan_size) {
    scan_buf[tid] = 0;
  }

  __syncthreads();

  // Blelloch scan
  int offset = 1;
#pragma unroll
  for (int d = scan_size >> 1; d > 0; d >>= 1) {
    if (tid < d) {
      int ai = offset * (2 * tid + 1) - 1;
      int bi = offset * (2 * tid + 2) - 1;
      scan_buf[bi] += scan_buf[ai];
    }
    offset <<= 1;
    __syncthreads();
  }

  // down-sweep
  if (tid == 0) {
    prefix[num_experts] = scan_buf[scan_size - 1];
    scan_buf[scan_size - 1] = 0;
  }
  __syncthreads();

#pragma unroll
  for (int d = 1; d < scan_size; d <<= 1) {
    offset >>= 1;
    if (tid < d) {
      int ai = offset * (2 * tid + 1) - 1;
      int bi = offset * (2 * tid + 2) - 1;
      if (bi < scan_size) {
        int temp = scan_buf[ai];
        scan_buf[ai] = scan_buf[bi];
        scan_buf[bi] += temp;
      }
    }
    __syncthreads();
  }

  if (tid < num_experts) {
    prefix[tid] = scan_buf[tid];
  }

  if (tid == 0) {
    s_total_tokens_post_pad = prefix[num_experts];
    *total_tokens_post_pad = s_total_tokens_post_pad;
  }
  __syncthreads();

#else  // CUDA

  // Intra warp prefix sum
  int32_t* warp_sums = scan_buf + scan_size;  // [<= 32]
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid & (WARP_SIZE - 1);
  const int num_warps_for_scan = (scan_size + WARP_SIZE - 1) / WARP_SIZE;
  const int warp_sum = warp_exclusive_scan(padded_count) + padded_count;
  if (lane_id == WARP_SIZE - 1) warp_sums[warp_id] = warp_sum;
  __syncthreads();

  // warp0 accumulate all the block's prefix sum
  if (tid < WARP_SIZE) {
    int val = (tid < num_warps_for_scan) ? warp_sums[tid] : 0;
    int incl = warp_exclusive_scan(val) + val;
    warp_sums[tid] = incl;
  }
  __syncthreads();

  // Every thread obtains the whole block's sum
  if (tid == 0) {
    prefix[num_experts] = warp_sums[num_warps_for_scan - 1];
    s_total_tokens_post_pad = prefix[num_experts];
    *total_tokens_post_pad = s_total_tokens_post_pad;
  }
  __syncthreads();

  // Fill 0 to scan_buf extended area (tid >= num_expert)
  if (tid >= num_experts && tid < scan_size) scan_buf[tid] = 0;
  __syncthreads();

  // Perform 2 level exclusive-prefix-sum to scan_buf
  int v = (tid < scan_size) ? scan_buf[tid] : 0;
  int pre = warp_exclusive_scan(v);
  if (lane_id == WARP_SIZE - 1) warp_sums[warp_id] = pre + v;
  __syncthreads();

  if (warp_id == 0) {
    int val = (lane_id < num_warps_for_scan) ? warp_sums[lane_id] : 0;
    warp_sums[lane_id] = warp_exclusive_scan(val);
  }
  __syncthreads();

  int offset = warp_sums[warp_id];
  if (tid < scan_size) scan_buf[tid] = pre + offset;
  __syncthreads();

  // Write prefix[0..num_experts - 1] and cumsum
  if (tid < num_experts) prefix[tid] = scan_buf[tid];
#endif

  if (tid <= num_experts) {
    cumsum[tid] = prefix[tid];
  }
  // fill expert_ids
  const int32_t num_blocks = s_total_tokens_post_pad / block_size;
  for (int32_t i = tid; i < num_blocks; i += stride) {
    int32_t block_start = i * block_size;
    int left = 0, right = num_experts;
    while (left < right) {
      int mid = (left + right) >> 1;
      if (prefix[mid] <= block_start) {
        left = mid + 1;
      } else {
        right = mid;
      }
    }
    expert_ids[i] = left - 2;
  }
}

template <typename scalar_t, int32_t fill_threads>
__global__ void moe_align_block_size_small_batch_expert_kernel(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad,
    int32_t num_experts,
    int32_t block_size,
    size_t numel,
    bool pad_sorted_token_ids,
    bool ignore_invalid_expert,
    int32_t max_num_tokens_padded) {
  // Adapted from
  // https://github.com/vllm-project/vllm/pull/29642/files#diff-5647b1413f4ae9aacba904eca8f8a8aee9079321eadff4c10101a2c6962dcc53R226
  // Use an additional group of threads to fill sorted_token_ids.
  // Since the kernel will use sorted_token_ids afterward,
  // we fill sorted_token_ids within the same threadblock to make
  // synchronization easier.
  if (threadIdx.x < fill_threads) {
    // Initialize sorted_token_ids with numel
    if (pad_sorted_token_ids) {
      for (int32_t it = threadIdx.x; it < max_num_tokens_padded; it += fill_threads) {
        sorted_token_ids[it] = numel;
      }
    }
    // Three __syncthreads() corresponding to the other threads
    __syncthreads();
    __syncthreads();
    __syncthreads();
    return;
  }

  const size_t tid = threadIdx.x - fill_threads;
  const size_t stride = blockDim.x - fill_threads;

  extern __shared__ int32_t shared_mem[];
  int32_t* cumsum = shared_mem;
  int32_t* tokens_cnts = (int32_t*)(shared_mem + num_experts + 1);

  for (int i = 0; i < num_experts; ++i) {
    tokens_cnts[(tid + 1) * num_experts + i] = 0;
  }

  for (size_t i = tid; i < numel; i += stride) {
    if (ignore_invalid_expert && topk_ids[i] < 0) continue;
    int32_t expert_id = topk_ids[i] + 1;
    ++tokens_cnts[(tid + 1) * num_experts + expert_id];
  }

  __syncthreads();

  if (tid < num_experts) {
    tokens_cnts[tid] = 0;
    for (int i = 1; i <= stride; ++i) {
      tokens_cnts[i * num_experts + tid] += tokens_cnts[(i - 1) * num_experts + tid];
    }
  }

  __syncthreads();

  if (tid == 0) {
    cumsum[0] = 0;
    for (int i = 1; i <= num_experts; ++i) {
      cumsum[i] = cumsum[i - 1] + CEILDIV(tokens_cnts[stride * num_experts + i - 1], block_size) * block_size;
    }
    *total_tokens_post_pad = static_cast<int32_t>(cumsum[num_experts]);
  }

  __syncthreads();

  if (tid < num_experts) {
    for (int i = cumsum[tid]; i < cumsum[tid + 1]; i += block_size) {
      expert_ids[i / block_size] = tid - 1;
    }
  }

  for (size_t i = tid; i < numel; i += stride) {
    if (ignore_invalid_expert && topk_ids[i] < 0) continue;
    int32_t expert_id = topk_ids[i] + 1;
    int32_t rank_post_pad = tokens_cnts[tid * num_experts + expert_id] + cumsum[expert_id];
    sorted_token_ids[rank_post_pad] = i;
    ++tokens_cnts[tid * num_experts + expert_id];
  }
}

// Bit-exact parallel replacement for
// moe_align_block_size_small_batch_expert_kernel, selected with
// SGLANG_MOE_ALIGN_PARALLEL=1.
//
// Why it is bit-exact: the blocked MoE GEMM consumes expert_ids[],
// num_tokens_post_pad and sorted_token_ids[], and all three are a pure function
// of the *histogram* of expert buckets (padded count -> block layout -> expert
// ids and padded total). The histogram is integer adds, so it is identical. The
// only remaining freedom is which slot inside one expert's [E[b], E[b]+count)
// range a token lands in; each output row of the blocked GEMM is an independent
// dot product over K (BLOCK_SIZE_M only batches rows, padding rows are marked by
// sorted_token_ids == numel), so permuting rows inside one expert changes no
// arithmetic - only the order in which independent elements are moved.
//
// Why it is faster: the shipped small-batch kernel reduces its per-thread
// histograms with a `stride`-deep dependent shared-memory chain and then a
// num_experts-deep dependent chain for cumsum - about 113 dependent LDS round
// trips. Here the histogram lives in shared memory once and the reduction is a
// 6-step Hillis-Steele scan over 64 lanes (log depth).
//
// Only portable primitives are used (atomicAdd / __syncthreads / dynamic smem),
// and the kernel is only reachable behind the env flag and only on the
// small-batch path, so the CUDA path is untouched when the flag is off.
template <typename scalar_t>
__global__ void moe_align_block_size_parallel_kernel(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad,
    int32_t num_experts,
    int32_t block_size,
    size_t numel,
    bool pad_sorted_token_ids,
    bool ignore_invalid_expert,
    int32_t max_num_tokens_padded) {
  extern __shared__ int32_t smem[];
  // [0, num_experts)                        per-bucket counts
  // [num_experts, 2 * num_experts)          per-bucket write cursors
  // [2 * num_experts, 3 * num_experts + 1) exclusive offsets (num_experts + 1)
  // [3 * num_experts + 1, ... + 64)         scan lanes
  int32_t* counts = smem;
  int32_t* cursor = counts + num_experts;
  int32_t* excl = cursor + num_experts;
  int32_t* scan = excl + (num_experts + 1);

  const int32_t tid = threadIdx.x;

  if (tid < num_experts) {
    counts[tid] = 0;
    cursor[tid] = 0;
  }
  __syncthreads();

  // 1. histogram, one pass over numel
  for (size_t i = tid; i < numel; i += blockDim.x) {
    if (ignore_invalid_expert && topk_ids[i] < 0) continue;
    const int32_t bucket = static_cast<int32_t>(topk_ids[i]) + 1;
    atomicAdd(&counts[bucket], 1);
  }
  __syncthreads();

  // 2. padded counts -> inclusive scan over 64 lanes, log depth
  const int32_t padded = (tid < num_experts)
      ? ((counts[tid] + block_size - 1) / block_size) * block_size
      : 0;
  if (tid < 64) scan[tid] = padded;
  __syncthreads();
#pragma unroll
  for (int offset = 1; offset < 64; offset <<= 1) {
    int32_t add = 0;
    if (tid >= offset && tid < 64) add = scan[tid - offset];
    __syncthreads();
    if (tid < 64) scan[tid] += add;
    __syncthreads();
  }

  // 3. exclusive offsets; every offset is a multiple of block_size
  if (tid < num_experts) {
    excl[tid] = (tid == 0) ? 0 : scan[tid - 1];
  }
  if (tid == 0) {
    excl[num_experts] = scan[num_experts - 1];
    *total_tokens_post_pad = scan[num_experts - 1];
  }
  __syncthreads();

  // 4. pad sorted_token_ids with numel
  if (pad_sorted_token_ids) {
    for (int32_t i = tid; i < max_num_tokens_padded; i += blockDim.x) {
      sorted_token_ids[i] = static_cast<int32_t>(numel);
    }
  }
  __syncthreads();

  // 5. expert_ids, one thread per bucket
  if (tid < num_experts) {
    const int32_t lo = excl[tid] / block_size;
    const int32_t hi = excl[tid + 1] / block_size;
    for (int32_t i = lo; i < hi; ++i) expert_ids[i] = tid - 1;
  }

  // 6. scatter; the order inside one bucket is whatever the atomic gives and is
  //    irrelevant to the GEMM (see the header comment)
  for (size_t i = tid; i < numel; i += blockDim.x) {
    if (ignore_invalid_expert && topk_ids[i] < 0) continue;
    const int32_t bucket = static_cast<int32_t>(topk_ids[i]) + 1;
    const int32_t rank = atomicAdd(&cursor[bucket], 1);
    sorted_token_ids[excl[bucket] + rank] = static_cast<int32_t>(i);
  }
}

void moe_align_block_size(
    torch::Tensor topk_ids,
    int64_t num_experts,
    int64_t block_size,
    torch::Tensor sorted_token_ids,
    torch::Tensor experts_ids,
    torch::Tensor num_tokens_post_pad,
    torch::Tensor cumsum_buffer,
    bool pad_sorted_token_ids,
    bool ignore_invalid_expert) {
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int threads = 1024;
  threads = ((threads + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;

  int64_t max_num_tokens_padded = sorted_token_ids.size(0);

  DISPATCH_INTEGRAL_TYPES(topk_ids.scalar_type(), "moe_align_block_size_kernel", [&] {
    bool small_batch_expert_mode = (topk_ids.numel() < 1024) && (num_experts <= 64);

    // Read once per process: getenv on every MoE layer would be its own tax.
    static const bool use_parallel_align = getBoolEnv("SGLANG_MOE_ALIGN_PARALLEL");

    if (small_batch_expert_mode && use_parallel_align) {
      constexpr int32_t par_threads = 256;
      const int32_t shared_mem_size =
          (2 * num_experts + (num_experts + 1) + 64) * sizeof(int32_t);

      auto parallel_kernel = moe_align_block_size_parallel_kernel<scalar_t>;
      parallel_kernel<<<1, par_threads, shared_mem_size, stream>>>(
          topk_ids.data_ptr<scalar_t>(),
          sorted_token_ids.data_ptr<int32_t>(),
          experts_ids.data_ptr<int32_t>(),
          num_tokens_post_pad.data_ptr<int32_t>(),
          num_experts,
          block_size,
          topk_ids.numel(),
          pad_sorted_token_ids,
          ignore_invalid_expert,
          max_num_tokens_padded);
    } else if (small_batch_expert_mode) {
      const int32_t threads = max((int32_t)num_experts, WARP_SIZE);
      constexpr int32_t fill_threads = 256;
      const int32_t shared_mem_size = ((threads + 1) * num_experts + (num_experts + 1)) * sizeof(int32_t);

      auto small_batch_expert_kernel = moe_align_block_size_small_batch_expert_kernel<scalar_t, fill_threads>;
      small_batch_expert_kernel<<<1, fill_threads + threads, shared_mem_size, stream>>>(
          topk_ids.data_ptr<scalar_t>(),
          sorted_token_ids.data_ptr<int32_t>(),
          experts_ids.data_ptr<int32_t>(),
          num_tokens_post_pad.data_ptr<int32_t>(),
          num_experts,
          block_size,
          topk_ids.numel(),
          pad_sorted_token_ids,
          ignore_invalid_expert,
          max_num_tokens_padded);
    } else {
      auto align_kernel = moe_align_block_size_kernel<scalar_t>;

      const size_t scan_size = next_pow2(num_experts);
      const size_t shared_mem_size = (num_experts + (num_experts + 1) + scan_size + WARP_SIZE) * sizeof(int32_t);
      align_kernel<<<2, threads, shared_mem_size, stream>>>(
          topk_ids.data_ptr<scalar_t>(),
          sorted_token_ids.data_ptr<int32_t>(),
          experts_ids.data_ptr<int32_t>(),
          num_tokens_post_pad.data_ptr<int32_t>(),
          num_experts,
          block_size,
          topk_ids.numel(),
          cumsum_buffer.data_ptr<int32_t>(),
          pad_sorted_token_ids,
          ignore_invalid_expert,
          scan_size,
          max_num_tokens_padded);

      const int block_threads = std::min(256, (int)threads);
      const int num_blocks = (topk_ids.numel() + block_threads - 1) / block_threads;
      const int max_blocks = 65535;
      const int actual_blocks = std::min(num_blocks, max_blocks);

      auto sort_kernel = count_and_sort_expert_tokens_kernel<scalar_t>;
      sort_kernel<<<actual_blocks, block_threads, 0, stream>>>(
          topk_ids.data_ptr<scalar_t>(),
          sorted_token_ids.data_ptr<int32_t>(),
          cumsum_buffer.data_ptr<int32_t>(),
          topk_ids.numel(),
          ignore_invalid_expert);
    }
  });
}
