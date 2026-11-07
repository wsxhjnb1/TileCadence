// Strided Qwen3 GQA prefill. The generated rowblock derives from pinned FA2.
#include <ATen/ATen.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include "config.cuh"
#include "flash_fwd_launch_template.h"
#include "common.cuh"
#include "long_prefill_rowblock.cuh"

namespace TORCH_EXTENSION_NAME {
using Element = cutlass::bfloat16_t;
using CandidateTraits = Flash_fwd_kernel_traits<128, PREFILL_BLOCK_M, PREFILL_BLOCK_N,
    PREFILL_NWARPS, PREFILL_Q_IN_REGS, PREFILL_SHARE_QK, Element>;

template<class Traits>
__global__ void prefill_kernel(KERNEL_PARAM_MODIFIER const Flash_fwd_params p) {
    // Four adjacent logical block coordinates share one physical KV head.
    const int head = blockIdx.y * 4 + blockIdx.x % 4;
    const int tile = blockIdx.x / 4;
    const int m_block = PREFILL_REVERSE_TILES ? int(gridDim.x) / 4 - 1 - tile : tile;
    long_prefill_rowblock<Traits, false, true, false, false, true, true, false, false>(
        p, blockIdx.z, head, m_block);
}

const void* kernel_for() {
    return reinterpret_cast<const void*>(&prefill_kernel<CandidateTraits>);
}

std::vector<int64_t> model_workspace_spec(const at::Tensor& q, const at::Tensor& k,
    const at::Tensor& v, int64_t requested, int64_t mapping, bool causal) {
    check_inputs(q, k, v, causal, true);
    TORCH_CHECK(requested == 0 && mapping == 1,
        "Generated long prefill candidate requires S=0 and mapping=1");
    TORCH_CHECK(q.size(1) % CandidateTraits::kBlockM == 0 && k.size(1) % CandidateTraits::kBlockN == 0,
        "Generated prefill requires fully tiled equal-length input");
    return {1, 0, 0};
}

void model_forward_out(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    at::Tensor& out, at::Tensor& lse, at::Tensor& ws_o, at::Tensor& ws_lse,
    int64_t requested, int64_t mapping, double scale, bool causal) {
    check_outputs(q, k, v, out, lse, causal, true);
    model_workspace_spec(q, k, v, requested, mapping, causal);
    TORCH_CHECK(ws_o.numel() == 0 && ws_lse.numel() == 0,
        "Generated long prefill uses no split workspace");
    const c10::cuda::CUDAGuard guard(q.device());
    Flash_fwd_params p;
    set_common(p, q, k, v, out, lse, scale);
    p.is_causal = true;
    p.window_size_right = 0;
    p.num_splits = 1;
    const auto kernel = kernel_for();
    constexpr int shared = CandidateTraits::kSmemSize;
    if constexpr (shared >= 48 * 1024) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared));
    }
    const auto stream = c10::cuda::getCurrentCUDAStream(q.get_device()).stream();
    const auto tiles = (p.seqlen_q + CandidateTraits::kBlockM - 1) / CandidateTraits::kBlockM;
    const dim3 grid(tiles * 4, 8, p.b);
    void* arguments[] = {&p};
    C10_CUDA_CHECK(cudaLaunchKernel(kernel, grid, dim3(CandidateTraits::kNThreads), arguments, shared, stream));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::vector<int64_t> model_resources(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    int64_t requested, int64_t mapping, bool causal) {
    model_workspace_spec(q, k, v, requested, mapping, causal);
    const c10::cuda::CUDAGuard guard(q.device());
    constexpr int m = CandidateTraits::kBlockM;
    constexpr int n = CandidateTraits::kBlockN;
    std::vector<int64_t> resource = {
        1, q.size(1), 32, 4, 1, q.size(0) * 32 * ((q.size(1) + m - 1) / m), 0, 0, 0, m, n, 1
    };
    append_resource(resource, kernel_for(), CandidateTraits::kNThreads, CandidateTraits::kSmemSize);
    resource.insert(resource.end(), 8, 0);
    append_layout_resources(resource, q, k, v, true);
    return resource;
}
} // namespace TORCH_EXTENSION_NAME
