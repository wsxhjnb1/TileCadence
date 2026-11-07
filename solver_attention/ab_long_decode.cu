// CUDA launch adapter for B's generated strided, nonappend Qwen3 decode plan.
#include <ATen/ATen.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include "flash_fwd_launch_template.h"
#include "config.cuh"
#include "v7_traits.cuh"
#include "common.cuh"
#include "ab_long_splitkv.cuh"

namespace TORCH_EXTENSION_NAME {

using Element = cutlass::bfloat16_t;
using CombineTraits = Flash_fwd_kernel_traits<128, 64, 128, 4, false, false, Element>;
using CandidateBase = NativeQwenV7Traits<
    128, 16 * AB_DECODE_WARPS, AB_DECODE_BLOCK_N, AB_DECODE_WARPS,
    AB_DECODE_Q_RESIDENCY != 0, AB_DECODE_Q_RESIDENCY == 2,
    Element, AB_DECODE_CACHE_ALWAYS != 0>;
using CandidateTraits = NativeQwenV7DecodeOutputTraits<CandidateBase>;

template<bool Split>
constexpr int shared_bytes() {
    // The output epilogue reuses the tile allocation after the last PV.
    constexpr int output_bytes = CandidateTraits::kBlockM * 128 * (Split ? 4 : 2);
    return std::max(output_bytes, CandidateTraits::kSmemSize);
}

static_assert(shared_bytes<true>() == AB_DECODE_SPLIT_SHARED_BYTES);

std::vector<int64_t> model_workspace_spec(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    int64_t splits, int64_t mapping, bool causal) {
    check_inputs(q, k, v, causal, false);
    const int64_t tiles = (k.size(1) + CandidateTraits::kBlockN - 1) /
        CandidateTraits::kBlockN;
    TORCH_CHECK(mapping == 1, "Generated decode candidate requires mapping 1");
    TORCH_CHECK(splits >= 1 && splits <= 32 && splits <= tiles,
                "Requested splits exceed the generated decode partition domain");
    const int64_t rows = q.size(0) * 32;
    return {splits, splits > 1 ? splits * rows * 128 : 0,
            splits > 1 ? splits * rows : 0};
}

Flash_fwd_params parameters(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    at::Tensor& out, at::Tensor& lse, at::Tensor& workspace_o,
    at::Tensor& workspace_lse, int splits, double scale) {
    Flash_fwd_params p;
    set_common(p, q, k, v, out, lse, scale);
    p.q_row_stride = q.stride(2);
    p.q_head_stride = 4 * q.stride(2);
    p.o_row_stride = out.stride(2);
    p.o_head_stride = 4 * out.stride(2);
    p.h = 8;
    p.h_k = 8;
    p.h_h_k_ratio = 1;
    p.seqlen_q = 4;
    p.total_q = 4 * p.b;
    p.seqlen_q_rounded = 128;
    p.is_causal = false;
    p.num_splits = splits;
    if (splits > 1) {
        p.oaccum_ptr = workspace_o.data_ptr();
        p.softmax_lseaccum_ptr = workspace_lse.data_ptr();
    }
    return p;
}

template<bool Split>
__global__ void candidate_kernel(KERNEL_PARAM_MODIFIER const Flash_fwd_params p) {
    const int batch = Split ? blockIdx.z / 8 : blockIdx.z;
    const int head = Split ? blockIdx.z - batch * 8 : blockIdx.y;
    compute_long_ab_splitkv<CandidateTraits, false, false, false, false,
                            true, false, Split, false>(
        p, batch, head, blockIdx.x, Split ? blockIdx.y : 0, Split ? gridDim.y : 1);
}

template<int Log>
void combine(Flash_fwd_params& p, cudaStream_t stream) {
    flash_fwd_splitkv_combine_kernel<CombineTraits, 4, Log, true>
        <<<dim3((p.b * p.h * p.seqlen_q + 3) / 4), 128, 0, stream>>>(p);
}

void combine_dispatch(Flash_fwd_params& p, cudaStream_t stream) {
    if (p.num_splits <= 2) combine<1>(p, stream);
    else if (p.num_splits <= 4) combine<2>(p, stream);
    else if (p.num_splits <= 8) combine<3>(p, stream);
    else if (p.num_splits <= 16) combine<4>(p, stream);
    else combine<5>(p, stream);
}

template<bool Split>
void launch_candidate(Flash_fwd_params& p, cudaStream_t stream) {
    constexpr int shared = shared_bytes<Split>();
    if constexpr (shared >= 48 * 1024) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(&candidate_kernel<Split>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, shared));
    }
    candidate_kernel<Split>
        <<<dim3(1, Split ? p.num_splits : p.h, Split ? p.b * p.h : p.b),
           CandidateTraits::kNThreads, shared, stream>>>(p);
    if constexpr (Split) combine_dispatch(p, stream);
}

void model_forward_out(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    at::Tensor& out, at::Tensor& lse, at::Tensor& workspace_o,
    at::Tensor& workspace_lse, int64_t splits, int64_t mapping,
    double scale, bool causal) {
    check_outputs(q, k, v, out, lse, causal, false);
    const c10::cuda::CUDAGuard guard(q.device());
    const auto spec = model_workspace_spec(q, k, v, splits, mapping, causal);
    for (const auto* work : {&workspace_o, &workspace_lse}) {
        TORCH_CHECK(work->device() == q.device() && work->scalar_type() == at::kFloat &&
                    work->is_contiguous() && !work->requires_grad(),
                    "Contiguous FP32 split workspace required");
    }
    TORCH_CHECK(workspace_o.numel() >= spec[1] && workspace_lse.numel() >= spec[2],
                "Complete split workspace too small");
    const at::Tensor* writes[] = {&out, &lse, &workspace_o, &workspace_lse};
    const at::Tensor* reads[] = {&q, &k, &v};
    for (const auto* write : writes) {
        for (const auto* read : reads) {
            TORCH_CHECK(!overlaps(*write, *read), "Write aliases strided Q/K/V span");
        }
    }
    for (int i = 0; i < 4; ++i) {
        for (int j = i + 1; j < 4; ++j) {
            TORCH_CHECK(!overlaps(*writes[i], *writes[j]), "Writable buffers overlap");
        }
    }
    auto p = parameters(q, k, v, out, lse, workspace_o, workspace_lse, int(splits), scale);
    auto stream = c10::cuda::getCurrentCUDAStream(q.get_device()).stream();
    if (splits > 1) launch_candidate<true>(p, stream);
    else launch_candidate<false>(p, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<int Log>
void combine_resource(std::vector<int64_t>& values) {
    append_resource(values, &flash_fwd_splitkv_combine_kernel<CombineTraits, 4, Log, true>, 128, 0);
}

std::vector<int64_t> model_resources(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    int64_t splits, int64_t mapping, bool causal) {
    const c10::cuda::CUDAGuard guard(q.device());
    const auto spec = model_workspace_spec(q, k, v, splits, mapping, causal);
    const bool split = splits > 1;
    std::vector<int64_t> values = {
        splits, 4, 8, 4, split ? 2 : 1, q.size(0) * 8 * splits,
        split ? q.size(0) * 8 : 0, spec[1], spec[2],
        CandidateTraits::kBlockM, CandidateTraits::kBlockN, 1,
    };
    if (split) append_resource(values, &candidate_kernel<true>, CandidateTraits::kNThreads, shared_bytes<true>());
    else append_resource(values, &candidate_kernel<false>, CandidateTraits::kNThreads, shared_bytes<false>());
    if (!split) values.insert(values.end(), 8, 0);
    else if (splits <= 2) combine_resource<1>(values);
    else if (splits <= 4) combine_resource<2>(values);
    else if (splits <= 8) combine_resource<3>(values);
    else if (splits <= 16) combine_resource<4>(values);
    else combine_resource<5>(values);
    append_layout_resources(values, q, k, v, false);
    return values;
}

}  // namespace TORCH_EXTENSION_NAME
