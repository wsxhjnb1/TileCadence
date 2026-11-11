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

constexpr int kGroupedThreads = CandidateTraits::kNThreads * PREFILL_GROUP_HEADS;
constexpr int kGroupedShared = CandidateTraits::kSmemQSize * PREFILL_GROUP_HEADS + CandidateTraits::kSmemKVSize;
static_assert(kGroupedShared == PREFILL_GROUP_SHARED_BYTES);
static_assert(PREFILL_GROUP_HEADS == 1 || PREFILL_GROUP_HEADS == 2);
static_assert(4 % PREFILL_GROUP_HEADS == 0);

template<class Traits, bool EvenMN>
__global__ void prefill_kernel(KERNEL_PARAM_MODIFIER const Flash_fwd_params p) {
    const int head_slot = PREFILL_GROUP_HEADS == 1 ? 0 : threadIdx.x / Traits::kNThreads;
    static_assert(PREFILL_GROUP_HEADS == 2 && Traits::kBlockM == 128);
    constexpr int kStripeTiles = 12;
    constexpr int kGroups = 4;
    constexpr int kBandCtas = kStripeTiles * kGroups;
    const int x = int(blockIdx.x);
    const int tiles = int(gridDim.x >> 2);
    const int slab = x / kBandCtas;
    const int within = x % kBandCtas;
    const int remaining = tiles % kStripeTiles;
    const int bulk = (tiles - remaining) * kGroups;
    // Split a short final stripe evenly between both KV heads.
    const int width = x >= bulk ? remaining : kStripeTiles;
    const int head_within = within >= 2 * width;
    const int row = (within >> 1) - head_within * width;
    const int group = int(blockIdx.z) * kGroups + 2 * head_within + (within & 1);
    const int m_block = tiles - 1 - slab * kStripeTiles - row;
    const int head = group * PREFILL_GROUP_HEADS + head_slot;
    grouped_prefill_rowblock<Traits, false, true, false, false, EvenMN, true, false, false>(
        p, int(blockIdx.y), head, m_block, head_slot);
}

bool grouped_even(const at::Tensor& q, const at::Tensor& k) {
    return q.size(1) % CandidateTraits::kBlockM == 0 && k.size(1) % CandidateTraits::kBlockN == 0;
}

const void* kernel_for(bool even) {
    return even ? reinterpret_cast<const void*>(&prefill_kernel<CandidateTraits, true>)
                : reinterpret_cast<const void*>(&prefill_kernel<CandidateTraits, false>);
}

const cudaDeviceProp& device_properties(const at::Tensor& q) {
    // The framework's full CUDA context header also pulls in cuSPARSE.
    // Cache the runtime query per host thread, including the device key.
    thread_local int cached_device = -1;
    thread_local cudaDeviceProp cached_properties{};
    const int device = q.get_device();
    if (device != cached_device) {
        cudaDeviceProp properties{};
        C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
        cached_properties = properties;
        cached_device = device;
    }
    return cached_properties;
}

bool use_grouped(const at::Tensor& q, int64_t mapping) {
    const auto tokens = q.size(1);
    if (mapping == 0 || (q.size(0) != 1 && q.size(0) != 4) ||
        tokens < 128 || tokens > 32768) {
        return false;
    }
    const auto& device = device_properties(q);
    return device.major >= 8 && device.sharedMemPerBlockOptin >= kGroupedShared;
}

struct DispatchResources {
    const void* function;
    int m, n, threads, shared, grid_heads;
};

template<class Traits>
DispatchResources native_resources(const at::Tensor& q, const at::Tensor& k) {
    const bool even = q.size(1) % Traits::kBlockM == 0 && k.size(1) % Traits::kBlockN == 0;
    const void* kernel = even
        ? reinterpret_cast<const void*>(&flash_fwd_kernel<Traits, false, true, false, false, true, true, false, false>)
        : reinterpret_cast<const void*>(&flash_fwd_kernel<Traits, false, true, false, false, false, true, false, false>);
    return {kernel, Traits::kBlockM, Traits::kBlockN, Traits::kNThreads, Traits::kSmemSize, 32};
}

DispatchResources resources_for(const at::Tensor& q, const at::Tensor& k, int64_t mapping) {
    if (use_grouped(q, mapping)) {
        return {kernel_for(grouped_even(q, k)), CandidateTraits::kBlockM, CandidateTraits::kBlockN,
                kGroupedThreads, kGroupedShared, 32 / PREFILL_GROUP_HEADS};
    }
    // Match the pinned run_mha_fwd_hdim128 default causal branch for resources.
    const auto& device = device_properties(q);
    if (device.major == 8 && device.minor > 0) {
        return native_resources<Flash_fwd_kernel_traits<128, 64, 64, 4, false, false, Element>>(q, k);
    }
    return native_resources<Flash_fwd_kernel_traits<128, 128, 64, 4, false, false, Element>>(q, k);
}

std::vector<int64_t> model_workspace_spec(const at::Tensor& q, const at::Tensor& k,
    const at::Tensor& v, int64_t requested, int64_t mapping, bool causal) {
    check_inputs(q, k, v, causal, true);
    TORCH_CHECK(requested == 0 && (mapping == 0 || mapping == 1),
        "V40 requires S=0 and mapping=0 (native) or 1 (auto)");
    const auto& device = device_properties(q);
    TORCH_CHECK(device.major >= 8, "V40 requires CUDA SM80 or newer for BF16 prefill");
    TORCH_CHECK(q.size(0) <= device.maxGridSize[1], "V40 batch exceeds the CUDA grid Y dimension");
    TORCH_CHECK(mapping == 0 || use_grouped(q, mapping),
        "The shared-LSE schedule requires batch 1 or 4, 128–32768 tokens, and 96 KiB opt-in shared memory");
    const int64_t native_work = q.size(0) * q.size(2) * ((q.size(1) + 63) / 64);
    TORCH_CHECK(k.size(1) <= 128 || native_work >= 0.8f * (2 * device.multiProcessorCount),
        "V40 scope requires the upstream default prefill heuristic to select one split");
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
    TORCH_CHECK(std::isfinite(p.scale_softmax) && p.scale_softmax > 0.f && std::isfinite(p.scale_softmax_log2),
        "V40 requires a positive finite FP32 attention scale and log2 multiplier");
    const auto stream = c10::cuda::getCurrentCUDAStream(q.get_device()).stream();
    if (!use_grouped(q, mapping)) {
        run_mha_fwd_hdim128<Element, true>(p, stream);
        return;
    }
    const auto kernel = kernel_for(grouped_even(q, k));
    constexpr int shared = kGroupedShared;
    if constexpr (shared >= 48 * 1024) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared));
    }
    const auto tiles = (p.seqlen_q + CandidateTraits::kBlockM - 1) / CandidateTraits::kBlockM;
    const dim3 grid(4 * tiles, p.b, p.h / (4 * PREFILL_GROUP_HEADS));
    void* arguments[] = {&p};
    C10_CUDA_CHECK(cudaLaunchKernel(kernel, grid, dim3(kGroupedThreads), arguments, shared, stream));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::vector<int64_t> model_resources(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    int64_t requested, int64_t mapping, bool causal) {
    model_workspace_spec(q, k, v, requested, mapping, causal);
    const c10::cuda::CUDAGuard guard(q.device());
    const auto selected = resources_for(q, k, mapping);
    const auto tiles = (q.size(1) + selected.m - 1) / selected.m;
    std::vector<int64_t> resource = {
        1, q.size(1), 32, 4, 1, q.size(0) * selected.grid_heads * tiles, 0, 0, 0, selected.m, selected.n, 1
    };
    append_resource(resource, selected.function, selected.threads, selected.shared);
    resource.insert(resource.end(), 8, 0);
    append_layout_resources(resource, q, k, v, true);
    return resource;
}
} // namespace TORCH_EXTENSION_NAME
