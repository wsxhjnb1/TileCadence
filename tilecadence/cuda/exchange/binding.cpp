// Host interface only. No build/import operation launches a CUDA kernel.
#include <ATen/ATen.h>
#include <torch/library.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>
#include <cstdint>

// Keep every plan's host symbols private to its extension namespace as well as
// its dispatcher registry. Otherwise multiple loaded DSOs can interpose an
// identically named plan_forward_cuda/forward function.
namespace TORCH_EXTENSION_NAME {
void plan_forward_cuda(const at::Tensor&,const at::Tensor&,const at::Tensor&,
                       at::Tensor&,at::Tensor&,const at::Tensor&,bool);
std::vector<int64_t> plan_resources_cuda(const at::Tensor&,bool);
int plan_nonce_elements();
bool plan_diagnostic_guard();

namespace {
bool overlap(const at::Tensor& a,const at::Tensor& b) {
    auto x=reinterpret_cast<uintptr_t>(a.data_ptr());
    auto y=reinterpret_cast<uintptr_t>(b.data_ptr());
    return std::max(x,y)<std::min(x+a.numel()*a.element_size(),y+b.numel()*b.element_size());
}
void check_q(const at::Tensor& q) {
    TORCH_CHECK(q.is_cuda() && q.dim()==4 && q.is_contiguous() && q.size(1)==64 && q.size(3)==128,
                "Expected CUDA contiguous [B,64,H,128]; this initial graph is exactly two KV tiles");
    TORCH_CHECK(q.size(0)>0 && q.size(2)>0 && !q.requires_grad(),"Positive inference-only dimensions required");
    TORCH_CHECK(q.scalar_type()==at::kHalf || q.scalar_type()==at::kBFloat16,"FP16/BF16 required");
}
}

void forward_out(const at::Tensor& q,const at::Tensor& k,const at::Tensor& v,
                 at::Tensor out,at::Tensor lse,const at::Tensor& nonce,bool causal,
                 bool nonce_is_zero,bool diagnostic_non_nan_asserted) {
    check_q(q);
    const at::Tensor* matching_tensors[]={&k,&v,&out};
    for (const auto* t:matching_tensors) {
        TORCH_CHECK(t->device()==q.device() && t->scalar_type()==q.scalar_type()
            && t->sizes()==q.sizes() && t->is_contiguous() && !t->requires_grad(),"K/V/O must match Q");
    }
    TORCH_CHECK(lse.device()==q.device() && lse.scalar_type()==at::kFloat && lse.is_contiguous()
        && !lse.requires_grad() && lse.sizes()==at::IntArrayRef({q.size(0),q.size(2),64}),"LSE must be FP32 [B,H,64]");
    for (const auto* t:{&q,&k,&v}) TORCH_CHECK(!overlap(out,*t) && !overlap(lse,*t),"Outputs must not alias inputs");
    TORCH_CHECK(!overlap(out,lse),"Outputs must not alias one another");
    TORCH_CHECK(nonce.device()==q.device() && nonce.is_contiguous() && nonce.dim()==1
        && (nonce.scalar_type()==at::kInt || nonce.scalar_type()==at::ScalarType::UInt32)
        && nonce.numel()>=plan_nonce_elements(),"Supply root-owned contiguous zero int32/uint32 nonce vector");
    TORCH_CHECK(!overlap(nonce,out) && !overlap(nonce,lse),"Read-only nonce must not alias an output");
    TORCH_CHECK(plan_nonce_elements()==0 || nonce_is_zero,
        "Opaque recovery ABI requires explicit zero-nonce attestation; initialize before graph capture");
    TORCH_CHECK(!plan_diagnostic_guard() || diagnostic_non_nan_asserted,
        "trap_nan is a bounded non-NaN diagnostic, not arbitrary-nonfinite production behavior");
    const c10::cuda::CUDAGuard guard(q.device());
    plan_forward_cuda(q,k,v,out,lse,nonce,causal);
}

std::vector<at::Tensor> forward(const at::Tensor& q,const at::Tensor& k,const at::Tensor& v,
                              const at::Tensor& nonce,bool causal,bool nonce_is_zero,
                              bool diagnostic_non_nan_asserted) {
    check_q(q);
    auto out=at::empty_like(q);
    auto lse=at::empty({q.size(0),q.size(2),64},q.options().dtype(at::kFloat));
    forward_out(q,k,v,out,lse,nonce,causal,nonce_is_zero,diagnostic_non_nan_asserted);
    return {out,lse};
}

std::vector<int64_t> resources(const at::Tensor& q,bool causal) {
    check_q(q);
    const c10::cuda::CUDAGuard guard(q.device());
    return plan_resources_cuda(q,causal);
}
} // namespace TORCH_EXTENSION_NAME

// Expand the extension-name macro before TORCH_LIBRARY stringifies its token.
// cpp_extension.load(..., is_python_module=False) loads this dispatcher library
// without requiring Python development headers or an embedded PyInit function.
#define FA2_V3_LIBRARY_EXPAND(name,m) TORCH_LIBRARY(name,m)
FA2_V3_LIBRARY_EXPAND(TORCH_EXTENSION_NAME,m) {
    m.def("forward_out(Tensor q, Tensor k, Tensor v, Tensor(a!) out, Tensor(b!) lse, Tensor nonce, bool causal=False, bool nonce_is_zero=False, bool diagnostic_non_nan_asserted=False) -> ()");
    m.def("forward(Tensor q, Tensor k, Tensor v, Tensor nonce, bool causal=False, bool nonce_is_zero=False, bool diagnostic_non_nan_asserted=False) -> Tensor[]");
    m.def("resources(Tensor q, bool causal=False) -> int[]");
    m.impl("forward_out",torch::dispatch(c10::DispatchKey::CUDA,TORCH_FN(TORCH_EXTENSION_NAME::forward_out)));
    m.impl("forward",torch::dispatch(c10::DispatchKey::CUDA,TORCH_FN(TORCH_EXTENSION_NAME::forward)));
    m.impl("resources",torch::dispatch(c10::DispatchKey::CUDA,TORCH_FN(TORCH_EXTENSION_NAME::resources)));
}
