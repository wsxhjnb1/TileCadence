// New real-model host ABI; no vendor math/header is modified.
#pragma once
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>
#include "flash.h"

namespace TORCH_EXTENSION_NAME {
inline uint64_t touched_bytes(const at::Tensor& tensor) {
    TORCH_CHECK(tensor.layout()==at::kStrided,"Strided tensor storage required");
    if(tensor.numel()==0)return 0;
    std::vector<std::pair<int64_t,int64_t>> axes;
    for(int i=0;i<tensor.dim();++i) {
        TORCH_CHECK(tensor.stride(i)>=0,"Negative strides are outside this native ABI");
        if(tensor.size(i)>1)axes.emplace_back(tensor.stride(i),tensor.size(i));
    }
    std::sort(axes.begin(),axes.end());uint64_t span=1;
    const uint64_t limit=uint64_t(std::numeric_limits<int32_t>::max())/tensor.element_size();
    for(const auto& axis:axes) {
        TORCH_CHECK(uint64_t(axis.first)>=span,"Internal overlap/expanded tensor is outside this native ABI");
        TORCH_CHECK(uint64_t(axis.first)<=limit && uint64_t(axis.second-1)<=(limit-span)/uint64_t(axis.first),"Touched native storage exceeds bounded2GiB span");
        span+=uint64_t(axis.second-1)*uint64_t(axis.first);
    }
    return span*tensor.element_size();
}
inline bool overlaps(const at::Tensor& a,const at::Tensor& b) {
    const auto na=touched_bytes(a),nb=touched_bytes(b);if(!na||!nb)return false;
    const auto x=reinterpret_cast<uintptr_t>(a.data_ptr()),y=reinterpret_cast<uintptr_t>(b.data_ptr());
    return x<y+nb&&y<x+na;
}
inline void check_inputs(const at::Tensor& q,const at::Tensor& k,const at::Tensor& v,bool causal,bool prefill) {
    TORCH_CHECK(q.is_cuda()&&q.dim()==4&&q.size(0)>0&&q.size(1)>0&&q.size(2)==32&&q.size(3)==128,
        "Q must be actual CUDA BNHD[B,T,32,128]");
    TORCH_CHECK(q.size(0)<=std::numeric_limits<int>::max()/32&&q.size(1)<=std::numeric_limits<int>::max()-127,"Native Q geometry overflow");
    for(const auto* tensor:{&q,&k,&v}) {
        TORCH_CHECK(tensor->device()==q.device()&&tensor->scalar_type()==at::kBFloat16&&!tensor->requires_grad(),"Same-device inference BF16 Q/K/V required");
        TORCH_CHECK(tensor->dim()==4&&tensor->stride(3)==1,"BNHD inputs must have contiguous last dimension only");
        TORCH_CHECK(reinterpret_cast<uintptr_t>(tensor->data_ptr())%16==0,"Aligned native Q/K/V base required");
        touched_bytes(*tensor);
        for(int axis=0;axis<3;++axis){if(tensor->size(axis)>1){TORCH_CHECK(tensor->stride(axis)%8==0,"Native vector row/head alignment requires strides divisible by8 elements");}}
    }
    TORCH_CHECK(k.size(0)==q.size(0)&&k.size(1)>0&&k.size(1)<=std::numeric_limits<int>::max()-127&&k.size(2)==8&&k.size(3)==128&&v.sizes()==k.sizes(),"Actual visible K/V must be BNHD[B,Tk,8,128]");
    if(prefill){TORCH_CHECK(causal&&q.size(1)>1&&q.size(1)==k.size(1),"Full causal no-past prefill requires Tq=Tk>1");}
    else {TORCH_CHECK(!causal&&q.size(1)==1,"Q1 decode uses the native effective all-visible mask");}
}
inline void check_outputs(const at::Tensor& q,const at::Tensor& k,const at::Tensor& v,
    const at::Tensor& out,const at::Tensor& lse,bool causal,bool prefill) {
    check_inputs(q,k,v,causal,prefill);
    TORCH_CHECK(out.device()==q.device()&&out.scalar_type()==q.scalar_type()&&out.sizes()==q.sizes()&&out.is_contiguous()&&!out.requires_grad(),"Common complete output must be contiguous BNHD with Q shape");
    TORCH_CHECK(lse.device()==q.device()&&lse.scalar_type()==at::kFloat&&lse.is_contiguous()&&!lse.requires_grad()
        &&lse.sizes()==at::IntArrayRef({q.size(0),32,q.size(1)}),"Complete natural-log LSE must be FP32[B,32,Tq]");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(out.data_ptr())%16==0,"Aligned output required");
    for(const auto* tensor:{&q,&k,&v}){TORCH_CHECK(!overlaps(out,*tensor)&&!overlaps(lse,*tensor),"Writable output aliases actual strided input storage span");}
    TORCH_CHECK(!overlaps(out,lse),"O/LSE overlap");
}
inline void check_scale(double scale) {
    TORCH_CHECK(std::isfinite(scale)&&scale>0,"Finite positive model attention scale required");
}
inline void set_common(Flash_fwd_params& p,const at::Tensor& q,const at::Tensor& k,const at::Tensor& v,
    at::Tensor& out,at::Tensor& lse,double scale) {
    check_scale(scale);p={};
    p.q_ptr=q.data_ptr();p.k_ptr=k.data_ptr();p.v_ptr=v.data_ptr();p.o_ptr=out.data_ptr();p.softmax_lse_ptr=lse.data_ptr();
    p.q_batch_stride=q.stride(0);p.k_batch_stride=k.stride(0);p.v_batch_stride=v.stride(0);p.o_batch_stride=out.stride(0);
    p.q_row_stride=q.stride(1);p.k_row_stride=k.stride(1);p.v_row_stride=v.stride(1);p.o_row_stride=out.stride(1);
    p.q_head_stride=q.stride(2);p.k_head_stride=k.stride(2);p.v_head_stride=v.stride(2);p.o_head_stride=out.stride(2);
    p.b=int(q.size(0));p.h=32;p.h_k=8;p.h_h_k_ratio=4;p.seqlen_q=int(q.size(1));p.seqlen_k=int(k.size(1));
    p.seqlen_q_rounded=((p.seqlen_q+127)/128)*128;p.seqlen_k_rounded=((p.seqlen_k+127)/128)*128;
    p.d=128;p.d_rounded=128;p.scale_softmax=float(scale);p.scale_softmax_log2=p.scale_softmax*M_LOG2E;
    p.p_dropout=1.f;p.p_dropout_in_uint8_t=255;p.rp_dropout=1.f;p.scale_softmax_rp_dropout=p.scale_softmax;
    p.is_bf16=true;p.window_size_left=-1;p.window_size_right=-1;p.is_seqlens_k_cumulative=true;
    p.unpadded_lse=false;p.seqlenq_ngroups_swapped=false;
    // cu_seqlens_k/knew_ptr remain nullptr: HF DynamicCache already appended.
}
inline void append_layout_resources(std::vector<int64_t>& values,const at::Tensor& q,const at::Tensor& k,const at::Tensor& v,bool prefill) {
    for(int64_t value:{int64_t(prefill?0:1),q.size(1),k.size(1),int64_t(32),int64_t(8),int64_t(prefill),int64_t(MODEL_FAMILY_CODE),
        q.stride(0),q.stride(1),q.stride(2),k.stride(0),k.stride(1),k.stride(2),v.stride(0),v.stride(1),v.stride(2),
        prefill?q.stride(1):q.stride(2),prefill?q.stride(2):4*q.stride(2)})values.push_back(value);
}
template<typename Kernel>inline void append_resource(std::vector<int64_t>& values,Kernel kernel,int threads,int shared) {
    if(shared>=48*1024)C10_CUDA_CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,shared));
    cudaFuncAttributes a{};C10_CUDA_CHECK(cudaFuncGetAttributes(&a,kernel));int active=0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,kernel,threads,shared));
    for(int64_t value:{int64_t(threads),int64_t(shared),int64_t(a.sharedSizeBytes),int64_t(a.numRegs),int64_t(a.localSizeBytes),int64_t(a.binaryVersion),int64_t(a.ptxVersion),int64_t(active)})values.push_back(value);
}
} // namespace TORCH_EXTENSION_NAME
