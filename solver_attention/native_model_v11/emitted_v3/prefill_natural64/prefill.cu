// Real-model strided BNHD prefill; original causal tensor math is unchanged.
#include <ATen/ATen.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include "config.cuh"
#include "v7_traits.cuh"
#include "flash_fwd_launch_template.h"
#include "common.cuh"
namespace TORCH_EXTENSION_NAME {
using Element=cutlass::bfloat16_t;
#if MODEL_PREFILL_V7_TRAITS
using CandidateTraits=NativeQwenV7Traits<128,PREFILL_BLOCK_M,PREFILL_BLOCK_N,4,PREFILL_Q_IN_REGS,PREFILL_SHARE_QK,Element,false>;
#else
using CandidateTraits=Flash_fwd_kernel_traits<128,PREFILL_BLOCK_M,PREFILL_BLOCK_N,4,PREFILL_Q_IN_REGS,PREFILL_SHARE_QK,Element>;
#endif
template<class Traits,bool EvenMN,bool HeadFast>
__global__ void prefill_kernel(KERNEL_PARAM_MODIFIER const Flash_fwd_params p){
    const int batch=HeadFast?blockIdx.z:blockIdx.y,head=HeadFast?blockIdx.y:blockIdx.z;
    compute_attn_1rowblock<Traits,false,true,false,false,EvenMN,true,false,false>(p,batch,head,blockIdx.x);
}
template<class Traits,bool Official>const void* kernel_for(bool even){
    if constexpr(Official){if(even)return reinterpret_cast<const void*>(&flash_fwd_kernel<Traits,false,true,false,false,true,true,false,false>);return reinterpret_cast<const void*>(&flash_fwd_kernel<Traits,false,true,false,false,false,true,false,false>);}
    else {if(even)return reinterpret_cast<const void*>(&prefill_kernel<Traits,true,PREFILL_HEAD_FAST>);return reinterpret_cast<const void*>(&prefill_kernel<Traits,false,PREFILL_HEAD_FAST>);}
}
struct Selected{const void* function;int m,n,threads,shared;};
template<class Traits,bool Official>Selected descriptor(const Flash_fwd_params& p){
    bool even=p.seqlen_q%Traits::kBlockM==0&&p.seqlen_k%Traits::kBlockN==0;
    return {kernel_for<Traits,Official>(even),Traits::kBlockM,Traits::kBlockN,Traits::kNThreads,Traits::kSmemSize};
}
Selected selected(const Flash_fwd_params& p,int64_t mapping){
    if(mapping==0){auto [major,minor]=get_compute_capability(get_current_device());if(major==8&&minor>0)return descriptor<Flash_fwd_kernel_traits<128,64,64,4,false,false,Element>,true>(p);return descriptor<Flash_fwd_kernel_traits<128,128,64,4,false,false,Element>,true>(p);}
    return descriptor<CandidateTraits,false>(p);
}
void configure(const Selected& value){if(value.shared>=48*1024)C10_CUDA_CHECK(cudaFuncSetAttribute(value.function,cudaFuncAttributeMaxDynamicSharedMemorySize,value.shared));}
std::vector<int64_t> model_workspace_spec(const at::Tensor& q,const at::Tensor& k,const at::Tensor& v,int64_t requested,int64_t mapping,bool causal){
    check_inputs(q,k,v,causal,true);TORCH_CHECK(requested==0&&(mapping==0||mapping==1),"Prefill exposes native default0 or frozen candidate1, with S0 only");return {1,0,0};
}
void model_forward_out(const at::Tensor& q,const at::Tensor& k,const at::Tensor& v,at::Tensor& out,at::Tensor& lse,at::Tensor& ws_o,at::Tensor& ws_lse,int64_t requested,int64_t mapping,double scale,bool causal){
    check_outputs(q,k,v,out,lse,causal,true);model_workspace_spec(q,k,v,requested,mapping,causal);
    TORCH_CHECK(ws_o.numel()==0&&ws_lse.numel()==0,"Prefill does not use split workspaces");
    const c10::cuda::CUDAGuard guard(q.device());Flash_fwd_params p;set_common(p,q,k,v,out,lse,scale);
    p.is_causal=true;p.window_size_right=0;p.num_splits=1;
    auto stream=c10::cuda::getCurrentCUDAStream(q.get_device()).stream();
    if(mapping==0)run_mha_fwd_hdim128<Element,true>(p,stream);
    else {const auto value=selected(p,mapping);configure(value);const dim3 grid((p.seqlen_q+value.m-1)/value.m,PREFILL_HEAD_FAST?p.h:p.b,PREFILL_HEAD_FAST?p.b:p.h);void* arguments[]={&p};C10_CUDA_CHECK(cudaLaunchKernel(value.function,grid,dim3(value.threads),arguments,value.shared,stream));}
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
std::vector<int64_t> model_resources(const at::Tensor& q,const at::Tensor& k,const at::Tensor& v,int64_t requested,int64_t mapping,bool causal){
    model_workspace_spec(q,k,v,requested,mapping,causal);const c10::cuda::CUDAGuard guard(q.device());
    Flash_fwd_params p{};p.seqlen_q=int(q.size(1));p.seqlen_k=int(k.size(1));const auto value=selected(p,mapping);configure(value);
    std::vector<int64_t> r={1,q.size(1),32,4,1,int64_t(q.size(0))*32*((q.size(1)+value.m-1)/value.m),0,0,0,value.m,value.n,1};
    append_resource(r,value.function,value.threads,value.shared);r.insert(r.end(),8,0);append_layout_resources(r,q,k,v,true);return r;
}
} // namespace TORCH_EXTENSION_NAME
