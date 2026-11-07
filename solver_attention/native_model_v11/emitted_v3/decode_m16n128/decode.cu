// Actual HF DynamicCache NONAPPEND decode; arbitrary validated BNHD strides.
#include <ATen/ATen.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include "flash_fwd_launch_template.h"
#include "decode_traits.cuh"
#include "common.cuh"
namespace TORCH_EXTENSION_NAME {
int official_splits(int blocks,int sms,int nblocks,int cap) {
    if(blocks>=0.8f*sms)return 1;cap=std::min({cap,sms,nblocks});float best=0.f;std::vector<float> efficiency(cap,0.f);
    auto divup=[](int a,int b){return(a+b-1)/b;};
    for(int s=1;s<=cap;++s){if(s!=1&&divup(nblocks,s)==divup(nblocks,s-1))continue;float waves=float(blocks*s)/sms;efficiency[s-1]=waves/std::ceil(waves);best=std::max(best,efficiency[s-1]);}
    for(int s=1;s<=cap;++s){if(s!=1&&divup(nblocks,s)==divup(nblocks,s-1))continue;if(efficiency[s-1]>=0.85*best)return s;}return 1;
}
int actual_splits(const at::Tensor& q,const at::Tensor& k,int64_t requested,int64_t mapping) {
    TORCH_CHECK(mapping==0||mapping==1,"Only native default0 and frozen candidate1 are exposed");
    if(mapping==0){TORCH_CHECK(requested==0,"Native default requires automatic S0");int sms=0;C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,q.get_device()));return official_splits(int(q.size(0))*8,2*sms,int((k.size(1)+127)/128),128);}
    TORCH_CHECK(requested>=1&&requested<=128,"Frozen candidate requires explicit S1..128");return int(requested);
}
std::vector<int64_t> model_workspace_spec(const at::Tensor& q,const at::Tensor& k,const at::Tensor& v,int64_t requested,int64_t mapping,bool causal) {
    check_inputs(q,k,v,causal,false);const c10::cuda::CUDAGuard guard(q.device());int s=actual_splits(q,k,requested,mapping);const int64_t rows=q.size(0)*32;
    return {s,s>1?s*rows*128:0,s>1?s*rows:0};
}
Flash_fwd_params parameters(const at::Tensor& q,const at::Tensor& k,const at::Tensor& v,at::Tensor& out,at::Tensor& lse,at::Tensor& ws_o,at::Tensor& ws_lse,int splits,double scale) {
    Flash_fwd_params p;set_common(p,q,k,v,out,lse,scale);
    p.q_row_stride=q.stride(2);p.q_head_stride=4*q.stride(2);p.o_row_stride=out.stride(2);p.o_head_stride=4*out.stride(2);
    p.h=8;p.h_k=8;p.h_h_k_ratio=1;p.seqlen_q=4;p.total_q=4*p.b;p.seqlen_q_rounded=128;p.is_causal=false;p.num_splits=splits;
    if(splits>1){p.oaccum_ptr=ws_o.data_ptr();p.softmax_lseaccum_ptr=ws_lse.data_ptr();}return p;
}
template<bool Split>__global__ void candidate_kernel(KERNEL_PARAM_MODIFIER const Flash_fwd_params p) {
    const int b=Split?blockIdx.z/8:blockIdx.z,h=Split?blockIdx.z-b*8:blockIdx.y;
    compute_attn_1rowblock_splitkv<CandidateTraits,false,false,false,false,true,false,Split,false>(p,b,h,blockIdx.x,Split?blockIdx.y:0,Split?gridDim.y:1);
}
template<int Log>void combine(Flash_fwd_params& p,cudaStream_t stream){flash_fwd_splitkv_combine_kernel<DecodeTraits,4,Log,true><<<dim3((p.b*p.h*p.seqlen_q+3)/4),128,0,stream>>>(p);}
void combine_dispatch(Flash_fwd_params& p,cudaStream_t stream){if(p.num_splits<=1)return;if(p.num_splits<=2)combine<1>(p,stream);else if(p.num_splits<=4)combine<2>(p,stream);else if(p.num_splits<=8)combine<3>(p,stream);else if(p.num_splits<=16)combine<4>(p,stream);else if(p.num_splits<=32)combine<5>(p,stream);else if(p.num_splits<=64)combine<6>(p,stream);else combine<7>(p,stream);}
template<bool Split>void launch_candidate(Flash_fwd_params& p,cudaStream_t stream){constexpr int shared=CandidateTraits::kSmemSize;auto kernel=&candidate_kernel<Split>;if(shared>=48*1024)C10_CUDA_CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,shared));kernel<<<dim3(1,Split?p.num_splits:p.h,Split?p.b*p.h:p.b),CandidateTraits::kNThreads,shared,stream>>>(p);combine_dispatch(p,stream);}
void model_forward_out(const at::Tensor& q,const at::Tensor& k,const at::Tensor& v,at::Tensor& out,at::Tensor& lse,at::Tensor& ws_o,at::Tensor& ws_lse,int64_t requested,int64_t mapping,double scale,bool causal){
    check_outputs(q,k,v,out,lse,causal,false);const c10::cuda::CUDAGuard guard(q.device());auto spec=model_workspace_spec(q,k,v,requested,mapping,causal);
    for(const auto* w:{&ws_o,&ws_lse}){TORCH_CHECK(w->device()==q.device()&&w->scalar_type()==at::kFloat&&w->is_contiguous()&&!w->requires_grad(),"Contiguous FP32 split workspace required");}
    TORCH_CHECK(ws_o.numel()>=spec[1]&&ws_lse.numel()>=spec[2],"Complete split workspace too small");
    const at::Tensor* writes[]={&out,&lse,&ws_o,&ws_lse};const at::Tensor* reads[]={&q,&k,&v};
    for(auto* w:writes){for(auto* r:reads){TORCH_CHECK(!overlaps(*w,*r),"Write aliases strided Q/K/V span");}}
    for(int i=0;i<4;++i){for(int j=i+1;j<4;++j){TORCH_CHECK(!overlaps(*writes[i],*writes[j]),"Writable buffers overlap");}}
    auto p=parameters(q,k,v,out,lse,ws_o,ws_lse,int(spec[0]),scale);auto stream=c10::cuda::getCurrentCUDAStream(q.get_device()).stream();
    if(mapping==0){if(p.num_splits>1)run_mha_fwd_splitkv_dispatch<DecodeElement,128,false>(p,stream);else run_mha_fwd_hdim128<DecodeElement,false>(p,stream);}
    else if(p.num_splits>1)launch_candidate<true>(p,stream);else launch_candidate<false>(p,stream);C10_CUDA_KERNEL_LAUNCH_CHECK();
}
template<int Log>void combine_resource(std::vector<int64_t>& r){append_resource(r,&flash_fwd_splitkv_combine_kernel<DecodeTraits,4,Log,true>,128,0);}
std::vector<int64_t> model_resources(const at::Tensor& q,const at::Tensor& k,const at::Tensor& v,int64_t requested,int64_t mapping,bool causal){
    check_inputs(q,k,v,causal,false);const c10::cuda::CUDAGuard guard(q.device());auto spec=model_workspace_spec(q,k,v,requested,mapping,causal);int s=int(spec[0]);
    std::vector<int64_t> r={s,4,8,4,s>1?2:1,int64_t(q.size(0))*8*(s>1?s:1),s>1?int64_t(q.size(0))*8:0,spec[1],spec[2],mapping?CandidateTraits::kBlockM:(s>1?64:128),mapping?CandidateTraits::kBlockN:(s>1?128:64),1};
    if(mapping==0){if(s==1)append_resource(r,&flash_fwd_kernel<OrdinaryTraits,false,false,false,false,false,true,false,false>,128,OrdinaryTraits::kSmemSize);else append_resource(r,&flash_fwd_splitkv_kernel<DecodeTraits,false,false,false,false,true,false,true,false>,128,DecodeTraits::kSmemSize);}
    else if(s>1)append_resource(r,&candidate_kernel<true>,CandidateTraits::kNThreads,CandidateTraits::kSmemSize);else append_resource(r,&candidate_kernel<false>,CandidateTraits::kNThreads,CandidateTraits::kSmemSize);
    if(s==1)r.insert(r.end(),8,0);else if(s<=2)combine_resource<1>(r);else if(s<=4)combine_resource<2>(r);else if(s<=8)combine_resource<3>(r);else if(s<=16)combine_resource<4>(r);else if(s<=32)combine_resource<5>(r);else if(s<=64)combine_resource<6>(r);else combine_resource<7>(r);
    append_layout_resources(r,q,k,v,false);return r;
}
} // namespace TORCH_EXTENSION_NAME
