// Isolated V10: full-capacity KV cache plus GPU logical old lengths, Qlen1.
// Both native-default and frozen-candidate paths use original fused Append_KV.
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>
#include "flash_fwd_launch_template.h"
#include "family_traits.cuh"
#include "ab_fused_splitkv.cuh"

namespace TORCH_EXTENSION_NAME {
constexpr int DecodeHkv=8,DecodeGroups=4,DecodeDim=128,DecodeQueryHeads=32;

// Same float arithmetic and eligibility/85%-of-best rule as the pinned API.
int official_splits(int blocks,int sms,int nblocks,int cap) {
    if(blocks>=0.8f*sms)return 1;
    cap=std::min({cap,sms,nblocks});
    float best=0.f;std::vector<float> efficiency(cap,0.f);
    auto divup=[](int a,int b){return(a+b-1)/b;};
    for(int splits=1;splits<=cap;++splits) {
        if(splits!=1&&divup(nblocks,splits)==divup(nblocks,splits-1))continue;
        float waves=float(blocks*splits)/sms;
        efficiency[splits-1]=waves/std::ceil(waves);
        best=std::max(best,efficiency[splits-1]);
    }
    for(int splits=1;splits<=cap;++splits) {
        if(splits!=1&&divup(nblocks,splits)==divup(nblocks,splits-1))continue;
        if(efficiency[splits-1]>=0.85*best)return splits;
    }
    return 1;
}

void check_q_cache_lengths(const at::Tensor& q,const at::Tensor& cache,const at::Tensor& lengths) {
    TORCH_CHECK(q.is_cuda()&&cache.is_cuda()&&lengths.is_cuda()&&q.device()==cache.device()
        &&q.device()==lengths.device(),"Q/cache/old_lengths must be CUDA on the same device");
    TORCH_CHECK(q.dim()==4&&q.size(0)>0&&q.size(1)==1&&q.size(2)==32&&q.size(3)==128,
        "Q must be [B,1,32,128]");
    TORCH_CHECK(cache.dim()==4&&cache.size(0)==q.size(0)&&cache.size(1)>=4
        &&cache.size(2)==8&&cache.size(3)==128,"Full capacity cache must be [B,C>=4,8,128]");
    TORCH_CHECK(q.scalar_type()==at::kBFloat16&&cache.scalar_type()==at::kBFloat16
        &&q.is_contiguous()&&cache.is_contiguous()&&!q.requires_grad()&&!cache.requires_grad(),
        "Contiguous BF16 inference Q and FULL capacity cache required");
    TORCH_CHECK(lengths.scalar_type()==at::kInt&&lengths.is_contiguous()&&lengths.dim()==1
        &&lengths.size(0)==q.size(0)&&!lengths.requires_grad(),"old_lengths must be contiguous CUDA int32[B]");
    // No host read or synchronization of device lengths is hidden in this ABI.
    // The caller supplies device old_lengths in [0, C-1] for each append.
    // Resource length fields describe the capacity boundary, not current device values.
    TORCH_CHECK(cache.size(1)<=std::numeric_limits<int>::max()-127
        &&q.size(0)<=std::numeric_limits<int>::max()/32,"Native int geometry overflow");
    for(const auto* tensor:{&q,&cache}) {
        TORCH_CHECK(tensor->numel()*tensor->element_size()<=std::numeric_limits<int32_t>::max(),
            "Each Q/cache allocation must fit the bounded <=2GiB wrapper domain");
        TORCH_CHECK(reinterpret_cast<std::uintptr_t>(tensor->data_ptr())%16==0,"16-byte aligned Q/cache required");
    }
}

int actual_splits(const at::Tensor& q,const at::Tensor& cache,int64_t requested,int64_t mapping) {
    TORCH_CHECK(mapping==0||mapping==1,"Expected native mapping0 or V40 candidate mapping1");
    if(mapping==0) {
        TORCH_CHECK(requested==0,"Native baseline exposes only official automatic num_splits=0");
    } else {
        TORCH_CHECK((q.size(0)==1||q.size(0)==4)&&cache.size(1)>=128&&cache.size(1)<=32776,
                    "V40 requires B1/B4 with a cache capacity in [128,32776]");
    }
    int sms=0;C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,q.get_device()));
    const int splits=official_splits(int(q.size(0))*8,2*sms,int((cache.size(1)+127)/128),128);
    TORCH_CHECK(requested==0||requested==splits,"V40 requires the upstream default split count");
    return splits;
}

std::vector<int64_t> decode_workspace_spec(const at::Tensor& q,const at::Tensor& cache,
    const at::Tensor& old_lengths,int64_t requested,int64_t mapping,bool append) {
    check_q_cache_lengths(q,cache,old_lengths);const c10::cuda::CUDAGuard guard(q.device());
    TORCH_CHECK(mapping==0||append,"V40 candidate requires a fused append");
    const int splits=actual_splits(q,cache,requested,mapping);
    const int64_t rows=q.size(0)*DecodeHkv*DecodeGroups;
    return {splits,splits>1?splits*rows*DecodeDim:0,splits>1?splits*rows:0};
}

bool overlap(const at::Tensor& a,const at::Tensor& b) {
    if(a.numel()==0||b.numel()==0)return false;
    const auto x=reinterpret_cast<std::uintptr_t>(a.data_ptr()),y=reinterpret_cast<std::uintptr_t>(b.data_ptr());
    return x<y+b.numel()*b.element_size()&&y<x+a.numel()*a.element_size();
}

Flash_fwd_params params_for(const at::Tensor& q,const at::Tensor& kcache,const at::Tensor& vcache,
    const at::Tensor& knew,const at::Tensor& vnew,const at::Tensor& old_lengths,
    at::Tensor& out,at::Tensor& lse,at::Tensor& ws_o,at::Tensor& ws_lse,int splits,bool append) {
    Flash_fwd_params p{};
    p.q_ptr=q.data_ptr();p.k_ptr=kcache.data_ptr();p.v_ptr=vcache.data_ptr();p.o_ptr=out.data_ptr();
    p.softmax_lse_ptr=lse.data_ptr();
    // Original q1 GQA reshape, via strides only: [B,1,32,128] -> [B,4,8,128].
    p.q_batch_stride=q.stride(0);p.o_batch_stride=out.stride(0);
    p.q_row_stride=128;p.o_row_stride=128;p.q_head_stride=4*128;p.o_head_stride=4*128;
    p.k_batch_stride=kcache.stride(0);p.v_batch_stride=vcache.stride(0);
    p.k_row_stride=kcache.stride(1);p.v_row_stride=vcache.stride(1);
    p.k_head_stride=kcache.stride(2);p.v_head_stride=vcache.stride(2);
    p.b=int(q.size(0));p.h=8;p.h_k=8;p.h_h_k_ratio=1;
    p.seqlen_q=4;p.seqlen_k=int(kcache.size(1));p.total_q=4*p.b;
    p.seqlen_q_rounded=128;p.seqlen_k_rounded=((p.seqlen_k+127)/128)*128;
    p.d=128;p.d_rounded=128;p.scale_softmax=1.f/std::sqrt(128.f);p.scale_softmax_log2=p.scale_softmax*M_LOG2E;
    p.p_dropout=1.f;p.p_dropout_in_uint8_t=255;p.rp_dropout=1.f;p.scale_softmax_rp_dropout=p.scale_softmax;
    p.is_bf16=true;p.is_causal=false;p.window_size_left=-1;p.window_size_right=-1;
    p.cu_seqlens_k=old_lengths.data_ptr<int>();p.is_seqlens_k_cumulative=false;p.num_splits=splits;
    p.unpadded_lse=false;p.seqlenq_ngroups_swapped=false;
    if(append) {
        p.seqlen_knew=1;p.knew_ptr=knew.data_ptr();p.vnew_ptr=vnew.data_ptr();
        p.knew_batch_stride=knew.stride(0);p.vnew_batch_stride=vnew.stride(0);
        p.knew_row_stride=knew.stride(1);p.vnew_row_stride=vnew.stride(1);
        p.knew_head_stride=knew.stride(2);p.vnew_head_stride=vnew.stride(2);
    }
    if(splits>1){p.oaccum_ptr=ws_o.data_ptr();p.softmax_lseaccum_ptr=ws_lse.data_ptr();}
    return p;
}

// B-generated candidate with the pinned fused append and split combiner.
template<bool Split,bool Append>
__global__ void decode_candidate_kernel(KERNEL_PARAM_MODIFIER const Flash_fwd_params p) {
    const int m=blockIdx.x;
    const int b=Split?blockIdx.y/8:blockIdx.z;
    const int h=Split?blockIdx.y-b*8:blockIdx.y;
    const int split=Split?blockIdx.z:0,count=Split?gridDim.z:1;
    compute_long_ab_splitkv<CandidateTraits,false,false,false,false,true,false,Split,Append>(p,b,h,m,split,count);
}

template<int Log>void combine(Flash_fwd_params& p,cudaStream_t stream) {
    const dim3 grid((p.b*p.h*p.seqlen_q+3)/4);
    flash_fwd_splitkv_combine_kernel<DecodeTraits,4,Log,true><<<grid,128,0,stream>>>(p);
}
void combine_dispatch(Flash_fwd_params& p,cudaStream_t stream) {
    if(p.num_splits<=1)return;
    if(p.num_splits<=2)combine<1>(p,stream);else if(p.num_splits<=4)combine<2>(p,stream);
    else if(p.num_splits<=8)combine<3>(p,stream);else if(p.num_splits<=16)combine<4>(p,stream);
    else if(p.num_splits<=32)combine<5>(p,stream);else if(p.num_splits<=64)combine<6>(p,stream);else combine<7>(p,stream);
}
template<bool Split>
constexpr int candidate_shared() {
    constexpr int output = CandidateTraits::kBlockM * DecodeDim * (Split ? 4 : 2);
    constexpr int kv_bytes = AB_DECODE_TILE_SLOTS
        * CandidateTraits::kBlockN * DecodeDim * sizeof(DecodeElement);
    return std::max(kv_bytes, output);
}
static_assert(candidate_shared<true>() == AB_DECODE_SPLIT_SHARED_BYTES);

template<bool Split,bool Append>void launch_candidate(Flash_fwd_params& p,cudaStream_t stream) {
    constexpr int shared=candidate_shared<Split>();
    auto kernel=&decode_candidate_kernel<Split,Append>;
    if(shared>=48*1024)C10_CUDA_CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,shared));
    const dim3 grid(1,Split?p.b*p.h:p.h,Split?p.num_splits:p.b);
    kernel<<<grid,CandidateTraits::kNThreads,shared,stream>>>(p);combine_dispatch(p,stream);
}

void decode_forward_out(const at::Tensor& q,at::Tensor& kcache,at::Tensor& vcache,
    const at::Tensor& knew,const at::Tensor& vnew,const at::Tensor& old_lengths,
    at::Tensor& out,at::Tensor& lse,at::Tensor& ws_o,at::Tensor& ws_lse,
    int64_t requested,int64_t mapping,bool append) {
    check_q_cache_lengths(q,kcache,old_lengths);const c10::cuda::CUDAGuard guard(q.device());
    TORCH_CHECK(vcache.device()==q.device()&&vcache.sizes()==kcache.sizes()&&vcache.scalar_type()==q.scalar_type()
        &&vcache.is_contiguous()&&!vcache.requires_grad(),"Vcache must match FULL contiguous BF16 Kcache");
    for(const auto* tensor:{&knew,&vnew})TORCH_CHECK(tensor->device()==q.device()&&tensor->scalar_type()==q.scalar_type()
        &&tensor->sizes()==at::IntArrayRef({q.size(0),1,8,128})&&tensor->is_contiguous()&&!tensor->requires_grad(),
        "knew/vnew must be contiguous BF16[B,1,8,128], also supplied for the A nonappend call");
    TORCH_CHECK(out.device()==q.device()&&out.sizes()==q.sizes()&&out.scalar_type()==q.scalar_type()
        &&out.is_contiguous()&&!out.requires_grad(),"Invalid complete output");
    TORCH_CHECK(lse.device()==q.device()&&lse.scalar_type()==at::kFloat&&lse.is_contiguous()
        &&lse.sizes()==at::IntArrayRef({q.size(0),32,1})&&!lse.requires_grad(),"Invalid full float32 LSE");
    const auto spec=decode_workspace_spec(q,kcache,old_lengths,requested,mapping,append);
    for(const auto* workspace:{&ws_o,&ws_lse})TORCH_CHECK(workspace->is_cuda()&&workspace->device()==q.device()
        &&workspace->scalar_type()==at::kFloat&&workspace->is_contiguous()&&!workspace->requires_grad(),"Contiguous CUDA FP32 workspace required");
    TORCH_CHECK(ws_o.numel()>=spec[1]&&ws_lse.numel()>=spec[2],"Preallocated complete split workspace is too small");
    const at::Tensor* tensors[]={&q,&kcache,&vcache,&knew,&vnew,&old_lengths,&out,&lse,&ws_o,&ws_lse};
    for(int i=0;i<10;++i)for(int j=i+1;j<10;++j)TORCH_CHECK(!overlap(*tensors[i],*tensors[j]),"All nonempty ABI tensor allocations must be disjoint");
    const at::Tensor* aligned[]={&vcache,&knew,&vnew,&out};
    for(const auto* tensor:aligned)TORCH_CHECK(reinterpret_cast<std::uintptr_t>(tensor->data_ptr())%16==0,"16-byte aligned BF16 storage required");
    Flash_fwd_params p=params_for(q,kcache,vcache,knew,vnew,old_lengths,out,lse,ws_o,ws_lse,int(spec[0]),append);
    auto stream=c10::cuda::getCurrentCUDAStream(q.get_device()).stream();
    if(mapping==0) {
        // Exact DEFAULT fwd_kvcache dispatch: append forces native splitKV,
        // including S1. A nonappend/S1 call uses the ordinary native path.
        if(append||p.num_splits>1)run_mha_fwd_splitkv_dispatch<DecodeElement,128,false>(p,stream);
        else run_mha_fwd_hdim128<DecodeElement,false>(p,stream);
    } else if(append) {
        if(p.num_splits>1)launch_candidate<true,true>(p,stream);else launch_candidate<false,true>(p,stream);
    } else {
        if(p.num_splits>1)launch_candidate<true,false>(p,stream);else launch_candidate<false,false>(p,stream);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<typename Kernel>void append_resource(std::vector<int64_t>& result,Kernel kernel,int threads,int shared) {
    if(shared>=48*1024)C10_CUDA_CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,shared));
    cudaFuncAttributes a{};C10_CUDA_CHECK(cudaFuncGetAttributes(&a,kernel));int active=0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,kernel,threads,shared));
    for(int64_t value:{int64_t(threads),int64_t(shared),int64_t(a.sharedSizeBytes),int64_t(a.numRegs),
        int64_t(a.localSizeBytes),int64_t(a.binaryVersion),int64_t(a.ptxVersion),int64_t(active)})result.push_back(value);
}
template<int Log>void combine_resource(std::vector<int64_t>& values) {
    append_resource(values,&flash_fwd_splitkv_combine_kernel<DecodeTraits,4,Log,true>,128,0);
}
template<bool Split,bool Append>void main_resource(std::vector<int64_t>& values,int64_t mapping) {
    if(mapping==0)append_resource(values,&flash_fwd_splitkv_kernel<DecodeTraits,false,false,false,false,true,false,Split,Append>,128,DecodeTraits::kSmemSize);
    else append_resource(values,&decode_candidate_kernel<Split,Append>,CandidateTraits::kNThreads,candidate_shared<Split>());
}

std::vector<int64_t> decode_resources(const at::Tensor& q,const at::Tensor& kcache,const at::Tensor& old_lengths,
    int64_t requested,int64_t mapping,bool append) {
    check_q_cache_lengths(q,kcache,old_lengths);const c10::cuda::CUDAGuard guard(q.device());
    int major=0;C10_CUDA_CHECK(cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,q.get_device()));
    TORCH_CHECK(major>=8 && append,"Resource reporting requires SM80 or newer and fused KV append");
    const auto spec=decode_workspace_spec(q,kcache,old_lengths,requested,mapping,append);const int splits=int(spec[0]);
    const bool split_path=append||splits>1;
    std::vector<int64_t> values={splits,4,8,4,splits>1?2:1,int64_t(q.size(0))*8*(splits>1?splits:1),splits>1?int64_t(q.size(0))*8:0,
        spec[1],spec[2],mapping==1?CandidateTraits::kBlockM:(split_path?64:128),mapping==1?CandidateTraits::kBlockN:(split_path?128:64),1};
    if(mapping==0&&!split_path)append_resource(values,&flash_fwd_kernel<OrdinaryTraits,false,false,false,false,false,true,false,false>,128,OrdinaryTraits::kSmemSize);
    else if(append) {if(splits>1)main_resource<true,true>(values,mapping);else main_resource<false,true>(values,mapping);}
    else {if(splits>1)main_resource<true,false>(values,mapping);else main_resource<false,false>(values,mapping);}
    if(splits<=1)values.insert(values.end(),8,0);
    else if(splits<=2)combine_resource<1>(values);else if(splits<=4)combine_resource<2>(values);
    else if(splits<=8)combine_resource<3>(values);else if(splits<=16)combine_resource<4>(values);
    else if(splits<=32)combine_resource<5>(values);else if(splits<=64)combine_resource<6>(values);else combine_resource<7>(values);
    for(int64_t value:{int64_t(kcache.size(1)),int64_t(kcache.size(1)-1),int64_t(kcache.size(1)-1+(append?1:0)),
        int64_t(append),int64_t(mapping==0),int64_t(V10_FAMILY_CODE)})values.push_back(value);
    return values;
}
} // namespace TORCH_EXTENSION_NAME
