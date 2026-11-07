// Bounded V10 family selection; original native tensor math is included unchanged.
#pragma once
#include "config.cuh"
#include "v7_traits.cuh"

namespace TORCH_EXTENSION_NAME {
using DecodeElement=cutlass::bfloat16_t;
using DecodeTraits=Flash_fwd_kernel_traits<128,64,128,4,false,false,DecodeElement>;
using OrdinaryTraits=Flash_fwd_kernel_traits<128,128,64,4,false,false,DecodeElement>;
#if MODEL_FAMILY_CODE == 0
using CandidateTraits=DecodeTraits;
#elif MODEL_FAMILY_CODE == 1
using CandidateTraits=Flash_fwd_kernel_traits<128,32,128,2,false,false,DecodeElement>;
#elif MODEL_FAMILY_CODE == 2
struct CandidateTraits : Flash_fwd_kernel_traits<128,16,64,1,false,false,DecodeElement> {
    // Exact V62 four-GQA-row accumulator ownership repair, unchanged.
    using GmemTiledCopyOaccum=decltype(cute::make_tiled_copy(
        cute::Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<128>,float>{},
        cute::Layout<cute::Shape<cute::_4,cute::_8>,cute::Stride<cute::_8,cute::_1>>{},
        cute::Layout<cute::Shape<cute::_1,cute::_4>>{}));
};
static_assert(CandidateTraits::kNThreads==32 && CandidateTraits::kSmemSize==36864);
#elif MODEL_FAMILY_CODE == 3 || MODEL_FAMILY_CODE == 4
using CandidateBase=NativeQwenV7Traits<128,16,MODEL_CANDIDATE_BLOCK_N,1,
    false,false,DecodeElement,false>;
using CandidateTraits=NativeQwenV7DecodeOutputTraits<CandidateBase>;
#else
#error Unsupported bounded V10 family
#endif
static_assert(!CandidateTraits::Is_Q_in_regs && !CandidateTraits::Share_Q_K_smem);
} // namespace TORCH_EXTENSION_NAME
