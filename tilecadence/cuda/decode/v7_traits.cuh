// New isolated trait adapter; pinned vendor headers and tensor math are unchanged.
#pragma once
#include <type_traits>
#include "kernel_traits.h"

namespace TORCH_EXTENSION_NAME {
template<int D,int M,int N,int W,bool QRegs,bool ShareQK,class Element,bool CacheAlways,
         class Base=Flash_kernel_traits<D,M,N,W,Element>>
struct NativeQwenV7Traits : Flash_fwd_kernel_traits<D,M,N,W,QRegs,ShareQK,Element,Base> {
    using Parent=Flash_fwd_kernel_traits<D,M,N,W,QRegs,ShareQK,Element,Base>;
    using Gmem_copy_struct=std::conditional_t<CacheAlways && Parent::Has_cp_async,
        cute::SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>,typename Parent::Gmem_copy_struct>;
    using GmemTiledCopyQKV=decltype(cute::make_tiled_copy(
        cute::Copy_Atom<Gmem_copy_struct,typename Parent::Element>{},
        typename Parent::GmemLayoutAtom{},cute::Layout<cute::Shape<cute::_1,cute::_8>>{}));
    static_assert(CacheAlways || std::is_same_v<GmemTiledCopyQKV,typename Parent::GmemTiledCopyQKV>);
    static_assert(std::is_same_v<typename Parent::SmemLayoutKV,
        typename Flash_fwd_kernel_traits<D,M,N,W,QRegs,ShareQK,Element,Base>::SmemLayoutKV>);
};

template<class Base,bool OneWarp=(Base::kNThreads==32)>
struct NativeQwenV7DecodeOutputTraits : Base {};

template<class Base>
struct NativeQwenV7DecodeOutputTraits<Base,true> : Base {
    static_assert(Base::kBlockM==16 && Base::kHeadDim==128);
    // Inherited from the validated V62 M16 repair: all four effective GQA rows
    // receive FP32 split accumulators; the native 128-thread combiner is retained.
    using GmemTiledCopyOaccum=decltype(cute::make_tiled_copy(
        cute::Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<128>,float>{},
        cute::Layout<cute::Shape<cute::_4,cute::_8>,cute::Stride<cute::_8,cute::_1>>{},
        cute::Layout<cute::Shape<cute::_1,cute::_4>>{}));
};
} // namespace TORCH_EXTENSION_NAME
