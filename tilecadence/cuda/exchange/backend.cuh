// Revision V5-ready: compile-time acquires and full-V views over V5 math.
// CUDA/CuTe primitives for the bounded action-level striped attention emitter.
// Uses FlashAttention-2's pinned accumulator-layout conversion helper in utils.h
// (copyright Tri Dao). The action graph and storage are emitted separately.
#pragma once

#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <type_traits>
#include <cute/tensor.hpp>
#include <cutlass/numeric_types.h>
#include "output_bits.cuh"
// Pinned utils.h uses unqualified CuTe names, as does native kernel_traits.h.
using namespace cute;
#include "utils.h"

#ifndef FA2_V5_OUTPUT_PACKED_CVT
#define FA2_V5_OUTPUT_PACKED_CVT 0
#endif
#ifndef FA2_V5_OUTPUT_STREAMING_STORE
#define FA2_V5_OUTPUT_STREAMING_STORE 0
#endif
static_assert(FA2_V5_OUTPUT_PACKED_CVT==0 || FA2_V5_OUTPUT_PACKED_CVT==1);
static_assert(FA2_V5_OUTPUT_STREAMING_STORE==0 || FA2_V5_OUTPUT_STREAMING_STORE==1);

namespace fa2_solver_striped_v5_ready {
using namespace cute;
using fa2_solver_striped_v5::output_pair_rn;
using fa2_solver_striped_v5::output_store_streaming128;

constexpr float kScale = 0.08838834764831845f;
constexpr float kScaleLog2 = 0.12751743082459868f;

template<class T>
struct Backend {
    using Atom = std::conditional_t<std::is_same_v<T, cutlass::half_t>,
        MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
        MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>>;
    using Mma = TiledMMA<Atom, Layout<Shape<_4,_1,_1>>, Tile<_64,_16,_16>>;
    using SmemAtom = decltype(composition(Swizzle<3,3,3>{},
        Layout<Shape<_8,_64>, Stride<_64,_1>>{}));
    using SmemQ = decltype(tile_to_shape(SmemAtom{}, Shape<_64,_128>{}));
    using SmemK = decltype(tile_to_shape(SmemAtom{}, Shape<_32,_128>{}));
    using SmemV = decltype(tile_to_shape(SmemAtom{}, Shape<_32,_64>{}));
    // Full V parent shares the exact32x128 layout used by K. Its two32x64
    // local_tile views have scalar16 origins0 and2048 (0/4096 bytes).
    using SmemV128 = decltype(tile_to_shape(SmemAtom{}, Shape<_32,_128>{}));
    using SmemVFull = SmemV128;
    using SmemVt = decltype(composition(SmemV{},
        make_layout(Shape<_64,_32>{}, GenRowMajor{})));
    using SmemVtNoSwizzle = decltype(get_nonswizzle_portion(SmemVt{}));
    using Checkpoint = Layout<Shape<_64,_32>, Stride<_32,_1>>;
    using SmemO64 = decltype(tile_to_shape(SmemAtom{}, Shape<_64,_64>{}));
    using Rows = decltype(make_tensor<float>(Shape<_2>{}));
};

template<class T, int R, int C, class Shared>
__device__ __forceinline__ void issue_g2s(const T* source, int row_stride, Shared destination) {
    auto global = make_tensor(make_gmem_ptr(source), Shape<Int<R>,Int<C>>{},
                              make_stride(row_stride, _1{}));
    auto tiled = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>,T>{},
        Layout<Shape<_16,_8>,Stride<_8,_1>>{}, Layout<Shape<_1,_8>>{});
    auto thread = tiled.get_thread_slice(threadIdx.x);
    cute::copy(tiled, thread.partition_S(global), thread.partition_D(destination));
    cute::cp_async_fence();
}

// Completion/acquire belongs to the selected consumer action. The first V4
// primitive conservatively waits for all older issue groups. The solver can
// place independent compute between issue and consumer; no timing guarantee is
// inferred from these source boundaries.
__device__ __forceinline__ void acquire_issued_shared() {
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
}

__device__ __forceinline__ void release_shared_reads() {
    __syncthreads();
}

template<class T, bool OperandA, bool Acquire, class Shared, class Mma>
__device__ __forceinline__ auto load_operand_ready(const Shared& source, Mma mma) {
    // Acquire=false is legal only with the solver's global committed-copy
    // readiness certificate. It emits neither wait_group nor a CTA barrier.
    if constexpr(Acquire) acquire_issued_shared();
    auto thread = mma.get_thread_slice(threadIdx.x);
    if constexpr (OperandA) {
        auto result = thread.partition_fragment_A(source);
        auto tiled = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N,T>{}, mma);
        auto copy_thread = tiled.get_thread_slice(threadIdx.x);
        cute::copy(tiled, copy_thread.partition_S(source), copy_thread.retile_D(result));
        return result;
    } else {
        auto result = thread.partition_fragment_B(source);
        auto tiled = make_tiled_copy_B(Copy_Atom<SM75_U32x4_LDSM_N,T>{}, mma);
        auto copy_thread = tiled.get_thread_slice(threadIdx.x);
        cute::copy(tiled, copy_thread.partition_S(source), copy_thread.retile_D(result));
        return result;
    }
}

template<class T, bool Acquire, class Shared, class Mma>
__device__ __forceinline__ auto load_v_ready(const Shared& source, Mma mma) {
    if constexpr(Acquire) acquire_issued_shared();
    auto transposed = make_tensor(source.data(), typename Backend<T>::SmemVt{});
    auto no_swizzle = make_tensor(source.data().get(), typename Backend<T>::SmemVtNoSwizzle{});
    auto result = mma.get_thread_slice(threadIdx.x).partition_fragment_B(no_swizzle);
    auto tiled = make_tiled_copy_B(Copy_Atom<SM75_U16x8_LDSM_T,T>{}, mma);
    auto thread = tiled.get_thread_slice(threadIdx.x);
    cute::copy(tiled, thread.partition_S(transposed), thread.retile_D(result));
    return result;
}

template<class T, int Stripe, bool Acquire, class SharedParent, class Mma>
__device__ __forceinline__ auto load_v_parent_ready(const SharedParent& parent,Mma mma) {
    static_assert(Stripe==0 || Stripe==1);
    // The parent is an8192-byte SmemV128 allocation. local_tile respects its
    // nontrivial swizzle; Stripe1 is4096bytes from the parent, not128bytes.
    auto view=local_tile(parent,Shape<_32,_64>{},make_coord(_0{},Int<Stripe>{}));
    return load_v_ready<T,Acquire>(view,mma);
}

// Full acquire compatibility wrappers are deliberately explicit. New ready
// emitters should call the compile-time helpers and bind each bool to a replay.
template<class T, bool OperandA, class Shared, class Mma>
__device__ __forceinline__ auto load_operand(const Shared& source,Mma mma) {
    return load_operand_ready<T,OperandA,true>(source,mma);
}
template<class T, class Shared, class Mma>
__device__ __forceinline__ auto load_v(const Shared& source,Mma mma) {
    return load_v_ready<T,true>(source,mma);
}

template<class Element, class Shared, class Mma>
__device__ __forceinline__ auto load_checkpoint(const Shared& source, Mma mma) {
    auto result = make_tensor<Element>(partition_fragment_C(mma, Shape<_64,_32>{}).layout());
    auto coordinates=mma.get_thread_slice(threadIdx.x).partition_C(make_identity_tensor(Shape<_64,_32>{}));
    #pragma unroll
    for(int i=0;i<size(result);++i) {
        const int index=get<0>(coordinates(i))*32+get<1>(coordinates(i));
        if constexpr(std::is_same_v<Element,float>) {
            result(i)=reinterpret_cast<const float*>(source.data().get())[index];
        } else {
            const uint16_t bits=reinterpret_cast<const uint16_t*>(source.data().get())[index];
            result(i)=Element::bitcast(bits);
        }
    }
    return result;
}

template<class Fragment, class Shared, class Mma>
__device__ __forceinline__ void store_checkpoint(const Fragment& source, Shared destination, Mma mma) {
    using Element=typename Fragment::value_type;
    auto coordinates=mma.get_thread_slice(threadIdx.x).partition_C(make_identity_tensor(Shape<_64,_32>{}));
    #pragma unroll
    for(int i=0;i<size(source);++i) {
        const int index=get<0>(coordinates(i))*32+get<1>(coordinates(i));
        if constexpr(std::is_same_v<Element,float>) {
            reinterpret_cast<float*>(destination.data().get())[index]=source(i);
        } else {
            reinterpret_cast<uint16_t*>(destination.data().get())[index]=source(i).raw();
        }
    }
    __syncthreads();
}

__device__ __forceinline__ uint32_t load_nonce(const uint32_t* nonce, int event_slot) {
    uint32_t value;
    asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(value) : "l"(nonce+event_slot) : "memory");
    return value;
}

template<class Fragment>
__device__ __forceinline__ auto opaque_half_operand(const Fragment& source,
                                                  const uint32_t* nonce, int event_slot) {
    using Element=typename Fragment::value_type;
    auto result=make_fragment_like(source);
    const uint16_t mask=static_cast<uint16_t>(load_nonce(nonce,event_slot));
    #pragma unroll
    for(int i=0;i<size(source);++i) result(i)=Element::bitcast(source(i).raw() ^ mask);
    return result;
}

template<class Fragment>
__device__ __forceinline__ auto opaque_score_operand(const Fragment& source,
                                                   const uint32_t* nonce, int event_slot) {
    auto result = make_fragment_like(source);
    const uint32_t mask = load_nonce(nonce,event_slot);
    #pragma unroll
    for (int i=0; i<size(source); ++i) result(i)=__uint_as_float(__float_as_uint(source(i)) ^ mask);
    return result;
}

template<class Mma, class A, class B>
__device__ __forceinline__ auto qk(Mma mma, const A& a, const B& b) {
    auto result = partition_fragment_C(mma, Shape<_64,_32>{});
    cute::clear(result);
    #pragma unroll
    for (int k=0; k<size<2>(a); ++k) cute::gemm(mma,a(_,_,k),b(_,_,k),result);
    return result;
}

__device__ __forceinline__ float quad_max(float value) {
    value=fmaxf(value,__shfl_xor_sync(0xffffffffu,value,2));
    return fmaxf(value,__shfl_xor_sync(0xffffffffu,value,1));
}
__device__ __forceinline__ float quad_sum(float value) {
    value+=__shfl_xor_sync(0xffffffffu,value,2);
    return value+__shfl_xor_sync(0xffffffffu,value,1);
}

template<bool Causal, int Tile, class S, class Mma>
__device__ __forceinline__ auto masked_scores(const S& source, Mma mma) {
    auto result = make_fragment_like(source);
    auto coordinates = mma.get_thread_slice(threadIdx.x).partition_C(make_identity_tensor(Shape<_64,_32>{}));
    #pragma unroll
    for (int i=0; i<size(source); ++i) {
        const int row=get<0>(coordinates(i));
        const int key=Tile*32+get<1>(coordinates(i));
        result(i)=Causal && key>row ? -INFINITY : source(i);
    }
    return result;
}

template<bool Causal, int Tile, class S, class Row, class P, class Mma>
__device__ __forceinline__ void stats_first(const S& source, Row& maximum, Row& denominator, P& probability, Mma mma) {
    auto temporary=masked_scores<Causal,Tile>(source,mma);
    auto rows=make_tensor(temporary.data(),FLASH_NAMESPACE::convert_layout_acc_rowcol(temporary.layout()));
    CUTE_STATIC_ASSERT_V(size<0>(rows)==size(maximum));
    #pragma unroll
    for (int r=0; r<size<0>(rows); ++r) {
        float value=-INFINITY;
        #pragma unroll
        for (int c=0; c<size<1>(rows); ++c) value=fmaxf(value,rows(r,c));
        maximum(r)=quad_max(value);
        const float safe=maximum(r)==-INFINITY ? 0.f : maximum(r);
        float sum=0.f;
        #pragma unroll
        for (int c=0; c<size<1>(rows); ++c) {
            rows(r,c)=exp2f(rows(r,c)*kScaleLog2-safe*kScaleLog2);
            sum+=rows(r,c);
        }
        denominator(r)=quad_sum(sum);
    }
    using Element=typename P::value_type;
    #pragma unroll
    for (int i=0; i<size(temporary); ++i) probability(i)=Element(temporary(i));
}

template<bool Causal, int Tile, class S, class Row, class P, class Mma>
__device__ __forceinline__ void stats_next(const S& source, const Row& old_max, const Row& old_sum,
                                         Row& maximum, Row& denominator, Row& alpha, P& probability, Mma mma) {
    auto temporary=masked_scores<Causal,Tile>(source,mma);
    auto rows=make_tensor(temporary.data(),FLASH_NAMESPACE::convert_layout_acc_rowcol(temporary.layout()));
    #pragma unroll
    for (int r=0; r<size<0>(rows); ++r) {
        float value=-INFINITY;
        #pragma unroll
        for (int c=0; c<size<1>(rows); ++c) value=fmaxf(value,rows(r,c));
        maximum(r)=fmaxf(old_max(r),quad_max(value));
        const float safe=maximum(r)==-INFINITY ? 0.f : maximum(r);
        alpha(r)=exp2f((old_max(r)-safe)*kScaleLog2);
        float sum=0.f;
        #pragma unroll
        for (int c=0; c<size<1>(rows); ++c) {
            rows(r,c)=exp2f(rows(r,c)*kScaleLog2-safe*kScaleLog2);
            sum+=rows(r,c);
        }
        denominator(r)=alpha(r)*old_sum(r)+quad_sum(sum);
    }
    using Element=typename P::value_type;
    #pragma unroll
    for (int i=0; i<size(temporary); ++i) probability(i)=Element(temporary(i));
}

template<bool Causal, int Tile, class S, class Row, class P, class Mma>
__device__ __forceinline__ void recover_p(const S& source, const Row& maximum, P& probability, Mma mma) {
    auto temporary=masked_scores<Causal,Tile>(source,mma);
    auto rows=make_tensor(temporary.data(),FLASH_NAMESPACE::convert_layout_acc_rowcol(temporary.layout()));
    #pragma unroll
    for (int r=0; r<size<0>(rows); ++r) {
        const float safe=maximum(r)==-INFINITY ? 0.f : maximum(r);
        #pragma unroll
        for (int c=0; c<size<1>(rows); ++c) rows(r,c)=exp2f(rows(r,c)*kScaleLog2-safe*kScaleLog2);
    }
    using Element=typename P::value_type;
    #pragma unroll
    for (int i=0; i<size(temporary); ++i) probability(i)=Element(temporary(i));
}

template<class Mma, class P, class V, class O>
__device__ __forceinline__ void pv(Mma mma, const P& probability, const V& value, O& output) {
    auto operand=make_tensor(probability.data(),FLASH_NAMESPACE::convert_layout_acc_Aregs<Mma>(probability.layout()));
    #pragma unroll
    for (int k=0; k<size<2>(operand); ++k) cute::gemm(mma,operand(_,_,k),value(_,_,k),output);
}

template<class O, class Row>
__device__ __forceinline__ void rescale(O& output, const Row& alpha) {
    auto rows=make_tensor(output.data(),FLASH_NAMESPACE::convert_layout_acc_rowcol(output.layout()));
    #pragma unroll
    for (int r=0; r<size<0>(rows); ++r) {
        #pragma unroll
        for (int c=0; c<size<1>(rows); ++c) rows(r,c)*=alpha(r);
    }
}

template<class Element, int Stripe, class O, class Row, class Mma>
__device__ __forceinline__ void normalize_store(O& output, const Row& denominator,
                                               Element* target, int row_stride, Mma mma) {
    auto rows=make_tensor(output.data(),FLASH_NAMESPACE::convert_layout_acc_rowcol(output.layout()));
    #pragma unroll
    for (int r=0; r<size<0>(rows); ++r) {
        float sum=denominator(r);
        float inverse=(sum==0.f || sum!=sum) ? 1.f : 1.f/sum;
        #pragma unroll
        for (int c=0; c<size<1>(rows); ++c) rows(r,c)*=inverse;
    }
    // Explicit raw16 stores avoid generic CuTe half-to-byte numeric conversion.
    // C-coordinate ownership gives exactly one writer for each output element.
    auto coordinates=mma.get_thread_slice(threadIdx.x).partition_C(make_identity_tensor(Shape<_64,_64>{}));
    #pragma unroll
    for(int i=0;i<size(output);++i) {
        const int row=get<0>(coordinates(i));
        const int column=get<1>(coordinates(i));
        const Element converted=Element(output(i));
        reinterpret_cast<uint16_t*>(target)[row*row_stride+Stripe*64+column]=converted.raw();
    }
}

template<class O,class Row>
__device__ __forceinline__ void normalize_rows(O& output,const Row& denominator) {
    auto rows=make_tensor(output.data(),FLASH_NAMESPACE::convert_layout_acc_rowcol(output.layout()));
    #pragma unroll
    for(int r=0;r<size<0>(rows);++r) {
        const float sum=denominator(r);
        const float inverse=(sum==0.f || sum!=sum) ? 1.f : 1.f/sum;
        #pragma unroll
        for(int c=0;c<size<1>(rows);++c) rows(r,c)*=inverse;
    }
}

// The SM80 C layout owns adjacent two-element pairs in each thread. All bit
// packing is integral; no generic half-to-byte conversion is used.
template<class Element,int Stripe,class O,class Row,class Mma>
__device__ __forceinline__ void normalize_store_packed32(O& output,const Row& denominator,
                                                        Element* target,int row_stride,Mma mma) {
    normalize_rows(output,denominator);
    auto coordinates=mma.get_thread_slice(threadIdx.x).partition_C(make_identity_tensor(Shape<_64,_64>{}));
    static_assert(decltype(size(output))::value%2==0);
    #pragma unroll
    for(int i=0;i<size(output);i+=2) {
        const int row=get<0>(coordinates(i));
        const int column=get<1>(coordinates(i));
#if FA2_V5_OUTPUT_PACKED_CVT
        const uint32_t pair=output_pair_rn<Element>(output(i),output(i+1));
#else
        const uint32_t low=Element(output(i)).raw();
        const uint32_t high=Element(output(i+1)).raw();
        const uint32_t pair=low|(high<<16);
#endif
        reinterpret_cast<uint32_t*>(target)[(row*row_stride+Stripe*64+column)/2]=pair;
    }
}

// 8KiB scratch is an explicit solver allocation for exactly this close action.
// C-layout pairs first store raw32 into the native swizzle. A CTA exchange then
// gives each thread four contiguous aligned128-bit vectors for global output.
// A later overlapping write must emit a release barrier, derived from concrete
// address hazards by the emitter. No unnecessary final barrier is inserted.
template<class Element,int Stripe,class O,class Row,class Mma>
__device__ __forceinline__ void normalize_store_coalesced128(O& output,const Row& denominator,
    Element* target,int row_stride,Mma mma,unsigned char* scratch) {
    normalize_rows(output,denominator);
    auto coordinates=mma.get_thread_slice(threadIdx.x).partition_C(make_identity_tensor(Shape<_64,_64>{}));
    typename Backend<Element>::SmemO64 layout;
    static_assert(decltype(cosize(layout))::value*sizeof(Element)==8192);
    static_assert(decltype(size(output))::value%2==0);
    #pragma unroll
    for(int i=0;i<size(output);i+=2) {
        const int row=get<0>(coordinates(i));
        const int column=get<1>(coordinates(i));
#if FA2_V5_OUTPUT_PACKED_CVT
        const uint32_t pair=output_pair_rn<Element>(output(i),output(i+1));
#else
        const uint32_t low=Element(output(i)).raw();
        const uint32_t high=Element(output(i+1)).raw();
        const uint32_t pair=low|(high<<16);
#endif
        const int index=layout(make_coord(row,column));
        *reinterpret_cast<uint32_t*>(scratch+index*sizeof(Element))=pair;
    }
    __syncthreads();
    #pragma unroll
    for(int block=0;block<4;++block) {
        const int row=threadIdx.x/8+block*16;
        const int column=(threadIdx.x%8)*8;
        const int index=layout(make_coord(row,column));
        const uint4 vector=*reinterpret_cast<const uint4*>(scratch+index*sizeof(Element));
#if FA2_V5_OUTPUT_STREAMING_STORE
        output_store_streaming128(target+row*row_stride+Stripe*64+column,vector);
#else
        *reinterpret_cast<uint4*>(target+row*row_stride+Stripe*64+column)=vector;
#endif
    }
}

template<class Row>
__device__ __forceinline__ void store_lse(const Row& maximum, const Row& denominator, float* target) {
    const int lane=threadIdx.x%32;
    const int first_row=(threadIdx.x/32)*16+lane/4;
    if (lane%4==0) {
        #pragma unroll
        for (int r=0; r<size(maximum); ++r) {
            const float sum=denominator(r);
            target[first_row+r*8]=(sum==0.f || sum!=sum) ? INFINITY : maximum(r)*kScale+logf(sum);
        }
    }
}

template<class Fragment>
__device__ __forceinline__ void diagnostic_consume_non_nan(const Fragment& value) {
    #pragma unroll
    for (int i=0; i<size(value); ++i) {
        const float scalar=float(value(i));
        asm volatile("{ .reg .pred p; testp.notanumber.f32 p, %0; @p trap; }" :: "f"(scalar) : "memory");
    }
}
} // namespace fa2_solver_striped_v5_ready
