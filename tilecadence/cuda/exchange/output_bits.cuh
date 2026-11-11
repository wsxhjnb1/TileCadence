// V5 ordinary-output conversion policy; no reduced-precision representation.
#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <type_traits>
#include <cutlass/numeric_types.h>

namespace fa2_solver_striped_v5 {

// PTX cvt.{f16x2,bf16x2}.f32 places its first input in HIGH16 and its
// second input in LOW16. Passing (high,low) preserves [low,high] memory order.
// RN with no ftz/relu/sat modifier matches the ordinary output conversion.
// Primary specification:
// https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt
template<class Element>
__device__ __forceinline__ uint32_t output_pair_rn(float low,float high) {
    uint32_t result;
    if constexpr(std::is_same_v<Element,cutlass::half_t>) {
        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(result) : "f"(high), "f"(low));
    } else {
        static_assert(std::is_same_v<Element,cutlass::bfloat16_t>);
        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(result) : "f"(high), "f"(low));
    }
    return result;
}

// A cache policy affects replacement priority, not mathematical values or
// global visibility. The caller proves an aligned16-byte address and one writer.
__device__ __forceinline__ void output_store_streaming128(void* target,uint4 value) {
    asm volatile("st.global.cs.v4.b32 [%0], {%1, %2, %3, %4};"
        :: "l"(target),"r"(value.x),"r"(value.y),"r"(value.z),"r"(value.w) : "memory");
}

} // namespace fa2_solver_striped_v5
