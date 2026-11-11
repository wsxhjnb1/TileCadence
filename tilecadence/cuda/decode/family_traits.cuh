#pragma once
#include "config.cuh"
#include "v7_traits.cuh"

namespace TORCH_EXTENSION_NAME {
using DecodeElement = cutlass::bfloat16_t;
using DecodeTraits = Flash_fwd_kernel_traits<128, 64, 128, 4, false, false, DecodeElement>;
using OrdinaryTraits = Flash_fwd_kernel_traits<128, 128, 64, 4, false, false, DecodeElement>;
using CandidateBase = NativeQwenV7Traits<128, 16 * AB_DECODE_WARPS, AB_DECODE_BLOCK_N,
    AB_DECODE_WARPS, AB_DECODE_Q_RESIDENCY != 0, AB_DECODE_Q_RESIDENCY == 2, DecodeElement, true>;
using CandidateTraits = NativeQwenV7DecodeOutputTraits<CandidateBase>;
} // namespace TORCH_EXTENSION_NAME
