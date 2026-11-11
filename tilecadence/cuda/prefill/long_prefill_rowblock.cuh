/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once
#include "flash_fwd_kernel.h"


// Replace cache priority only. The underlying 16-byte .cg copy and
// 128-byte L2 prefetch, layouts, and completion groups remain the same.
namespace cute {
template<bool Retain>
struct PrefillV72CacheHint {
    using SRegisters = uint128_t[1];
    using DRegisters = uint128_t[1];

    CUTE_HOST_DEVICE static void copy(const uint128_t& source, uint128_t& destination) {
#if defined(CUTE_ARCH_CP_ASYNC_SM80_ENABLED)
        const uint32_t shared = cast_smem_ptr_to_uint(&destination);
        if constexpr (Retain) {
            asm volatile(
                "{ .reg .b64 policy;\n"
                "  createpolicy.fractional.L2::evict_last.b64 policy, 0f3E000000;\n"
                "  cp.async.cg.shared.global.L2::cache_hint.L2::128B [%0], [%1], 16, policy;\n"
                "}\n" :: "r"(shared), "l"(&source));
        } else {
            asm volatile(
                "{ .reg .b64 policy;\n"
                "  createpolicy.fractional.L2::evict_first.b64 policy;\n"
                "  cp.async.cg.shared.global.L2::cache_hint.L2::128B [%0], [%1], 16, policy;\n"
                "}\n" :: "r"(shared), "l"(&source));
        }
#else
        CUTE_INVALID_CONTROL_PATH("V72 cache hints require SM80 cp.async");
#endif
    }
};

template<bool Retain>
struct Copy_Traits<PrefillV72CacheHint<Retain>>
    : Copy_Traits<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>> {};
}  // namespace cute

namespace FLASH_NAMESPACE {


template<class Full, class Packet, class Mma>
CUTE_HOST_DEVICE constexpr bool entry_packet_layout_matches() {
    constexpr Full full{};
    constexpr Packet packet{};
    constexpr decltype(convert_layout_acc_Aregs<Mma>(Full{})) full_p{};
    constexpr decltype(convert_layout_acc_Aregs<Mma>(Packet{})) packet_p{};
    for (int k = 0; k < 4; ++k) {
        for (int m = 0; m < 2; ++m) {
            for (int n = 0; n < 2; ++n) {
                for (int a = 0; a < 4; ++a) {
                    if (full(a, m, 2 * k + n) != 16 * k + packet(a, m, n)) return false;
                }
            }
            for (int v = 0; v < 8; ++v) {
                if (full_p(v, m, k) != 16 * k + packet_p(v, m, 0)) return false;
            }
        }
    }
    return true;
}

template<int kNRows>
struct RecordGateSoftmax : Softmax<kNRows> {
    template<typename TensorS, typename TensorO>
    __device__ __forceinline__ void rescale_steady(TensorS &acc_s, TensorO &acc_o, float scale) {
        auto &row_max = this->row_max;
        auto &row_sum = this->row_sum;
        auto scores = make_tensor(acc_s.data(), convert_layout_acc_rowcol(acc_s.layout()));
        auto out = make_tensor(acc_o.data(), convert_layout_acc_rowcol(acc_o.layout()));
        static_assert(decltype(size<0>(scores))::value == kNRows);
        static_assert(decltype(size<0>(out))::value == kNRows);
        auto previous = make_fragment_like(row_max);
        cute::copy(row_max, previous);
        MaxOp<float> max_op;
        thread_reduce_</*zero_init=*/false>(scores, row_max, max_op);
        bool recover = false;
        #pragma unroll
        for (int row = 0; row < kNRows; ++row) {
            const unsigned int old = __float_as_uint(previous(row));
            const unsigned int abs_old = old & 0x7fffffffu;
            recover |= abs_old < 0x00800000u || abs_old >= 0x7f800000u;
            recover |= old != __float_as_uint(row_max(row));
        }
        recover |= (__float_as_uint(scale) & 0x7f800000u) == 0x7f800000u;
        auto factors = make_fragment_like(row_max);
        cute::fill(factors, 1.0f);
        if (__any_sync(0xffffffffu, recover)) {
            quad_allreduce_(row_max, row_max, max_op);
            #pragma unroll
            for (int row = 0; row < kNRows; ++row) {
                const float factor = exp2f((previous(row) - row_max(row)) * scale);
                factors(row) = factor;
                #pragma unroll
                for (int col = 0; col < size<1>(out); ++col) out(row, col) *= factor;
            }
        }
        scale_apply_exp2(scores, row_max, scale);
        #pragma unroll
        for (int row = 0; row < kNRows; ++row) {
            float sum;
            // Match the compiled V33 update, with one rounding for alpha*l + p0.
            asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
                : "=f"(sum) : "f"(row_sum(row)), "f"(factors(row)), "f"(scores(row, 0)));
            #pragma unroll
            for (int col = 1; col < size<1>(scores); ++col) sum += scores(row, col);
            row_sum(row) = sum;
        }
    }
};

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params>
inline __device__ void grouped_prefill_rowblock(const Params &params, const int bidb, const int bidh, const int m_block, const int head_slot) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    static_assert(!Kernel_traits::Is_Q_in_regs && !Kernel_traits::Share_Q_K_smem);
    static_assert(!Is_dropout && Is_causal && !Is_local && !Has_alibi &&
                  Is_even_K && !Is_softcap && !Return_softmax);
    // Local IDs preserve each head's original CuTe MMA/copy partition.
    const int tidx = threadIdx.x - head_slot * Kernel_traits::kNThreads;
    const bool kv_copy_owner = head_slot == 0;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kNWarps = Kernel_traits::kNWarps;

    auto seed_offset = at::cuda::philox::unpack(params.philox_args);
    FLASH_NAMESPACE::Dropout dropout(std::get<0>(seed_offset), std::get<1>(seed_offset), params.p_dropout_in_uint8_t,
                           bidb, bidh, tidx, params.h);

    // Save seed and offset for backward, before any early exiting. Otherwise the 0-th thread block might
    // exit early and no one saves the rng states.
    if (Is_dropout && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tidx == 0) {
        params.rng_state[0] = std::get<0>(seed_offset);
        params.rng_state[1] = std::get<1>(seed_offset);
    }

    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const int n_block_min = !Is_local ? 0 : std::max(0, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);
    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);
    if (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max,
                               cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
        // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        //     printf("m_block = %d, n_block_max = %d\n", m_block, n_block_max);
        // }
    }
    // We exit early and write 0 to gO and gLSE. This also covers the case where actual_seqlen_k == 0.
    // Otherwise we might read OOB elements from gK and gV.
    if ((Is_causal || Is_local || !Is_even_MN) && n_block_max <= n_block_min) {
        Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
                                              + binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)),
                                make_shape(binfo.actual_seqlen_q, params.h, params.d),
                                make_stride(params.o_row_stride, params.o_head_stride, _1{}));
        Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                              make_coord(m_block, 0));  // (kBlockM, kHeadDim)

        Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);

        typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
        Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
        Tensor tOrO = make_tensor<Element>(shape(tOgO));
        clear(tOrO);
        // Construct identity layout for sO
        Tensor cO = make_identity_tensor(make_shape(size<0>(gO), size<1>(gO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
        if (!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d; }
        }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
        );
        #pragma unroll
        for (int m = 0; m < size<1>(tOgO); ++m) {
            const int row = get<0>(tOcO(0, m, 0));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM && get<1>(tOcO(0, m, 0)) == 0) { gLSE(row) = INFINITY; }
        }
        return;
    }
    // if (tidx == 0) { printf("m_block = %d, n_block_min = %d, n_block_max = %d\n", m_block, n_block_min, n_block_max); }

    // We iterate over the blocks in reverse order. This is because the last block is the only one
    // that needs masking when we read K and V from global memory. Moreover, iterating in reverse
    // might save us 1 register (we just need n_block instead of both n_block and n_block_max).

    const index_t row_offset_p = ((bidb * params.h + bidh) * params.seqlen_q_rounded
        + m_block * kBlockM) * params.seqlen_k_rounded + (n_block_max - 1) * kBlockN;

    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr)
                                          + binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)
    Tensor mK = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.k_ptr)
                                          + binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)),
                            make_shape(binfo.actual_seqlen_k, params.h_k, params.d),
                            make_stride(params.k_row_stride, params.k_head_stride, _1{}));
    Tensor gK = local_tile(mK(_, bidh / params.h_h_k_ratio, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                           make_coord(_, 0));  // (kBlockN, kHeadDim, nblocksN)
    Tensor mV = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.v_ptr)
                                          + binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb)),
                            make_shape(binfo.actual_seqlen_k, params.h_k, params.d),
                            make_stride(params.v_row_stride, params.v_head_stride, _1{}));
    Tensor gV = local_tile(mV(_, bidh / params.h_h_k_ratio, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                           make_coord(_, 0));  // (kBlockN, kHeadDim, nblocksN)
    Tensor gP = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.p_ptr) + row_offset_p),
                            Shape<Int<kBlockM>, Int<kBlockN>>{},
                            make_stride(params.seqlen_k_rounded, _1{}));

    auto shared_base = make_smem_ptr(reinterpret_cast<Element *>(smem_));
    constexpr int q_elements = Kernel_traits::kSmemQSize / sizeof(Element);
    Tensor sQ = make_tensor(shared_base + head_slot * q_elements,
                            typename Kernel_traits::SmemLayoutQ{});
    // K/V have one physical destination per CTA. Q/O remain head-private.
    Tensor sK = make_tensor(shared_base + PREFILL_GROUP_HEADS * q_elements,
                            typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle = make_tensor(sV.data().get(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});

    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    using KVCacheCopy = decltype(cute::make_tiled_copy(
        cute::Copy_Atom<cute::PrefillV72CacheHint<true>, Element>{},
        typename Kernel_traits::GmemLayoutAtom{}, cute::Layout<cute::Shape<cute::_1, cute::_8>>{}));
    static_assert(Kernel_traits::kGmemElemsPerLoad == 8);
    KVCacheCopy gmem_tiled_copy_hint;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K, nblocksN)
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K, nblocksN)
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

    typename Kernel_traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    Tensor tSrQ  = thr_mma.partition_fragment_A(sQ);                           // (MMA,MMA_M,MMA_K)
    Tensor tSrK  = thr_mma.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)
    Tensor tOrVt  = thr_mma.partition_fragment_B(sVtNoSwizzle);                // (MMA, MMA_K,MMA_N)

    Tensor tSgS  = thr_mma.partition_C(gP);

    Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});  // MMA, MMA_M, MMA_K

    //
    // Copy Atom retiling
    //

    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
    // if (cute::thread0()) {smem_thr_copy_Q.print_all();}
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);
    // if (cute::thread0()) {print(tSsQ.layout()); printf("\n");}

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

    //
    // PREDICATES
    //

    // // Allocate predicate tensors for m and n
    // Tensor tQpQ = make_tensor<bool>(make_shape(size<1>(tQsQ), size<2>(tQsQ)), Stride<_1,_0>{});
    // Tensor tKVpKV = make_tensor<bool>(make_shape(size<1>(tKsK), size<2>(tKsK)), Stride<_1,_0>{});

    // Construct identity layout for sQ and sK
    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    // Tensor tScQ = thr_mma.partition_A(cQ);                           // (MMA,MMA_M,MMA_K)
    // if (cute::thread0()) {
    //     print(tScQ.layout()); printf("\n");
    //     for (int i = 0; i < size(tScQ); ++i) {
    //         printf("%d ", get<0>(tScQ(i)));
    //     }
    //     printf("\n");
    //     for (int i = 0; i < size(tScQ); ++i) {
    //         printf("%d ", get<1>(tScQ(i)));
    //     }
    //     printf("\n");
    // }

    // Repeat the partitioning with identity layouts
    Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);   // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)

    // Allocate predicate tensors for k
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));

    // Set predicates for k bounds
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
        #pragma unroll
        for (int k = 0; k < size(tKVpKV); ++k) { tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d; }
    }

    // Prologue

    // We don't need to clear the sQ smem tiles since we'll only write out the valid outputs
    FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ,
                                       binfo.actual_seqlen_q - m_block * kBlockM);
    if (Kernel_traits::Is_Q_in_regs) { cute::cp_async_fence(); }

    // // if (cute::thread(1, 0)) { print(tQsQ); }
    // // Tensor sQNoSwizzle = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename Kernel_traits::SmemLayoutQNoSwizzle{});
    // // if (cute::thread0()) { print(sQNoSwizzle); }

    if (Kernel_traits::Share_Q_K_smem) {
        FLASH_NAMESPACE::cp_async_wait<0>();
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));            // M
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
        __syncthreads();
    }

    int n_block = n_block_max - 1;
    // We don't need to clear the sK smem tiles since we'll mask out the scores anyway.
    if (kv_copy_owner) {
        FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_hint, tKgK(_, _, _, n_block), tKsK, tKVcKV, tKVpKV,
                                           binfo.actual_seqlen_k - n_block * kBlockN);
    }
    cute::cp_async_fence();
    // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z < 2) { print(tKgK); }
    // __syncthreads();

    if (Kernel_traits::Is_Q_in_regs && !Kernel_traits::Share_Q_K_smem) {
        FLASH_NAMESPACE::cp_async_wait<1>();
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));            // M
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
    }

    clear(acc_o);

    FLASH_NAMESPACE::RecordGateSoftmax<2 * size<1>(acc_o)> softmax;

    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    FLASH_NAMESPACE::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right, alibi_slope);

    // For performance reason, we separate out two kinds of iterations:
    // those that need masking on S, and those that don't.
    // We need masking on S for the very last block when K and V has length not multiple of kBlockN.
    // We also need masking on S if it's causal, for the last ceil_div(kBlockM, kBlockN) blocks.
    // We will have at least 1 "masking" iteration.

    // If not even_N, then seqlen_k might end in the middle of a block. In that case we need to
    // mask 2 blocks (e.g. when kBlockM == kBlockN), not just 1.
    constexpr int n_masking_steps = (!Is_causal && !Is_local)
        ? 1
        : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);
    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        clear(acc_s);
        FLASH_NAMESPACE::cp_async_wait<0>();
        __syncthreads();

        if (kv_copy_owner) {
            // Advance gV
            if (masking_step > 0) {
                FLASH_NAMESPACE::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_hint, tVgV(_, _, _, n_block), tVsV, tKVcKV, tKVpKV);
            } else {
                // Clear the smem tiles to account for predicated off loads
                FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
                    gmem_tiled_copy_hint, tVgV(_, _, _, n_block), tVsV, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN
                );
            }
            cute::cp_async_fence();
        }

        FLASH_NAMESPACE::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
            acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K
        );
        // if (cute::thread0()) { print(acc_s); }
        if constexpr (Is_softcap){
            FLASH_NAMESPACE::apply_softcap(acc_s, params.softcap);
        }

        mask.template apply_mask<Is_causal, Is_even_MN>(
            acc_s, n_block * kBlockN, m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4, kNWarps * 16
        );

        FLASH_NAMESPACE::cp_async_wait<0>();
        __syncthreads();
        if (n_block > n_block_min && kv_copy_owner) {
            FLASH_NAMESPACE::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_hint, tKgK(_, _, _, n_block - 1), tKsK, tKVcKV, tKVpKV);
            // This cp_async_fence needs to be in the if block, otherwise the synchronization
            // isn't right and we get race conditions.
            cute::cp_async_fence();
        }

        // TODO: when we have key_padding_mask we'll need to Check_inf
        masking_step == 0
            ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2)
            : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2);

        // Convert acc_s from fp32 to fp16/bf16
        Tensor rP = FLASH_NAMESPACE::convert_type<Element>(acc_s);
        int block_row_idx = m_block * (kBlockM / 16) + tidx / 32;
        int block_col_idx = n_block * (kBlockN / 32);
        if (Return_softmax) {
            Tensor rP_drop = make_fragment_like(rP);
            cute::copy(rP, rP_drop);
            dropout.template apply_dropout</*encode_dropout_in_sign_bit=*/true>(
                rP_drop, block_row_idx, block_col_idx, kNWarps
            );
            cute::copy(rP_drop, tSgS);
            tSgS.data() = tSgS.data() + (-kBlockN);
        }
        if (Is_dropout) {
            dropout.apply_dropout(rP, block_row_idx, block_col_idx, kNWarps);
        }

        // Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
        // if using m16n8k16 or (4, MMA_M, MMA_N) if using m16n8k8.
        Tensor tOrP = make_tensor(rP.data(), FLASH_NAMESPACE::convert_layout_acc_Aregs<typename Kernel_traits::TiledMma>(rP.layout()));
        // if (cute::thread0()) { print(tOrP); }
        FLASH_NAMESPACE::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
        // if (cute::thread0()) { print(scores); }

        // This check is at the end of the loop since we always have at least 1 iteration
        if (n_masking_steps > 1 && n_block <= n_block_min) {
            --n_block;
            break;
        }
    }

    // These are the iterations where we don't need masking on S
    for (; n_block >= n_block_min; --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        clear(acc_s);
        FLASH_NAMESPACE::cp_async_wait<0>();
        __syncthreads();
        if (kv_copy_owner) {
            FLASH_NAMESPACE::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_hint, tVgV(_, _, _, n_block), tVsV, tKVcKV, tKVpKV);
            cute::cp_async_fence();
        }

        FLASH_NAMESPACE::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
            acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K
        );
        if constexpr (Is_softcap){
            FLASH_NAMESPACE::apply_softcap(acc_s, params.softcap);
        }

        FLASH_NAMESPACE::cp_async_wait<0>();
        __syncthreads();
        if (n_block > n_block_min && kv_copy_owner) {
            FLASH_NAMESPACE::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_hint, tKgK(_, _, _, n_block - 1), tKsK, tKVcKV, tKVpKV);
            // This cp_async_fence needs to be in the if block, otherwise the synchronization
            // isn't right and we get race conditions.
            cute::cp_async_fence();
        }

        mask.template apply_mask</*Causal_mask=*/false>(
            acc_s, n_block * kBlockN, m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4, kNWarps * 16
        );

        softmax.rescale_steady(acc_s, acc_o, params.scale_softmax_log2);
        constexpr auto packet_layout = make_layout(Shape<_4, _2, _2>{});
        static_assert(FLASH_NAMESPACE::entry_packet_layout_matches<typename decltype(acc_s)::layout_type,
            decltype(make_layout(Shape<_4, _2, _2>{})), typename Kernel_traits::TiledMma>());
        auto v_dst = smem_thr_copy_V.retile_D(tOrVt);
        cute::copy(smem_tiled_copy_V, tOsVt(_, _, _0{}), v_dst(_, _, _0{}));
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const auto scores_k = make_tensor(acc_s.data() + 16 * k, packet_layout);
            auto p_k = FLASH_NAMESPACE::convert_type<Element>(scores_k);
            auto mma_p = make_tensor(p_k.data(),
                FLASH_NAMESPACE::convert_layout_acc_Aregs<typename Kernel_traits::TiledMma>(p_k.layout()));
            if (k + 1 < 4) cute::copy(smem_tiled_copy_V, tOsVt(_, _, k + 1), v_dst(_, _, k + 1));
            cute::gemm(tiled_mma, mma_p(_, _, _0{}), tOrVt(_, _, k), acc_o);
        }
    }

    // Epilogue

    Tensor lse = softmax.template normalize_softmax_lse<Is_dropout>(acc_o, params.scale_softmax, params.rp_dropout);

    // Convert acc_o from fp32 to fp16/bf16
    Tensor rO = FLASH_NAMESPACE::convert_type<Element>(acc_o);
    Tensor sO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutO{});    // (SMEM_M,SMEM_N)
    // Partition sO to match the accumulator partitioning
    auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
    Tensor taccOrO = smem_thr_copy_O.retile_S(rO);        // ((Atom,AtomNum), MMA_M, MMA_N)
    Tensor taccOsO = smem_thr_copy_O.partition_D(sO);     // ((Atom,AtomNum),PIPE_M,PIPE_N)

    // sO has the same size as sQ, so we don't need to sync here.
    if (Kernel_traits::Share_Q_K_smem) { __syncthreads(); }

    cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

    Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
                                          + binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.o_row_stride, params.o_head_stride, _1{}));
    Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)
    Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);

    typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
    Tensor tOsO = gmem_thr_copy_O.partition_S(sO);        // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

    // The terminal QK has read K and passed its CTA barrier. The terminal
    // branch skips the next K copy, so both heads can stage LSE in that K span.
    static_assert(PREFILL_GROUP_HEADS == 2 && kBlockM == 128 && kBlockN == 64 && kHeadDim == 128);
    constexpr int kKStart = PREFILL_GROUP_HEADS * kBlockM * kHeadDim * sizeof(Element);
    constexpr int kKEnd = kKStart + kBlockN * kHeadDim * sizeof(Element);
    constexpr int kLseBytes = kBlockM * sizeof(ElementAccum);
    constexpr int kHead0 = PREFILL_LSE_HEAD0_OFFSET_BYTES;
    constexpr int kHead1 = PREFILL_LSE_HEAD1_OFFSET_BYTES;
    static_assert(kHead0 % 128 == 0 && kHead1 % 128 == 0);
    static_assert(kKStart <= kHead0 && kHead0 + kLseBytes <= kKEnd);
    static_assert(kKStart <= kHead1 && kHead1 + kLseBytes <= kKEnd);
    static_assert(kHead0 + kLseBytes <= kHead1 || kHead1 + kLseBytes <= kHead0);
    auto* lse_shared = reinterpret_cast<ElementAccum*>(smem_ + (head_slot == 0 ? kHead0 : kHead1));

    Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor taccOcO = thr_mma.partition_C(caccO);                           // (MMA,MMA_M,MMA_K)
    static_assert(decltype(size<0>(taccOcO))::value == 4);
    // Convert to ((2, 2), MMA_M, MMA_K) then take only the row indices.
    Tensor taccOcO_row = logical_divide(taccOcO, Shape<_2>{})(make_coord(0, _), _, 0);
    CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));                     // MMA_M
    if (get<1>(taccOcO_row(0)) == 0) {
        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = get<0>(taccOcO_row(mi));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM) { lse_shared[row] = lse(mi); }
        }
    }

    // Publish both LSE exchanges and the existing per-head O tiles together.
    __syncthreads();
    const int valid_lse_rows = binfo.actual_seqlen_q - m_block * kBlockM;
    // The dense ABI sets both unpadded_lse and seqlenq_ngroups_swapped to false.
    // get_lse_tile therefore has stride 1 and this pointer is its first row.
    auto* lse_global = gLSE.data().get();
    if (valid_lse_rows >= kBlockM && (reinterpret_cast<uintptr_t>(lse_global) & 15u) == 0) {
        static_assert(sizeof(ElementAccum) == 4 && kBlockM == 128);
        if (tidx < kBlockM / 4) {
            const int first_row = 4 * tidx;
            const uint32_t shared_address = cute::cast_smem_ptr_to_uint(lse_shared + first_row);
            const uint64_t global_address = __cvta_generic_to_global(lse_global + first_row);
            asm volatile(
                "{ .reg .b32 l0, l1, l2, l3;\n"
                "  ld.shared.v4.b32 {l0, l1, l2, l3}, [%0];\n"
                "  st.global.v4.b32 [%1], {l0, l1, l2, l3};\n"
                "}\n" :: "r"(shared_address), "l"(global_address) : "memory");
        }
    } else if (tidx < valid_lse_rows) {
        gLSE(tidx) = lse_shared[tidx];
    }

    Tensor tOrO = make_tensor<Element>(shape(tOgO));
    cute::copy(gmem_tiled_copy_O, tOsO, tOrO);

    // Construct identity layout for sO
    Tensor cO = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    // Repeat the partitioning with identity layouts
    Tensor tOcO = gmem_thr_copy_O.partition_D(cO);                           // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d; }
    }
    // Clear_OOB_K must be false since we don't want to write zeros to gmem
    FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
    );
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace FLASH_NAMESPACE
