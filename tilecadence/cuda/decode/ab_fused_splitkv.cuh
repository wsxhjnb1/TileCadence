// Derived from flash_fwd_kernel.h, Copyright (c) 2024, Tri Dao.
#pragma once

namespace FLASH_NAMESPACE {

template<typename Kernel_traits, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Split, bool Append_KV, typename Params>
inline __device__ void compute_long_ab_splitkv(const Params &params, const int bidb, const int bidh, const int m_block, const int n_split_idx, const int num_n_splits) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kNWarps = Kernel_traits::kNWarps;

    using GmemTiledCopyO = std::conditional_t<
        !Split,
        typename Kernel_traits::GmemTiledCopyO,
        typename Kernel_traits::GmemTiledCopyOaccum
    >;
    using ElementO = std::conditional_t<!Split, Element, ElementAccum>;

    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) { printf("Is_even_MN = %d, is_cumulativ = %d, seqlen_k_cache = %d, actual_seqlen_k = %d\n", Is_even_MN, params.is_seqlens_k_cumulative, binfo.seqlen_k_cache, binfo.actual_seqlen_k); }
    // if (threadIdx.x == 0 && blockIdx.y == 1 && blockIdx.z == 0) { printf("params.knew_ptr = %p, seqlen_k_cache + seqlen_knew = %d\n", params.knew_ptr, binfo.seqlen_k_cache + (params.knew_ptr == nullptr ? 0 : params.seqlen_knew)); }
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    static_assert(!Is_causal && !Is_local);
    const int n_blocks_per_split = ((params.seqlen_k + kBlockN - 1) / kBlockN + num_n_splits - 1) / num_n_splits;
    const int n_block_min = !Is_local
        ? n_split_idx * n_blocks_per_split
        : std::max(n_split_idx * n_blocks_per_split, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);
    int n_block_max = std::min(cute::ceil_div(binfo.actual_seqlen_k, kBlockN), (n_split_idx + 1) * n_blocks_per_split);
    if (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max,
                               cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
    }
    if (n_block_min >= n_block_max) {  // This also covers the case where n_block_max <= 0
        // We exit early and write 0 to gOaccum and -inf to gLSEaccum.
        // Otherwise we might read OOB elements from gK and gV,
        // or get wrong results when we combine gOaccum from different blocks.
        const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
            + m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
        const index_t row_offset_oaccum = (((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q
            + m_block * kBlockM) * params.d_rounded;
        const index_t row_offset_lseaccum = ((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q + m_block * kBlockM;
        Tensor gOaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO *>(Split ? params.oaccum_ptr : params.o_ptr) + (Split ? row_offset_oaccum : row_offset_o)),
                                      Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                     make_stride(Split ? kHeadDim : params.o_row_stride, _1{}));
        Tensor gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(Split ? params.softmax_lseaccum_ptr : params.softmax_lse_ptr) + row_offset_lseaccum),
                                      Shape<Int<kBlockM>>{}, Stride<_1>{});

        GmemTiledCopyO gmem_tiled_copy_Oaccum;
        auto gmem_thr_copy_Oaccum = gmem_tiled_copy_Oaccum.get_thread_slice(tidx);
        Tensor tOgOaccum = gmem_thr_copy_Oaccum.partition_D(gOaccum);
        Tensor tOrOaccum = make_tensor<ElementO>(shape(tOgOaccum));
        clear(tOrOaccum);
        // Construct identity layout for sO
        Tensor cO = make_identity_tensor(make_shape(size<0>(gOaccum), size<1>(gOaccum)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tOcO = gmem_thr_copy_Oaccum.partition_D(cO);
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgOaccum)));
        if (!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d; }
        }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_Oaccum, tOrOaccum, tOgOaccum, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
        );
        #pragma unroll
        for (int m = 0; m < size<1>(tOgOaccum); ++m) {
            const int row = get<0>(tOcO(0, m, 0));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM && get<1>(tOcO(0, m, 0)) == 0) { gLSEaccum(row) = Split ? -INFINITY : INFINITY; }
        }
        return;
    }

    // We iterate over the blocks in reverse order. This is because the last block is the only one
    // that needs masking when we read K and V from global memory. Moreover, iterating in reverse
    // might save us 1 register (we just need n_block instead of both n_block and n_block_max).

    // We move K and V to the last block.
    const int bidb_cache = params.cache_batch_idx == nullptr ? bidb : params.cache_batch_idx[bidb];
    const int *block_table = params.block_table == nullptr ? nullptr : params.block_table + bidb * params.block_table_batch_stride;
    const int block_table_idx = block_table == nullptr ? 0 : (n_block_max - 1) * kBlockN / params.page_block_size;
    const int block_table_offset = block_table == nullptr ? 0 : (n_block_max - 1) * kBlockN - block_table_idx * params.page_block_size;
    const index_t row_offset_k = block_table == nullptr
        ? binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb_cache)
          + (n_block_max - 1) * kBlockN * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride
        : block_table[block_table_idx] * params.k_batch_stride + block_table_offset * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t row_offset_v = block_table == nullptr
        ? binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb_cache)
          + (n_block_max - 1) * kBlockN * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride
        : block_table[block_table_idx] * params.v_batch_stride + block_table_offset * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride;

    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr) + binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));
    // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) { printf("k_ptr = %p, row_offset_k = %d, gK_ptr = %p\n", params.k_ptr, row_offset_k, gK.data()); }
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.v_row_stride, _1{}));

    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQ{});
    Tensor sK = make_tensor(sQ.data() + (Kernel_traits::Share_Q_K_smem ? 0 : size(sQ)),
                            typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutKV{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle = make_tensor(sV.data().get(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});

    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K)
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K)
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

    typename Kernel_traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    Tensor tSrQ  = thr_mma.partition_fragment_A(sQ);                           // (MMA,MMA_M,MMA_K)
    Tensor tSrK  = thr_mma.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)
    Tensor tOrVt  = thr_mma.partition_fragment_B(sVtNoSwizzle);                // (MMA, MMA_K,MMA_N)

    Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});  // MMA, MMA_M, MMA_K

    //
    // Copy Atom retiling
    //

    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

    // PREDICATES
    //

    // // Allocate predicate tensors for m and n
    // Tensor tQpQ = make_tensor<bool>(make_shape(size<1>(tQsQ), size<2>(tQsQ)), Stride<_1,_0>{});
    // Tensor tKVpKV = make_tensor<bool>(make_shape(size<1>(tKsK), size<2>(tKsK)), Stride<_1,_0>{});

    // Construct identity layout for sQ and sK
    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)

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

    // The host passes pre-rotated Q/K, dense contiguous KV and one new row.
    if constexpr (Append_KV) {
        static_assert(kHeadDim == 128 && Is_even_K);
        const int cache_row = binfo.seqlen_k_cache;
        const int append_tile = cache_row / kBlockN;
        if (n_block_min <= append_tile && append_tile < n_block_max) {
            const int vector_column = tidx * (sizeof(uint4) / sizeof(Element));
            if (vector_column < kHeadDim) {
                const int kv_head = bidh / params.h_h_k_ratio;
                const auto dst_k = binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)
                    + cache_row * params.k_row_stride + kv_head * params.k_head_stride + vector_column;
                const auto dst_v = binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb)
                    + cache_row * params.v_row_stride + kv_head * params.v_head_stride + vector_column;
                const auto src_k = bidb * params.knew_batch_stride + kv_head * params.knew_head_stride + vector_column;
                const auto src_v = bidb * params.vnew_batch_stride + kv_head * params.vnew_head_stride + vector_column;
                *reinterpret_cast<uint4*>(static_cast<Element*>(params.k_ptr) + dst_k)
                    = *reinterpret_cast<const uint4*>(static_cast<const Element*>(params.knew_ptr) + src_k);
                *reinterpret_cast<uint4*>(static_cast<Element*>(params.v_ptr) + dst_v)
                    = *reinterpret_cast<const uint4*>(static_cast<const Element*>(params.vnew_ptr) + src_v);
            }
            // Publish the entire row before this CTA asynchronously loads the tile.
            __syncthreads();
        }
    }

    static_assert(kNWarps == 4 && kBlockM == 64 && kBlockN == 128);
    static_assert(AB_DECODE_PRODUCER_MASK == 15 && AB_DECODE_BARRIER_MASK == 15);
    const bool computes_q = (AB_DECODE_COMPUTE_MASK >> (tidx / 32)) & 1;

    FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K>(
        gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ, binfo.actual_seqlen_q - m_block * kBlockM);

    // When Q and K alias, every Q shared read finishes before any K write.
    if constexpr (Kernel_traits::Is_Q_in_regs) { cute::cp_async_fence(); }
    if constexpr (Kernel_traits::Share_Q_K_smem) {
        FLASH_NAMESPACE::cp_async_wait<0>();
        __syncthreads();
        Tensor q_register_copy = smem_thr_copy_Q.retile_D(tSrQ);
        if (computes_q) { cute::copy(smem_tiled_copy_Q, tSsQ, q_register_copy); }
        __syncthreads();
    }
    // Q is in registers. K and V reuse the same shared tile in order.
    static_assert(!Is_causal && !Is_local && !Has_alibi && !Is_even_MN && Is_even_K && !Is_softcap);
    static_assert(Kernel_traits::Share_Q_K_smem && Kernel_traits::Is_Q_in_regs);
    static_assert(AB_DECODE_TILE_SLOTS == 1);
    constexpr int arena_bytes = kBlockN * kHeadDim * sizeof(Element);
    static_assert(arena_bytes == 32 * 1024);
    static_assert(Kernel_traits::kSmemKVSize == 2 * arena_bytes);
    static_assert(Kernel_traits::kSmemQSize <= arena_bytes);
    static_assert(kBlockM * kHeadDim * sizeof(ElementO) <= arena_bytes);
    const int k_tile_stride = kBlockN * params.k_row_stride;
    const int v_tile_stride = kBlockN * params.v_row_stride;
    const auto first_gK = tKgK.data();
    const auto first_gV = tVgV.data();
    const int first_tile = n_block_max - 1;

    auto load_k = [&](auto first, int tile) {
        tKgK.data() = first_gK + (tile - first_tile) * k_tile_stride;
        if constexpr (decltype(first)::value) {
            FLASH_NAMESPACE::copy<false, true>(
                gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV,
                binfo.actual_seqlen_k - tile * kBlockN);
        } else {
            FLASH_NAMESPACE::copy<true, true>(gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV);
        }
    };
    auto load_v = [&](auto first, int tile) {
        tVgV.data() = first_gV + (tile - first_tile) * v_tile_stride;
        if constexpr (decltype(first)::value) {
            FLASH_NAMESPACE::copy<false, true, /*Clear_OOB_MN=*/true>(
                gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV,
                binfo.actual_seqlen_k - tile * kBlockN);
        } else {
            FLASH_NAMESPACE::copy<true, true>(gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV);
        }
    };

    clear(acc_o);
    FLASH_NAMESPACE::Softmax<2 * size<1>(acc_o)> softmax;
    FLASH_NAMESPACE::Mask<false, false, false> mask(
        binfo.actual_seqlen_k, binfo.actual_seqlen_q,
        params.window_size_left, params.window_size_right, 0.0f);
    auto step = [&](auto first, int tile) {
        Tensor scores = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
        load_k(first, tile);
        cute::cp_async_fence();
        FLASH_NAMESPACE::cp_async_wait<0>();
        __syncthreads();
        if (computes_q) {
            clear(scores);
            FLASH_NAMESPACE::gemm<true>(
                scores, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q,
                smem_tiled_copy_K, smem_thr_copy_Q, smem_thr_copy_K);
        }
        // K cannot be overwritten until the final QK shared reads retire.
        __syncthreads();
        load_v(first, tile);
        cute::cp_async_fence();
        Tensor probabilities = make_tensor<Element>(scores.layout());
        if (computes_q) {
            if constexpr (decltype(first)::value) {
                mask.template apply_mask<false, false>(
                    scores, tile * kBlockN, m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4, kNWarps * 16);
                softmax.template softmax_rescale_o<true, true>(scores, acc_o, params.scale_softmax_log2);
            } else {
                softmax.template softmax_rescale_o<false, false>(scores, acc_o, params.scale_softmax_log2);
            }
            cute::copy(FLASH_NAMESPACE::convert_type<Element>(scores), probabilities);
        }
        FLASH_NAMESPACE::cp_async_wait<0>();
        __syncthreads();
        if (computes_q) {
            Tensor tOrP = make_tensor(probabilities.data(),
                FLASH_NAMESPACE::convert_layout_acc_Aregs<typename Kernel_traits::TiledMma>(probabilities.layout()));
            FLASH_NAMESPACE::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
        }
        // The original epilogue barrier retires the terminal V readers.
        // Nonterminal tiles must retire V before the next K copy is issued.
        if (tile > n_block_min) __syncthreads();
    };

    step(std::true_type{}, first_tile);
    for (int tile = first_tile - 1; tile >= n_block_min; --tile) {
        step(std::false_type{}, tile);
    }

    // Epilogue

    Tensor lse = make_fragment_like(softmax.row_sum);
    if (computes_q) {
        Tensor active_lse = softmax.template normalize_softmax_lse<false, Split>(acc_o, params.scale_softmax);
        cute::copy(active_lse, lse);
    } else {
        cute::fill(lse, Split ? -INFINITY : INFINITY);
    }
    // if (cute::thread0()) { print(lse); }


    if constexpr (AB_DECODE_COMPACT_EPILOGUE) {
        // The host GQA reshape always supplies four valid rows per head CTA.
        constexpr int rows = 4;
        constexpr int output_offset = AB_DECODE_O_OFFSET_BYTES;
        constexpr int lse_offset = AB_DECODE_LSE_OFFSET_BYTES;
        constexpr int max_output_bytes = rows * kHeadDim * sizeof(ElementAccum);
        constexpr int lse_bytes = rows * sizeof(ElementAccum);
        static_assert(kHeadDim == 128 && Kernel_traits::kNThreads == 128);
        static_assert(output_offset >= 0 && output_offset % 16 == 0);
        static_assert(lse_offset >= 0 && lse_offset % 16 == 0);
        static_assert(output_offset + max_output_bytes <= arena_bytes);
        static_assert(lse_offset + lse_bytes <= arena_bytes);
        static_assert(output_offset + max_output_bytes <= lse_offset || lse_offset + lse_bytes <= output_offset);

        auto* shared_output = reinterpret_cast<ElementO*>(smem_ + output_offset);
        auto* shared_lse = reinterpret_cast<ElementAccum*>(smem_ + lse_offset);
        // Release terminal V before either independent scratch value overwrites it.
        __syncthreads();
        if (computes_q) {
            const Tensor coordinates = thr_mma.partition_C(
                make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{}));
            Tensor output = FLASH_NAMESPACE::convert_type<ElementO>(acc_o);
            // XOR only column bits above the 16-byte vector. This reduces
            // write bank conflicts and keeps the cooperative load vector aligned.
            #pragma unroll
            for (int i = 0; i < size(output); ++i) {
                const auto coordinate = coordinates(i);
                const int row = get<0>(coordinate);
                const int column = get<1>(coordinate);
                if (row < rows) shared_output[row * kHeadDim + (column ^ (row * 8))] = output(i);
            }
            const Tensor row_coordinates = logical_divide(coordinates, Shape<_2>{})(make_coord(0, _), _, 0);
            CUTE_STATIC_ASSERT_V(size(lse) == size(row_coordinates));
            if (get<1>(row_coordinates(0)) == 0) {
                #pragma unroll
                for (int i = 0; i < size(lse); ++i) {
                    const int row = get<0>(row_coordinates(i));
                    if (row < rows) shared_lse[row] = lse(i);
                }
            }
        }
        // O and LSE share this one publication barrier and remain live through export.
        __syncthreads();

        const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
            + m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
        const index_t row_offset_oaccum = (((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q
            + m_block * kBlockM) * params.d_rounded;
        const index_t row_offset_lseaccum = (Split || !params.unpadded_lse
            ? ((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q
            : bidh * params.total_q + binfo.q_offset(params.seqlen_q, 1, bidb)) + m_block * kBlockM;
        auto* output = reinterpret_cast<ElementO*>(Split ? params.oaccum_ptr : params.o_ptr)
            + (Split ? row_offset_oaccum : row_offset_o);
        auto* lse_output = reinterpret_cast<ElementAccum*>(Split ? params.softmax_lseaccum_ptr : params.softmax_lse_ptr)
            + row_offset_lseaccum;

        constexpr int vector_elements = sizeof(uint4) / sizeof(ElementO);
        constexpr int vectors_per_row = kHeadDim / vector_elements;
        if (tidx < rows * vectors_per_row) {
            const int row = tidx / vectors_per_row;
            const int column = (tidx % vectors_per_row) * vector_elements;
            const auto* source = shared_output + row * kHeadDim + (column ^ (row * 8));
            auto* destination = output + row * (Split ? kHeadDim : params.o_row_stride) + column;
            if (reinterpret_cast<std::uintptr_t>(destination) % alignof(uint4) == 0) {
                *reinterpret_cast<uint4*>(destination) = *reinterpret_cast<const uint4*>(source);
            } else {
                #pragma unroll
                for (int i = 0; i < vector_elements; ++i) destination[i] = source[i];
            }
        }
        if (tidx < rows) lse_output[tidx] = shared_lse[tidx];
    } else {
    Tensor sOaccum = make_tensor(make_smem_ptr(reinterpret_cast<ElementO *>(smem_)), typename Kernel_traits::SmemLayoutO{}); // (SMEM_M,SMEM_N)
    // Partition sO to match the accumulator partitioning
    using SmemTiledCopyO = std::conditional_t<
        !Split,
        typename Kernel_traits::SmemCopyAtomO,
        typename Kernel_traits::SmemCopyAtomOaccum
    >;
    auto smem_tiled_copy_Oaccum = make_tiled_copy_C(SmemTiledCopyO{}, tiled_mma);
    auto smem_thr_copy_Oaccum = smem_tiled_copy_Oaccum.get_thread_slice(tidx);
    Tensor rO = FLASH_NAMESPACE::convert_type<ElementO>(acc_o);
    Tensor taccOrOaccum = smem_thr_copy_Oaccum.retile_S(rO);        // ((Atom,AtomNum), MMA_M, MMA_N)
    Tensor taccOsOaccum = smem_thr_copy_Oaccum.partition_D(sOaccum);     // ((Atom,AtomNum),PIPE_M,PIPE_N)

    // sOaccum is larger than sQ, so we need to syncthreads here
    // TODO: allocate enough smem for sOaccum
    if constexpr (Split || Kernel_traits::Share_Q_K_smem) { __syncthreads(); }

    cute::copy(smem_tiled_copy_Oaccum, taccOrOaccum, taccOsOaccum);

    const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
        + m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
    const index_t row_offset_oaccum = (((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q
                                         + m_block * kBlockM) * params.d_rounded;
    const index_t row_offset_lseaccum = (Split || !params.unpadded_lse ?
            ((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q : bidh * params.total_q + binfo.q_offset(params.seqlen_q, 1, bidb)
        ) + m_block * kBlockM;

    Tensor gOaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO *>(Split ? params.oaccum_ptr : params.o_ptr) + (Split ? row_offset_oaccum : row_offset_o)),
                                 Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                 make_stride(Split ? kHeadDim : params.o_row_stride, _1{}));
    Tensor gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(Split ? params.softmax_lseaccum_ptr : params.softmax_lse_ptr) + row_offset_lseaccum),
                                   Shape<Int<kBlockM>>{}, Stride<_1>{});
    // if (tidx == 0) { printf("row_offset_o = %d, bidh = %d, gOaccum = %p\n", row_offset_o, bidh, gOaccum.data()); }

    GmemTiledCopyO gmem_tiled_copy_Oaccum;
    auto gmem_thr_copy_Oaccum = gmem_tiled_copy_Oaccum.get_thread_slice(tidx);
    Tensor tOsOaccum = gmem_thr_copy_Oaccum.partition_S(sOaccum);        // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tOgOaccum = gmem_thr_copy_Oaccum.partition_D(gOaccum);

    __syncthreads();

    Tensor tOrOaccum = make_tensor<ElementO>(shape(tOgOaccum));
    cute::copy(gmem_tiled_copy_Oaccum, tOsOaccum, tOrOaccum);

    Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor taccOcO = thr_mma.partition_C(caccO);                           // (MMA,MMA_M,MMA_K)
    static_assert(decltype(size<0>(taccOcO))::value == 4);
    // Convert to ((2, 2), MMA_M, MMA_K) then take only the row indices.
    Tensor taccOcO_row = logical_divide(taccOcO, Shape<_2>{})(make_coord(0, _), _, 0);
    CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));                     // MMA_M
    if (computes_q && get<1>(taccOcO_row(0)) == 0) {
        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = get<0>(taccOcO_row(mi));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM) { gLSEaccum(row) = lse(mi); }
        }
    }

    // Construct identity layout for sO
    Tensor cO = make_identity_tensor(make_shape(size<0>(sOaccum), size<1>(sOaccum)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    // Repeat the partitioning with identity layouts
    Tensor tOcO = gmem_thr_copy_Oaccum.partition_D(cO);                           // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgOaccum)));
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d; }
    }
    // Clear_OOB_K must be false since we don't want to write zeros to gmem
    FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_Oaccum, tOrOaccum, tOgOaccum, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
    );
    }
}

}  // namespace FLASH_NAMESPACE
