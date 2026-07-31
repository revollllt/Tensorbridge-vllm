#pragma once

#include <tensorbridge/utils/all.cuh>


#ifndef TENSORBRIDGE_S2R_DUMP_MAPPING
#define TENSORBRIDGE_S2R_DUMP_MAPPING 0
#endif

#ifndef TENSORBRIDGE_NVFP4_SWZ64_B_PRMT_INTERLEAVE
#define TENSORBRIDGE_NVFP4_SWZ64_B_PRMT_INTERLEAVE 1
#endif

#ifndef TENSORBRIDGE_NVFP4_SWZ64_B_DUAL_MMA_PREINT_LOAD
#define TENSORBRIDGE_NVFP4_SWZ64_B_DUAL_MMA_PREINT_LOAD 0
#endif

#if TENSORBRIDGE_S2R_DUMP_MAPPING
// One-shot probe: dumps per-thread (warp_id, lane_id, iter_id, smem_idx, val_lo, val_hi)
// for blockIdx.x==0, all iter_ids, all 256 math threads. Layout per slot (6 int32):
//   slot_offset = (iter_id * 256 + math_thread_id) * 6;  // skip 4 header words
//   dst = tensorbridge_s2r_dump_buf + 4 + slot_offset
// Buffer total: 4 + (4 iters * 256 threads * 6 int32) = 6148 int32 = 24KB.
__device__ int32_t *tensorbridge_s2r_dump_buf = nullptr;
#endif


template <class BlockShape, class WarpShape, class ElementA, class ElementB, class LayerConfig, class TuningConfig>
class S2RMemoryLoaderB {
private:
  static constexpr uint32_t kNumMathThreads = TuningConfig::kNumMathThreads;
  static constexpr uint32_t kPartMmaShapeK = 256 / ElementA::kBits;
  static constexpr uint32_t kWarpItersK = WarpShape::K / kPartMmaShapeK;

  static constexpr uint32_t M_WARPS = BlockShape::M / WarpShape::M;
  static constexpr uint32_t N_WARPS = BlockShape::N / WarpShape::N;
  static constexpr uint32_t K_WARPS = BlockShape::K / WarpShape::K;

  static constexpr bool kIsWarpHalfGroup = WarpShape::N == ElementA::kBits * 2;
  static constexpr bool kLoadHalfGroup = ElementB::kBits % 2 == 0 && kIsWarpHalfGroup;
  static constexpr bool kUseNvfp4RawS2RDeint = LayerConfig::kUseNvfp4RawS2RDeint;
  static constexpr bool kUseNvfp4TmaSwizzle64 = TuningConfig::kUseTmaBSwizzle64;
  static constexpr uint32_t TRUE_N_WARPS = kIsWarpHalfGroup ? N_WARPS / 2 : N_WARPS;
  static constexpr uint32_t kWarpWeightBlocks = MAX(WarpShape::N / (ElementA::kBits * 4), 1);
  static constexpr uint32_t kSmemStride = BlockShape::N * kPartMmaShapeK * ElementB::kBits / 32 / 4;
  static constexpr uint32_t kNumIntsPerThread = ElementB::kBits / (kLoadHalfGroup ? 2 : 1);
  using LoadType = typename LoadTypeChooser<kNumIntsPerThread * 4>::Type;
  static constexpr uint32_t kLoadIters = kNumIntsPerThread / (sizeof(LoadType) / 4);

public:
  CUDA_INLINE
  static uint32_t interleave_swizzle64_chunks(uint16_t lo_src, uint16_t hi_src) {
#if TENSORBRIDGE_NVFP4_SWZ64_B_PRMT_INTERLEAVE
    uint32_t lo_dup = __byte_perm(
        static_cast<uint32_t>(lo_src),
        static_cast<uint32_t>(lo_src),
        0x1100);
    uint32_t hi_dup = __byte_perm(
        static_cast<uint32_t>(hi_src),
        static_cast<uint32_t>(hi_src),
        0x1100);
    uint32_t lo_part = (lo_dup & 0x000F000Fu) | ((lo_dup >> 4) & 0x0F000F00u);
    uint32_t hi_part = ((hi_dup << 4) & 0x00F000F0u) | (hi_dup & 0xF000F000u);
    return lo_part | hi_part;
#else
    uint32_t out = 0;
    out |= ((uint32_t)(lo_src       & 0x0Fu));
    out |= ((uint32_t)((hi_src      & 0x0Fu))) << 4;
    out |= ((uint32_t)((lo_src >> 4) & 0x0Fu)) << 8;
    out |= ((uint32_t)((hi_src >> 4) & 0x0Fu)) << 12;
    out |= ((uint32_t)((lo_src >> 8) & 0x0Fu)) << 16;
    out |= ((uint32_t)((hi_src >> 8) & 0x0Fu)) << 20;
    out |= ((uint32_t)((lo_src >>12) & 0x0Fu)) << 24;
    out |= ((uint32_t)((hi_src >>12) & 0x0Fu)) << 28;
    return out;
#endif
  }

  CUDA_INLINE
  void load_pair(const int4 *smem_ptr, uint32_t *regs_ptr0, uint32_t *regs_ptr1, uint32_t iter_id) {
#if TENSORBRIDGE_NVFP4_SWZ64_B_DUAL_MMA_PREINT_LOAD
    if constexpr (kUseNvfp4TmaSwizzle64) {
      static_assert(ElementA::kBits == 8 && ElementB::kBits == 4,
                    "dual-MMA preinterleaved raw B S2R supports only NVFP4 W4A8");
      static_assert(BlockShape::N == 128 && WarpShape::N == 16,
                    "dual-MMA preinterleaved raw B S2R supports only BlockN=128/WarpN=16");
      static_assert(BlockShape::K == 128 && WarpShape::K == 128 && kWarpItersK == 4,
                    "dual-MMA preinterleaved raw B S2R supports only four K-iters");
      static_assert(kLoadIters == 1);
      static_assert(kWarpWeightBlocks == 1);

      uint32_t pair_id = iter_id / 2;
      const uint8_t *smem_byte = reinterpret_cast<const uint8_t *>(smem_ptr);
      uint4 packed = *reinterpret_cast<const uint4 *>(smem_byte + pair_id * 4096 + threadIdx.x * 16);
      regs_ptr0[0] = packed.x;
      regs_ptr0[1] = packed.y;
      regs_ptr1[0] = packed.z;
      regs_ptr1[1] = packed.w;
      return;
    }
#endif
    load(smem_ptr, regs_ptr0, iter_id);
    load(smem_ptr, regs_ptr1, iter_id + 1);
  }

  CUDA_INLINE
  void load(const int4 *smem_ptr, uint32_t *regs_ptr, uint32_t iter_id) {
    static_assert(
        !kUseNvfp4RawS2RDeint ||
            (ElementA::kBits == 8 && ElementB::kBits == 4 && kLoadHalfGroup),
        "nvfp4_raw_s2r_deint_v1 is only validated for NVFP4 W4A8 half-group raw-B loads");

    uint32_t warp_id = (threadIdx.x / 32);
    uint32_t n_warp_id = warp_id % N_WARPS;
    if (kIsWarpHalfGroup) n_warp_id = n_warp_id / 2;
    uint32_t lane_id = threadIdx.x % 32;
    uint32_t idx = kWarpWeightBlocks * 32 * n_warp_id + lane_id;
    uint32_t k_warp_base = 0;

    if constexpr (K_WARPS > 1) {
      uint32_t k_warp_id = (threadIdx.x / (kNumMathThreads / K_WARPS));
      k_warp_base = TRUE_N_WARPS * 32 * kWarpWeightBlocks * kWarpItersK * k_warp_id;
      idx = k_warp_base + idx;
    }

    uint32_t smem_start_idx = idx * kLoadIters;
    // For non-swizzle paths, advance smem_ptr per iter (existing contract).
    // For 2D+64B-swizzle TMA-B path, do NOT shift here — iter_id is folded
    // into the per-thread byte offset because the 2D layout interleaves
    // K-iters within each N-row rather than as contiguous chunks.
    const int4 *smem_ptr_base = smem_ptr;
    if constexpr (!kUseNvfp4TmaSwizzle64) {
      smem_ptr = smem_ptr + kSmemStride * iter_id;
    }
    const LoadType *smem_ptr_load = reinterpret_cast<const LoadType *>(smem_ptr);
    LoadType *reg_ptr_load = reinterpret_cast<LoadType *>(regs_ptr);

    PRAGMA_UNROLL
    for (uint32_t i = 0; i < kWarpWeightBlocks; i++) {
      PRAGMA_UNROLL
      for (uint32_t j = 0; j < kLoadIters; j++) {
        uint32_t smem_idx;
#if TENSORBRIDGE_NVFP4_SWZ64_B_DUAL_MMA_PREINT_LOAD
        if constexpr (kUseNvfp4TmaSwizzle64) {
          static_assert(ElementA::kBits == 8 && ElementB::kBits == 4,
                        "preinterleaved swizzle64 raw B S2R supports only NVFP4 W4A8");
          static_assert(BlockShape::N == 128 && WarpShape::N == 16,
                        "preinterleaved swizzle64 raw B S2R supports only BlockN=128/WarpN=16");
          static_assert(BlockShape::K == 128 && WarpShape::K == 128 && kWarpItersK == 4,
                        "preinterleaved swizzle64 raw B S2R supports only four K-iters");
          static_assert(kLoadIters == 1);
          static_assert(kWarpWeightBlocks == 1);
          uint32_t pair_id = iter_id / 2;
          uint32_t iter_in_pair = iter_id & 1u;
          const uint8_t *smem_byte = reinterpret_cast<const uint8_t *>(smem_ptr_base);
          uint32_t byte_offset = pair_id * 4096 + threadIdx.x * 16 + iter_in_pair * 8;
          uint2 packed = *reinterpret_cast<const uint2 *>(smem_byte + byte_offset);
          regs_ptr[0] = packed.x;
          regs_ptr[1] = packed.y;
          smem_idx = byte_offset;  // for dump (informational only)
        } else
#endif
        if constexpr (kUseNvfp4TmaSwizzle64) {
          static_assert(ElementA::kBits == 8 && ElementB::kBits == 4,
                        "swizzle64 raw B S2R supports only NVFP4 W4A8");
          static_assert(BlockShape::N == 128 && WarpShape::N == 16,
                        "phase-1 swizzle64 raw B S2R supports only BlockN=128/WarpN=16");
          static_assert(BlockShape::K == 128 && WarpShape::K == 128,
                        "phase-1 swizzle64 raw B S2R supports only BlockK=WarpK=128");
          static_assert(LayerConfig::kWeightScaleGroupSize == 16,
                        "phase-1 swizzle64 raw B S2R supports only g16 scales");
          // 2D + 64B-swizzle TMA-B path. Per-thread WGMMA m64n256k32_RS_TN
          // A-fragment coverage (TensorBridge uses operand-swap so WGMMA-M = weight-N):
          //   N_base = lane/4 + (warp%4)*16 + (warp/4)*64
          //   K_base = (lane%4)*4
          // Thread covers (N_base, N_base+8) × (K_base..K_base+3, K_base+16..K_base+19).
          //
          // To match the dequant kSplitKScale K-low/K-high nibble semantics
          // (low nibble of each byte = K-group-low, high nibble = K-group-high),
          // load 4 disjoint 2-byte chunks from raw (N, K_int4_packed) SMEM and
          // interleave nibbles into regs_qb[0..1]:
          //   regs_qb[0] (4 bytes for M=N_base):
          //     byte_b: low nib = weight[N_base, K_base+b]
          //             high nib = weight[N_base, K_base+b+16]
          //   regs_qb[1] (same for M=N_base+8).
          static_assert(kLoadIters == 1);
          static_assert(kWarpWeightBlocks == 1);
          constexpr uint32_t kBytesPerNRow = BlockShape::K / 2;
          uint32_t n_base = lane_id / 4 + (warp_id % 4) * 16 + (warp_id / 4) * 64;
          uint32_t k_byte_in_iter = (lane_id % 4) * 2;
          uint32_t k_byte_lo = k_byte_in_iter + iter_id * 16;
          uint32_t k_byte_hi = k_byte_in_iter + 8 + iter_id * 16;
          uint32_t byte_a = n_base * kBytesPerNRow + k_byte_lo;
          uint32_t byte_b = n_base * kBytesPerNRow + k_byte_hi;
          uint32_t byte_c = (n_base + 8) * kBytesPerNRow + k_byte_lo;
          uint32_t byte_d = (n_base + 8) * kBytesPerNRow + k_byte_hi;
          byte_a ^= ((byte_a >> 7) & 0x3u) << 4;
          byte_b ^= ((byte_b >> 7) & 0x3u) << 4;
          byte_c ^= ((byte_c >> 7) & 0x3u) << 4;
          byte_d ^= ((byte_d >> 7) & 0x3u) << 4;
          const uint8_t *smem_byte = reinterpret_cast<const uint8_t *>(smem_ptr_base);
          uint16_t chunk_a = *reinterpret_cast<const uint16_t *>(smem_byte + byte_a);
          uint16_t chunk_b = *reinterpret_cast<const uint16_t *>(smem_byte + byte_b);
          uint16_t chunk_c = *reinterpret_cast<const uint16_t *>(smem_byte + byte_c);
          uint16_t chunk_d = *reinterpret_cast<const uint16_t *>(smem_byte + byte_d);
          // chunk_x layout (16 bits = 4 nibbles): byte0_low=n0, byte0_high=n1,
          // byte1_low=n2, byte1_high=n3 where n_i is K_base+i (lo) or +16+i (hi).
          // Produce qb[0] = (lo_nibbles_a interleaved with lo_nibbles_b):
          //   bit  0..3  = a.n0  (K_base+0 of N_base)
          //   bit  4..7  = b.n0  (K_base+16 of N_base)
          //   bit  8..11 = a.n1  (K_base+1)
          //   bit 12..15 = b.n1  (K_base+17)
          //   etc.
          uint32_t qb0 = interleave_swizzle64_chunks(chunk_a, chunk_b);
          uint32_t qb1 = interleave_swizzle64_chunks(chunk_c, chunk_d);
          regs_ptr[0] = qb0;
          regs_ptr[1] = qb1;
          smem_idx = byte_a;  // for dump (informational only)
        } else if constexpr (kLoadHalfGroup) {
          if constexpr (kUseNvfp4RawS2RDeint) {
            static_assert(kLoadIters == 1);
            static_assert(kWarpWeightBlocks == 1);
            smem_idx = k_warp_base * 2 + 64 * n_warp_id + 32 * (warp_id % 2) + lane_id;
          } else {
            smem_idx = (smem_start_idx + 32 * kLoadIters * i) * 2 + warp_id % 2 * kLoadIters + j;
          }
          reg_ptr_load[i * kLoadIters + j] = smem_ptr_load[smem_idx];
        } else {
          smem_idx = smem_start_idx + 32 * kLoadIters * i + j;
          reg_ptr_load[i * kLoadIters + j] = smem_ptr_load[smem_idx];
        }

#if TENSORBRIDGE_S2R_DUMP_MAPPING
        // Dump only first slot (i=0, j=0), first CTA, all iters & all math threads.
        // Layout per math thread per iter: 6 int32 (warp_id, lane_id, iter_id,
        // smem_idx_unit, val_lo, val_hi). val is 8 bytes = full 16 int4s loaded.
        if (i == 0 && j == 0 && blockIdx.x == 0 && tensorbridge_s2r_dump_buf != nullptr) {
          uint32_t math_tid = threadIdx.x;  // [0, kNumMathThreads)
          if (math_tid < 256) {
            uint64_t val_u64 = *reinterpret_cast<uint64_t *>(&reg_ptr_load[0]);
            uint32_t slot = iter_id * 256 + math_tid;
            int32_t *dst = tensorbridge_s2r_dump_buf + 4 + slot * 6;
            dst[0] = (int32_t)warp_id;
            dst[1] = (int32_t)lane_id;
            dst[2] = (int32_t)iter_id;
            dst[3] = (int32_t)smem_idx;
            dst[4] = (int32_t)(val_u64 & 0xFFFFFFFFu);
            dst[5] = (int32_t)(val_u64 >> 32);
          }
        }
#endif
      }
    }
  };
};
