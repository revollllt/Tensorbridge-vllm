#pragma once

#include <tensorbridge/utils/all.cuh>


#ifndef TENSORBRIDGE_NVFP4_SWZ64_BS_PAIR_DIRECT_LOAD
#define TENSORBRIDGE_NVFP4_SWZ64_BS_PAIR_DIRECT_LOAD 1
#endif

#ifndef TENSORBRIDGE_NVFP4_SWZ64_BS_PREBCAST_LOAD
#define TENSORBRIDGE_NVFP4_SWZ64_BS_PREBCAST_LOAD 0
#endif

template <
    class MmaOpClass,
    class BlockShape, class WarpShape,
    class ElementA, class ElementBS,
    class LayerConfig, class TuningConfig>
class S2RMemoryLoaderBS {
private:
  static constexpr uint32_t kNumThreads = TuningConfig::kNumThreads;
  static constexpr bool kUseWgmma = MmaOpClass::kMmaType == MmaType::WGMMA;

  static constexpr bool kIsChannel = LayerConfig::kIsChannelWeightScale;
  static constexpr bool kIsGroup = LayerConfig::kIsGroupWeightScale;
  static constexpr bool kIsBlock = LayerConfig::kIsBlockWeightScale;
  static constexpr bool kUseFusedE8m0Scale = LayerConfig::kUseFusedE8m0Scale;
  static constexpr bool kUseFusedE4m3Scale = LayerConfig::kUseFusedE4m3Scale;
  static constexpr uint32_t kGroupSize = kIsChannel ? BlockShape::K : LayerConfig::kWeightScaleGroupSize;
  static constexpr uint32_t kGroupSizeN = LayerConfig::kWeightScaleGroupSizeN;

  static constexpr uint32_t kPartMmaShapeK = 256 / ElementA::kBits;
  static constexpr uint32_t M_WARPS = BlockShape::M / WarpShape::M;
  static constexpr uint32_t N_WARPS = BlockShape::N / WarpShape::N;
  static constexpr uint32_t K_WARPS = BlockShape::K / WarpShape::K;

  // Fused-E8M0 with a scale group smaller than the K-tile: each K-tile spans
  // multiple scale groups, so loads must split into K-low / K-high halves.
  static constexpr bool kUseAnyFusedScale = kUseFusedE8m0Scale || kUseFusedE4m3Scale;
  static constexpr bool kSplitKScale =
      kUseAnyFusedScale && kGroupSize > 0 && kGroupSize < kPartMmaShapeK;
  static_assert(
      !kUseFusedE4m3Scale ||
      kGroupSize * 2 == kPartMmaShapeK ||
      kGroupSize == kPartMmaShapeK,
      "NVFP4 W4A8 fused-E4M3 supports group_size=16 or group_size=32");

  static constexpr uint32_t kNumSubBlocks = WarpShape::N / 16;
  static constexpr uint32_t kNumScalesPerSubBlock = !kUseFusedE8m0Scale && (kIsChannel || (ElementA::kBits != 16 && !kUseWgmma)) ? 4 : 2;
  static constexpr uint32_t kNumScales = kNumSubBlocks * kNumScalesPerSubBlock;
  static constexpr uint32_t kNumBytesPerThread = kNumScales * ElementBS::kBits / 8;

  static constexpr uint32_t kNumRowsPerMiniBlock = 128 / kNumScalesPerSubBlock;
  static constexpr uint32_t kNumWarpsPerMiniBlock = CEIL_DIV(kNumRowsPerMiniBlock, WarpShape::N);
  static constexpr uint32_t kMaxBytesPerLoad = ElementBS::kBits / kNumWarpsPerMiniBlock;
  static constexpr uint32_t kNumBytesPerLoad = MIN(kNumBytesPerThread, kMaxBytesPerLoad);
  using LoadType = typename LoadTypeChooser<kMaxBytesPerLoad>::Type;

  static constexpr uint32_t kLoadItersPerGroup = CEIL_DIV(kNumBytesPerThread, sizeof(LoadType));
  static constexpr uint32_t kSmemStride = BlockShape::N * ElementBS::kBits / 32 / 4;
  static constexpr uint32_t kSmemStrideLoadType = kSmemStride * 16 / sizeof(LoadType);

  static constexpr bool kUseNvfp4TmaSwizzle64 = TuningConfig::kUseTmaBSwizzle64;

  CUDA_INLINE
  static uint32_t pack_scale_bytes(uint8_t s0, uint8_t s1, uint8_t s2, uint8_t s3) {
    return static_cast<uint32_t>(s0) |
           (static_cast<uint32_t>(s1) << 8) |
           (static_cast<uint32_t>(s2) << 16) |
           (static_cast<uint32_t>(s3) << 24);
  }

  CUDA_INLINE
  void load_one_group(LoadType *dst, const LoadType *src, uint32_t base_offset) {
    constexpr uint32_t warp_load_delta = (16 / kNumScalesPerSubBlock);
    PRAGMA_UNROLL
    for (uint32_t j = 0; j < kLoadItersPerGroup; j++) {
      uint32_t smem_idx = warp_load_delta * j + base_offset;
      dst[j] = src[smem_idx];
    }
  }

public:

  CUDA_INLINE
  void load_pair(const int4 *smem_ptr, uint32_t *regs_ptr0, uint32_t *regs_ptr1, int32_t iter_id) {
#if TENSORBRIDGE_NVFP4_SWZ64_BS_PAIR_DIRECT_LOAD && !TENSORBRIDGE_NVFP4_SWZ64_BS_PREBCAST_LOAD
    if constexpr (kUseNvfp4TmaSwizzle64 &&
                  kIsGroup &&
                  BlockShape::M == 128) {
      static_assert(ElementA::kBits == 8 && ElementBS::kBits == 8,
                    "swizzle64 raw BS pair-direct S2R supports only FP8 activations/scales");
      static_assert(BlockShape::N == 128 && WarpShape::N == 16,
                    "swizzle64 raw BS pair-direct S2R supports only BlockN=128/WarpN=16");
      static_assert(BlockShape::K == 128 && WarpShape::K == 128,
                    "swizzle64 raw BS pair-direct S2R supports only BlockK=WarpK=128");
      static_assert(LayerConfig::kWeightScaleGroupSize == 16,
                    "swizzle64 raw BS pair-direct S2R supports only g16 scales");
      static_assert(kNumSubBlocks == 1 && kNumScalesPerSubBlock == 2,
                    "swizzle64 raw BS pair-direct S2R expects one WarpN16 sub-block");
      uint32_t warp_id = threadIdx.x / 32;
      uint32_t lane_id = threadIdx.x % 32;
      uint32_t n_base = lane_id / 4 + (warp_id % 4) * 16 + (warp_id / 4) * 64;
      uint32_t k_group0 = static_cast<uint32_t>(iter_id) * 2;
      const uint8_t *smem_byte = reinterpret_cast<const uint8_t *>(smem_ptr);
      const uint8_t *g0 = smem_byte + k_group0 * BlockShape::N;
      const uint8_t *g1 = g0 + BlockShape::N;
      const uint8_t *g2 = g1 + BlockShape::N;
      const uint8_t *g3 = g2 + BlockShape::N;
      regs_ptr0[0] = pack_scale_bytes(g0[n_base], g0[n_base + 8],
                                      g1[n_base], g1[n_base + 8]);
      regs_ptr1[0] = pack_scale_bytes(g2[n_base], g2[n_base + 8],
                                      g3[n_base], g3[n_base + 8]);
      return;
    }
#endif
    load(smem_ptr, regs_ptr0, iter_id);
    load(smem_ptr, regs_ptr1, iter_id + 1);
  }

  CUDA_INLINE
  void load(const int4 *smem_ptr, uint32_t *regs_ptr, int32_t iter_id) {
    if constexpr (kIsBlock) {
      load_block(smem_ptr, regs_ptr, iter_id);
    } else {
      load_group_or_channel(smem_ptr, regs_ptr, iter_id);
    }
  }

  CUDA_INLINE
  void load_block(const int4 *smem_ptr, uint32_t *regs_ptr, int32_t iter_id) {
    static_assert(ElementA::kBits != 16);
    static_assert(kGroupSizeN >= 64);

    uint32_t warp_id = threadIdx.x / 32;
    uint32_t n_warp_id = warp_id % N_WARPS;

    uint32_t index = (n_warp_id * WarpShape::N) / kGroupSizeN;
    if constexpr (BlockShape::K >= kGroupSize) {
      uint32_t k_index = (warp_id / (M_WARPS * N_WARPS)) * WarpShape::K + iter_id * kPartMmaShapeK;
      uint32_t group_index = k_index / kGroupSize;
      index += group_index * CEIL_DIV(BlockShape::N, kGroupSizeN);
    }
    regs_ptr[0] = reinterpret_cast<const uint32_t *>(smem_ptr)[index];
  };

  CUDA_INLINE
  void load_group_or_channel(const int4 *smem_ptr, uint32_t *regs_ptr, int32_t iter_id) {
    uint32_t warp_id = threadIdx.x / 32;

    if constexpr (kUseNvfp4TmaSwizzle64) {
      static_assert(ElementA::kBits == 8 && ElementBS::kBits == 8,
                    "swizzle64 raw BS S2R supports only FP8 activations/scales");
      static_assert(BlockShape::N == 128 && WarpShape::N == 16,
                    "phase-1 swizzle64 raw BS S2R supports only BlockN=128/WarpN=16");
      static_assert(BlockShape::K == 128 && WarpShape::K == 128,
                    "phase-1 swizzle64 raw BS S2R supports only BlockK=WarpK=128");
      static_assert(LayerConfig::kWeightScaleGroupSize == 16,
                    "phase-1 swizzle64 raw BS S2R supports only g16 scales");
      // SMEM layout for swizzle path: (K_groups, BlockShape::N) row-major FP8,
      // produced by cp.async from prepare_tensorbridge_weight_scale(swizzle64_raw)
      // which transposes (N, K_groups) -> (K_groups, N) without the 8-stride
      // permutation. Per-thread coverage matches the v2 weight S2R rewrite:
      //   N_base = lane/4 + (warp%4)*16 + (warp/4)*64
      // Thread holds 2 N rows (N_base, N_base+8) × 2 K-groups (iter*2, iter*2+1).
      // Dequant kSplitKScale reads 4 bytes from arith.bs[buffer_id]:
      //   scales_lo[0] = scale[N_base    ][K-group-low]    (applied to qb[0] low nibs)
      //   scales_lo[1] = scale[N_base + 8][K-group-low]    (applied to qb[1] low nibs)
      //   scales_hi[0] = scale[N_base    ][K-group-high]   (applied to qb[0] high nibs)
      //   scales_hi[1] = scale[N_base + 8][K-group-high]   (applied to qb[1] high nibs)
      static_assert(ElementBS::kBits == 8, "swizzle path expects FP8 scales");
      uint32_t lane_id = threadIdx.x % 32;
      uint32_t n_base = lane_id / 4 + (warp_id % 4) * 16 + (warp_id / 4) * 64;
      uint32_t k_group_lo = static_cast<uint32_t>(iter_id) * 2;
      uint32_t k_group_hi = k_group_lo + 1;
      const uint8_t *smem_byte = reinterpret_cast<const uint8_t *>(smem_ptr);
      uint8_t s_lo_a = smem_byte[k_group_lo * BlockShape::N + n_base];
      uint8_t s_lo_b = smem_byte[k_group_lo * BlockShape::N + n_base + 8];
      uint8_t s_hi_a = smem_byte[k_group_hi * BlockShape::N + n_base];
      uint8_t s_hi_b = smem_byte[k_group_hi * BlockShape::N + n_base + 8];
#if TENSORBRIDGE_NVFP4_SWZ64_BS_PREBCAST_LOAD
      regs_ptr[0] = uint32_t(s_lo_a) * 0x01010101u;
      regs_ptr[1] = uint32_t(s_lo_b) * 0x01010101u;
      regs_ptr[2] = uint32_t(s_hi_a) * 0x01010101u;
      regs_ptr[3] = uint32_t(s_hi_b) * 0x01010101u;
#else
      uint8_t *regs_byte = reinterpret_cast<uint8_t *>(regs_ptr);
      regs_byte[0] = s_lo_a;
      regs_byte[1] = s_lo_b;
      regs_byte[2] = s_hi_a;
      regs_byte[3] = s_hi_b;
#endif
      return;
    }

    uint32_t n_warp_id = warp_id % N_WARPS / kNumWarpsPerMiniBlock;
    constexpr uint32_t warp_load_delta = (16 / kNumScalesPerSubBlock);
    uint32_t s_sh_rd = (kLoadItersPerGroup * warp_load_delta * kNumWarpsPerMiniBlock) * n_warp_id;

    if constexpr (kUseFusedE8m0Scale) {
      s_sh_rd += (threadIdx.x % 32) / 4 * kNumWarpsPerMiniBlock + warp_id % kNumWarpsPerMiniBlock;
    } else if constexpr (kUseWgmma && kIsChannel) {
      s_sh_rd += (threadIdx.x % 32) / 8 * kNumWarpsPerMiniBlock + warp_id % kNumWarpsPerMiniBlock;
    } else if constexpr (kUseWgmma && ElementA::kBits != 16) {
      s_sh_rd += (threadIdx.x % 32) / 4 * kNumWarpsPerMiniBlock + warp_id % kNumWarpsPerMiniBlock;
    } else if constexpr (!kUseWgmma && (kIsChannel || ElementA::kBits != 16)) {
      s_sh_rd += threadIdx.x % 4 * kNumWarpsPerMiniBlock + warp_id % kNumWarpsPerMiniBlock;
    } else if constexpr (kIsGroup && ElementA::kBits == 16) {
      s_sh_rd += (threadIdx.x % 32) / 4 * kNumWarpsPerMiniBlock + warp_id % kNumWarpsPerMiniBlock;
    }

    if constexpr (kGroupSize < BlockShape::K) {
      uint32_t k_index = (warp_id / (M_WARPS * N_WARPS)) * WarpShape::K + iter_id * kPartMmaShapeK;
      uint32_t group_index = k_index / kGroupSize;
      s_sh_rd += group_index * kSmemStrideLoadType;
    };

    LoadType *reg_ptr_load = reinterpret_cast<LoadType *>(regs_ptr);
    const LoadType *smem_ptr_load = reinterpret_cast<const LoadType *>(smem_ptr);
    load_one_group(reg_ptr_load, smem_ptr_load, s_sh_rd);

    if constexpr (kSplitKScale) {
      // K-high half lives in the next scale group along K.
      load_one_group(reg_ptr_load + kLoadItersPerGroup,
                     smem_ptr_load,
                     s_sh_rd + kSmemStrideLoadType);
    }
  };
};
