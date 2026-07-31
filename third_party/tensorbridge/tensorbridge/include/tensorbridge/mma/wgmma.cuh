#pragma once

#include <tensorbridge/utils/all.cuh>

#ifndef TENSORBRIDGE_NVFP4_SWZ64_BS_PREBCAST_LOAD
#define TENSORBRIDGE_NVFP4_SWZ64_BS_PREBCAST_LOAD 0
#endif

#ifndef TENSORBRIDGE_WGMMA_CONST_SCALE_D_FUSED
#define TENSORBRIDGE_WGMMA_CONST_SCALE_D_FUSED 0
#endif

#ifndef TENSORBRIDGE_WGMMA_INIT_ACCUM_WITH_SCALE_D
#define TENSORBRIDGE_WGMMA_INIT_ACCUM_WITH_SCALE_D 0
#endif

#ifndef TENSORBRIDGE_WGMMA_ZERO_FINAL_ACCUM_ONLY
#define TENSORBRIDGE_WGMMA_ZERO_FINAL_ACCUM_ONLY 0
#endif

template <uint32_t swizzle_bytes = 128>
CUDA_INLINE uint64_t make_wgmma_smem_desc(void *smem_ptr, uint32_t iter_id) {
  static_assert(swizzle_bytes == 128 || swizzle_bytes == 64);

  constexpr uint64_t swizzle_type = swizzle_bytes == 128 ? 1 : 2;
  constexpr uint64_t stride = (swizzle_bytes * 8) >> 4;
  constexpr uint64_t desc_base = (swizzle_type << 62) | (stride << 32);

  uint32_t addr = cast_smem_ptr_to_uint(smem_ptr);
  uint64_t desc = desc_base;

  reinterpret_cast<uint32_t *>(&desc)[0] = (addr >> 4);

  return desc;
};


template <
    class MmaOpClass_, class SharedStorage, class ArithClass,
    class BlockShape, class WarpShape,
    class ElementA, class ElementB,
    class LayerConfig>
struct WGMMA {
public:
  using MmaOpClass = MmaOpClass_;
  using Arithmetic = ArithClass;
  using MmaShape = typename MmaOpClass::MmaShape;

  static constexpr bool kHasZeroPoint = LayerConfig::kHasZeroPoint;
  static constexpr bool kIsFpZeroPoint = LayerConfig::kIsFpZeroPoint;
  static constexpr bool kUseFusedE8m0Scale = LayerConfig::kUseFusedE8m0Scale;
  static constexpr bool kUseFusedE4m3Scale = LayerConfig::kUseFusedE4m3Scale;
  static constexpr bool kUseAnyFusedScale = kUseFusedE8m0Scale || kUseFusedE4m3Scale;

  static constexpr uint32_t kPartMmaShapeK = 256 / ElementA::kBits;
  static constexpr bool kSplitKScale =
      kUseAnyFusedScale && LayerConfig::kWeightScaleGroupSize > 0
      && LayerConfig::kWeightScaleGroupSize < kPartMmaShapeK;
  static constexpr uint32_t M_WARPS = BlockShape::M / WarpShape::M;
  static constexpr uint32_t N_WARPS = BlockShape::N / WarpShape::N;
  static constexpr uint32_t K_WARPS = BlockShape::K / WarpShape::K;
  static constexpr uint32_t kWarpItersK = WarpShape::K / (256 / ElementA::kBits);
  static constexpr uint32_t kSwizzleBytes = ElementA::kBits * BlockShape::K >= 1024 ? 128 : 64;
  static constexpr uint32_t kNumWarpShapeNSplits = WarpShape::N == ElementA::kBits * 2 ? 2 : 1;
  // Optimal NVFP4 W4A8 path predicate, generalized from the single 256-token tile to a
  // family of token tiles that preserve the load-bearing invariants (K-TMPL-E005 R8):
  //   * fused-E4M3 fp8 activation, half-group WarpShape::N==16, MmaShape m64*k32
  //   * M_WARPS==1 (WarpShape::M==BlockShape::M) and MmaShape::N==WarpShape::M (the
  //     wgmma.cuh:146 static_assert), so the accumulator/register layout matches the WGMMA
  //   * warp_k_iters == WarpShape::K/kPartMmaShapeK == 128/32 == 4 (BlockShape::K==128),
  //     which is what makes kNumRegisterBuffers==4 / wait<3> (=bufs-1) valid.
  //   * BlockShape::N==128 channel tile family keeps 256 math threads; R8 also permits
  //     BlockShape::N==256 at token64/token128, which moves to 512 math threads/deint512.
  // At BlockShape::M==256 every term reduces to the original literals -> bitwise-identical.
  static constexpr bool kUseFourRegisterBuffers =
      kUseFusedE4m3Scale &&
      ElementA::kBits == 8 &&
      MmaShape::M == 64 &&
      MmaShape::N == WarpShape::M &&
      MmaShape::K == 32 &&
      WarpShape::M == BlockShape::M &&
      is_nvfp4_w4a8_register_buffer_tile<BlockShape>() &&
      BlockShape::K == 128 &&
      WarpShape::N == 16 &&
      WarpShape::K == 128;
  static constexpr uint32_t kNumRegisterBuffers = kUseFourRegisterBuffers ? 4 : 2;
  static constexpr uint32_t kRegsQbWords = ElementB::kBits * (16 / ElementA::kBits);

  SharedStorage &smem;
  ArithClass &arith;
  uint32_t regs_qb[kNumRegisterBuffers][kRegsQbWords];
  typename MmaOpClass::ARegisters regs_b[kNumRegisterBuffers][WarpShape::N * 4 / MmaShape::M][kPartMmaShapeK / MmaShape::K];
  typename MmaOpClass::CRegisters regs_c[2][WarpShape::N * 4 / MmaShape::M][WarpShape::M / MmaShape::N];
  uint32_t smem_offset = 0;

  CUDA_INLINE
  WGMMA(SharedStorage &smem, ArithClass &arith)
      : smem(smem), arith(arith) {
    uint32_t warp_id = threadIdx.x / 32;
    uint32_t m_warp_id = warp_id / N_WARPS % M_WARPS;
    uint32_t k_warp_id = warp_id / (N_WARPS * M_WARPS);

    constexpr uint32_t kSwizzleSizeK = kSwizzleBytes * 8 / ElementA::kBits;
    static_assert(kSwizzleSizeK >= WarpShape::K);

    const uint32_t row_offset = M_WARPS > 1 ? WarpShape::M * m_warp_id : 0;
    const uint32_t col_offset = K_WARPS > 1 ? WarpShape::K * k_warp_id : 0;

    smem_offset = row_offset * (kSwizzleBytes / 16);
    smem_offset += (col_offset % kSwizzleSizeK) * ElementA::kBits / 128;
    smem_offset += (col_offset / kSwizzleSizeK) * (BlockShape::M * kSwizzleBytes / 16);
  }

  CUDA_INLINE
  void zero_accum() {
    // Fused-scale paths write the only epilogue-visible accumulator buffer.
    // The generic double-buffered C fragment is needed by unfused scale paths
    // that apply scales on C, but zeroing the unused second buffer forces
    // extra CS2R zero moves and register liveness in NVFP4 fused-scale kernels.
    constexpr bool kZeroFinalOnly =
        TENSORBRIDGE_WGMMA_ZERO_FINAL_ACCUM_ONLY &&
        kUseAnyFusedScale;
    uint32_t *regs_c_ptr = regs_c_as_ptr();
    constexpr uint32_t kWords =
        kZeroFinalOnly ? sizeof(regs_c[0]) / 4 : sizeof(regs_c) / 4;
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < kWords; i++) {
      regs_c_ptr[i] = 0;
    };
  };

  CUDA_INLINE
  void transform_b(uint32_t buffer_id) {
    if constexpr (std::is_same<ElementA, ElementB>::value) return;

    if constexpr (kUseFusedE8m0Scale) {
      constexpr uint32_t kCount = WarpShape::N / 16;
      uint32_t *regs_b_ptr = reinterpret_cast<uint32_t *>(regs_b[buffer_id]);
      if constexpr (kSplitKScale) {
        auto *bs_bytes = reinterpret_cast<const uint8_t *>(arith.bs[buffer_id]);
        fused_dequant_for_mxfp4_split_k<ElementA, kCount>(
            regs_qb[buffer_id], regs_b_ptr, bs_bytes, bs_bytes + kCount * 2);
      } else {
        fused_dequant_for_mxfp4<ElementA, kCount, true>(
            regs_qb[buffer_id], regs_b_ptr, arith.bs[buffer_id]);
      }
    } else if constexpr (kUseFusedE4m3Scale) {
      static_assert(
          LayerConfig::kWeightScaleGroupSize * 2 == kPartMmaShapeK ||
          LayerConfig::kWeightScaleGroupSize == kPartMmaShapeK,
          "NVFP4 W4A8 fused-E4M3 supports group_size=16 or group_size=32");
      constexpr uint32_t kCount = WarpShape::N / 16;
      uint32_t *regs_b_ptr = reinterpret_cast<uint32_t *>(regs_b[buffer_id]);
      auto *bs_bytes = reinterpret_cast<const uint8_t *>(arith.bs[buffer_id]);
      if constexpr (kSplitKScale) {
#if TENSORBRIDGE_NVFP4_SWZ64_BS_PREBCAST_LOAD
        if constexpr (kUseFourRegisterBuffers && LayerConfig::kWeightScaleGroupSize == 16) {
          auto *bs_words = reinterpret_cast<const uint32_t *>(arith.bs[buffer_id]);
          fused_dequant_for_nvfp4_a8_split_k_prebcast<kCount>(
              regs_qb[buffer_id], regs_b_ptr, bs_words, bs_words + kCount * 2);
        } else
#endif
        {
          fused_dequant_for_nvfp4_a8_split_k<kCount>(
              regs_qb[buffer_id], regs_b_ptr, bs_bytes, bs_bytes + kCount * 2);
        }
      } else {
        fused_dequant_for_nvfp4_a8<kCount>(
            regs_qb[buffer_id], regs_b_ptr, bs_bytes);
      }
    } else {
      if constexpr (ElementB::kBits == 1 && kNumWarpShapeNSplits == 2) {
        regs_qb[buffer_id][0] = regs_qb[buffer_id][0] >> (threadIdx.x / 32 % 2 * 8);
      }

      PRAGMA_UNROLL
      for (uint32_t i = 0; i < WarpShape::N / (MmaShape::M / 4); i++) {
        uint32_t *regs_b_ptr = reinterpret_cast<uint32_t *>(regs_b[buffer_id][i * 64 / MmaShape::M]);
        uint4 zp_vals = arith.prepare_zp_for_dequant(buffer_id, i);
        uint32_t *zp_vals_ptr = reinterpret_cast<uint32_t *>(&zp_vals);
        dequant<ElementB, ElementA, kHasZeroPoint, kIsFpZeroPoint, kNumWarpShapeNSplits>(regs_qb[buffer_id], regs_b_ptr, i, zp_vals_ptr);
        arith.may_apply_bs_and_zp_on_b(regs_b_ptr, i, buffer_id);
      };
    }
  };

  CUDA_INLINE
  void issue(uint32_t stage_id, uint32_t iter_id, bool accum_scale_d = true) {
    static_assert(WarpShape::M == MmaShape::N);
    uint32_t buffer_id = iter_id % kNumRegisterBuffers;

    PRAGMA_UNROLL
    for (uint32_t k = 0; k < kPartMmaShapeK / MmaShape::K; k++) {
      int4 *smem_ptr = smem.a[stage_id] + smem_offset + iter_id * 2 + k;
      uint64_t desc = make_wgmma_smem_desc<kSwizzleBytes>(smem_ptr, iter_id);

      constexpr uint32_t kNumIters = WarpShape::N / (MmaShape::M / 4);

      bool scale_d = accum_scale_d;
      constexpr bool kApplyScaleOnC = ElementA::kBits != 16 && (LayerConfig::kInputScaleGroupSize > 0 || LayerConfig::kWeightScaleGroupSize > 0);
      if constexpr (!kUseAnyFusedScale && ElementA::kBits != 16 && LayerConfig::kInputScaleGroupSize > 0) {
        scale_d = (iter_id * kPartMmaShapeK) % LayerConfig::kInputScaleGroupSize > 0;
      }
      if constexpr (!kUseAnyFusedScale && ElementA::kBits != 16 && LayerConfig::kWeightScaleGroupSize > 0) {
        scale_d = scale_d && (iter_id * kPartMmaShapeK) % LayerConfig::kWeightScaleGroupSize > 0;
      }

      wgmma_fence();
      if constexpr (kUseAnyFusedScale) {
        // Fused-scale paths bake the scale into regs_b in transform_b, so
        // may_apply_as_and_bs_on_wgmma_c is a no-op and the per-fma drain
        // is unnecessary — issue all atoms as a single commit-group, then
        // let the caller run transform_b for the next buffer DURING the
        // WGMMA wait by deferring the drain to a separate call.
        PRAGMA_UNROLL
        for (uint32_t j = 0; j < kNumIters; j++) {
#if TENSORBRIDGE_WGMMA_CONST_SCALE_D_FUSED
          if constexpr (TENSORBRIDGE_WGMMA_INIT_ACCUM_WITH_SCALE_D) {
            if (scale_d) {
              MmaOpClass::fma_scale_one(regs_b[buffer_id][j][k], desc, regs_c[0][j][0]);
            } else {
              MmaOpClass::fma_scale_zero(regs_b[buffer_id][j][k], desc, regs_c[0][j][0]);
            }
          } else {
            MmaOpClass::fma_scale_one(regs_b[buffer_id][j][k], desc, regs_c[0][j][0]);
          }
#else
          MmaOpClass::fma(regs_b[buffer_id][j][k], desc, regs_c[0][j][0], scale_d);
#endif
        }
        wgmma_commit();
      } else {
        PRAGMA_UNROLL
        for (uint32_t j = 0; j < kNumIters; j++) {
          if constexpr (kApplyScaleOnC) fence_regs(regs_c[0][j][0]);
          MmaOpClass::fma(regs_b[buffer_id][j][k], desc, regs_c[0][j][0], scale_d);
          wgmma_commit();
          wgmma_wait<0>();
          if constexpr (kApplyScaleOnC) fence_regs(regs_c[0][j][0]);
          arith.may_apply_as_and_bs_on_wgmma_c(regs_c_as_ptr(), j, k, iter_id);
        }
      }
    }
  };

  CUDA_INLINE
  void drain() {
    if constexpr (kUseAnyFusedScale) {
      wgmma_wait<0>();
    }
  };

  template <uint32_t kGroups>
  CUDA_INLINE void wait() {
    if constexpr (kUseAnyFusedScale) {
      wgmma_wait<kGroups>();
    }
  };

  template <class T>
  CUDA_INLINE void fence_regs(T &regs) {
    PRAGMA_UNROLL
    for (uint32_t r = 0; r < sizeof(T) / 4; r++) {
      warpgroup_fence_operand(reinterpret_cast<uint32_t *>(regs)[r]);
    }
  };

  template <class T = uint32_t>
  CUDA_INLINE T *regs_qb_as_ptr(uint32_t buffer_id) {
    if constexpr (std::is_same<ElementA, ElementB>::value) {
      return reinterpret_cast<T *>(regs_b[buffer_id]);
    } else {
      return reinterpret_cast<T *>(regs_qb[buffer_id]);
    };
  };

  template <class T = uint32_t>
  CUDA_INLINE T *regs_c_as_ptr(uint32_t buffer_id = 0) {
    return reinterpret_cast<T *>(regs_c[buffer_id]);
  };

  template <class T = uint32_t>
  CUDA_INLINE T *final_regs_c_as_ptr() {
    uint32_t index = 0;
    constexpr bool kIsGroupInputScale = LayerConfig::kInputScaleGroupSize > 0;
    constexpr bool kIsGroupWeightScale = LayerConfig::kIsGroupWeightScale;
    constexpr bool kIsBlockWeightScale = LayerConfig::kIsBlockWeightScale;

    if constexpr (ElementA::kBits < 16 && kIsGroupInputScale) {
      index = 1;
    }

    if constexpr (ElementA::kBits < 16 && !kUseAnyFusedScale && (kIsGroupWeightScale || kIsBlockWeightScale)) {
      index = 1;
    }

    return regs_c_as_ptr<T>(index);
  };
};
