#pragma once

#include <tensorbridge/utils/all.cuh>


// CUTLASS-aligned stmatrix (STSM) epilogue for the NVFP4 W4A8 target config
// (docs/changes/26-05-29_epilogue_spill_optimization_plan.md). When set, the
// register->SMEM store uses stmatrix.x2.trans into a 128B-swizzled [token][channel]
// staging tile (matching make_tma_desc_c's Swizzle<3,4,3> TMA-C box) instead of the
// scalar per-element store. This eliminates the per-token-scale register spill of the
// scalar path (STL 26->3) and removes its hand-rolled swizzle-address arithmetic,
// recovering ~12us at the target shape (bit-exact). The path is GATED to exactly the
// verified NVFP4 tile family — every other config falls through to the scalar loop,
// since the WGMMA-fragment->GMEM map the addresses rely on is config-specific. In a
// StreamK-tail-enabled kernel, full-K DP tiles (slice_count==1) may still use STSM;
// split-K tail slices fall back to scalar SMEM writes before the ordered GMEM reduce.
// See tensorbridge/kernel/tensorbridge_ws.cuh.
#ifndef TENSORBRIDGE_NVFP4_PIPELINE_BASELINE
#define TENSORBRIDGE_NVFP4_PIPELINE_BASELINE 0
#endif
#ifndef TENSORBRIDGE_NVFP4_EPI_SUBTILE_STSM
#define TENSORBRIDGE_NVFP4_EPI_SUBTILE_STSM (TENSORBRIDGE_NVFP4_PIPELINE_BASELINE == 0)
#endif
#ifndef TENSORBRIDGE_NVFP4_EPI_STSM_X4
#define TENSORBRIDGE_NVFP4_EPI_STSM_X4 0
#endif


CUDA_INLINE void shlf_trans_mma_c_32b(void *vals_ptr) {
  uint32_t *vals_uint_ptr = reinterpret_cast<uint32_t *>(vals_ptr);

  uint32_t val;
  uint32_t idx = (threadIdx.x / 4) % 2;
  switch (idx) {
    case 0: {
      val = vals_uint_ptr[1];
      break;
    };
    case 1: {
      val = vals_uint_ptr[0];
      break;
    }
  }

  uint32_t swapped_val = __shfl_xor_sync(0xffffffff, val, 4);

  switch (idx) {
    case 0: {
      vals_uint_ptr[1] = swapped_val;
      break;
    };
    case 1: {
      vals_uint_ptr[0] = swapped_val;
      break;
    }
  }
}


CUDA_INLINE void shlf_trans_mma_c_16b(void *vals_ptr) {
  uint32_t *vals_uint_ptr = reinterpret_cast<uint32_t *>(vals_ptr);

  uint32_t &val = vals_uint_ptr[0];
  uint32_t swapped_val = __shfl_xor_sync(0xffffffff, val, 4);
  uint32_t idx = (threadIdx.x / 4) % 2;

  uint16_t *vals_ushort_ptr = reinterpret_cast<uint16_t *>(&val);
  uint16_t *swapped_vals_ushort_ptr = reinterpret_cast<uint16_t *>(&swapped_val);

  switch (idx) {
    case 0: {
      vals_ushort_ptr[1] = swapped_vals_ushort_ptr[0];
      break;
    };
    case 1: {
      vals_ushort_ptr[0] = swapped_vals_ushort_ptr[1];
      break;
    }
  }
}


template <typename T>
CUDA_INLINE void shlf_trans_mma_c(T &vals) {
  static_assert(sizeof(T) == 8 || sizeof(T) == 4);
  if constexpr (sizeof(T) == 8) {
    shlf_trans_mma_c_32b(&vals);
  } else {
    shlf_trans_mma_c_16b(&vals);
  }
}


template <
    class MmaOpClass, class ArithClass,
    class BlockShape, class WarpShape,
    class ElementA, class ElementC,
    class LayerConfig, class TuningConfig>
class EpilogueSmemWriter : F16Conversion<ElementC> {
private:
  static constexpr bool kUseWgmma = MmaOpClass::kMmaType == MmaType::WGMMA;

  using scalar_t = typename F16Conversion<ElementC>::scalar_t;
  using scalar_t2 = typename F16Conversion<ElementC>::scalar_t2;
  using MmaShape = typename MmaOpClass::MmaShape;
  using ValTypeC = typename MmaOpClass::ValTypeC;
  using CRegistersType = typename MmaOpClass::CRegisters;
  using MMA_CRegistersArrayType = CRegistersType[MAX(WarpShape::M / MmaShape::M, 1)][MAX(WarpShape::N / MmaShape::N, 1)];
  using WGMMA_CRegistersArrayType = CRegistersType[WarpShape::N * 4 / MmaShape::M][WarpShape::M / MmaShape::N];
  using CRegistersArrayType = std::conditional_t<kUseWgmma, WGMMA_CRegistersArrayType, MMA_CRegistersArrayType>;

  static constexpr uint32_t kNumWriteSplits = TuningConfig::kNumWriteSplits;
  static constexpr uint32_t kNumMathThreads = TuningConfig::kNumMathThreads;
  static constexpr bool kHasInputScale = ElementA::kBits != 16;
  static constexpr bool kIsGroupInputScale = kHasInputScale && LayerConfig::kInputScaleGroupSize > 0;
  static constexpr bool kIsGroupWeightScale = LayerConfig::kIsGroupWeightScale;
  static constexpr bool kIsBlockWeightScale = LayerConfig::kIsBlockWeightScale;
  static constexpr bool kUseIntWeightScale = LayerConfig::kUseIntWeightScale;
  static constexpr bool kUseFusedE8m0Scale = LayerConfig::kUseFusedE8m0Scale;
  static constexpr bool kHasGroupScale = kIsGroupInputScale || kIsGroupWeightScale || kIsBlockWeightScale;
  static constexpr bool kIsIntAccum = std::is_same<ValTypeC, int32_t>::value && (!kHasGroupScale || kUseIntWeightScale || kUseFusedE8m0Scale);

  static constexpr uint32_t M_WARPS = BlockShape::M / WarpShape::M;
  static constexpr uint32_t N_WARPS = BlockShape::N / WarpShape::N;
  static constexpr uint32_t K_WARPS = BlockShape::K / WarpShape::K;

public:
  int4 *smem_ptr;
  ArithClass &arith;

  CUDA_INLINE
  EpilogueSmemWriter(int4 *smem_ptr, ArithClass &arith)
      : smem_ptr(smem_ptr),
        arith(arith) {
  }

  CUDA_INLINE
  void write(uint32_t *regs_ptr, uint32_t slice_count, uint32_t split_idx) {
    if (threadIdx.x >= kNumMathThreads / K_WARPS) return;

    auto &regs = *reinterpret_cast<CRegistersArrayType *>(regs_ptr);

    scalar_t2 *smem_half2_ptr = reinterpret_cast<scalar_t2 *>(smem_ptr);
    uint32_t smem = cast_smem_ptr_to_uint(smem_ptr) / 128;
    using PackTypeC = std::conditional_t<
        sizeof(ValTypeC) == 2, scalar_t2,
        std::conditional_t<kIsIntAccum, int2, float2>>;

    uint32_t laneid = threadIdx.x % 32;
    uint32_t warpid = threadIdx.x / 32;
    uint32_t warp_delta_row = (warpid / N_WARPS % M_WARPS) * WarpShape::M;
    uint32_t n_warp_id = warpid % N_WARPS;
    uint32_t group_warp_id = warpid % 4;

#if TENSORBRIDGE_NVFP4_EPI_SUBTILE_STSM
    // Verified fragment->GMEM map for this config (token = 8*cb + 2*(lane%4) + half,
    // half = float2 .x/.y = 2 adjacent tokens; channel = box*64 + (2*wq+rb)*8 + lane//4,
    // box=n_warp_id/4, wq=n_warp_id%4). Each warp writes a 128B-swizzled [token][channel]
    // (channel-inner, 64 ch/box) tile via x2.trans with NO cross-lane shuffle; the
    // per-token input scale stays register-resident (no spill). make_tma_desc_c's 128B
    // Swizzle<3,4,3> box round-trips it to row-major GMEM[token][channel] bit-exact.
    // GATED to exactly the verified config; every other config falls through to scalar.
    // Generalized token-tile family (K-TMPL-E005 R8): the generic constexpr address map
    // above is valid for M_WARPS==1, half-group WarpShape::N==16, MmaShape m64, and a
    // channel tile of 128 with token tile 256, 128, 64, or 32, plus channel256-token64/128. token16 currently falls
    // through to scalar epilogue because the attempted STSM split-store map failed bitwise.
    // MmaShape::N==WarpShape::M (the
    // wgmma static_assert). Reduces to the original 256-only gate at BlockShape::M==256 ->
    // bitwise-identical; narrower token tiles now also take the STSM path instead of scalar.
    if constexpr (kUseWgmma && sizeof(ValTypeC) == 4 && !kIsIntAccum &&
                  TuningConfig::kUseTmaC &&
                  kHasInputScale && !kIsGroupInputScale &&
                  WarpShape::M == BlockShape::M &&
                  is_nvfp4_w4a8_stsm_tile<BlockShape>() &&
                  BlockShape::K == 128 &&
                  WarpShape::N == 16 &&
                  MmaShape::M == 64 && MmaShape::N == WarpShape::M) {
      if (!TuningConfig::kUseStreamK || slice_count == 1) {
      static_assert(kNumWriteSplits == 1, "STSM path assumes a single write split");
      static_assert(M_WARPS == 1, "STSM path assumes M_WARPS==1");
      auto part = reinterpret_cast<PackTypeC *>(&regs[0][0]);  // kFloat2PerThread float2 / thread
      scalar_t *smem_ct = reinterpret_cast<scalar_t *>(smem_ptr);  // [token][channel] b16
      // --- Generic (templatized) STSM fragment->GMEM map ---------------------------------
      // Parameterized over MmaShape::N (token-tile) and the TMA-C swizzle box, reducing to
      // the verified 256 constants (64/4/16/8/2/4/8) at this config. Two structural classes:
      //  * swizzle/box geometry: kChanPerBox = swizzle_bytes / sizeof(ElementC); the box
      //    inner dim MUST equal make_tma_desc_c's 128B Swizzle<3,4,3> (asserted below).
      //  * WGMMA m64nNk32 D-fragment loop counts. token16 has only one physical
      //    WGMMA call but still needs two channel-half store groups, so the store
      //    groups are intentionally decoupled from the physical call count.
      // The intra-warp lane map (%4, /4, shfl_xor 4, (lane%8)/4, lane%8, cb_p*8) is FIXED by
      // the PTX k32 D-fragment and stays literal; x2.trans (not x4) avoids a ptxas defect.
      constexpr uint32_t kCSwizzleBytes = 128;  // lockstep with make_tma_desc_c(..., 128)
      constexpr uint32_t kElemsPerSwzGroup = 8; // stmatrix m8n8 b16 inner row (channels/group)
      constexpr uint32_t kChanPerBox = kCSwizzleBytes / sizeof(scalar_t);  // TMA-C box inner dim
      constexpr uint32_t kSwzGroups = kChanPerBox / kElemsPerSwzGroup;     // 8 channel-groups/box
      constexpr uint32_t kFloat2PerThread = MmaShape::N / 4;  // PTX: N/2 f32 = N/4 float2
      constexpr uint32_t kPhysicalCalls = MmaShape::N / 16;   // physical WGMMA n16 calls
      constexpr uint32_t kTokBlocks = MAX(MmaShape::N / 32, 1); // token-8-block groups
      constexpr uint32_t kRbPerWarp = 2;                      // channel-pair halves (rb range)
      constexpr uint32_t kCbPerTokBlock = MmaShape::N / (8 * kTokBlocks); // float2 per token-block
      constexpr uint32_t kStoreGroups = kRbPerWarp * kTokBlocks;
      constexpr uint32_t kChanPerWarp = kRbPerWarp * kElemsPerSwzGroup; // channels owned per warp
      constexpr uint32_t kWarpsPerBox = kChanPerBox / kChanPerWarp;     // warps spanning one box
      static_assert(kChanPerBox * sizeof(scalar_t) == kCSwizzleBytes, "C-box must match TMA-C swizzle bytes");
      static_assert(kSwzGroups * kElemsPerSwzGroup == kChanPerBox, "invalid swizzle grouping");
      static_assert(kPhysicalCalls >= 1, "token tile must have at least one WGMMA call");
      static_assert(kStoreGroups * kCbPerTokBlock == kFloat2PerThread, "fragment packing mismatch");
      static_assert(kTokBlocks * kCbPerTokBlock * 8 == MmaShape::N, "token block decomposition mismatch");
      static_assert(kCbPerTokBlock % 2 == 0, "x2.trans requires pairs of token-8 blocks");
      static_assert(kWarpsPerBox * kChanPerWarp == kChanPerBox, "warp/box channel split mismatch");
      const uint32_t box = n_warp_id / kWarpsPerBox;  // warpgroup -> which channel box
      const uint32_t wq = n_warp_id % kWarpsPerBox;   // warp-in-box 0..kWarpsPerBox-1
      // Fold the global scale into as[] once, then read the per-token scale from registers.
      arith.may_process_f32_on_smem_write(0, 0);
      float *asf = arith.template regs_as_as_ptr<float>();
      PRAGMA_UNROLL
      for (uint32_t store_group = 0; store_group < kStoreGroups; store_group++) {
        uint32_t rb = store_group / kTokBlocks;
        uint32_t cbg = store_group % kTokBlocks;
        uint32_t g = kRbPerWarp * wq + rb;      // channel-8-group within the box (0..kSwzGroups-1)
        uint32_t src[kCbPerTokBlock];
        PRAGMA_UNROLL
        for (uint32_t k = 0; k < kCbPerTokBlock; k++) {
          uint32_t cb = cbg * kCbPerTokBlock + k; // col_8x8block (token-8-block)
          PackTypeC v = part[cb * kRbPerWarp + rb]; // verified part index = cb*kRbPerWarp + rb
          // .x/.y are token_even/token_odd; each gets its own per-token scale (the odd
          // token's scale lives in lane^4's as[cb], by the loader's sub_row parity).
          float as_self = asf[cb];
          float as_other = __shfl_xor_sync(0xffffffff, as_self, 4);
          uint32_t par = (laneid % 8) / 4;
          float *vf = reinterpret_cast<float *>(&v);
          vf[0] *= par ? as_other : as_self;
          vf[1] *= par ? as_self : as_other;
          scalar_t2 h = this->float22num2(v);
          src[k] = *reinterpret_cast<uint32_t *>(&h);
        }
        // x2.trans (2 matrices/call): token-col addresses come from lanes 0..15. The 128B
        // swizzle XORs the channel-8-group by the token row's low 3 bits (one box row =
        // kChanPerBox ch = swizzle_bytes). The optional x4 probe is restricted to the
        // original 256-token tile where each store group owns exactly four matrices.
#if TENSORBRIDGE_NVFP4_EPI_STSM_X4
        if constexpr (BlockShape::M == 256 && MmaShape::N == 256 && kCbPerTokBlock == 4) {
          uint32_t mm = laneid / 8;
          uint32_t cb_p = cbg * kCbPerTokBlock + mm;
          uint32_t token_x4 = cb_p * 8 + (laneid % 8);
          uint32_t chan_grp = g ^ (token_x4 % kSwzGroups);  // Swizzle<3,4,3> within the box
          uint32_t off = box * (BlockShape::M * kChanPerBox) + token_x4 * kChanPerBox + chan_grp * kElemsPerSwzGroup;
          uint32_t addr = cast_smem_ptr_to_uint(smem_ct + off);
          st_shared_trans<4>(addr, &src[0]);
        } else
#endif
        {
          PRAGMA_UNROLL
          for (uint32_t pair = 0; pair < 2; pair++) {
            uint32_t m0 = pair * 2;
            uint32_t mm = m0 + (laneid / 8) % 2;
            uint32_t cb_p = cbg * kCbPerTokBlock + mm;
            uint32_t token_x2 = cb_p * 8 + (laneid % 8);
            uint32_t chan_grp = g ^ (token_x2 % kSwzGroups);  // Swizzle<3,4,3> within the box
            uint32_t off = box * (BlockShape::M * kChanPerBox) + token_x2 * kChanPerBox + chan_grp * kElemsPerSwzGroup;
            uint32_t addr = cast_smem_ptr_to_uint(smem_ct + off);
            st_shared_trans<2>(addr, &src[m0]);
          }
        }
      }
      return;
      }
    }
#endif

    auto write_to_smem = [&](PackTypeC val, uint32_t row_8x8block, uint32_t col_8x8block) {
      scalar_t2 val_half2;

      static_assert(kNumWriteSplits == 1 || kNumWriteSplits == 2);
      if constexpr (kNumWriteSplits == 2) {
        static_assert(M_WARPS == 1);
        uint32_t m_8x8block = kUseWgmma ? col_8x8block : row_8x8block;
        if (split_idx == 0 && m_8x8block >= BlockShape::M / 8 / 2) return;
        if (split_idx == 1 && m_8x8block < BlockShape::M / 8 / 2) return;
      }

      if constexpr (kUseWgmma) shlf_trans_mma_c(val);
      if constexpr (sizeof(ValTypeC) != 4) {
        val_half2 = val;
      } else if constexpr (kIsIntAccum) {
        float2 val_float2 = {__int2float_rn(val.x), __int2float_rn(val.y)};
        if constexpr (kUseWgmma) {
          arith.may_apply_f32_on_smem_write(val_float2, col_8x8block, row_8x8block);
        } else {
          arith.may_apply_f32_on_smem_write(val_float2, row_8x8block, col_8x8block);
        }
        val_half2 = this->float22num2(val_float2);
      } else {
        if constexpr (kUseWgmma) {
          arith.may_apply_f32_on_smem_write(val, col_8x8block, row_8x8block);
        } else {
          arith.may_apply_f32_on_smem_write(val, row_8x8block, col_8x8block);
        }
        val_half2 = this->float22num2(val);
      };

      uint32_t &val_uint = *reinterpret_cast<uint32_t *>(&val_half2);
      if constexpr (kUseWgmma) {
        arith.may_apply_on_smem_write(val_uint, col_8x8block, row_8x8block);
        col_8x8block = col_8x8block - BlockShape::M / 8 / 2 * split_idx;
      } else {
        arith.may_apply_on_smem_write(val_uint, row_8x8block, col_8x8block);
        row_8x8block = row_8x8block - BlockShape::M / 8 / 2 * split_idx;
      }

      if constexpr (!kUseWgmma) {
        uint32_t sub_row = laneid / 4;
        uint32_t row = warp_delta_row + 8 * row_8x8block + sub_row;
        uint32_t col = col_8x8block * 4 + WarpShape::N / 2 * n_warp_id;

        row = row + (BlockShape::M / kNumWriteSplits) * (col / 32);
        col = ((col % 32 / 4) ^ ((sub_row + smem) % 8)) * 4 + laneid % 4;

        uint32_t idx = row * 32 + col;
        smem_half2_ptr[idx] = val_half2;
      } else {
        uint32_t sub_row = (laneid % 4) * 2 + (laneid % 8) / 4;
        uint32_t row = warp_delta_row + 8 * col_8x8block + sub_row;

        uint32_t count = (64 / WarpShape::N);
        uint32_t col1 = ((n_warp_id % count * (8 / count) + row_8x8block) ^ ((sub_row + smem) % 8)) * 4 + laneid / 8;
        uint32_t col2 = (n_warp_id / count) * (BlockShape::M / kNumWriteSplits * 64 / 2);
        uint32_t idx = row * 32 + col1 + col2;
        smem_half2_ptr[idx] = val_half2;
      }
    };

    PRAGMA_UNROLL
    for (uint32_t i = 0; i < sizeof(regs) / sizeof(regs[0]); i++) {
      PRAGMA_UNROLL
      for (uint32_t j = 0; j < sizeof(regs[0]) / sizeof(regs[0][0]); j++) {
        auto part_regs = reinterpret_cast<PackTypeC *>(&regs[i][j]);
        constexpr uint32_t inner_m = (kUseWgmma ? (MmaShape::M / 4) : MmaShape::M) / 8;
        constexpr uint32_t inner_n = sizeof(regs[0][0]) / sizeof(PackTypeC) / inner_m;

        PRAGMA_UNROLL
        for (uint32_t m = 0; m < inner_m; m++) {
          PRAGMA_UNROLL
          for (uint32_t n = 0; n < inner_n; n++) {
            uint32_t row_index = i * inner_m + m;
            uint32_t col_index = j * inner_n + n;
            write_to_smem(part_regs[n * inner_m + m], row_index, col_index);
          }
        }
      }
    }
  }
};
