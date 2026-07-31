#pragma once

#include <tensorbridge/datatype/base_conversion.cuh>
#include <tensorbridge/datatype/dtypes.cuh>
#include <tensorbridge/utils/all.cuh>


template <class TargetType>
CUDA_INLINE uint2 fused_dequant_single_for_mxfp4(const uint32_t qb, const uint32_t exp_offset) {
  static_assert(std::is_same<TargetType, Float8E4M3>::value || std::is_same<TargetType, Int8>::value);
  return {0, 0};
}


template <>
CUDA_INLINE uint2 fused_dequant_single_for_mxfp4<Float8E4M3>(const uint32_t qb, const uint32_t exp_offset) {
  uint32_t qb_ls4 = qb << 4;
  uint32_t qb_rs4 = qb >> 4;

  uint32_t res[2];
  uint32_t signs[2] = {qb_ls4, qb};
  uint32_t others[2] = {qb & 0x07070707, qb_rs4 & 0x07070707};

  uint32_t exp_offset_buffer1 = (exp_offset * 0x08080800) + (exp_offset ? -0x00000400 : 0);
  uint32_t exp_offset_buffer2 = exp_offset * 0x08080808;

  uint32_t exp_offsets[2] = {
      __byte_perm(exp_offset_buffer1, exp_offset_buffer2, qb),
      __byte_perm(exp_offset_buffer1, exp_offset_buffer2, qb >> 16)};

  PRAGMA_UNROLL
  for (uint32_t i = 0; i < 2; i++) {
    uint32_t val = lop3_and_or(signs[i], 0x80808080, others[i] << 2);
    val = val + __byte_perm(exp_offsets[0], exp_offsets[1], 0x6420 + 0x1111 * i);
    res[i] = val;
  }

  return *reinterpret_cast<uint2 *>(res);
}


template <>
CUDA_INLINE uint2 fused_dequant_single_for_mxfp4<Int8>(const uint32_t qb, const uint32_t exp_offset) {
  uint32_t buffer1 = 0x03020100 << exp_offset;
  uint32_t buffer2 = 0x0C080604 << exp_offset;

  uint32_t res[2];
  uint32_t int8s[2] = {
      __byte_perm(buffer1, buffer2, qb),
      __byte_perm(buffer1, buffer2, qb >> 16)};

  PRAGMA_UNROLL
  for (uint32_t i = 0; i < 2; i++) {
    uint32_t val = __byte_perm(int8s[0], int8s[1], 0x6420 + 0x1111 * i);
    uint32_t flag = i == 0 ? (qb & 0x08080808) >> 3 : (qb & 0x80808080) >> 7;
    uint32_t mask = flag * 0xFF;
    val = (val ^ mask) + flag;
    res[i] = val;
  }

  return *reinterpret_cast<uint2 *>(res);
}


template <class TargetType>
CUDA_INLINE void dequant_one_qb(uint32_t qb, uint32_t exp_offset,
                                uint32_t *res_lo, uint32_t *res_hi) {
  uint2 res = fused_dequant_single_for_mxfp4<TargetType>(qb, exp_offset);
  *res_lo = res.x;
  *res_hi = res.y;
}


template <class TargetType, uint32_t kCount, bool kUseWgmma>
CUDA_INLINE void fused_dequant_for_mxfp4(const uint32_t *qb_ptrs, uint32_t *res_ptrs, uint32_t *scales_ptr) {
  PRAGMA_UNROLL
  for (uint32_t i = 0; i < kCount * 2; i++) {
    uint32_t exp_offset = reinterpret_cast<uint8_t *>(scales_ptr)[i];
    dequant_one_qb<TargetType>(qb_ptrs[i], exp_offset, &res_ptrs[i * 2], &res_ptrs[i * 2 + 1]);
  }

  if constexpr (kUseWgmma) {
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < kCount; i++) {
      uint32_t tmp = res_ptrs[i * 4 + 1];
      res_ptrs[i * 4 + 1] = res_ptrs[i * 4 + 2];
      res_ptrs[i * 4 + 2] = tmp;
    }
  }
}


template <class TargetType, uint32_t kCount>
CUDA_INLINE void fused_dequant_for_mxfp4_split_k(
    const uint32_t *qb_ptrs, uint32_t *res_ptrs,
    const uint8_t *scales_lo, const uint8_t *scales_hi) {
  PRAGMA_UNROLL
  for (uint32_t i = 0; i < kCount * 2; i++) {
    uint2 lo = fused_dequant_single_for_mxfp4<TargetType>(qb_ptrs[i], scales_lo[i]);
    uint2 hi = fused_dequant_single_for_mxfp4<TargetType>(qb_ptrs[i], scales_hi[i]);
    res_ptrs[i * 2]     = lo.x;
    res_ptrs[i * 2 + 1] = hi.y;
  }

  PRAGMA_UNROLL
  for (uint32_t i = 0; i < kCount; i++) {
    uint32_t tmp = res_ptrs[i * 4 + 1];
    res_ptrs[i * 4 + 1] = res_ptrs[i * 4 + 2];
    res_ptrs[i * 4 + 2] = tmp;
  }
}


#ifndef TENSORBRIDGE_NVFP4_SNC_MASK_PRMT_LUT_PAIR
#define TENSORBRIDGE_NVFP4_SNC_MASK_PRMT_LUT_PAIR 0
#endif

#ifndef TENSORBRIDGE_USE_NVFP4_SNC
#define TENSORBRIDGE_USE_NVFP4_SNC 0
#endif

#ifndef TENSORBRIDGE_NVFP4_FPMA_ULP_CORRECTION
#define TENSORBRIDGE_NVFP4_FPMA_ULP_CORRECTION 0
#endif

#ifndef TENSORBRIDGE_NVFP4_FPMA_ULP_SCALE_MSB_FLAG_V1
#define TENSORBRIDGE_NVFP4_FPMA_ULP_SCALE_MSB_FLAG_V1 0
#endif

#if TENSORBRIDGE_NVFP4_FPMA_ULP_CORRECTION != \
    TENSORBRIDGE_NVFP4_FPMA_ULP_SCALE_MSB_FLAG_V1
#error "NVFP4 FPMA ULP correction requires scale-MSB flag ABI V1"
#endif

#if TENSORBRIDGE_NVFP4_FPMA_ULP_CORRECTION && TENSORBRIDGE_USE_NVFP4_SNC
// The NVRTC flag is process-wide, while accuracy harnesses also instantiate
// ordinary FP8 reference kernels. Host validation restricts correction use to
// SNC g16 NVFP4 layers; non-SNC translation units must still compile normally.
CUDA_INLINE uint32_t nvfp4_fpma_ulp_scale_flag4(uint32_t scale4) {
  return scale4 >> 7;
}
#endif

CUDA_INLINE uint32_t nvfp4_nonzero_mask_from_shifted_mag(uint32_t mag_shifted) {
  uint32_t mask;
  asm volatile("prmt.b32 %0, %1, 0, 0xBA98;"
               : "=r"(mask)
               : "r"(mag_shifted + 0x7F7F7F7Fu));
  return mask;
}

CUDA_INLINE uint32_t nvfp4_snc_nonzero_mask_from_shifted_mag(uint32_t mag_shifted) {
  uint32_t mask;
  asm volatile(
      "{ .reg .u32 t; "
      "xor.b32 t, %1, 0x04040404; "
      "add.u32 t, t, 0x7f7f7f7f; "
      "prmt.b32 %0, t, 0, 0xBA98; }"
      : "=r"(mask)
      : "r"(mag_shifted));
  return mask;
}

CUDA_INLINE uint2 nvfp4_snc_nonzero_masks_from_packed_prmt_lut(uint32_t packed) {
  uint32_t masks_01;
  uint32_t masks_23;
  uint32_t masks_lo;
  uint32_t masks_hi;
  asm volatile("prmt.b32 %0, 0xffff00ff, 0xffffffff, %1;"
               : "=r"(masks_01)
               : "r"(packed));
  asm volatile("prmt.b32 %0, 0xffff00ff, 0xffffffff, %1;"
               : "=r"(masks_23)
               : "r"(packed >> 16));
  asm volatile("prmt.b32 %0, %1, %2, 0x6420;"
               : "=r"(masks_lo)
               : "r"(masks_01), "r"(masks_23));
  asm volatile("prmt.b32 %0, %1, %2, 0x7531;"
               : "=r"(masks_hi)
               : "r"(masks_01), "r"(masks_23));
  return {masks_lo, masks_hi};
}

CUDA_INLINE void dequant_nvfp4_a8_one_split_k_prebcast(
    uint32_t packed, uint32_t scale_lo4, uint32_t scale_hi4,
    uint32_t &res_lo, uint32_t &res_hi) {
  uint32_t lo_mag_shifted = (packed & 0x07070707u) << 2;
  uint32_t hi_mag_shifted = (packed & 0x70707070u) >> 2;
  uint32_t lo_sign_shifted = packed << 4;

  uint32_t lo_addend = lop3_and_or(lo_sign_shifted, 0x80808080u, lo_mag_shifted);
  uint32_t hi_addend = lop3_and_or(packed, 0x80808080u, hi_mag_shifted);
#if TENSORBRIDGE_NVFP4_FPMA_ULP_CORRECTION && TENSORBRIDGE_USE_NVFP4_SNC
  uint32_t ulp_flag_lo = nvfp4_fpma_ulp_scale_flag4(scale_lo4);
  uint32_t ulp_flag_hi = nvfp4_fpma_ulp_scale_flag4(scale_hi4);
  scale_lo4 &= 0x7F7F7F7Fu;
  scale_hi4 &= 0x7F7F7F7Fu;
#endif
  uint32_t sum_lo = scale_lo4 + lo_addend;
  uint32_t sum_hi = scale_hi4 + hi_addend;

#if TENSORBRIDGE_NVFP4_FPMA_ULP_CORRECTION && TENSORBRIDGE_USE_NVFP4_SNC
  // Bit 7 carries the load-time scale-residue decision. SNC stores zero as
  // odd magnitude 1, so only real even-magnitude lanes receive correction.
  sum_lo -= lop3<0x20>(ulp_flag_lo, packed, 0x01010101u);
  sum_hi -= lop3<0x20>(ulp_flag_hi, packed >> 4, 0x01010101u);
#endif

#if TENSORBRIDGE_USE_NVFP4_SNC
#if TENSORBRIDGE_NVFP4_SNC_MASK_PRMT_LUT_PAIR
  uint2 nonzero_masks = nvfp4_snc_nonzero_masks_from_packed_prmt_lut(packed);
  uint32_t lo_nonzero = nonzero_masks.x;
  uint32_t hi_nonzero = nonzero_masks.y;
#else
  uint32_t lo_nonzero = nvfp4_snc_nonzero_mask_from_shifted_mag(lo_mag_shifted);
  uint32_t hi_nonzero = nvfp4_snc_nonzero_mask_from_shifted_mag(hi_mag_shifted);
#endif
#else
  uint32_t lo_nonzero = nvfp4_nonzero_mask_from_shifted_mag(lo_mag_shifted);
  uint32_t hi_nonzero = nvfp4_nonzero_mask_from_shifted_mag(hi_mag_shifted);
#endif

  res_lo = sum_lo & lo_nonzero;
  res_hi = sum_hi & hi_nonzero;
}

// NVFP4 W4A8 fused-scale bridge (split-K variant). Produces fp8e4m3 bytes
// b such that fp8(b) ~= s * fp4 / 6, where s is `prefolded_scale =
// raw_e4m3_byte - 0x1C` and fp4 is the low 4 bits of each byte in `qb`.
// The /6 factor is compensated by multiplying global_scale by 6 in
// `may_process_fused_e4m3_scale`.
template <uint32_t kCount>
CUDA_INLINE void fused_dequant_for_nvfp4_a8_split_k(
    const uint32_t *qb_ptrs, uint32_t *res_ptrs,
    const uint8_t *scales_lo, const uint8_t *scales_hi) {
  PRAGMA_UNROLL
  for (uint32_t i = 0; i < kCount * 2; i++) {
    uint32_t res_lo;
    uint32_t res_hi;
    dequant_nvfp4_a8_one_split_k_prebcast(
        qb_ptrs[i],
        uint32_t(scales_lo[i]) * 0x01010101u,
        uint32_t(scales_hi[i]) * 0x01010101u,
        res_lo,
        res_hi);

    uint32_t k_idx = i >> 1;
    uint32_t pair_idx = i & 1;
    res_ptrs[k_idx * 4 + pair_idx]     = res_lo;
    res_ptrs[k_idx * 4 + 2 + pair_idx] = res_hi;
  }
}

// Accepted aggregate path: the S2R scale loader has already broadcast each
// scale byte into four byte lanes. This avoids per-operand scale broadcast in
// the dequant bridge while preserving compact global storage.
template <uint32_t kCount>
CUDA_INLINE void fused_dequant_for_nvfp4_a8_split_k_prebcast(
    const uint32_t *qb_ptrs, uint32_t *res_ptrs,
    const uint32_t *scales_lo4, const uint32_t *scales_hi4) {
  PRAGMA_UNROLL
  for (uint32_t i = 0; i < kCount * 2; i++) {
    uint32_t res_lo;
    uint32_t res_hi;
    dequant_nvfp4_a8_one_split_k_prebcast(qb_ptrs[i], scales_lo4[i], scales_hi4[i], res_lo, res_hi);

    uint32_t k_idx = i >> 1;
    uint32_t pair_idx = i & 1;
    res_ptrs[k_idx * 4 + pair_idx]     = res_lo;
    res_ptrs[k_idx * 4 + 2 + pair_idx] = res_hi;
  }
}

// NVFP4 W4A8 fused-scale bridge (non-split K variant). This is for the
// group_size == kPartMmaShapeK path: one E4M3 scale byte applies to both
// K-low and K-high halves of each packed qb word.
template <uint32_t kCount>
CUDA_INLINE void fused_dequant_for_nvfp4_a8(
    const uint32_t *qb_ptrs, uint32_t *res_ptrs,
    const uint8_t *scales) {
  PRAGMA_UNROLL
  for (uint32_t i = 0; i < kCount * 2; i++) {
    uint32_t scale4 = uint32_t(scales[i]) * 0x01010101u;
    uint32_t res_lo;
    uint32_t res_hi;
    dequant_nvfp4_a8_one_split_k_prebcast(qb_ptrs[i], scale4, scale4, res_lo, res_hi);

    uint32_t k_idx = i >> 1;
    uint32_t pair_idx = i & 1;
    res_ptrs[k_idx * 4 + pair_idx]     = res_lo;
    res_ptrs[k_idx * 4 + 2 + pair_idx] = res_hi;
  }
}
