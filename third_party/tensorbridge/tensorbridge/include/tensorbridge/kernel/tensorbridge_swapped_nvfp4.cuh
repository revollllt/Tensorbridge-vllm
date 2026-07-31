#pragma once

#include <tensorbridge/datatype/base_conversion.cuh>
#include <tensorbridge/datatype/dequant_fused.cuh>
#include <tensorbridge/utils/all.cuh>


template <bool kUseTma>
class KernelTensorParamType {
public:
  using Type = std::conditional_t<kUseTma, CUtensorMap const, void *const>;
};


template <uint32_t swizzle_bytes = 128>
CUDA_INLINE uint64_t make_swapped_probe_wgmma_smem_desc(void *smem_ptr) {
  static_assert(swizzle_bytes == 128 || swizzle_bytes == 64);
  constexpr uint64_t swizzle_type = swizzle_bytes == 128 ? 1 : 2;
  constexpr uint64_t stride = (swizzle_bytes * 8) >> 4;
  constexpr uint64_t desc_base = (swizzle_type << 62) | (stride << 32);

  uint32_t addr = cast_smem_ptr_to_uint(smem_ptr);
  uint64_t desc = desc_base;
  reinterpret_cast<uint32_t *>(&desc)[0] = addr >> 4;
  return desc;
}


template <
    class MmaOpClass,
    class ProblemShape, class BlockShape, class WarpShape, class PadShape,
    class ElementA, class ElementB, class ElementC, class ElementBS,
    class LayerConfig, class ComputeConfig, class TuningConfig>
__global__ __launch_bounds__(TuningConfig::kNumThreads, TuningConfig::kNumCtasPerSm) void tensorbridge(
    const __grid_constant__ typename KernelTensorParamType<TuningConfig::kUseTmaA>::Type A,
    const __grid_constant__ typename KernelTensorParamType<TuningConfig::kUseTmaB>::Type B,
    const __grid_constant__ typename KernelTensorParamType<TuningConfig::kUseTmaC>::Type C,
    const uint32_t *AS,
    const __grid_constant__ typename KernelTensorParamType<TuningConfig::kUseTmaBS>::Type BS,
    const __grid_constant__ typename KernelTensorParamType<TuningConfig::kUseTmaBZP>::Type BZP,
    const __grid_constant__ typename KernelTensorParamType<TuningConfig::kUseTmaBias>::Type Bias,
    const uint32_t *GS,
    const uint32_t *sorted_ids_ptr,
    const uint32_t *expert_ids_ptr,
    const uint32_t *num_tokens_padded_ptr,
    const uint32_t *expert_layout_ptr,
    CUtensorMap *tensor_map_buffer,
    int32_t *locks,
    uint32_t shape_m,
    uint32_t top_k,
    bool use_int64_expert_layout) {
  static_assert(TuningConfig::kUseSwappedLargeNvfp4);
  static_assert(MmaOpClass::kMmaType == MmaType::WGMMA);
  static_assert(MmaOpClass::MmaShape::M == 64);
  static_assert(MmaOpClass::MmaShape::N == 256);
  static_assert(MmaOpClass::MmaShape::K == 32);
  static_assert(BlockShape::M == 128 && BlockShape::N == 256 && BlockShape::K == 128);
  static_assert(WarpShape::M == 128 && WarpShape::N == 32 && WarpShape::K == 128);
  static_assert(TuningConfig::kNumMathThreads == 256);
  static_assert(!TuningConfig::kUseTmaA && !TuningConfig::kUseTmaB && !TuningConfig::kUseTmaC);
  static_assert(!TuningConfig::kUseTmaBS && !TuningConfig::kUseTmaBZP && !TuningConfig::kUseTmaBias);
  static_assert(LayerConfig::kUseFusedE4m3Scale);
  static_assert(LayerConfig::kWeightScaleGroupSize == 16);
  static_assert(std::is_same<ElementC, BFloat16>::value || std::is_same<ElementC, Float16>::value);

  (void)AS;
  (void)BZP;
  (void)Bias;
  (void)sorted_ids_ptr;
  (void)expert_ids_ptr;
  (void)num_tokens_padded_ptr;
  (void)expert_layout_ptr;
  (void)tensor_map_buffer;
  (void)locks;
  (void)shape_m;
  (void)top_k;
  (void)use_int64_expert_layout;

  constexpr uint32_t kOrigN = ProblemShape::N - PadShape::N;
  constexpr uint32_t kOrigK = ProblemShape::K - PadShape::K;
  constexpr uint32_t kLogicalM = BlockShape::M;
  constexpr uint32_t kLogicalN = BlockShape::N;
  constexpr uint32_t kTileK = BlockShape::K;
  constexpr uint32_t kGroupSize = LayerConfig::kWeightScaleGroupSize;
  constexpr uint32_t kKSubblocks = kTileK / MmaOpClass::MmaShape::K;
  constexpr uint32_t kOrigGroups = CEIL_DIV(kOrigK, kGroupSize);
  constexpr uint32_t kRawWordsPerKSub = 2u * 128u * 2u;
  constexpr uint32_t kSmemStrideInt4 = 8u;
  constexpr uint32_t kSmemInt4PerKSub = kLogicalN * kSmemStrideInt4;
  constexpr uint32_t kSmemInt4Total = kKSubblocks * kSmemInt4PerKSub;
  static_assert(kOrigK <= kTileK);
  static_assert(kOrigN <= kLogicalM);
  static_assert(kKSubblocks == 4);

  if (blockIdx.x != 0) return;
  if (shape_m > kLogicalN) return;

  extern __shared__ __align__(128) uint4 smem[];
  uint32_t tid = threadIdx.x;
  uint32_t wg = tid / 128u;
  uint32_t local = tid & 127u;
  const uint8_t *activation = reinterpret_cast<const uint8_t *>(A);
  const uint32_t *raw_frag = reinterpret_cast<const uint32_t *>(B);
  const uint8_t *scales = reinterpret_cast<const uint8_t *>(BS);

  uint32_t *smem_words = reinterpret_cast<uint32_t *>(smem);
  for (uint32_t i = tid; i < kSmemInt4Total * 4u; i += blockDim.x) {
    smem_words[i] = 0u;
  }
  __syncthreads();

  for (uint32_t idx = tid; idx < kKSubblocks * kLogicalN * 2u; idx += blockDim.x) {
    uint32_t k_sub = idx / (kLogicalN * 2u);
    uint32_t rem = idx - k_sub * kLogicalN * 2u;
    uint32_t row = rem / 2u;
    uint32_t k_chunk = rem & 1u;
    uint32_t swizzled_chunk = k_chunk ^ (row & 7u);
    uint32_t base_word = (k_sub * kSmemInt4PerKSub + row * kSmemStrideInt4 + swizzled_chunk) * 4u;
    uint32_t m = row;
    uint32_t k_base = k_sub * MmaOpClass::MmaShape::K + k_chunk * 16u;

    PRAGMA_UNROLL
    for (uint32_t w = 0; w < 4u; ++w) {
      uint32_t word = 0u;
      PRAGMA_UNROLL
      for (uint32_t b = 0; b < 4u; ++b) {
        uint32_t k = k_base + w * 4u + b;
        uint8_t value = (m < shape_m && k < kOrigK) ? activation[m * kOrigK + k] : 0u;
        word |= uint32_t(value) << (8u * b);
      }
      smem_words[base_word + w] = word;
    }
  }
  __syncthreads();

  typename MmaOpClass::CRegisters d;
  constexpr uint32_t kARegs = sizeof(typename MmaOpClass::ARegisters) / sizeof(uint32_t);
  constexpr uint32_t kDRegs = sizeof(typename MmaOpClass::CRegisters) / sizeof(float);
  uint32_t a[kKSubblocks][kARegs];

  uint32_t *d_ptr = reinterpret_cast<uint32_t *>(d);
  PRAGMA_UNROLL
  for (uint32_t i = 0; i < sizeof(d) / sizeof(uint32_t); ++i) {
    d_ptr[i] = 0;
  }

  uint32_t row0_in_wg = (local / 32u) * 16u + ((local % 32u) / 4u);
  uint32_t row1_in_wg = row0_in_wg + 8u;
  uint32_t n0 = wg * 64u + row0_in_wg;
  uint32_t n1 = wg * 64u + row1_in_wg;
  PRAGMA_UNROLL
  for (uint32_t k_sub = 0; k_sub < kKSubblocks; ++k_sub) {
    uint32_t raw_base = k_sub * kRawWordsPerKSub + ((wg * 128u + local) * 2u);
    uint32_t qb[2] = {raw_frag[raw_base + 0], raw_frag[raw_base + 1]};
    uint32_t group_lo = k_sub * 2u;
    uint32_t group_hi = group_lo + 1u;
    uint8_t scales_lo[2] = {
        (n0 < kOrigN && group_lo < kOrigGroups) ? scales[n0 * kOrigGroups + group_lo] : 0u,
        (n1 < kOrigN && group_lo < kOrigGroups) ? scales[n1 * kOrigGroups + group_lo] : 0u};
    uint8_t scales_hi[2] = {
        (n0 < kOrigN && group_hi < kOrigGroups) ? scales[n0 * kOrigGroups + group_hi] : 0u,
        (n1 < kOrigN && group_hi < kOrigGroups) ? scales[n1 * kOrigGroups + group_hi] : 0u};
    fused_dequant_for_nvfp4_a8_split_k<1>(qb, a[k_sub], scales_lo, scales_hi);
  }

  uint64_t desc0 = make_swapped_probe_wgmma_smem_desc<128>(reinterpret_cast<void *>(&smem[0 * kSmemInt4PerKSub]));
  uint64_t desc1 = make_swapped_probe_wgmma_smem_desc<128>(reinterpret_cast<void *>(&smem[1 * kSmemInt4PerKSub]));
  uint64_t desc2 = make_swapped_probe_wgmma_smem_desc<128>(reinterpret_cast<void *>(&smem[2 * kSmemInt4PerKSub]));
  uint64_t desc3 = make_swapped_probe_wgmma_smem_desc<128>(reinterpret_cast<void *>(&smem[3 * kSmemInt4PerKSub]));

  wgmma_fence();
  MmaOpClass::fma(a[0], desc0, d, true);
  wgmma_commit();
  wgmma_fence();
  MmaOpClass::fma(a[1], desc1, d, true);
  wgmma_commit();
  wgmma_fence();
  MmaOpClass::fma(a[2], desc2, d, true);
  wgmma_commit();
  wgmma_fence();
  MmaOpClass::fma(a[3], desc3, d, true);
  wgmma_commit();
  wgmma_wait<3>();
  wgmma_wait<0>();

  using OutputConversion = F16Conversion<ElementC>;
  using OutputScalar = typename OutputConversion::scalar_t;
  OutputScalar *out = reinterpret_cast<OutputScalar *>(C);
  float global_scale = GS == nullptr ? 1.0f : reinterpret_cast<const float *>(GS)[0];
  float *d_float = reinterpret_cast<float *>(d);

  PRAGMA_UNROLL
  for (uint32_t reg = 0; reg < kDRegs; ++reg) {
    uint32_t col_group = reg / 4u;
    uint32_t reg_in_group = reg & 3u;
    uint32_t row = (local / 32u) * 16u + (local % 32u) / 4u + (reg_in_group / 2u) * 8u;
    uint32_t col = col_group * 8u + (local % 4u) * 2u + (reg_in_group & 1u);
    uint32_t logical_m = wg * 64u + row;
    uint32_t logical_n = col;
    if (logical_m < kOrigN && logical_n < shape_m) {
      float value = d_float[reg] * global_scale;
      if constexpr (std::is_same<ElementC, BFloat16>::value) {
        out[logical_n * kOrigN + logical_m] = __float2bfloat16_rn(value);
      } else {
        out[logical_n * kOrigN + logical_m] = __float2half_rn(value);
      }
    }
  }
}
