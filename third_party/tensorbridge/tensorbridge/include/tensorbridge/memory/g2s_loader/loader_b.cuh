#pragma once

#include <tensorbridge/utils/all.cuh>

template <
    class ProblemShape, class BlockShape,
    class ElementA, class ElementB,
    class ComputeConfig, class TuningConfig>
class G2SMemoryLoaderB {
private:
  static constexpr bool kUseWarpSpec = TuningConfig::kUseWarpSpec;
  static constexpr bool kUseTma = TuningConfig::kUseTmaB;
  static constexpr bool kUseTmaSwizzle64 = TuningConfig::kUseTmaBSwizzle64;
  static constexpr bool kUseCpAsync = TuningConfig::kUseCpAsync;
  static constexpr uint32_t kNumLoadThreads = TuningConfig::kNumLoadThreads;
  static constexpr uint32_t kLoadThreadOffset = TuningConfig::kNumThreads - kNumLoadThreads;
  static constexpr uint32_t kMultiCastSizeB = TuningConfig::kMultiCastSizeB;

  static constexpr uint32_t kPartMmaShapeK = 256 / ElementA::kBits;
  static constexpr uint32_t kSmemStride = BlockShape::N * kPartMmaShapeK * ElementB::kBits / 32 / 4;
  static constexpr uint32_t kGmemStride = ProblemShape::N * kPartMmaShapeK * ElementB::kBits / 32 / 4;
  static constexpr uint32_t kGmemExpertStride = ProblemShape::N * ProblemShape::K * ElementB::kBits / 32 / 4;
  static constexpr uint32_t kNumInt4s = kSmemStride * BlockShape::K / kPartMmaShapeK;
  // For the 2D + 64B-swizzle TMA-B path: 2D descriptor uses (K-int32 col, N-row).
  // K-int32 per K-iter (kPartMmaShapeK = 32 int4 K) = 32 * ElementB::kBits / 32 = 4 int32.
  // So per stage (BlockShape::K/kPartMmaShapeK = 4 K-iters), TMA K-coord increments by
  // 4 * 4 = 16 int32 = 64 bytes = one swizzle row. Matches CUTLASS layout.
  static constexpr uint32_t kSwizzleKInt32PerKIter = kPartMmaShapeK * ElementB::kBits / 32;  // = 4
  static constexpr uint32_t kSwizzleKInt32PerStage = kSwizzleKInt32PerKIter * (BlockShape::K / kPartMmaShapeK);  // = 16

public:
  const CUtensorMap *tensor_map_ptr;
  const int4 *gmem_ptr_raw;
  const int4 *gmem_ptr;

  uint32_t row_offset;
  uint32_t col_offset;
  uint32_t cluster_rank = blockIdx.x % kMultiCastSizeB;

  CUDA_INLINE
  G2SMemoryLoaderB(const void *ptr) {
    if constexpr (kUseTma) {
      tensor_map_ptr = reinterpret_cast<const CUtensorMap *>(ptr);
    } else {
      gmem_ptr_raw = reinterpret_cast<const int4 *>(ptr);
    }
  }

  template <bool kShouldAdvance = true>
  CUDA_INLINE void load(int4 *smem_ptr, void *mbar_ptr) {
    if constexpr (kUseTma) load_tma(smem_ptr, mbar_ptr);
    else load_legacy(smem_ptr);
    if constexpr (kShouldAdvance) advance();
  }

  CUDA_INLINE
  void load_tma(int4 *smem_ptr, void *mbar_ptr) {
    if (threadIdx.x == kLoadThreadOffset) {
      if constexpr (kUseTmaSwizzle64) {
        // 2D descriptor: smem_dims = (K-int32 inner = 16, N rows = block_shape_n).
        // Coords: (K-int32 offset, N-row offset). K-coord scales row_offset
        // (in K-iter units) by kSwizzleKInt32PerKIter (= 4); N-coord is
        // col_offset * 32 (col_offset units = block_shape_n / 32).
        uint32_t k_coord = row_offset * kSwizzleKInt32PerKIter;
        uint32_t n_coord = col_offset * 32;
        if constexpr (kMultiCastSizeB == 1) {
          tma_load_2d(tensor_map_ptr, smem_ptr, mbar_ptr, k_coord, n_coord);
        } else if (cluster_rank == 0) {
          tma_load_2d<kMultiCastSizeB>(tensor_map_ptr, smem_ptr, mbar_ptr, k_coord, n_coord);
        }
      } else if constexpr (kMultiCastSizeB == 1) {
        tma_load_3d(tensor_map_ptr, smem_ptr, mbar_ptr, 0, col_offset, row_offset);
      } else if (cluster_rank == 0) {
        tma_load_3d<kMultiCastSizeB>(tensor_map_ptr, smem_ptr, mbar_ptr, 0, col_offset, row_offset);
      }
    }
  }

  CUDA_INLINE
  void load_legacy(int4 *smem_ptr) {
    legacy_load_2d<
        kUseCpAsync, kNumInt4s, kNumLoadThreads,
        kGmemStride, kSmemStride, kLoadThreadOffset>(gmem_ptr, smem_ptr);
  }

  CUDA_INLINE
  void advance() {
    row_offset += BlockShape::K / kPartMmaShapeK;
    gmem_ptr += kGmemStride * BlockShape::K / kPartMmaShapeK;
  }

  CUDA_INLINE
  void seek(uint32_t expert_id, uint32_t n_block_id, uint32_t k_block_id) {
    row_offset = expert_id * (ProblemShape::K / kPartMmaShapeK) + k_block_id * (BlockShape::K / kPartMmaShapeK);
    col_offset = n_block_id * (BlockShape::N / 32);

    uint64_t gmem_offset = expert_id * kGmemExpertStride;
    gmem_offset += n_block_id * kSmemStride + k_block_id * (kGmemStride * BlockShape::K / kPartMmaShapeK);
    gmem_ptr = gmem_ptr_raw + gmem_offset;
  }
};
