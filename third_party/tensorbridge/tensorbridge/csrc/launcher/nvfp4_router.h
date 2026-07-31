#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

#include "./utils.h"

struct Nvfp4Swizzle64RawPlan {
  int64_t block_m = 128;
  int64_t block_n = 128;
  int64_t block_k = 128;
  int64_t warp_m = 128;
  int64_t warp_n = 16;
  int64_t warp_k = 128;
  bool use_stream_k = false;
  int64_t num_stages = 4;
  bool use_tma_c = true;
  int64_t multi_cast_size_a = 1;
  int64_t multi_cast_size_b = 1;
  bool use_m_fast_tile_order = false;
  bool use_nvfp4_prefetch_raw_wait3_issue_auto = false;
  int64_t warp_spec_producer_regs = 0;
  bool nvfp4_swz64_prebcast_prmt_const_variant = false;
};

inline int64_t ceil_div_i64(int64_t a, int64_t b) {
  return (a + b - 1) / b;
}

inline bool shape_eq(int64_t m, int64_t n, int64_t k, int64_t em, int64_t en, int64_t ek) {
  return m == em && n == en && k == ek;
}

inline bool nk_eq(int64_t n, int64_t k, int64_t en, int64_t ek) {
  return n == en && k == ek;
}

inline int64_t select_block_m(int64_t shape_m, int64_t max_block_m) {
  int64_t best_block_m = 8;
  int64_t best_blocks = ceil_div_i64(shape_m, best_block_m);
  for (int64_t block_m = 16; block_m <= max_block_m; block_m += 8) {
    int64_t blocks = ceil_div_i64(shape_m, block_m);
    if (blocks < best_blocks) {
      best_blocks = blocks;
      best_block_m = block_m;
    }
  }
  return best_block_m;
}

inline bool is_measured_m256_token256_island(
    int64_t shape_m,
    int64_t shape_n,
    int64_t shape_k,
    int64_t block_m) {
  return shape_m == 256 && block_m == 256 &&
         (((shape_n > 8192) && (shape_k >= 4096)) ||
          nk_eq(shape_n, shape_k, 32768, 512) ||
          nk_eq(shape_n, shape_k, 24576, 1536));
}

inline bool use_stream_k_tail_for_nvfp4_interleave_cpp(
    int64_t shape_m,
    int64_t shape_n,
    int64_t shape_k,
    const Nvfp4Swizzle64RawPlan &plan,
    int64_t num_sms) {
  if (is_measured_m256_token256_island(shape_m, shape_n, shape_k, plan.block_m)) {
    return false;
  }

  int64_t cta_groups = std::max<int64_t>(
      1, num_sms / std::max<int64_t>(1, plan.multi_cast_size_a * plan.multi_cast_size_b));
  int64_t m_blocks = ceil_div_i64(shape_m, plan.block_m * std::max<int64_t>(1, plan.multi_cast_size_b));
  int64_t n_blocks = ceil_div_i64(shape_n, plan.block_n * std::max<int64_t>(1, plan.multi_cast_size_a));
  int64_t k_blocks = ceil_div_i64(shape_k, plan.block_k);
  int64_t mn_tiles = m_blocks * n_blocks;
  int64_t dp_waves = mn_tiles ? ceil_div_i64(mn_tiles, cta_groups) : 0;
  int64_t dp_wave_fill_den = dp_waves * cta_groups;

  int64_t streamk_mn_tiles = mn_tiles;
  if (mn_tiles > cta_groups) {
    streamk_mn_tiles = mn_tiles % cta_groups;
    if (streamk_mn_tiles && streamk_mn_tiles * 10 <= cta_groups) {
      streamk_mn_tiles += cta_groups;
    }
  }
  if (streamk_mn_tiles == 0) {
    return false;
  }

  // Severe underfill: dp_wave_fill <= 3/4.
  if (mn_tiles * 4 <= 3 * dp_wave_fill_den) {
    return true;
  }
  // Well-filled DP wave: dp_wave_fill > 112/132.
  if (mn_tiles * 132 > 112 * dp_wave_fill_den) {
    return false;
  }

  int64_t streamk_mnk_iters = streamk_mn_tiles * k_blocks;
  int64_t streamk_total_iters_per_cta =
      streamk_mnk_iters ? ceil_div_i64(streamk_mnk_iters, cta_groups) : 0;
  int64_t estimated_slice_count = 1;
  if (streamk_total_iters_per_cta) {
    estimated_slice_count = std::max<int64_t>(1, ceil_div_i64(k_blocks, streamk_total_iters_per_cta));
  }
  int64_t estimated_reduce_tiles = streamk_mn_tiles * std::max<int64_t>(0, estimated_slice_count - 1);
  int64_t estimated_reduce_elems = estimated_reduce_tiles * plan.block_m * plan.block_n;
  return estimated_reduce_elems <= 1500000;
}

inline bool has_stream_k_tail_work_for_nvfp4_interleave_cpp(
    int64_t shape_m,
    int64_t shape_n,
    int64_t shape_k,
    const Nvfp4Swizzle64RawPlan &plan,
    int64_t num_sms) {
  int64_t cta_groups = std::max<int64_t>(
      1, num_sms / std::max<int64_t>(1, plan.multi_cast_size_a * plan.multi_cast_size_b));
  int64_t m_blocks = ceil_div_i64(shape_m, plan.block_m * std::max<int64_t>(1, plan.multi_cast_size_b));
  int64_t n_blocks = ceil_div_i64(shape_n, plan.block_n * std::max<int64_t>(1, plan.multi_cast_size_a));
  int64_t mn_tiles = m_blocks * n_blocks;
  int64_t streamk_mn_tiles = mn_tiles;
  if (mn_tiles > cta_groups) {
    streamk_mn_tiles = mn_tiles % cta_groups;
    if (streamk_mn_tiles && streamk_mn_tiles * 10 <= cta_groups) {
      streamk_mn_tiles += cta_groups;
    }
  }
  return streamk_mn_tiles != 0;
}

inline bool bm256_rescues_bm176_unified_tail_cpp(
    int64_t shape_m,
    int64_t shape_n,
    int64_t block_m,
    bool use_stream_k,
    int64_t num_sms) {
  if (use_stream_k || block_m != 176 || shape_m < 512) {
    return false;
  }
  int64_t mn_tiles = ceil_div_i64(shape_m, 256) * ceil_div_i64(shape_n, 128);
  int64_t waves = mn_tiles ? ceil_div_i64(mn_tiles, num_sms) : 0;
  int64_t den = waves * num_sms;
  return den > 0 && mn_tiles * 132 >= 96 * den;
}

inline Nvfp4Swizzle64RawPlan select_nvfp4_swizzle64_raw_plan_cpp(
    int64_t shape_m,
    int64_t shape_n,
    int64_t shape_k,
    int64_t use_stream_k_override,
    int64_t num_sms) {
  Nvfp4Swizzle64RawPlan plan;
  const bool large_mlp_shape =
      ((shape_n >= 8192 && shape_k >= 3584) || (shape_n >= 3584 && shape_k >= 8192));
  const bool target_path =
      shape_m >= 1024 && large_mlp_shape && shape_n % 128 == 0 && shape_k % 128 == 0;

  if (target_path) {
    plan.block_m = 256;
    plan.warp_m = 256;
    plan.multi_cast_size_a = 1;
    if (shape_m >= 512 && shape_m <= 2048 && shape_n >= 14336 && shape_k >= 3072) {
      plan.use_m_fast_tile_order = true;
    }
  } else {
    int64_t max_block_m = 176;
    if (shape_m >= 8192 && shape_n >= 8192 && shape_k >= 8192) {
      max_block_m = 184;
    }
    plan.block_m = select_block_m(shape_m, max_block_m);
    plan.warp_m = plan.block_m;

    int64_t pre_swizzle_block_n = 256;
    if (shape_n <= 4096 && plan.block_m <= 64) {
      pre_swizzle_block_n = 128;
    }
    bool wide_n_decode = shape_n > 8192 && shape_m <= 128;
    bool mid_n_smallbatch = (shape_n >= 4096 && shape_n < 8192) && shape_m <= 512;
    if (wide_n_decode || mid_n_smallbatch) {
      pre_swizzle_block_n = 128;
      plan.num_stages = wide_n_decode ? 6 : 4;
    }

    bool disable_multicast_a = shape_m >= 1024;
    if (shape_n % (pre_swizzle_block_n * 2) == 0 &&
        shape_m >= 4 * plan.block_m &&
        !disable_multicast_a) {
      plan.multi_cast_size_a = 2;
    }
  }

  // Apply the swizzle64 raw BN128 kernel-family contract.
  plan.block_n = 128;
  plan.block_k = 128;
  plan.warp_n = 16;
  plan.warp_k = 128;
  plan.multi_cast_size_b = 1;

  if (shape_m == 2048 && shape_n == 18944 && shape_k == 3584) {
    plan.use_nvfp4_prefetch_raw_wait3_issue_auto = true;
  }

  // Pre-route sanitization.
  if (!plan.use_stream_k && shape_m == 256 &&
      (((shape_n > 8192) && (shape_k >= 4096)) ||
       nk_eq(shape_n, shape_k, 32768, 512) ||
       nk_eq(shape_n, shape_k, 24576, 1536))) {
    plan.block_m = 256;
    plan.warp_m = 256;
  }
  if (!plan.use_stream_k && shape_m == 128 &&
      (nk_eq(shape_n, shape_k, 57344, 8192) ||
       nk_eq(shape_n, shape_k, 28672, 8192) ||
       nk_eq(shape_n, shape_k, 13824, 5120) ||
       nk_eq(shape_n, shape_k, 15360, 5120))) {
    plan.num_stages = 5;
  }

  bool force_tail_off = bm256_rescues_bm176_unified_tail_cpp(
      shape_m, shape_n, plan.block_m, plan.use_stream_k, num_sms);
  if (force_tail_off) {
    plan.block_m = 256;
    plan.warp_m = 256;
    plan.use_stream_k = false;
  }

  if (use_stream_k_override < 0) {
    plan.use_stream_k = force_tail_off ? false :
        use_stream_k_tail_for_nvfp4_interleave_cpp(shape_m, shape_n, shape_k, plan, num_sms);
  } else {
    plan.use_stream_k = (use_stream_k_override != 0) &&
        has_stream_k_tail_work_for_nvfp4_interleave_cpp(shape_m, shape_n, shape_k, plan, num_sms);
  }

  // Post-route measured islands.
  if (!plan.use_stream_k && plan.block_m == 176 &&
      ((shape_m == 512 && shape_n >= 13824) || shape_m >= 2048)) {
    plan.block_m = 256;
    plan.warp_m = 256;
  }
  if (!plan.use_stream_k && plan.block_m == 256 &&
      (shape_m == 4096 || shape_m == 8192) && shape_n == 32768 && shape_k == 512) {
    plan.use_m_fast_tile_order = true;
  }
  if (!plan.use_stream_k && plan.block_m == 256 &&
      shape_m == 1024 && shape_n == 4096 && shape_k == 7168) {
    plan.use_m_fast_tile_order = true;
  }
  if (!plan.use_stream_k && plan.block_m == 256 &&
      (shape_eq(shape_m, shape_n, shape_k, 1024, 8192, 28672) ||
       shape_eq(shape_m, shape_n, shape_k, 2048, 8192, 28672) ||
       shape_eq(shape_m, shape_n, shape_k, 4096, 24576, 4096))) {
    plan.use_m_fast_tile_order = true;
  }
  if (!plan.use_stream_k && plan.block_m == 256 &&
      shape_m == 512 && shape_k >= 4096 && shape_k <= 18432 &&
      (shape_n >= 8192 || shape_k >= 8192)) {
    plan.use_m_fast_tile_order = true;
  }
  if (!plan.use_stream_k && plan.block_m == 128 &&
      shape_m == 256 && (shape_n == 7168 || shape_n == 8192) && shape_k >= 5120) {
    plan.use_m_fast_tile_order = true;
  }

  if (!plan.use_stream_k && (shape_m == 16 || shape_m == 32 || shape_m == 64 || shape_m == 128) &&
      shape_n == 32768 && shape_k == 512) {
    plan.use_tma_c = false;
  }
  if (!plan.use_stream_k && shape_m == 128 &&
      (nk_eq(shape_n, shape_k, 57344, 8192) ||
       nk_eq(shape_n, shape_k, 28672, 8192) ||
       nk_eq(shape_n, shape_k, 13824, 5120) ||
       nk_eq(shape_n, shape_k, 15360, 5120))) {
    plan.use_tma_c = false;
  }
  if (!plan.use_stream_k && plan.block_m == 256 &&
      (shape_eq(shape_m, shape_n, shape_k, 512, 8192, 28672) ||
       shape_eq(shape_m, shape_n, shape_k, 2048, 8192, 28672) ||
       shape_eq(shape_m, shape_n, shape_k, 2048, 4096, 12288))) {
    plan.use_tma_c = false;
  }

  if (!plan.use_stream_k && plan.block_m == 256 &&
      (shape_eq(shape_m, shape_n, shape_k, 4096, 8192, 5120) ||
       shape_eq(shape_m, shape_n, shape_k, 1024, 24576, 4096) ||
       shape_eq(shape_m, shape_n, shape_k, 2048, 22016, 4096) ||
       shape_eq(shape_m, shape_n, shape_k, 2048, 8192, 5120) ||
       shape_eq(shape_m, shape_n, shape_k, 4096, 8192, 8192) ||
       shape_eq(shape_m, shape_n, shape_k, 2048, 12288, 4096) ||
       shape_eq(shape_m, shape_n, shape_k, 1024, 4096, 7168) ||
       shape_eq(shape_m, shape_n, shape_k, 1024, 8192, 28672) ||
       shape_eq(shape_m, shape_n, shape_k, 1024, 15360, 5120) ||
       shape_eq(shape_m, shape_n, shape_k, 1024, 17408, 5120) ||
       shape_eq(shape_m, shape_n, shape_k, 1024, 18432, 7168) ||
       shape_eq(shape_m, shape_n, shape_k, 1024, 22016, 4096) ||
       shape_eq(shape_m, shape_n, shape_k, 1024, 25600, 5120) ||
       shape_eq(shape_m, shape_n, shape_k, 1024, 27648, 5120) ||
       shape_eq(shape_m, shape_n, shape_k, 1024, 28672, 8192) ||
       shape_eq(shape_m, shape_n, shape_k, 1024, 34816, 5120) ||
       shape_eq(shape_m, shape_n, shape_k, 1024, 36864, 5120) ||
       shape_eq(shape_m, shape_n, shape_k, 1024, 36864, 7168) ||
       shape_eq(shape_m, shape_n, shape_k, 1024, 51200, 5120) ||
       shape_eq(shape_m, shape_n, shape_k, 2048, 7168, 18432) ||
       shape_eq(shape_m, shape_n, shape_k, 2048, 25600, 5120) ||
       shape_eq(shape_m, shape_n, shape_k, 2048, 28672, 8192) ||
       shape_eq(shape_m, shape_n, shape_k, 2048, 34816, 5120) ||
       shape_eq(shape_m, shape_n, shape_k, 2048, 51200, 5120))) {
    plan.warp_spec_producer_regs = 56;
  }

  if (!plan.use_stream_k && plan.block_m == 256 && shape_k > 512) {
    plan.nvfp4_swz64_prebcast_prmt_const_variant = true;
  }
  if (!plan.use_stream_k && plan.block_m == 128 &&
      shape_m == 256 && (shape_n == 7168 || shape_n == 8192) && shape_k >= 5120) {
    plan.nvfp4_swz64_prebcast_prmt_const_variant = true;
  }

  if (!plan.use_stream_k && plan.block_m == 256 &&
      (shape_eq(shape_m, shape_n, shape_k, 4096, 36864, 7168) ||
       shape_eq(shape_m, shape_n, shape_k, 8192, 36864, 7168) ||
       shape_eq(shape_m, shape_n, shape_k, 4096, 51200, 5120) ||
       shape_eq(shape_m, shape_n, shape_k, 8192, 51200, 5120) ||
       shape_eq(shape_m, shape_n, shape_k, 4096, 57344, 8192) ||
       shape_eq(shape_m, shape_n, shape_k, 8192, 57344, 8192))) {
    plan.multi_cast_size_b = 2;
  }
  return plan;
}

inline std::vector<int64_t> select_nvfp4_swizzle64_raw_config(
    int64_t shape_m,
    int64_t shape_n,
    int64_t shape_k,
    int64_t use_stream_k_override,
    int64_t num_sms) {
  ASSERT_CHECK(num_sms > 0, "num_sms must be positive, got ", num_sms);
  Nvfp4Swizzle64RawPlan plan = select_nvfp4_swizzle64_raw_plan_cpp(
      shape_m, shape_n, shape_k, use_stream_k_override, num_sms);
  return {
      plan.block_m,
      plan.block_n,
      plan.block_k,
      plan.warp_m,
      plan.warp_n,
      plan.warp_k,
      static_cast<int64_t>(plan.use_stream_k),
      0,  // use_f16_accum
      plan.num_stages,
      1,  // use_warp_spec
      1,  // use_tma
      1,  // use_tma_b
      static_cast<int64_t>(plan.use_tma_c),
      0,  // use_tma_bs
      0,  // use_tma_bzp
      1,  // use_mbarrier
      1,  // num_ctas_per_sm
      plan.multi_cast_size_a,
      plan.multi_cast_size_b,
      1,  // use_tma_b_swizzle_64
      static_cast<int64_t>(plan.use_m_fast_tile_order),
      static_cast<int64_t>(plan.use_nvfp4_prefetch_raw_wait3_issue_auto),
      plan.warp_spec_producer_regs,
      static_cast<int64_t>(plan.nvfp4_swz64_prebcast_prmt_const_variant),
  };
}
