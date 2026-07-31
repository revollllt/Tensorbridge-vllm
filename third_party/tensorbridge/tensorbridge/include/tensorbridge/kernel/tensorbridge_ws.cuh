#pragma once

#include <tensorbridge/scheduler.cuh>
#include <tensorbridge/utils/all.cuh>

#include <tensorbridge/arith/epilogue_arith.cuh>
#include <tensorbridge/arith/mainloop_arith.cuh>

#include <tensorbridge/epilogue/pipeline.cuh>
#include <tensorbridge/memory/g2s_pipeline.cuh>
#include <tensorbridge/memory/s2r_pipeline.cuh>
#include <tensorbridge/mma/wgmma.cuh>
#include <tensorbridge/mma/wmma.cuh>

#include <tensorbridge/datatype/dequant.cuh>

// NVFP4 W4A8 mainloop + epilogue selection. Two configs only:
//   TENSORBRIDGE_NVFP4_PIPELINE_BASELINE == 0 (default): the optimal path --
//       CUTLASS-aligned lagged-release mainloop + STSM subtile epilogue.
//   TENSORBRIDGE_NVFP4_PIPELINE_BASELINE != 0: ablation baseline --
//       A-register wait3 mainloop + scalar epilogue.
// The optimal path engages only on the verified m64n256 fused-E4M3 config
// (see kUseNvfp4Optimal below); every other config uses the baseline path.
#ifndef TENSORBRIDGE_NVFP4_PIPELINE_BASELINE
#define TENSORBRIDGE_NVFP4_PIPELINE_BASELINE 0
#endif

#ifndef TENSORBRIDGE_PRODUCER_REGS
#define TENSORBRIDGE_PRODUCER_REGS 0
#endif


template <bool kUseTma>
class KernelTensorParamType {
public:
  using Type = std::conditional_t<kUseTma, CUtensorMap const, void *const>;
};


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

#if TENSORBRIDGE_PROFILE_TILE_PHASES
  static_assert(!TuningConfig::kUseStreamK, "TENSORBRIDGE_PROFILE_TILE_PHASES only supports non-StreamK");
#endif
  constexpr uint32_t kNumThreads = TuningConfig::kNumThreads;
  constexpr uint32_t kNumStages = TuningConfig::kNumStages;

  using SharedStorage = SharedStorage<
      MmaOpClass, BlockShape, WarpShape, ElementA, ElementB, ElementBS,
      LayerConfig, ComputeConfig, TuningConfig>;
  using Scheduler = Scheduler<
      SharedStorage, ProblemShape, BlockShape,
      LayerConfig, ComputeConfig, TuningConfig>;
  using ProducerPipeline = ProducerPipeline<
      SharedStorage, ProblemShape, BlockShape, PadShape, ElementA, ElementB, ElementBS,
      LayerConfig, ComputeConfig, TuningConfig>;
  using ConsumerPipeline = ConsumerPipeline<SharedStorage, ElementA, LayerConfig, TuningConfig>;
  using MainloopArithmetic = MainloopArithmetic<
      MmaOpClass, BlockShape, WarpShape,
      ElementA, ElementB, ElementC, ElementBS, LayerConfig>;
  using EpilogueArithmetic = EpilogueArithmetic<
      MmaOpClass, BlockShape, WarpShape,
      ElementA, ElementB, ElementC, ElementBS,
      LayerConfig, TuningConfig>;
  using WMMA = WMMA<MmaOpClass, SharedStorage, MainloopArithmetic, WarpShape, ElementA, ElementB, LayerConfig>;
  using WGMMA = WGMMA<MmaOpClass, SharedStorage, MainloopArithmetic, BlockShape, WarpShape, ElementA, ElementB, LayerConfig>;
  using MMA = std::conditional_t<MmaOpClass::kMmaType == MmaType::WGMMA, WGMMA, WMMA>;
  using Epilogue = EpiloguePipeline<
      MmaOpClass, SharedStorage, EpilogueArithmetic, ProblemShape, BlockShape, WarpShape, PadShape,
      ElementA, ElementC, LayerConfig, ComputeConfig, TuningConfig>;
  using S2RMemoryPipeline = S2RMemoryPipeline<
      SharedStorage, MMA, Epilogue, BlockShape, WarpShape, ElementA, ElementB, ElementBS,
      LayerConfig, TuningConfig>;

  extern __shared__ int4 shared_memory[];
  auto &smem = *reinterpret_cast<SharedStorage *>(shared_memory);

  auto pa = [&]() {if constexpr (TuningConfig::kUseTmaA) return &A; else return A; };
  auto pb = [&]() {if constexpr (TuningConfig::kUseTmaB) return &B; else return B; };
  auto pc = [&]() {if constexpr (TuningConfig::kUseTmaC) return &C; else return C; };
  auto pas = [&]() { return AS; };
  auto pbs = [&]() {if constexpr (TuningConfig::kUseTmaBS) return &BS; else return BS; };
  auto pbzp = [&]() {if constexpr (TuningConfig::kUseTmaBZP) return &BZP; else return BZP; };
  auto pbias = [&]() {if constexpr (TuningConfig::kUseTmaBias) return &Bias; else return Bias; };
  auto scheduler = Scheduler(smem, pc(), tensor_map_buffer, shape_m, top_k, sorted_ids_ptr, expert_ids_ptr, num_tokens_padded_ptr, expert_layout_ptr, use_int64_expert_layout);
#if TENSORBRIDGE_S2R_DUMP_MAPPING
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    tensorbridge_s2r_dump_buf = locks;
  }
  __syncthreads();
#endif
  if (threadIdx.x >= TuningConfig::kNumMathThreads) {
    if constexpr (TuningConfig::kNumMathThreads > 256) {
      asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(40));
    } else if constexpr (TuningConfig::kNumCtasPerSm == 1 && ElementA::kBits != 16) {
      static_assert(
          TuningConfig::kWarpSpecProducerRegs == 0 ||
              (TuningConfig::kWarpSpecProducerRegs >= 24 &&
               TuningConfig::kWarpSpecProducerRegs <= 128 &&
               TuningConfig::kWarpSpecProducerRegs % 8 == 0),
          "warp_spec_producer_regs must be 0 or an 8-register multiple in [24, 128]");
      constexpr int kProducerRegs = TuningConfig::kWarpSpecProducerRegs > 0
          ? TuningConfig::kWarpSpecProducerRegs
          : (TENSORBRIDGE_PRODUCER_REGS > 0 ? TENSORBRIDGE_PRODUCER_REGS : 40);
      asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(kProducerRegs));
    } else {
      asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(24));
    }

    auto producer = ProducerPipeline(smem, pa(), pb(), pas(), pbs(), pbzp(), pbias(), shape_m);
    producer.init_mbarrir();
    __syncthreads();
    while (scheduler.get_next_block()) {
      uint32_t &slice_iters = scheduler.slice_iters;

      producer.seek(scheduler.expert_id, scheduler.m_block_id, scheduler.n_block_id, scheduler.k_block_id, scheduler.current_shape_m, scheduler.m_offset);
      producer.wait_math_epilogue();
      producer.load_stage<true, true>(0);
      PRAGMA_UNROLL
      for (uint32_t stage_id = 1; stage_id < kNumStages - 1; stage_id++) {
        producer.load_stage(stage_id, stage_id < slice_iters);
      };

      while (slice_iters) {
        PRAGMA_UNROLL
        for (uint32_t stage_id = 0; stage_id < kNumStages; stage_id++) {
          if (slice_iters == 1) producer.load_channel();
          producer.wait_stage(stage_id);
          producer.load_stage(stage_id + kNumStages - 1, slice_iters >= kNumStages);
          slice_iters--;
          if (!slice_iters) break;
        }
      }
    }
  } else {
    if constexpr (TuningConfig::kNumMathThreads > 256) {
      asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(96));
    } else {
      // 65536 RF / (4 prod_warps * 32 + 8 cons_warps * 32) -> cons = (256 - prod/2) / 8 * 8
      constexpr int kProducerRegs = TuningConfig::kWarpSpecProducerRegs > 0
          ? TuningConfig::kWarpSpecProducerRegs
          : (TENSORBRIDGE_PRODUCER_REGS > 0 ? TENSORBRIDGE_PRODUCER_REGS : 40);
      constexpr int kConsumerRegs = kProducerRegs > 0
          ? ((256 - kProducerRegs / 2) / 8 * 8)
          : 232;
      asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(kConsumerRegs));
    }

    auto mainloop_arith = MainloopArithmetic();
    auto epilogue_arith = EpilogueArithmetic();
    auto mma = MMA(smem, mainloop_arith);
    auto epilogue = Epilogue(smem, pc(), tensor_map_buffer, epilogue_arith, GS, locks, shape_m, top_k);
    auto consumer = ConsumerPipeline(smem);
    auto s2r_pipe = S2RMemoryPipeline(smem, mma, epilogue);

    consumer.init_mbarrir();
    __syncthreads();
    consumer.arrive(kNumStages);
#if TENSORBRIDGE_PROFILE_TILE_PHASES
    uint32_t __profile_iter = 0;
    if (threadIdx.x == 0 && blockIdx.x == 0 && locks != nullptr) {
      locks[0] = MIN(gridDim.x, 48);
      locks[1] = 9;
    }
#endif

    while (scheduler.get_next_block()) {
      constexpr uint32_t kPartMmaShapeK = 256 / ElementA::kBits;
      constexpr uint32_t kFullSliceIters = ProblemShape::K / BlockShape::K;
      constexpr uint32_t warp_k_iters = WarpShape::K / kPartMmaShapeK;
      constexpr bool kUseNativeM64N256Wait3 =
          MMA::kUseFourRegisterBuffers &&
          warp_k_iters == 4 &&
          kNumStages > 2;
      constexpr bool kCanUseNativeM64N256RawS2R =
          kUseNativeM64N256Wait3 &&
          LayerConfig::kUseFusedE4m3Scale &&
          ElementA::kBits == 8 &&
          LayerConfig::kInputScaleGroupSize == 0 &&
          MMA::kNumRegisterBuffers == 4 &&
          !std::is_same<ElementA, ElementB>::value &&
          !LayerConfig::kHasZeroPoint;
      constexpr bool kUseNvfp4Optimal =
          TENSORBRIDGE_NVFP4_PIPELINE_BASELINE == 0 &&
          // use_stream_k now means "enable StreamK tail". DP full waves in the
          // same kernel still use the lagged CUTLASS-aligned NVFP4 mainloop;
          // split-K tail slices fall back only where runtime reduction requires it.
          kCanUseNativeM64N256RawS2R &&
          kNumStages >= 3 &&
          // Generalized token-tile family (K-TMPL-E005 R8): M_WARPS==1, channel tile fixed
          // at 128 for token {256,128,64,32,16}; R9 also allows channel256-token{64,128}
          // (512 math threads). Reduces to the 256 literals
          // at BlockShape::M==256 (bitwise). kCanUseNativeM64N256RawS2R already carries the
          // 4-buffer/wait<3>/fused-E4M3 invariants via kUseFourRegisterBuffers.
          WarpShape::M == BlockShape::M &&
          is_nvfp4_w4a8_register_buffer_tile<BlockShape>() &&
          BlockShape::K == 128 &&
          WarpShape::N == 16 && WarpShape::K == 128;
      constexpr bool kUseNvfp4PairLoad =
          TENSORBRIDGE_NVFP4_SWZ64_B_DUAL_MMA_PREINT_LOAD != 0 &&
          kUseNvfp4Optimal;
      constexpr bool kUseNvfp4DefaultLaggedLoop =
          kUseNvfp4PairLoad && warp_k_iters == 4;
      constexpr bool kInitAccumWithGmma =
          TENSORBRIDGE_WGMMA_INIT_ACCUM_WITH_SCALE_D &&
          kUseNvfp4DefaultLaggedLoop &&
          MMA::kUseAnyFusedScale;

      uint32_t &slice_iters = scheduler.slice_iters;
      const bool use_nvfp4_optimal_tile =
          kUseNvfp4Optimal &&
          (scheduler.k_block_id == 0 && slice_iters == kFullSliceIters && scheduler.slice_count == 1);
      if constexpr (!kInitAccumWithGmma) {
        mma.zero_accum();
      } else if (!use_nvfp4_optimal_tile) {
        mma.zero_accum();
      }

      epilogue.seek(scheduler.expert_id, scheduler.m_block_id, scheduler.n_block_id, scheduler.current_shape_m, scheduler.m_offset);
      epilogue.set_streamk_state(scheduler.slice_count, scheduler.slice_id, scheduler.locks_offset);

      consumer.wait_stage<true>(kNumStages);
      if constexpr (!kUseNativeM64N256Wait3) {
        s2r_pipe.load_stage_iter<true>(0, 0);
        mma.transform_b(0);
      } else if constexpr (kUseNvfp4Optimal) {
        // CUTLASS-aligned prologue: S2R(k=0), S2R(k=1), dequant(k=0)
        if (use_nvfp4_optimal_tile) {
          if constexpr (kUseNvfp4PairLoad) {
            s2r_pipe.load_stage_iter_pair(0, 0);
          } else {
            s2r_pipe.load_stage_iter(0, 0);
            if constexpr (warp_k_iters > 1) {
              s2r_pipe.load_stage_iter(0, 1);
            }
          }
          mma.transform_b(0);
        }
      }

      bool accum_scale_d = true;
      if constexpr (kInitAccumWithGmma) {
        accum_scale_d = !use_nvfp4_optimal_tile;
      }

      while (slice_iters) {
        PRAGMA_UNROLL
        for (uint32_t stage_id = 0; stage_id < kNumStages; stage_id++) {
          if constexpr (kUseNativeM64N256Wait3) {
            uint32_t next_stage_id = (stage_id + 1) % kNumStages;
            if constexpr (kUseNvfp4Optimal) {
              if (use_nvfp4_optimal_tile) {
                static_assert(
                    kCanUseNativeM64N256RawS2R,
                    "optimal lagged schedule is only audited for native NVFP4 fused-E4M3 m64n256k32");
                // CUTLASS-aligned lagged-release schedule
                // (sm90_mma_array_tma_gmma_rs_warpspecialized_mixed_input.hpp:1026-1059):
                // issue->wait<3>->S2R(k+2)->dequant(k+1) per K-iter, with the stage seam
                // (consumer release + wait next stage + S2R next stage + dequant next stage
                // iter 0) packed into the LAST iter to overlap the still-in-flight WGMMAs.
                // Buffer safety: with kNumRegisterBuffers=4 and wait<3>, iter 0 has retired
                // by the last iter so regs_b[0] is free; transform_b(0) for the next stage's
                // iter 0 is safe. No PRAGMA_UNROLL: full unroll causes register spill.
                typename ConsumerPipeline::ConsumerWaitToken consumer_wait_token{1};
                for (uint32_t warp_k_iter_id = 0; warp_k_iter_id < warp_k_iters; warp_k_iter_id++) {
                  mma.issue(stage_id, warp_k_iter_id, accum_scale_d);
                  if constexpr (kInitAccumWithGmma) {
                    accum_scale_d = true;
                  }
                  mma.template wait<3>();

                  if (warp_k_iter_id == 0 && slice_iters > 1) {
                    consumer_wait_token = consumer.try_wait_stage(next_stage_id);
                  }

                  if (warp_k_iter_id == warp_k_iters - 1) {
                    consumer.arrive(stage_id);
                    if (slice_iters > 1) {
                      consumer.wait_stage(next_stage_id, consumer_wait_token);
                      if constexpr (kUseNvfp4PairLoad) {
                        s2r_pipe.load_stage_iter_pair(next_stage_id, 0);
                      } else {
                        s2r_pipe.load_stage_iter(next_stage_id, 0);
                        if constexpr (warp_k_iters > 1) {
                          s2r_pipe.load_stage_iter(next_stage_id, 1);
                        }
                      }
                      mma.transform_b(0);
                    }
                  } else {
                    if constexpr (kUseNvfp4PairLoad) {
                      if (warp_k_iter_id == 0) {
                        consumer.wait_bs_pair(stage_id);
                        s2r_pipe.load_stage_iter_pair(stage_id, 2);
                      }
                    } else if (warp_k_iter_id < warp_k_iters - 2) {
                      s2r_pipe.load_stage_iter(stage_id, warp_k_iter_id + 2);
                    }
                    mma.transform_b(warp_k_iter_id + 1);
                  }
                }
              } else {
                // Baseline fallback for split-K tail slices: wait before
                // overwriting each register buffer and use runtime GMEM reduce.
                mma.template wait<3>();
                s2r_pipe.load_stage_iter(stage_id, 0);
                mma.transform_b(0);
                mma.template wait<2>();
                s2r_pipe.load_stage_iter(stage_id, 1);
                mma.transform_b(1);
                mma.template wait<1>();
                s2r_pipe.load_stage_iter(stage_id, 2);
                mma.transform_b(2);
                mma.template wait<0>();
                s2r_pipe.load_stage_iter(stage_id, 3);
                mma.transform_b(3);

                PRAGMA_UNROLL
                for (uint32_t warp_k_iter_id = 0; warp_k_iter_id < warp_k_iters; warp_k_iter_id++) {
                  mma.issue(stage_id, warp_k_iter_id);
                }

                consumer.arrive(stage_id);
                if (slice_iters > 1) {
                  consumer.wait_stage(next_stage_id);
                }
              }
            } else {
              // Baseline ablation: wait before overwriting each A-register buffer; QGMMA
              // consumes register operands asynchronously until its group completes.
              mma.template wait<3>();
              s2r_pipe.load_stage_iter(stage_id, 0);
              mma.transform_b(0);
              mma.template wait<2>();
              s2r_pipe.load_stage_iter(stage_id, 1);
              mma.transform_b(1);
              mma.template wait<1>();
              s2r_pipe.load_stage_iter(stage_id, 2);
              mma.transform_b(2);
              mma.template wait<0>();
              s2r_pipe.load_stage_iter(stage_id, 3);
              mma.transform_b(3);

              PRAGMA_UNROLL
              for (uint32_t warp_k_iter_id = 0; warp_k_iter_id < warp_k_iters; warp_k_iter_id++) {
                mma.issue(stage_id, warp_k_iter_id);
              }

              consumer.arrive(stage_id);
              if (slice_iters > 1) {
                consumer.wait_stage(next_stage_id);
              }
            }
          } else {
            PRAGMA_UNROLL
            for (uint32_t warp_k_iter_id = 0; warp_k_iter_id < warp_k_iters; warp_k_iter_id++) {
              s2r_pipe.load_stage_iter(stage_id, warp_k_iter_id + 1);
              mma.issue(stage_id, warp_k_iter_id);
              if (warp_k_iter_id == warp_k_iters - 2) {
                consumer.arrive(stage_id);
                if (slice_iters > 1) {
                  consumer.wait_stage((stage_id + 1) % kNumStages);
                }
              }

              mma.transform_b((warp_k_iter_id + 1) % MMA::kNumRegisterBuffers);
              mma.drain();
            }
          }

          slice_iters--;
          if (!slice_iters) break;
        };
      };

#if TENSORBRIDGE_PROFILE_TILE_PHASES
      long long __phase_t0 = 0, __phase_t1 = 0, __phase_t2 = 0, __phase_t3 = 0;
      long long __epilogue_ticks[7];
      if (threadIdx.x == 0) { asm volatile("mov.u64 %0, %%clock64;" : "=l"(__phase_t0)); }
#endif
      consumer.wait_channel();
      s2r_pipe.load_channel(scheduler.slice_id);
      if constexpr (kUseNativeM64N256Wait3) mma.drain();
#if TENSORBRIDGE_PROFILE_TILE_PHASES
      if (threadIdx.x == 0) { asm volatile("mov.u64 %0, %%clock64;" : "=l"(__phase_t1)); }
      epilogue.call(mma.final_regs_c_as_ptr(), __epilogue_ticks);
#else
      epilogue.call(mma.final_regs_c_as_ptr());
#endif
#if TENSORBRIDGE_PROFILE_TILE_PHASES
      if (threadIdx.x == 0) { asm volatile("mov.u64 %0, %%clock64;" : "=l"(__phase_t2)); }
#endif
      if constexpr (TuningConfig::kUseTmaC) tma_wait_store_group<0, true>();
#if TENSORBRIDGE_PROFILE_TILE_PHASES
      if (threadIdx.x == 0) {
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(__phase_t3));
        uint32_t slot = blockIdx.x;
        if (locks != nullptr && __profile_iter == 0 && slot < 48) {
          long long *dst = reinterpret_cast<long long*>(locks) + 2 + slot * 9;
          dst[0] = __phase_t0;
          dst[1] = __phase_t1 - __phase_t0;
          dst[2] = __epilogue_ticks[1] - __epilogue_ticks[0];
          dst[3] = __epilogue_ticks[2] - __epilogue_ticks[1];
          dst[4] = __epilogue_ticks[3] - __epilogue_ticks[2];
          dst[5] = __epilogue_ticks[4] - __epilogue_ticks[3];
          dst[6] = __epilogue_ticks[5] - __epilogue_ticks[4];
          dst[7] = __epilogue_ticks[6] - __epilogue_ticks[5];
          dst[8] = __phase_t3 - __phase_t2;
        }
      }
      __profile_iter++;
#endif
      consumer.arrive(kNumStages);
    }
  }

  __syncthreads();
  if constexpr (TuningConfig::kMultiCastSizeA > 0 || TuningConfig::kMultiCastSizeB > 0) {
    asm volatile("barrier.cluster.arrive;\n");
    asm volatile("barrier.cluster.wait;\n");
  }
};
