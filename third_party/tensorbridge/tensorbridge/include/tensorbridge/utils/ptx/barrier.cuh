#pragma once

#include <tensorbridge/utils/base.cuh>

#ifndef TENSORBRIDGE_MBAR_FUSED_TOKEN_WAIT
#define TENSORBRIDGE_MBAR_FUSED_TOKEN_WAIT 0
#endif


template <uint32_t kNumSyncThreads, uint32_t kNumThreads, uint32_t kBarrierId = 1>
CUDA_INLINE uint32_t sync_part_threads() {
  if constexpr (kNumSyncThreads == kNumThreads) {
    __syncthreads();
  } else {
    static_assert(kNumThreads >= kNumSyncThreads);
    static_assert(kNumSyncThreads > 0);
    asm volatile("bar.sync %0, %1;" ::"r"(kBarrierId), "r"(kNumSyncThreads));
  }
}

template <uint32_t kNumSyncThreads = 0, uint32_t kNumThreads = 0>
CUDA_INLINE void barrier_acquire(int *lock, int count) {
  if (threadIdx.x == 0) {
    int state = -1;
    do {
      asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n"
                   : "=r"(state)
                   : "l"(lock));
    } while (state != count);
  }
  sync_part_threads<kNumSyncThreads, kNumThreads>();
}

template <uint32_t kNumSyncThreads = 0, uint32_t kNumThreads = 0>
CUDA_INLINE void barrier_acquire2(int *lock, int count) {
  if (threadIdx.x == 0) {
    int state = 1;
    do {
      asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n"
                   : "=r"(state)
                   : "l"(lock));
    } while (state > count);
  }
  sync_part_threads<kNumSyncThreads, kNumThreads>();
}

template <uint32_t kNumSyncThreads = 0, uint32_t kNumThreads = 0>
CUDA_INLINE void barrier_release(int *lock, bool reset = false) {
  sync_part_threads<kNumSyncThreads, kNumThreads>();
  if (threadIdx.x == 0) {
    if (reset) {
      __stcg(&lock[0], 0);
    } else {
      int32_t val = 1;
      asm volatile("fence.acq_rel.gpu;\n");
      asm volatile("red.relaxed.gpu.global.add.s32 [%0], %1;\n"
                   :
                   : "l"(lock), "r"(val));
    }
  }
}

template <uint32_t kNumSyncThreads = 0, uint32_t kNumThreads = 0>
CUDA_INLINE void barrier_release2(int *lock, int32_t val) {
  sync_part_threads<kNumSyncThreads, kNumThreads>();
  if (threadIdx.x == 0) {
    if (val < 0) {
      __stcg(&lock[0], val);
    } else {
      int32_t val = 1;
      asm volatile("fence.acq_rel.gpu;\n");
      asm volatile("red.relaxed.gpu.global.add.s32 [%0], %1;\n"
                   :
                   : "l"(lock), "r"(val));
    }
  }
}

CUDA_INLINE
void mbarrier_wait(void *barrier, bool phase_parity) {
  uint32_t smem_int_mbar = cast_smem_ptr_to_uint(barrier);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("{\n"
               "  .reg .pred p;\n"
               "  waitLoop:\n"
               "  mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
               "  @p bra done;\n"
               "  bra waitLoop;\n"
               "  done:\n"
               "}\n"
               :
               : "r"(smem_int_mbar), "r"((uint32_t)phase_parity)
               : "memory");
#else
  asm volatile("{\n"
               "  .reg .pred p;\n"
               "  waitLoop:\n"
               "  mbarrier.test_wait.parity.shared::cta.b64 p, [%0], %1;\n"
               "  @p bra done;\n"
               "  bra waitLoop;\n"
               "  done:\n"
               "}\n"
               :
               : "r"(smem_int_mbar), "r"((uint32_t)phase_parity)
               : "memory");
#endif
};

CUDA_INLINE
uint32_t mbarrier_try_wait_parity(void *barrier, bool phase_parity) {
  uint32_t smem_int_mbar = cast_smem_ptr_to_uint(barrier);
  uint32_t ready = 0;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("{\n"
               "  .reg .pred p;\n"
               "  mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
               "  selp.u32 %0, 1, 0, p;\n"
               "}\n"
               : "=r"(ready)
               : "r"(smem_int_mbar), "r"((uint32_t)phase_parity)
               : "memory");
#else
  asm volatile("{\n"
               "  .reg .pred p;\n"
               "  mbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2;\n"
               "  selp.u32 %0, 1, 0, p;\n"
               "}\n"
               : "=r"(ready)
               : "r"(smem_int_mbar), "r"((uint32_t)phase_parity)
               : "memory");
#endif
  return ready;
};

CUDA_INLINE
void mbarrier_wait_with_ready_token(void *barrier, bool phase_parity, uint32_t ready) {
  uint32_t smem_int_mbar = cast_smem_ptr_to_uint(barrier);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("{\n"
               "  .reg .pred p_ready;\n"
               "  .reg .pred p_wait;\n"
               "  setp.ne.u32 p_ready, %2, 0;\n"
               "  @p_ready bra done;\n"
               "  waitLoop:\n"
               "  mbarrier.try_wait.parity.shared::cta.b64 p_wait, [%0], %1;\n"
               "  @p_wait bra done;\n"
               "  bra waitLoop;\n"
               "  done:\n"
               "}\n"
               :
               : "r"(smem_int_mbar), "r"((uint32_t)phase_parity), "r"(ready)
               : "memory");
#else
  asm volatile("{\n"
               "  .reg .pred p_ready;\n"
               "  .reg .pred p_wait;\n"
               "  setp.ne.u32 p_ready, %2, 0;\n"
               "  @p_ready bra done;\n"
               "  waitLoop:\n"
               "  mbarrier.test_wait.parity.shared::cta.b64 p_wait, [%0], %1;\n"
               "  @p_wait bra done;\n"
               "  bra waitLoop;\n"
               "  done:\n"
               "}\n"
               :
               : "r"(smem_int_mbar), "r"((uint32_t)phase_parity), "r"(ready)
               : "memory");
#endif
};


CUDA_INLINE void mbarrier_arrive_cluster_pred(void *barrier, uint32_t dst_cta_id, bool pred) {
  if (pred) {
    uint32_t smem_int_mbar = cast_smem_ptr_to_uint(barrier);
    asm volatile(
        "{\n\t"
        ".reg .b32 remAddr32;\n\t"
        "mapa.shared::cluster.u32  remAddr32, %0, %1;\n\t"
        "mbarrier.arrive.shared::cluster.b64  _, [remAddr32];\n\t"
        "}"
        :
        : "r"(smem_int_mbar), "r"(dst_cta_id)
        : "memory");
  }
}

template <bool kUseCluster = false>
CUDA_INLINE void mbarrier_arrive(void *barrier) {
  uint32_t smem_int_mbar = cast_smem_ptr_to_uint(barrier);

  if constexpr (kUseCluster) {
    asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];"
                 :
                 : "r"(smem_int_mbar)
                 : "memory");
  } else {
    asm volatile("mbarrier.arrive.shared.b64 _, [%0];"
                 :
                 : "r"(smem_int_mbar)
                 : "memory");
  }
};
