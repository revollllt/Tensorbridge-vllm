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


template <bool kUseTma>
class KernelTensorParamType {
public:
  using Type = std::conditional_t<kUseTma, CUtensorMap const, void *const>;
};


template <class MmaOpClass, uint32_t kRepeatCount, uint32_t kUnrollCount>
__global__ void tops_bench(uint32_t *out_ptr) {

  typename MmaOpClass::ARegisters regs_a;
  typename MmaOpClass::CRegisters regs_c;

  if constexpr (MmaOpClass::kMmaType == MmaType::WGMMA) {
    __shared__ alignas(1024) int4 smem[2048];
    uint64_t desc = make_wgmma_smem_desc<128>(smem, 0);
    PRAGMA_UNROLL_COUNT(kUnrollCount)
    for (uint32_t i = 0; i < kRepeatCount; i++) {
      wgmma_fence();
      MmaOpClass::fma(regs_a, desc, regs_c);
      wgmma_commit();
      wgmma_wait<0>();
    }
  } else {
    typename MmaOpClass::BRegisters regs_b;
    PRAGMA_UNROLL_COUNT(kUnrollCount)
    for (uint32_t i = 0; i < kRepeatCount; i++)
      MmaOpClass::fma(regs_a, regs_b, regs_c, regs_c);
  }

  if (threadIdx.x == 8192) out_ptr[0] = reinterpret_cast<uint32_t *>(&regs_c)[0];
};
