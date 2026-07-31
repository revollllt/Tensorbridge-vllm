#pragma once

#include <tensorbridge/utils/base.cuh>

template <int count>
CUDA_INLINE void ld_shared(const int4 *smem_ptr, int4 *regs_ptr) {
  uint32_t *a = reinterpret_cast<uint32_t *>(regs_ptr);
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  if constexpr (count == 4) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                 : "r"(smem));
  } else if constexpr (count == 2) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(a[0]), "=r"(a[1])
                 : "r"(smem));
  } else if constexpr (count == 1) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
                 : "=r"(a[0])
                 : "r"(smem));
  } else {
    static_assert(count == 1 || count == 2 || count == 4, "invalid count");
  }
}

// Non-transposed shared store (stmatrix). The matrix data is in registers
// (inputs); the warp supplies per-row base addresses via `smem`. NOTE: the
// previous body used ldmatrix syntax (`=r` outputs, regs-before-addr) which is
// wrong for a store — stmatrix is `[addr], {regs}` with regs as inputs.
template <int count>
CUDA_INLINE void st_shared(uint32_t smem, const uint32_t *regs) {
  if constexpr (count == 4) {
    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1,%2,%3,%4};\n"
                 :: "r"(smem), "r"(regs[0]), "r"(regs[1]), "r"(regs[2]), "r"(regs[3]));
  } else if constexpr (count == 2) {
    asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1,%2};\n"
                 :: "r"(smem), "r"(regs[0]), "r"(regs[1]));
  } else if constexpr (count == 1) {
    asm volatile("stmatrix.sync.aligned.m8n8.x1.shared.b16 [%0], {%1};\n"
                 :: "r"(smem), "r"(regs[0]));
  } else {
    static_assert(count == 1 || count == 2 || count == 4, "invalid count");
  }
}

template <int count>
CUDA_INLINE void st_shared_trans(uint32_t smem_addr, const uint32_t *regs) {
  if constexpr (count == 4) {
    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1,%2,%3,%4};\n"
                 :: "r"(smem_addr), "r"(regs[0]), "r"(regs[1]), "r"(regs[2]), "r"(regs[3]));
  } else if constexpr (count == 2) {
    asm volatile("stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%0], {%1,%2};\n"
                 :: "r"(smem_addr), "r"(regs[0]), "r"(regs[1]));
  } else if constexpr (count == 1) {
    asm volatile("stmatrix.sync.aligned.m8n8.x1.trans.shared.b16 [%0], {%1};\n"
                 :: "r"(smem_addr), "r"(regs[0]));
  } else {
    static_assert(count == 1 || count == 2 || count == 4, "invalid count");
  }
}
