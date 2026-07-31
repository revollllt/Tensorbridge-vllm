#pragma once

#include <tensorbridge/utils/base.cuh>

// 128-bit (v4.b32) global store with .cs (cache streaming) modifier.
// `.cs` is a STATIC PTX modifier that ptxas cannot elide (unlike the
// dynamic `.L2::cache_hint` + createpolicy which we tried before — sass
// dump showed the createpolicy form gets stripped at SASS level).
//
// `.cs` semantics: write streams through L2 with evict-first priority, so
// the cache line is the first candidate for eviction when L2 is full. This
// prevents C-stores from displacing weight (B) cache lines mid-launch when
// sizeof(C) > L2 capacity (50MB on H100). See
// docs/changes/26-05-22_l2_thrashing_diagnosis.md for the L2 capacity
// phase-change evidence (gap +9µs jump at exactly N=12800 where C=L2).
CUDA_INLINE void st_global_v4_b32_cs(int4 *ptr, int4 val) {
  asm volatile("st.global.cs.v4.b32 [%0], {%1, %2, %3, %4};"
               :
               : "l"(ptr), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w)
               : "memory");
}
