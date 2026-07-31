#pragma once

#include <tensorbridge/utils/base.cuh>

CUDA_INLINE void launch_dependent_grids() {
  asm volatile("griddepcontrol.launch_dependents;\n");
}
