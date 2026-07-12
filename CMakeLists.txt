cmake_minimum_required(VERSION 3.22)

project(hygemm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CUTLASS_DIR "" CACHE PATH "Path to a CUTLASS checkout")
if(NOT CUTLASS_DIR)
  message(FATAL_ERROR "Set CUTLASS_DIR to the pinned CUTLASS checkout")
endif()
if(NOT EXISTS "${CUTLASS_DIR}/include/cutlass/cutlass.h")
  message(FATAL_ERROR "CUTLASS_DIR does not look like a CUTLASS checkout: ${CUTLASS_DIR}")
endif()

# CMake 3.22 does not understand architecture suffixes such as 120a. Disable
# its architecture flag generation and pass the feature-specific target to nvcc.
set(CMAKE_CUDA_ARCHITECTURES OFF)

add_executable(hygemm_sm120_baselines
  src/kernel/sm120_baselines.cu
)

target_include_directories(hygemm_sm120_baselines PRIVATE
  "${CUTLASS_DIR}/include"
  "${CUTLASS_DIR}/tools/util/include"
)

target_compile_options(hygemm_sm120_baselines PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
    -gencode=arch=compute_120a,code=sm_120a
    --expt-relaxed-constexpr
    --expt-extended-lambda
    -lineinfo
  >
)

target_compile_definitions(hygemm_sm120_baselines PRIVATE
  CUTLASS_ENABLE_TENSOR_CORE_MMA=1
)
