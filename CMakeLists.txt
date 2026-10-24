cmake_minimum_required(VERSION 3.22)

project(hygemm LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

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

add_executable(hygemm_sm120_mma_mixing
  src/kernel/sm120_mma_mixing.cu
)

target_include_directories(hygemm_sm120_mma_mixing PRIVATE
  "${CUTLASS_DIR}/include"
)

target_compile_options(hygemm_sm120_mma_mixing PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
    -gencode=arch=compute_120a,code=sm_120a
    --expt-relaxed-constexpr
    --expt-extended-lambda
    -lineinfo
  >
)

target_compile_definitions(hygemm_sm120_mma_mixing PRIVATE
  CUTLASS_ENABLE_TENSOR_CORE_MMA=1
)

add_executable(hygemm_sm120_execution_policy_bench
  src/kernel/sm120_execution_policy_bench.cu
)

target_include_directories(hygemm_sm120_execution_policy_bench PRIVATE
  "${CUTLASS_DIR}/include"
)

target_compile_options(hygemm_sm120_execution_policy_bench PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
    -gencode=arch=compute_120a,code=sm_120a
    --expt-relaxed-constexpr
    --expt-extended-lambda
    -lineinfo
  >
)

target_compile_definitions(hygemm_sm120_execution_policy_bench PRIVATE
  CUTLASS_ENABLE_TENSOR_CORE_MMA=1
)

add_executable(hygemm_sm120_k64_paired_hybrid_gemm
  src/kernel/sm120_k64_paired_hybrid_gemm.cu
)

target_include_directories(hygemm_sm120_k64_paired_hybrid_gemm PRIVATE
  "${CUTLASS_DIR}/include"
  "${CUTLASS_DIR}/tools/util/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/kernel"
)

target_compile_options(hygemm_sm120_k64_paired_hybrid_gemm PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
    -gencode=arch=compute_120a,code=sm_120a
    --expt-relaxed-constexpr
    --expt-extended-lambda
    -lineinfo
  >
)

target_compile_definitions(hygemm_sm120_k64_paired_hybrid_gemm PRIVATE
  CUTLASS_ENABLE_TENSOR_CORE_MMA=1
)

add_executable(hygemm_sm120_format_probe src/kernel/sm120_format_probe.cu)
target_include_directories(hygemm_sm120_format_probe PRIVATE
  "${CUTLASS_DIR}/include" "${CUTLASS_DIR}/tools/util/include")
target_compile_options(hygemm_sm120_format_probe PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_120a,code=sm_120a;--expt-relaxed-constexpr;--expt-extended-lambda;-lineinfo;-Xptxas=-v>)
target_compile_definitions(hygemm_sm120_format_probe PRIVATE CUTLASS_ENABLE_TENSOR_CORE_MMA=1)
target_link_libraries(hygemm_sm120_format_probe PRIVATE CUDA::cublasLt)

add_executable(hygemm_sm120_format_probe_tuned src/kernel/sm120_format_probe.cu)
target_include_directories(hygemm_sm120_format_probe_tuned PRIVATE
  "${CUTLASS_DIR}/include" "${CUTLASS_DIR}/tools/util/include")
target_compile_options(hygemm_sm120_format_probe_tuned PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_120a,code=sm_120a;--expt-relaxed-constexpr;--expt-extended-lambda;-lineinfo;-Xptxas=-v>)
target_compile_definitions(hygemm_sm120_format_probe_tuned PRIVATE CUTLASS_ENABLE_TENSOR_CORE_MMA=1 HYGEMM_TUNED_VARIANTS=1)
target_link_libraries(hygemm_sm120_format_probe_tuned PRIVATE CUDA::cublasLt)

add_executable(hygemm_sm120_format_mma_probe src/kernel/sm120_format_mma_probe.cu)
target_include_directories(hygemm_sm120_format_mma_probe PRIVATE "${CUTLASS_DIR}/include")
target_compile_options(hygemm_sm120_format_mma_probe PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_120a,code=sm_120a;--expt-relaxed-constexpr;-lineinfo>)
