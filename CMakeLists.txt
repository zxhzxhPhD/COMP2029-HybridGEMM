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

# Small, fixed candidate family. Compile variants separately to bound compiler memory.
set(OC_TILES_N 16 32 64 128 16 32 64 64)
set(OC_TILES_K 128 128 128 128 256 256 128 128)
set(OC_STAGES 0 0 0 0 0 0 2 0)
set(OC_PINGPONG 0 0 0 0 0 0 0 1)
set(OC_OBJECTS)
foreach(id RANGE 0 7)
  list(GET OC_TILES_N ${id} tile_n)
  list(GET OC_TILES_K ${id} tile_k)
  list(GET OC_STAGES ${id} stages)
  list(GET OC_PINGPONG ${id} pingpong)
  add_library(hygemm_oc_${id} OBJECT src/kernel/output_channel_variant.cu)
  target_include_directories(hygemm_oc_${id} PRIVATE "${CUTLASS_DIR}/include" "${CUTLASS_DIR}/tools/util/include")
  target_compile_definitions(hygemm_oc_${id} PRIVATE CUTLASS_ENABLE_TENSOR_CORE_MMA=1
    HYGEMM_OC_VARIANT_ID=${id} HYGEMM_OC_TILE_N=${tile_n} HYGEMM_OC_TILE_K=${tile_k}
    HYGEMM_OC_STAGES=${stages} HYGEMM_OC_PINGPONG=${pingpong})
  target_compile_options(hygemm_oc_${id} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_120a,code=sm_120a;--expt-relaxed-constexpr;--expt-extended-lambda;-lineinfo;-Xptxas=-v>)
  list(APPEND OC_OBJECTS $<TARGET_OBJECTS:hygemm_oc_${id}>)
endforeach()
add_library(hygemm_output_channel STATIC src/kernel/output_channel_gemm.cpp ${OC_OBJECTS})
target_link_libraries(hygemm_output_channel PUBLIC CUDA::cudart)
target_include_directories(hygemm_output_channel PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src/kernel")

add_executable(hygemm_output_channel_bench src/kernel/output_channel_bench.cu)
target_include_directories(hygemm_output_channel_bench PRIVATE "${CUTLASS_DIR}/include" "${CUTLASS_DIR}/tools/util/include")
target_compile_options(hygemm_output_channel_bench PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_120a,code=sm_120a;--expt-relaxed-constexpr>)
target_link_libraries(hygemm_output_channel_bench PRIVATE hygemm_output_channel CUDA::cublasLt)

set(OC_FUSED_OBJECTS)
foreach(id RANGE 0 7)
  list(GET OC_TILES_N ${id} tile_n)
  list(GET OC_TILES_K ${id} tile_k)
  list(GET OC_STAGES ${id} stages)
  list(GET OC_PINGPONG ${id} pingpong)
  add_library(hygemm_oc_fused_${id} OBJECT src/kernel/output_channel_fused_variant.cu)
  target_include_directories(hygemm_oc_fused_${id} PRIVATE "${CUTLASS_DIR}/include" "${CUTLASS_DIR}/tools/util/include")
  target_compile_definitions(hygemm_oc_fused_${id} PRIVATE CUTLASS_ENABLE_TENSOR_CORE_MMA=1
    HYGEMM_OC_VARIANT_ID=${id} HYGEMM_OC_TILE_N=${tile_n} HYGEMM_OC_TILE_K=${tile_k}
    HYGEMM_OC_STAGES=${stages} HYGEMM_OC_PINGPONG=${pingpong})
  target_compile_options(hygemm_oc_fused_${id} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_120a,code=sm_120a;--expt-relaxed-constexpr;--expt-extended-lambda;-lineinfo;-Xptxas=-v>)
  list(APPEND OC_FUSED_OBJECTS $<TARGET_OBJECTS:hygemm_oc_fused_${id}>)
endforeach()
add_library(hygemm_output_channel_fused STATIC src/kernel/output_channel_fused.cpp ${OC_FUSED_OBJECTS})
target_link_libraries(hygemm_output_channel_fused PUBLIC CUDA::cudart)
target_include_directories(hygemm_output_channel_fused PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src/kernel")

add_executable(hygemm_output_channel_fused_bench src/kernel/output_channel_fused_bench.cu)
target_include_directories(hygemm_output_channel_fused_bench PRIVATE "${CUTLASS_DIR}/include" "${CUTLASS_DIR}/tools/util/include")
target_compile_options(hygemm_output_channel_fused_bench PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_120a,code=sm_120a;--expt-relaxed-constexpr>)
target_link_libraries(hygemm_output_channel_fused_bench PRIVATE hygemm_output_channel_fused hygemm_output_channel CUDA::cublasLt)
