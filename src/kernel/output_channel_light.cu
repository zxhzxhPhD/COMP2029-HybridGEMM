// Small-output-tile MX GEMM. All CTA warps cooperate in a cp.async ring;
// no dedicated producer warpgroup, TMA descriptor, or output scratch buffer.
#include "output_channel_gemm.h"
#include <stdexcept>
#include <cuda_fp16.h>
#include "cute/arch/mma_sm120.hpp"
#include "cutlass/numeric_types.h"

namespace hygemm::oc::light {
using F4 = cutlass::float_e2m1_t;
using F8 = cutlass::float_e4m3_t;
using SF = cutlass::float_ue8m0_t;

__device__ __forceinline__ int sf_index(int row, int group, int k) {
  return ((row / 128) * (k / 128) + group / 4) * 512
       + (row % 32) * 16 + ((row % 128) / 32) * 4 + group % 4;
}

template<int Bytes>
__device__ __forceinline__ void copy_async(void* dst, void const* src, bool valid) {
  unsigned smem = static_cast<unsigned>(__cvta_generic_to_shared(dst));
  int bytes = valid ? Bytes : 0;
  asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %3;\n"
               :: "r"(smem), "l"(src), "n"(Bytes), "r"(bytes) : "memory");
}

template<int BM, int BN, int BK, int Warps, int Stages, bool FP8>
struct Storage {
  static constexpr int Threads = Warps * 32;
  static constexpr int WarpM = BM / Warps;
  static constexpr int Groups = BK / 32;
  static constexpr int ASize = BM * BK;
  static constexpr int BSize = BN * BK;
  static constexpr int SFASize = BM * Groups;
  static constexpr int StageBytes = ASize + BSize + SFASize + BN * Groups;
  static constexpr int Bytes = Stages * StageBytes;
  static_assert(WarpM % 16 == 0 && BN % 8 == 0 && BK % 128 == 0);

  __device__ static int swizzle(int row, int col) {
    return row * BK + (col ^ ((row & 7) * 16));
  }

  __device__ static void load(uint8_t* dst, DeviceProblem const& p, int m0, int n0, int k0) {
    // Each 16 logical FP4 values occupy a 16-byte shared container: eight
    // packed bytes plus padding. LDSM expands the nibbles in hardware.
    for (int i = threadIdx.x; i < BM * (BK / 16); i += Threads) {
      int row = i / (BK / 16), col = (i % (BK / 16)) * 16;
      bool valid = m0 + row < p.rows && k0 + col < p.input_channels;
      auto src = static_cast<uint8_t const*>(p.weight)
               + (valid ? (int64_t(m0 + row) * p.input_channels + k0 + col) / (FP8 ? 1 : 2) : 0);
      copy_async<FP8 ? 16 : 8>(dst + swizzle(row, col), src, valid);
    }
    for (int i = threadIdx.x; i < BN * (BK / 16); i += Threads) {
      int row = i / (BK / 16), col = (i % (BK / 16)) * 16;
      bool valid = n0 + row < p.tokens && k0 + col < p.input_channels;
      auto src = static_cast<uint8_t const*>(p.activation)
               + (valid ? int64_t(n0 + row) * p.input_channels + k0 + col : 0);
      copy_async<16>(dst + ASize + swizzle(row, col), src, valid);
    }
    // Four adjacent K32 scale factors of one row are contiguous in the
    // existing CUTLASS K128 scale block. Keep the external packing unchanged.
    for (int i = threadIdx.x; i < (BM + BN) * (BK / 128); i += Threads) {
      int row = i / (BK / 128), chunk = i % (BK / 128);
      bool weight = row < BM;
      int local = weight ? row : row - BM;
      int global = (weight ? m0 : n0) + local;
      bool valid = global < (weight ? p.rows : p.tokens) && k0 + chunk * 128 < p.input_channels;
      auto src = static_cast<uint8_t const*>(weight ? p.weight_scales : p.activation_scales)
               + (valid ? sf_index(global, k0 / 32 + chunk * 4, p.input_channels) : 0);
      copy_async<4>(dst + ASize + BSize + (weight ? 0 : SFASize) + local * Groups + chunk * 4, src, valid);
    }
  }
};

template<int BM, int BN, int BK, int Warps, int Stages, bool FP8, bool CacheScale = false>
__global__ __launch_bounds__(Warps * 32)
void kernel(DeviceProblem p) {
  using S = Storage<BM, BN, BK, Warps, Stages, FP8>;
  __shared__ __align__(16) uint8_t storage[S::Bytes];
  int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  int m0 = blockIdx.x * BM, n0 = blockIdx.y * BN;
  int tiles = (p.input_channels + BK - 1) / BK;
  float accum[S::WarpM / 16][BN / 8][4] = {};
  if constexpr (!FP8) {
    for (int i = threadIdx.x; i < Stages * BM * (BK / 16); i += S::Threads) {
      int stage = i / (BM * (BK / 16));
      int index = i % (BM * (BK / 16));
      int row = index / (BK / 16), col = index % (BK / 16) * 16;
      *reinterpret_cast<uint64_t*>(storage + stage * S::StageBytes + S::swizzle(row, col) + 8) = 0;
    }
  }
  for (int s = 0; s < Stages - 1; ++s) {
    if (s < tiles) S::load(storage + s * S::StageBytes, p, m0, n0, s * BK);
    asm volatile("cp.async.commit_group;\n" ::);
  }
  for (int tile = 0; tile < tiles; ++tile) {
    if (tile + Stages - 2 >= tiles) {
      asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    } else {
      asm volatile("cp.async.wait_group %0;\n" :: "n"(Stages - 2) : "memory");
    }
    __syncthreads();
    int next = tile + Stages - 1;
    if (next < tiles) S::load(storage + (next % Stages) * S::StageBytes, p, m0, n0, next * BK);
    asm volatile("cp.async.commit_group;\n" ::);
    uint8_t* current = storage + (tile % Stages) * S::StageBytes;
    uint32_t cached_sa[S::WarpM / 16], cached_sb[BN / 8];
#pragma unroll
    for (int kk = 0; kk < BK / 32; ++kk) {
      if constexpr (CacheScale) {
        if (kk % 4 == 0) {
#pragma unroll
          for (int mi = 0; mi < S::WarpM / 16; ++mi) {
            int r = warp * S::WarpM + mi * 16 + (lane % 2) * 8 + lane / 4;
            cached_sa[mi] = *reinterpret_cast<uint32_t const*>(current + S::ASize + S::BSize + r * S::Groups + kk);
          }
#pragma unroll
          for (int ni = 0; ni < BN / 8; ++ni)
            cached_sb[ni] = *reinterpret_cast<uint32_t const*>(current + S::ASize + S::BSize + S::SFASize + (ni * 8 + lane / 4) * S::Groups + kk);
        }
      }
      uint32_t b[BN / 8][2];
      uint8_t sb[BN / 8];
#pragma unroll
      for (int ni = 0; ni < BN / 8; ++ni) {
        int row = ni * 8 + lane % 8;
        int col = kk * 32 + ((lane / 8) % 2) * 16;
        unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(current + S::ASize + S::swizzle(row, col)));
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                     : "=r"(b[ni][0]), "=r"(b[ni][1]) : "r"(address));
        if constexpr (CacheScale) sb[ni] = uint8_t(cached_sb[ni] >> (8 * (kk % 4)));
        else sb[ni] = current[S::ASize + S::BSize + S::SFASize + (ni * 8 + lane / 4) * S::Groups + kk];
      }
#pragma unroll
      for (int mi = 0; mi < S::WarpM / 16; ++mi) {
        int row = warp * S::WarpM + mi * 16 + lane % 16;
        int col = kk * 32 + (lane / 16) * 16;
        unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(current + S::swizzle(row, col)));
        uint32_t a[4];
        if constexpr (FP8) {
          asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                       : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(address));
        } else {
          asm volatile("ldmatrix.sync.aligned.m8n16.x4.shared.b8x16.b4x16_p64 {%0, %1, %2, %3}, [%4];\n"
                       : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(address));
#pragma unroll
          for (int j = 0; j < 4; ++j) a[j] <<= 2;
        }
        int sfrow = warp * S::WarpM + mi * 16 + (lane % 2) * 8 + lane / 4;
        uint8_t sa;
        if constexpr (CacheScale) sa = uint8_t(cached_sa[mi] >> (8 * (kk % 4)));
        else sa = current[S::ASize + S::BSize + sfrow * S::Groups + kk];
        using A = std::conditional_t<FP8, F8, F4>;
#pragma unroll
        for (int ni = 0; ni < BN / 8; ++ni) {
          auto& c = accum[mi][ni];
          cute::SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<A, F8, float, SF, 32>::fma(
              c[0], c[1], c[2], c[3], a[0], a[1], a[2], a[3], b[ni][0], b[ni][1],
              c[0], c[1], c[2], c[3], sa, sb[ni]);
        }
      }
    }
    __syncthreads();
  }
  asm volatile("cp.async.wait_group 0;\n" ::);
#pragma unroll
  for (int mi = 0; mi < S::WarpM / 16; ++mi) {
#pragma unroll
    for (int ni = 0; ni < BN / 8; ++ni) {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        int row = m0 + warp * S::WarpM + mi * 16 + lane / 4 + (j / 2) * 8;
        int token = n0 + ni * 8 + (lane % 4) * 2 + j % 2;
        if (row < p.rows && token < p.tokens) {
          static_cast<__half*>(p.output)[int64_t(token) * p.output_channels + p.channel_ids[row]] = __float2half_rn(accum[mi][ni][j]);
        }
      }
    }
  }
}

template<int Id, int BM, int BN, int BK, int Warps, int Stages, bool FP8, bool CacheScale = false>
class Instance final : public Kernel {
  DeviceProblem problem_;
  KernelInfo info_;
 public:
  Instance(DeviceProblem const& p) : problem_(p) {
    if (p.rows <= 0 || p.tokens <= 0 || p.input_channels <= 0 || p.input_channels % 128 || p.rows > p.output_channels)
      throw std::runtime_error("Invalid lightweight GEMM shape");
    cudaFuncAttributes attr{};
    auto status = cudaFuncGetAttributes(&attr, kernel<BM, BN, BK, Warps, Stages, FP8, CacheScale>);
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    info_ = {Id, BM, BN, BK, Stages, attr.numRegs, int(attr.sharedSizeBytes), int(attr.localSizeBytes), FP8, false};
  }
  void run(cudaStream_t stream) override {
    dim3 grid((problem_.rows + BM - 1) / BM, (problem_.tokens + BN - 1) / BN);
    kernel<BM, BN, BK, Warps, Stages, FP8, CacheScale><<<grid, Warps * 32, 0, stream>>>(problem_);
    auto status = cudaGetLastError();
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
  }
  KernelInfo info() const override { return info_; }
  bool update_operands(DeviceProblem const& p) override {
    if (p.rows != problem_.rows || p.tokens != problem_.tokens || p.input_channels != problem_.input_channels || p.output_channels != problem_.output_channels)
      throw std::runtime_error("Rebinding cannot change lightweight GEMM shape");
    problem_ = p;
    return true;
  }
};
} // namespace hygemm::oc::light

namespace hygemm::oc {
std::unique_ptr<Kernel> make_light(int id, bool fp8, DeviceProblem const& p, cudaStream_t) {
#define LIGHT(ID, M, N, K, W, S) case ID: if (fp8) return std::make_unique<light::Instance<ID,M,N,K,W,S,true>>(p); else return std::make_unique<light::Instance<ID,M,N,K,W,S,false>>(p);
#define CACHED(ID, M, N, K, W, S) case ID: if (fp8) return std::make_unique<light::Instance<ID,M,N,K,W,S,true,true>>(p); else return std::make_unique<light::Instance<ID,M,N,K,W,S,false,true>>(p);
  switch (id) {
    LIGHT(8, 32, 8, 128, 2, 2)
    LIGHT(9, 64, 8, 128, 4, 2)
    LIGHT(10, 64, 8, 128, 2, 2)
    LIGHT(11, 32, 16, 128, 2, 2)
    LIGHT(12, 64, 16, 128, 4, 2)
    LIGHT(13, 32, 8, 256, 2, 2)
    LIGHT(14, 64, 8, 256, 2, 2)
    LIGHT(15, 64, 16, 256, 4, 2)
    LIGHT(16, 32, 8, 128, 2, 3)
    LIGHT(17, 64, 8, 128, 2, 3)
    LIGHT(18, 16, 8, 128, 1, 2)
    LIGHT(19, 16, 16, 128, 1, 2)
    LIGHT(20, 32, 8, 128, 2, 4)
    LIGHT(21, 32, 8, 128, 2, 6)
    LIGHT(22, 64, 8, 128, 2, 4)
    LIGHT(23, 64, 8, 128, 4, 4)
    LIGHT(24, 16, 8, 128, 1, 4)
    LIGHT(25, 32, 16, 128, 2, 4)
    LIGHT(26, 64, 16, 128, 4, 4)
    LIGHT(27, 16, 8, 256, 1, 4)
    CACHED(35, 32, 8, 128, 2, 2)
    CACHED(36, 64, 8, 128, 2, 2)
    CACHED(37, 64, 16, 128, 4, 2)
    CACHED(38, 128, 8, 128, 4, 2)
    CACHED(39, 128, 16, 128, 4, 2)
    CACHED(40, 64, 8, 256, 2, 3)
    CACHED(41, 32, 8, 128, 2, 3)
    CACHED(42, 64, 16, 128, 4, 4)
    CACHED(43, 128, 8, 256, 4, 2)
    CACHED(44, 128, 8, 128, 8, 2)
    default: throw std::runtime_error("Invalid lightweight variant ID");
  }
#undef LIGHT
#undef CACHED
}
} // namespace hygemm::oc
