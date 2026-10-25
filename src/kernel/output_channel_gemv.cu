// Direct packed MX GEMV experiments. No expanded weight copy or shared memory.
#include "output_channel_gemm.h"
#include <cuda_fp16.h>
#include <stdexcept>

namespace hygemm::oc::gemv {
__device__ __forceinline__ int sf_index(int r, int g, int k) {
  return ((r / 128) * (k / 128) + g / 4) * 512
       + (r % 32) * 16 + ((r % 128) / 32) * 4 + g % 4;
}
__device__ __forceinline__ float2 fp8pair(uint16_t v) {
  uint32_t h;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h) : "h"(v));
  return __half22float2(*reinterpret_cast<__half2*>(&h));
}
template<bool FP8>
__device__ __forceinline__ void weight8(void const* ptr, float2 (&w)[4]) {
  if constexpr (FP8) {
    uint2 a = *static_cast<uint2 const*>(ptr);
    w[0] = fp8pair(a.x); w[1] = fp8pair(a.x >> 16);
    w[2] = fp8pair(a.y); w[3] = fp8pair(a.y >> 16);
  } else {
    uint32_t a = *static_cast<uint32_t const*>(ptr), h[4];
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.f16x2.e2m1x2 %0,b0; cvt.rn.f16x2.e2m1x2 %1,b1;"
        "cvt.rn.f16x2.e2m1x2 %2,b2; cvt.rn.f16x2.e2m1x2 %3,b3; }"
        : "=r"(h[0]), "=r"(h[1]), "=r"(h[2]), "=r"(h[3]) : "r"(a));
#pragma unroll
    for (int i = 0; i < 4; ++i) w[i] = __half22float2(*reinterpret_cast<__half2*>(&h[i]));
  }
}
template<bool FP8, int NT, int Splits>
__device__ __forceinline__ void row(DeviceProblem p, int r, int token0, int split, float* scratch) {
  if (r >= p.rows) return; // Uniform across the warp.
  int lane = threadIdx.x % 32, k = p.input_channels;
  float sum[NT] = {};
  for (int base = split * 256; base < k; base += 256 * Splits) {
    int col = base + lane * 8;
    if (col < k) {
      float2 w[4];
      weight8<FP8>(static_cast<uint8_t const*>(p.weight) + (int64_t(r) * k + col) / (FP8 ? 1 : 2), w);
      int ws = static_cast<uint8_t const*>(p.weight_scales)[sf_index(r,col/32,k)];
#pragma unroll
      for (int n = 0; n < NT; ++n) {
        int t = token0 + n;
        if (t < p.tokens) {
          uint2 a = *reinterpret_cast<uint2 const*>(static_cast<uint8_t const*>(p.activation) + int64_t(t)*k + col);
          float2 x[4] = {fp8pair(a.x),fp8pair(a.x>>16),fp8pair(a.y),fp8pair(a.y>>16)};
          float dot = 0;
#pragma unroll
          for(int i=0;i<4;++i) { dot = fmaf(w[i].x,x[i].x,dot); dot = fmaf(w[i].y,x[i].y,dot); }
          int as = static_cast<uint8_t const*>(p.activation_scales)[sf_index(t,col/32,k)];
          sum[n] += ldexpf(dot,ws+as-254);
        }
      }
    }
  }
#pragma unroll
  for(int n=0;n<NT;++n) {
    for(int d=16;d;d/=2) sum[n] += __shfl_down_sync(0xffffffff,sum[n],d);
    int t=token0+n;
    if(lane==0 && t<p.tokens) {
      if constexpr(Splits==1) static_cast<__half*>(p.output)[int64_t(t)*p.output_channels+p.channel_ids[r]]=__float2half_rn(sum[n]);
      else scratch[(int64_t(split)*p.tokens+t)*p.rows+r]=sum[n];
    }
  }
}
template<bool FP8,int NT,int Warps,int Splits,bool Persistent=false>
__global__ __launch_bounds__(Warps*32) void kernel(DeviceProblem p,float* scratch) {
  for(int tile=blockIdx.x;tile<(p.rows+Warps-1)/Warps;tile+=gridDim.x)
    row<FP8,NT,Splits>(p,tile*Warps+threadIdx.x/32,blockIdx.y*NT,blockIdx.z,scratch);
}
template<int Splits>
__global__ void reduce(DeviceProblem p,float const* scratch) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=p.rows*p.tokens)return;
  float sum=0;
#pragma unroll
  for(int s=0;s<Splits;++s)sum+=scratch[int64_t(s)*p.rows*p.tokens+i];
  static_cast<__half*>(p.output)[int64_t(i/p.rows)*p.output_channels+p.channel_ids[i%p.rows]]=__float2half_rn(sum);
}
template<int NT>
__global__ __launch_bounds__(128) void pair_kernel(DeviceProblem l,DeviceProblem h,int low_tiles) {
  if(blockIdx.x<low_tiles)row<false,NT,1>(l,blockIdx.x*4+threadIdx.x/32,blockIdx.y*NT,0,nullptr);
  else row<true,NT,1>(h,(blockIdx.x-low_tiles)*4+threadIdx.x/32,blockIdx.y*NT,0,nullptr);
}
template<int Id,bool FP8,int NT,int Warps,int Splits,bool Persistent=false>
class Instance final:public Kernel {
  DeviceProblem p_; KernelInfo info_; float* scratch_=nullptr; int max_blocks_=0;
 public:
  Instance(DeviceProblem p):p_(p) {
    if(p.rows<=0||p.tokens<=0||p.input_channels<=0||p.input_channels%128||p.rows>p.output_channels)
      throw std::runtime_error("Invalid GEMV shape");
    cudaFuncAttributes a{};
    auto e=cudaFuncGetAttributes(&a,kernel<FP8,NT,Warps,Splits,Persistent>);
    if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));
    info_={Id,Warps,NT,256,1,a.numRegs,int(a.sharedSizeBytes),int(a.localSizeBytes),FP8,false};
    if constexpr(Splits>1) {
      e=cudaMalloc(&scratch_,int64_t(Splits)*p.rows*p.tokens*sizeof(float));
      if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));
    }
    if constexpr(Persistent) {int dev,sm;cudaGetDevice(&dev);cudaDeviceGetAttribute(&sm,cudaDevAttrMultiProcessorCount,dev);max_blocks_=sm*4;}
  }
  ~Instance(){if(scratch_)cudaFree(scratch_);}
  void run(cudaStream_t s) override {
    int blocks=(p_.rows+Warps-1)/Warps;
    if constexpr(Persistent)blocks=std::min(blocks,max_blocks_);
    kernel<FP8,NT,Warps,Splits,Persistent><<<dim3(blocks,(p_.tokens+NT-1)/NT,Splits),Warps*32,0,s>>>(p_,scratch_);
    if constexpr(Splits>1)reduce<Splits><<<(p_.rows*p_.tokens+255)/256,256,0,s>>>(p_,scratch_);
  }
  KernelInfo info()const override{return info_;}
  bool update_operands(DeviceProblem const&p)override {
    if(p.rows!=p_.rows||p.tokens!=p_.tokens||p.input_channels!=p_.input_channels||p.output_channels!=p_.output_channels)
      throw std::runtime_error("GEMV rebind changes shape");
    p_=p;return true;
  }
};
}
namespace hygemm::oc {
std::unique_ptr<Kernel> make_gemv(int id,bool fp8,DeviceProblem const&p,cudaStream_t) {
#define GEMV(ID,N,W,S,P) case ID: if(fp8)return std::make_unique<gemv::Instance<ID,true,N,W,S,P>>(p);else return std::make_unique<gemv::Instance<ID,false,N,W,S,P>>(p);
  switch(id) {
    GEMV(28,1,4,1,false) GEMV(29,2,4,1,false) GEMV(30,4,4,1,false)
    GEMV(31,1,8,1,false) GEMV(32,1,4,4,false) GEMV(33,1,4,8,false)
    GEMV(34,1,4,1,true)
    default:throw std::runtime_error("Invalid GEMV variant");
  }
#undef GEMV
}
void run_gemv_pair(DeviceProblem const&l,DeviceProblem const&h,cudaStream_t s,int nt) {
  int lt=(l.rows+3)/4,ht=(h.rows+3)/4;
  if(!lt&&!ht)return;
  if(nt==1)gemv::pair_kernel<1><<<dim3(lt+ht,l.tokens),128,0,s>>>(l,h,lt);
  else if(nt==2)gemv::pair_kernel<2><<<dim3(lt+ht,(l.tokens+1)/2),128,0,s>>>(l,h,lt);
  else gemv::pair_kernel<4><<<dim3(lt+ht,(l.tokens+3)/4),128,0,s>>>(l,h,lt);
}
}
