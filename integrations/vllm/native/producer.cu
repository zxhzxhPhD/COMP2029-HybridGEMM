// Experimental SwiGLU -> MXFP8 producer. Match both FP16 rounding points of
// vLLM 0.26's packed SiluAndMul before computing the unchanged K32 scale.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "cutlass/float8.h"

__device__ __forceinline__ int sf_address(int r,int g,int k) {
  return ((r/128)*(k/128)+g/4)*512+(r%32)*16+((r%128)/32)*4+g%4;
}
template<int Math>
__global__ void silu_mx(__half const* input,uint8_t* q,uint8_t* sf,__half* debug,int rows,int k) {
  int tid=blockIdx.x*blockDim.x+threadIdx.x,lane=threadIdx.x%8,group=tid/8;
  if(group>=rows*(k/32))return;
  int r=group/(k/32),col=(group%(k/32))*32+lane*4;
  if(r==(rows/128)*128 && rows%128 && group%4==0) {
    auto* words=reinterpret_cast<uint32_t*>(sf+sf_address(r,col/32,k));
#pragma unroll
    for(int s=0;s<16;++s) {int word=lane+s*8,sr=(word%4)*32+word/4;if(sr>=rows%128)words[word]=0x7f7f7f7fu;}
  }
  uint2 ag=*reinterpret_cast<uint2 const*>(input+int64_t(r)*k*2+col);
  uint2 au=*reinterpret_cast<uint2 const*>(input+int64_t(r)*k*2+k+col);
  float2 g0=__half22float2(*reinterpret_cast<__half2*>(&ag.x)),g1=__half22float2(*reinterpret_cast<__half2*>(&ag.y));
  float2 u0=__half22float2(*reinterpret_cast<__half2*>(&au.x)),u1=__half22float2(*reinterpret_cast<__half2*>(&au.y));
  float g[4]={g0.x,g0.y,g1.x,g1.y},u[4]={u0.x,u0.y,u1.x,u1.y},v[4];
  float maximum=0;
#pragma unroll
  for(int i=0;i<4;++i) {
    float e=Math==0?expf(-g[i]):__expf(-g[i]);
    float act=Math==1?__fdividef(g[i],1.f+e):g[i]/(1.f+e);
    // vLLM adds its runtime beta (zero here) before the multiply. Preserve
    // +0 + -0 -> +0: dropping this addition changes the FP8 sign-of-zero bit.
    float up=__fadd_rn(u[i],0.f);
    __half rounded=__float2half_rn(__half2float(__float2half_rn(act))*up);
    v[i]=__half2float(rounded);maximum=fmaxf(maximum,fabsf(v[i]));
    if(debug)debug[int64_t(r)*k+col+i]=rounded;
  }
  for(int d=4;d;d/=2)maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,d,8));
  int exponent=maximum>0?int(ceilf(log2f(maximum/448.f))):0;
  exponent=max(-127,min(127,exponent));
  if(lane==0)sf[sf_address(r,col/32,k)]=uint8_t(exponent+127);
  uint32_t result=0;
#pragma unroll
  for(int i=0;i<4;++i)result|=uint32_t(cutlass::float_e4m3_t(ldexpf(v[i],-exponent)).raw())<<(i*8);
  *reinterpret_cast<uint32_t*>(q+int64_t(r)*k+col)=result;
}
extern "C" int hg_silu_pack(void const* input,void* q,void* sf,void* debug,int rows,int k,void* stream,int math) {
  if(!input||!q||!sf||rows<=0||k<=0||k%128||math<0||math>2)return int(cudaErrorInvalidValue);
  int blocks=(int64_t(rows)*k/4+127)/128;
  auto s=static_cast<cudaStream_t>(stream);
#define RUN(M) silu_mx<M><<<blocks,128,0,s>>>((__half const*)input,(uint8_t*)q,(uint8_t*)sf,(__half*)debug,rows,k)
  if(math==0){RUN(0);}else if(math==1){RUN(1);}else{RUN(2);}
#undef RUN
  return int(cudaGetLastError());
}
