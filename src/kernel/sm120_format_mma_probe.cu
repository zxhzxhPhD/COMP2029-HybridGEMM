// Register-resident MMA diagnostic, not an end-to-end GEMM benchmark.
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <vector>
#include "cute/arch/mma_sm80.hpp"
#include "cute/arch/mma_sm120.hpp"
#include "cutlass/numeric_types.h"
#define CK(x) do {auto s=(x); if(s!=cudaSuccess){std::cerr<<cudaGetErrorString(s)<<'\n';std::exit(1);}}while(0)
using F4=cutlass::float_e2m1_t;
using F8=cutlass::float_e4m3_t;
using SF=cutlass::float_ue8m0_t;
template<int P> __device__ __forceinline__ void mma(float(&a)[4]) {
  if constexpr(P==0) {
    constexpr uint32_t v=0x3c003c00;
    cute::SM80_16x8x16_F32F16F16F32_TN::fma(a[0],a[1],a[2],a[3],v,v,v,v,v,v,a[0],a[1],a[2],a[3]);
  } else if constexpr(P==3) {
    constexpr uint32_t v=0x22222222;
    cute::SM120::BLOCKSCALED::SM120_16x8x64_TN_VS<F4,F4,float,SF,32>::fma(
      a[0],a[1],a[2],a[3],v,v,v,v,v,v,a[0],a[1],a[2],a[3],uint16_t(0x7f7f),uint16_t(0x7f7f));
  } else {
    constexpr uint32_t av=P==1 ? 0x08080808 : 0x38383838, bv=0x38383838;
    using AT=typename std::conditional<P==1,F4,F8>::type;
    cute::SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<AT,F8,float,SF,32>::fma(
      a[0],a[1],a[2],a[3],av,av,av,av,bv,bv,a[0],a[1],a[2],a[3],uint8_t(0x7f),uint8_t(0x7f));
  }
}
template<int P> __global__ void probe(float* out,int iters) {
  float a[4][4]{};
  for(int i=0;i<iters;++i) {
#pragma unroll
    for(int j=0;j<4;++j) mma<P>(a[j]);
  }
  float sum=0;
#pragma unroll
  for(int j=0;j<4;++j) {
#pragma unroll
    for(int r=0;r<4;++r) sum+=a[j][r];
  }
  out[blockIdx.x*blockDim.x+threadIdx.x]=sum;
}
template<int P> void run(int grid,float* out) {
  constexpr int block=256,iters=2048,k=P==0 ? 16 : (P==3 ? 64 : 32);
  auto t0=std::chrono::steady_clock::now();
  do { probe<P><<<grid,block>>>(out,iters); CK(cudaDeviceSynchronize()); }
  while(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count()<100);
  cudaEvent_t a,b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  std::vector<float> times;
  for(int r=0;r<3;++r){
    CK(cudaEventRecord(a));
    for(int l=0;l<20;++l) probe<P><<<grid,block>>>(out,iters);
    CK(cudaEventRecord(b));CK(cudaEventSynchronize(b));
    float ms;CK(cudaEventElapsedTime(&ms,a,b));times.push_back(ms*1000/20);
  }
  std::vector<float> h(grid*block);CK(cudaMemcpy(h.data(),out,h.size()*sizeof(float),cudaMemcpyDeviceToHost));
  int failures=0; for(float v:h) failures+=(v!=float(iters*k*16));
  cudaFuncAttributes attr; CK(cudaFuncGetAttributes(&attr,probe<P>));
  std::vector<float> sorted=times;std::sort(sorted.begin(),sorted.end());
  const char* names[]={"fp16","w4a8","w8a8_mx","w4a4"};
  std::cout<<"{\"path\":\""<<names[P]<<"\",\"mma_k\":"<<k<<",\"median_us\":"<<sorted[1]
    <<",\"tflops\":"<<(2.0*16*8*k*iters*4*(block/32)*grid/sorted[1]/1e6)
    <<",\"samples_us\":["<<times[0]<<','<<times[1]<<','<<times[2]<<"],\"failures\":"<<failures
    <<",\"registers\":"<<attr.numRegs<<",\"grid\":"<<grid<<",\"block\":"<<block
    <<",\"iterations\":"<<iters<<",\"chains\":4}"<<std::endl;
  CK(cudaEventDestroy(a));CK(cudaEventDestroy(b)); if(failures) std::exit(2);
}
int main(){
  cudaDeviceProp p;CK(cudaGetDeviceProperties(&p,0));if(p.major!=12)return 1;
  int grid=p.multiProcessorCount*4;float* out;CK(cudaMalloc(&out,grid*256*sizeof(float)));
  run<0>(grid,out);run<1>(grid,out);run<2>(grid,out);run<3>(grid,out);CK(cudaFree(out));
}
