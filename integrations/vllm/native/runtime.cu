#include "output_channel_gemm.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <cuda_fp16.h>
#include "cute/tensor.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/float8.h"
#include "cutlass/float_subbyte.h"

using hygemm::oc::DeviceProblem;
using BS=cutlass::detail::Sm1xxBlockScaledConfig<32>;
static thread_local std::string last_error;
static void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
#define HG_TRY try {
#define HG_CATCH } catch(std::exception const&e){last_error=e.what();return -1;} return 0;

struct Runner {
  std::unique_ptr<hygemm::oc::Kernel> low,high;
  cudaStream_t side{};cudaEvent_t fork{},join{};
  Runner(int li,int hi,DeviceProblem const&l,DeviceProblem const&h,cudaStream_t s) {
    if(l.rows)low=hygemm::oc::make_kernel(li,false,l,s);
    if(h.rows)high=hygemm::oc::make_kernel(hi,true,h,s);
    check(cudaStreamCreateWithFlags(&side,cudaStreamNonBlocking));
    check(cudaEventCreateWithFlags(&fork,cudaEventDisableTiming));
    check(cudaEventCreateWithFlags(&join,cudaEventDisableTiming));
  }
  ~Runner(){cudaEventDestroy(fork);cudaEventDestroy(join);cudaStreamDestroy(side);}
  void run(DeviceProblem const&l,DeviceProblem const&h,cudaStream_t s,bool concurrent) {
    if(low && !low->update_operands(l))throw std::runtime_error("Low kernel cannot rebind");
    if(high && !high->update_operands(h))throw std::runtime_error("High kernel cannot rebind");
    if(concurrent && low && high){
      check(cudaEventRecord(fork,s));check(cudaStreamWaitEvent(side,fork));
      low->run(s);high->run(side);check(cudaEventRecord(join,side));check(cudaStreamWaitEvent(s,join));
    }else {if(low)low->run(s);if(high)high->run(s);}
    check(cudaGetLastError());
  }
};

template<bool FP4,class Layout>
__global__ void pack_mx(__half const*input,uint8_t*output,uint8_t*scales,int rows,int k,Layout layout) {
  int lane=threadIdx.x%32;
  int group=(blockIdx.x*blockDim.x+threadIdx.x)/32;
  if(group>=rows*(k/32))return;
  int row=group/(k/32),col=(group%(k/32))*32+lane;
  float value=__half2float(input[int64_t(row)*k+col]),maximum=fabsf(value);
  for(int d=16;d;d/=2)maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,d));
  // Conservative RTN smoke quantizer: smallest power-of-two scale that avoids
  // overflowing E2M1 (6) / E4M3 (448). Calibration/clipping is a separate task.
  int exponent=maximum>0?int(ceilf(log2f(maximum/(FP4?6.0f:448.0f)))):0;
  exponent=max(-127,min(127,exponent));
  if(lane==0)scales[layout(row,col,0)]=uint8_t(exponent+127);
  float scaled=ldexpf(value,-exponent);
  if constexpr(FP4){
    uint32_t q=cutlass::float_e2m1_t(scaled).raw();
    uint32_t next=__shfl_down_sync(0xffffffff,q,1);
    if(!(lane&1))output[(int64_t(row)*k+col)/2]=uint8_t(q|(next<<4));
  }else output[int64_t(row)*k+col]=cutlass::float_e4m3_t(scaled).raw();
}

extern "C" {
char const* hg_last_error(){return last_error.c_str();}
int64_t hg_scale_size(int rows,int k){
  return cute::cosize(BS::tile_atom_to_shape_SFA(cute::make_shape(rows,1,k,1)));
}
int hg_pack(void const*in,void*out,void*sf,int rows,int k,int fp4,void*stream) {HG_TRY
  if(rows<=0 || k<=0 || k%128 || !in || !out || !sf)throw std::runtime_error("Invalid MX pack arguments");
  auto s=static_cast<cudaStream_t>(stream);
  auto layout=BS::tile_atom_to_shape_SFA(cute::make_shape(rows,1,k,1));
  auto b=BS::tile_atom_to_shape_SFB(cute::make_shape(1,rows,k,1));
  static_assert(std::is_same_v<decltype(layout),decltype(b)>);
  check(cudaMemsetAsync(sf,127,cute::cosize(layout),s));
  int blocks=(int64_t(rows)*k+127)/128;
  if(fp4)pack_mx<true><<<blocks,128,0,s>>>((__half const*)in,(uint8_t*)out,(uint8_t*)sf,rows,k,layout);
  else pack_mx<false><<<blocks,128,0,s>>>((__half const*)in,(uint8_t*)out,(uint8_t*)sf,rows,k,layout);
  check(cudaGetLastError());HG_CATCH}
int hg_create(int low_id,int high_id,DeviceProblem const*l,DeviceProblem const*h,void*s,void**handle){HG_TRY
  *handle=new Runner(low_id,high_id,*l,*h,static_cast<cudaStream_t>(s));HG_CATCH}
int hg_run(void*handle,DeviceProblem const*l,DeviceProblem const*h,void*s,int concurrent){HG_TRY
  static_cast<Runner*>(handle)->run(*l,*h,static_cast<cudaStream_t>(s),concurrent);HG_CATCH}
void hg_destroy(void*handle){delete static_cast<Runner*>(handle);}
}
