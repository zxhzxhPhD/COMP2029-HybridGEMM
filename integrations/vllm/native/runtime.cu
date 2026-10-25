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
namespace hygemm::oc {void run_gemv_pair(DeviceProblem const&,DeviceProblem const&,cudaStream_t,int);}
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
  void run(DeviceProblem const&l,DeviceProblem const&h,cudaStream_t s,int policy) {
    if(low && !low->update_operands(l))throw std::runtime_error("Low kernel cannot rebind");
    if(high && !high->update_operands(h))throw std::runtime_error("High kernel cannot rebind");
    if(policy>=4 && policy<=6){
      hygemm::oc::run_gemv_pair(l,h,s,1<<(policy-4));
    }else if((policy==1 || policy==3) && low && high){
      check(cudaEventRecord(fork,s));check(cudaStreamWaitEvent(side,fork));
      if(policy==1){low->run(s);high->run(side);}else{high->run(s);low->run(side);}
      check(cudaEventRecord(join,side));check(cudaStreamWaitEvent(s,join));
    }else if(policy==2){if(high)high->run(s);if(low)low->run(s);}
    else {if(low)low->run(s);if(high)high->run(s);}
    check(cudaGetLastError());
  }
};

template<bool FP4,bool InitPadding=false,class Layout>
__global__ void pack_mx(__half const*input,uint8_t*output,uint8_t*scales,int rows,int k,Layout layout) {
  int lane=threadIdx.x%32;
  int group=(blockIdx.x*blockDim.x+threadIdx.x)/32;
  if(group>=rows*(k/32))return;
  int row=group/(k/32),col=(group%(k/32))*32+lane;
  if constexpr(InitPadding){
    // Small activation batches occupy one 128-row scale block. The first
    // warp of each K128 block owns all *padding* scale words for that block.
    // Its coalesced 32-bit stores never overlap any valid-row scale byte,
    // so no inter-CTA barrier or separate memset node is necessary.
    if(row==0 && group%4==0){
      auto* words=reinterpret_cast<uint32_t*>(scales+layout(0,col-lane,0));
#pragma unroll
      for(int step=0;step<4;++step){
        int word=lane+step*32;
        int scale_row=(word%4)*32+word/4;
        if(scale_row>=rows)words[word]=0x7f7f7f7fu;
      }
    }
  }
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

// Four FP16 values per lane, eight lanes per K32 group. Scale rounding and
// FP8 conversion deliberately match pack_mx; only loads/reduction/packing differ.
template<class Layout>
__global__ void pack_mx_vector(__half const* input,uint8_t* output,uint8_t* scales,int rows,int k,Layout layout) {
  int tid=blockIdx.x*blockDim.x+threadIdx.x,lane=threadIdx.x%8;
  int group=tid/8;
  if(group>=rows*(k/32))return;
  int row=group/(k/32),col=(group%(k/32))*32+lane*4;
  // Last 128-row scale block: the first four groups jointly initialize
  // disjoint padding words. No valid scale byte overlaps these stores.
  if(row==(rows/128)*128 && rows%128 && group%4==0) {
    auto* words=reinterpret_cast<uint32_t*>(scales+layout(row,col-lane*4,0));
#pragma unroll
    for(int step=0;step<16;++step){
      int word=lane+step*8,r=(word%4)*32+word/4;
      if(r>=rows%128)words[word]=0x7f7f7f7fu;
    }
  }
  uint2 raw=*reinterpret_cast<uint2 const*>(input+int64_t(row)*k+col);
  float2 a=__half22float2(*reinterpret_cast<__half2*>(&raw.x));
  float2 b=__half22float2(*reinterpret_cast<__half2*>(&raw.y));
  float v[4]={a.x,a.y,b.x,b.y};
  float maximum=fmaxf(fmaxf(fabsf(a.x),fabsf(a.y)),fmaxf(fabsf(b.x),fabsf(b.y)));
  for(int d=4;d;d/=2)maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,d,8));
  int exponent=maximum>0?int(ceilf(log2f(maximum/448.f))):0;
  exponent=max(-127,min(127,exponent));
  if(lane==0)scales[layout(row,col,0)]=uint8_t(exponent+127);
  uint32_t packed=0;
#pragma unroll
  for(int i=0;i<4;++i)packed|=uint32_t(cutlass::float_e4m3_t(ldexpf(v[i],-exponent)).raw())<<(i*8);
  *reinterpret_cast<uint32_t*>(output+int64_t(row)*k+col)=packed;
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
  int blocks=(int64_t(rows)*k+127)/128;
  if(!fp4 && rows<=32){
    pack_mx<false,true><<<blocks,128,0,s>>>((__half const*)in,(uint8_t*)out,(uint8_t*)sf,rows,k,layout);
  }else{
    check(cudaMemsetAsync(sf,127,cute::cosize(layout),s));
    if(fp4)pack_mx<true><<<blocks,128,0,s>>>((__half const*)in,(uint8_t*)out,(uint8_t*)sf,rows,k,layout);
    else pack_mx<false><<<blocks,128,0,s>>>((__half const*)in,(uint8_t*)out,(uint8_t*)sf,rows,k,layout);
  }
  check(cudaGetLastError());HG_CATCH}
int hg_pack_mode(void const*in,void*out,void*sf,int rows,int k,int fp4,void*stream,int mode) {HG_TRY
  if(fp4 || mode==0)return hg_pack(in,out,sf,rows,k,fp4,stream);
  if(rows<=0||k<=0||k%128||!in||!out||!sf)throw std::runtime_error("Invalid MX pack arguments");
  auto s=static_cast<cudaStream_t>(stream);
  auto layout=BS::tile_atom_to_shape_SFA(cute::make_shape(rows,1,k,1));
  if(mode==1 && rows<=128) {
    pack_mx<false,true><<<(int64_t(rows)*k+127)/128,128,0,s>>>((__half const*)in,(uint8_t*)out,(uint8_t*)sf,rows,k,layout);
  }else if(mode==2){
    pack_mx_vector<<<(int64_t(rows)*k/4+127)/128,128,0,s>>>((__half const*)in,(uint8_t*)out,(uint8_t*)sf,rows,k,layout);
  }else return hg_pack(in,out,sf,rows,k,fp4,stream);
  check(cudaGetLastError());HG_CATCH}
int hg_create(int low_id,int high_id,DeviceProblem const*l,DeviceProblem const*h,void*s,void**handle){HG_TRY
  *handle=new Runner(low_id,high_id,*l,*h,static_cast<cudaStream_t>(s));HG_CATCH}
int hg_run(void*handle,DeviceProblem const*l,DeviceProblem const*h,void*s,int concurrent){HG_TRY
  static_cast<Runner*>(handle)->run(*l,*h,static_cast<cudaStream_t>(s),concurrent);HG_CATCH}
// Optional offline inspection API. It does not participate in forward calls.
int hg_info(void*handle,int*output){HG_TRY
  auto* runner=static_cast<Runner*>(handle);
  std::fill(output,output+20,-1);
  for(int part=0;part<2;++part){
    auto* kernel=(part ? runner->high.get() : runner->low.get());
    if(!kernel)continue;
    auto i=kernel->info();int* dst=output+part*10;
    int values[]={i.id,i.tile_m,i.tile_n,i.tile_k,i.stages,i.registers_per_thread,i.shared_bytes,i.local_bytes,int(i.fp8_weight),int(i.pingpong)};
    std::copy(values,values+10,dst);
  }
  HG_CATCH}
void hg_destroy(void*handle){delete static_cast<Runner*>(handle);}
}
