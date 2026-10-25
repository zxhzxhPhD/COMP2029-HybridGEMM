// Fixed-pointer diagnostic ABI for the previously validated TMA switch kernel.
// This probe is not a serving adapter: operands must stay alive and unchanged.
#include "output_channel_fused.h"
#include <string>
#include <stdexcept>
static thread_local std::string error;
extern "C" {
char const* hg_fused_error(){return error.c_str();}
int hg_fused_create(int id,hygemm::oc::DeviceProblem const*l,hygemm::oc::DeviceProblem const*h,void*stream,void**handle){
  try{*handle=hygemm::oc::make_fused_kernel(id,*l,*h,static_cast<cudaStream_t>(stream)).release();return 0;}
  catch(std::exception const&e){error=e.what();return -1;}
}
int hg_fused_run(void*handle,void*stream){
  try{static_cast<hygemm::oc::FusedKernel*>(handle)->run(static_cast<cudaStream_t>(stream));return 0;}
  catch(std::exception const&e){error=e.what();return -1;}
}
int hg_fused_update(void*handle,hygemm::oc::DeviceProblem const*l,hygemm::oc::DeviceProblem const*h){
  try{if(!static_cast<hygemm::oc::FusedKernel*>(handle)->update_problems(*l,*h))throw std::runtime_error("Fused rebind unsupported");return 0;}
  catch(std::exception const&e){error=e.what();return -1;}
}
int hg_fused_info(void*handle,int*data){
  auto v=static_cast<hygemm::oc::FusedKernel*>(handle)->fused_info();
  int values[]={v.kernel.id,v.kernel.tile_m,v.kernel.tile_n,v.kernel.tile_k,v.kernel.stages,
    v.kernel.registers_per_thread,v.kernel.shared_bytes,v.kernel.local_bytes,
    int(v.low_transaction_bytes),int(v.high_transaction_bytes),v.virtual_rows};
  for(int i=0;i<11;++i)data[i]=values[i];return 0;
}
void hg_fused_destroy(void*handle){delete static_cast<hygemm::oc::FusedKernel*>(handle);}
}
