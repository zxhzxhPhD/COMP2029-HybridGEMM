#include "output_channel_fused.h"
#include <stdexcept>
namespace hygemm::oc {
#define DECLARE(N) std::unique_ptr<FusedKernel> make_fused_variant_##N(DeviceProblem const&,DeviceProblem const&,cudaStream_t,uint32_t*);
DECLARE(0) DECLARE(1) DECLARE(2) DECLARE(3)
DECLARE(4) DECLARE(5) DECLARE(6) DECLARE(7)
std::unique_ptr<FusedKernel> make_fused_kernel(int id,DeviceProblem const&l,DeviceProblem const&h,cudaStream_t s,uint32_t*a) {
  switch(id) {
#define DISPATCH(N) case N:return make_fused_variant_##N(l,h,s,a);
    DISPATCH(0) DISPATCH(1) DISPATCH(2) DISPATCH(3)
    DISPATCH(4) DISPATCH(5) DISPATCH(6) DISPATCH(7)
    default:throw std::runtime_error("Invalid fused variant ID");
  }
}
}
