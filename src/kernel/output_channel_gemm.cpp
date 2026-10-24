#include "output_channel_gemm.h"
#include <stdexcept>
namespace hygemm::oc {
#define DECLARE(N) std::unique_ptr<Kernel> make_variant_##N(bool,DeviceProblem const&,cudaStream_t);
DECLARE(0) DECLARE(1) DECLARE(2) DECLARE(3)
DECLARE(4) DECLARE(5) DECLARE(6) DECLARE(7)
std::unique_ptr<Kernel> make_light(int,bool,DeviceProblem const&,cudaStream_t);
std::unique_ptr<Kernel> make_kernel(int variant,bool fp8,DeviceProblem const&p,cudaStream_t stream) {
  switch(variant) {
#define DISPATCH(N) case N:return make_variant_##N(fp8,p,stream);
    DISPATCH(0) DISPATCH(1) DISPATCH(2) DISPATCH(3)
    DISPATCH(4) DISPATCH(5) DISPATCH(6) DISPATCH(7)
    default:
      if (variant >= 8 && variant < variant_count) return make_light(variant,fp8,p,stream);
      throw std::runtime_error("Invalid output-channel variant ID");
  }
}
} // namespace hygemm::oc
