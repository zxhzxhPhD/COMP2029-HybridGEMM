// Directional comparison of prepacked MXFP4/MXFP8 GEMMs and cuBLASLt.
// M = output channels, N = tokens, K = input channels; Y = W X^T,
// with column-major Y physically equal to the conventional token-major output.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <cublasLt.h>
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

#define CUDA_OK(x) do { auto s=(x); if(s!=cudaSuccess) { std::cerr << #x << ": " << cudaGetErrorString(s) << '\n'; std::exit(1); } } while(0)
#define CUTLASS_OK(x) do { auto s=(x); if(s!=cutlass::Status::kSuccess) { std::cerr << #x << ": " << cutlassGetStatusString(s) << '\n'; std::exit(1); } } while(0)
#define BLAS_OK(x) do { auto s=(x); if(s!=CUBLAS_STATUS_SUCCESS) { std::cerr << #x << ": " << int(s) << '\n'; std::exit(1); } } while(0)

struct Options {
  int m=4096, n=128, k=4096, tile_n=64, iterations=100, repeats=3;
  std::string path="w4a8";
  bool full_check=false;
  Options(int argc, char const** argv) {
    cutlass::CommandLine cmd(argc,argv);
    cmd.get_cmd_line_argument("m",m); cmd.get_cmd_line_argument("n",n);
    cmd.get_cmd_line_argument("k",k); cmd.get_cmd_line_argument("tile-n",tile_n);
    cmd.get_cmd_line_argument("iterations",iterations); cmd.get_cmd_line_argument("repeats",repeats);
    cmd.get_cmd_line_argument("path",path); full_check=cmd.check_cmd_line_flag("full-check");
    if(m<=0 || n<=0 || k<=0 || m%128 || k%128 || iterations<=0 || repeats<=0 ||
       (tile_n!=16 && tile_n!=64 && tile_n!=128)) { std::cerr << "Invalid options\n"; std::exit(1); }
  }
};

uint32_t hash32(uint32_t x) {
  x ^= x>>16; x*=0x7feb352du; x^=x>>15; x*=0x846ca68bu; return x^(x>>16);
}
// Exactly representable signed FP4 values, with nonconstant K32 power-of-two scales.
// Every path computes the same mathematical matrix (this is NOT an accuracy study).
float raw_value(int row,int col,int seed) {
  static constexpr float v[]={-2,-1.5f,-1,-0.5f,0.5f,1,1.5f,2};
  return v[hash32(uint32_t(row)*0x9e3779b9u+uint32_t(col)+uint32_t(seed))&7];
}
float scale_value(int row,int group,int seed) {
  static constexpr float s[]={0.5f,1,2};
  return s[hash32(uint32_t(row)*0x85ebca6bu+uint32_t(group)+uint32_t(seed))%3];
}
float value(int row,int col,int seed) { return raw_value(row,col,seed)*scale_value(row,col/32,seed+71); }

template<class T> std::vector<uint8_t> input_bytes(int rows,int k,int seed,bool scaled) {
  constexpr int bits=cutlass::sizeof_bits<T>::value;
  std::vector<uint8_t> b((int64_t(rows)*k*bits+7)/8,0);
  auto p=cute::recast_ptr<T>(b.data());
  for(int r=0;r<rows;++r) for(int c=0;c<k;++c)
    p[int64_t(r)*k+c]=T(scaled ? value(r,c,seed) : raw_value(r,c,seed));
  return b;
}
template<class T> void upload(cutlass::device_memory::allocation<T>& dst, std::vector<T> const& src) {
  dst.reset(src.size()); CUDA_OK(cudaMemcpy(dst.get(),src.data(),src.size()*sizeof(T),cudaMemcpyHostToDevice));
}

struct Check { int count=0, failures=0; float max_abs=0; };
Check check_output(Options const& o, cutlass::half_t const* ptr) {
  std::vector<cutlass::half_t> d(int64_t(o.m)*o.n);
  CUDA_OK(cudaMemcpy(d.data(),ptr,d.size()*sizeof(d[0]),cudaMemcpyDeviceToHost));
  Check c; c.count=o.full_check ? int(d.size()) : std::min<int64_t>(256,d.size());
  for(int i=0;i<c.count;++i) {
    int64_t at=o.full_check ? i : int64_t(hash32(i+3407))%d.size();
    if(!o.full_check && i==0) at=0;
    if(!o.full_check && i==1) at=d.size()-1;
    int r=at%o.m, n=at/o.m;
    float ref=0;
    for(int k=0;k<o.k;++k) ref+=value(r,k,17)*value(n,k,29);
    float expected=float(cutlass::half_t(ref)), got=float(d[at]);
    float error=std::abs(expected-got);
    c.max_abs=std::max(c.max_abs,error);
    c.failures+=(!std::isfinite(got) || error!=0);
  }
  return c;
}

// CUDA graph avoids measuring CPU submission gaps for microsecond kernels.
template<class F> std::vector<float> measure(F launch,cudaStream_t stream,int nodes,int repeats,int warm_ms=80) {
  launch(); CUDA_OK(cudaStreamSynchronize(stream));
  cudaGraph_t graph; cudaGraphExec_t executable;
  CUDA_OK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
  for(int i=0;i<nodes;++i) launch();
  CUDA_OK(cudaStreamEndCapture(stream,&graph));
  CUDA_OK(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
  auto t0=std::chrono::steady_clock::now();
  do {
    CUDA_OK(cudaGraphLaunch(executable,stream)); CUDA_OK(cudaStreamSynchronize(stream));
  } while(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count()<warm_ms);
  cudaEvent_t start,stop; CUDA_OK(cudaEventCreate(&start)); CUDA_OK(cudaEventCreate(&stop));
  std::vector<float> times;
  for(int i=0;i<repeats;++i) {
    CUDA_OK(cudaEventRecord(start,stream)); CUDA_OK(cudaGraphLaunch(executable,stream));
    CUDA_OK(cudaEventRecord(stop,stream)); CUDA_OK(cudaEventSynchronize(stop));
    float ms; CUDA_OK(cudaEventElapsedTime(&ms,start,stop)); times.push_back(ms*1000/nodes);
  }
  CUDA_OK(cudaEventDestroy(start)); CUDA_OK(cudaEventDestroy(stop));
  CUDA_OK(cudaGraphExecDestroy(executable)); CUDA_OK(cudaGraphDestroy(graph));
  return times;
}
float median(std::vector<float> t) { std::sort(t.begin(),t.end()); return t[t.size()/2]; }

void result(Options const&o,Check c,std::vector<float> const&t,std::string extra) {
  float us=median(t);
  std::cout << std::setprecision(9) << "{\"path\":\"" << o.path << "\",\"m_out\":" << o.m
    << ",\"n_tokens\":" << o.n << ",\"k_in\":" << o.k << ",\"median_us\":" << us
    << ",\"tflops\":" << (2.0*o.m*o.n*o.k/us/1e6) << ",\"checked\":" << c.count
    << ",\"failures\":" << c.failures << ",\"max_abs_error\":" << c.max_abs << ",\"samples_us\":[";
  for(size_t i=0;i<t.size();++i) std::cout << (i ? "," : "") << t[i];
  std::cout << "]," << extra << "}" << std::endl;
  if(c.failures) std::exit(2);
}

template<class EA,class EB,int TN,bool Pingpong=false> struct Config {
  using Tile=Shape<_128,Int<TN>,_128>; using Cluster=Shape<_1,_1,_1>;
  using Epilogue=typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm120,cutlass::arch::OpClassTensorOp,Tile,Cluster,
    cutlass::epilogue::collective::EpilogueTileAuto,float,float,
    cutlass::half_t,cutlass::layout::ColumnMajor,8,cutlass::half_t,cutlass::layout::ColumnMajor,8,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
  using Mainloop=typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm120,cutlass::arch::OpClassBlockScaledTensorOp,
    EA,cutlass::layout::RowMajor,128/cutlass::sizeof_bits<typename EA::DataType>::value,
    EB,cutlass::layout::ColumnMajor,128/cutlass::sizeof_bits<typename EB::DataType>::value,
    float,Tile,Cluster,
    cutlass::gemm::collective::StageCountAutoCarveout<int(sizeof(typename Epilogue::SharedStorage))>,
    cute::conditional_t<Pingpong,cutlass::gemm::KernelTmaWarpSpecializedPingpong,
      cutlass::gemm::collective::KernelScheduleAuto>>::CollectiveOp;
  using Kernel=cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>,Mainloop,Epilogue>;
  using Gemm=cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

template<class EA,class EB,int TN,bool Pingpong=false> void run_cutlass(Options const&o,cudaStream_t stream) {
  using Cfg=Config<EA,EB,TN,Pingpong>; using Gemm=typename Cfg::Gemm;
  using Kernel=typename Cfg::Kernel; using ML=typename Cfg::Mainloop;
  using BS=typename ML::Sm1xxBlkScaledConfig;
  auto sa=cutlass::make_cute_packed_stride(typename Kernel::StrideA{},{o.m,o.k,1});
  auto sb=cutlass::make_cute_packed_stride(typename Kernel::StrideB{},{o.n,o.k,1});
  auto sc=cutlass::make_cute_packed_stride(typename Kernel::StrideC{},{o.m,o.n,1});
  auto sd=cutlass::make_cute_packed_stride(typename Kernel::StrideD{},{o.m,o.n,1});
  auto shape=make_shape(o.m,o.n,o.k,1);
  auto la=BS::tile_atom_to_shape_SFA(shape), lb=BS::tile_atom_to_shape_SFB(shape);
  std::vector<cutlass::float_ue8m0_t> sfa(cosize(la),cutlass::float_ue8m0_t(1.0f));
  std::vector<cutlass::float_ue8m0_t> sfb(cosize(lb),cutlass::float_ue8m0_t(1.0f));
  auto ta=make_tensor(sfa.data(),la), tb=make_tensor(sfb.data(),lb);
  for(int r=0;r<o.m;++r) for(int k=0;k<o.k;k+=32) ta(r,k,0)=cutlass::float_ue8m0_t(scale_value(r,k/32,88));
  for(int r=0;r<o.n;++r) for(int k=0;k<o.k;k+=32) tb(r,k,0)=cutlass::float_ue8m0_t(scale_value(r,k/32,100));
  cutlass::device_memory::allocation<uint8_t> a,b;
  cutlass::device_memory::allocation<cutlass::float_ue8m0_t> da,db;
  upload(a,input_bytes<typename EA::DataType>(o.m,o.k,17,false));
  upload(b,input_bytes<typename EB::DataType>(o.n,o.k,29,false)); upload(da,sfa); upload(db,sfb);
  cutlass::device_memory::allocation<cutlass::half_t> d(int64_t(o.m)*o.n);
  CUDA_OK(cudaMemset(d.get(),0,d.size()*sizeof(cutlass::half_t)));
  typename Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,{o.m,o.n,o.k,1},
    {reinterpret_cast<typename EA::DataType*>(a.get()),sa,reinterpret_cast<typename EB::DataType*>(b.get()),sb,da.get(),la,db.get(),lb},
    {{1.0f,0.0f},d.get(),sc,d.get(),sd}};
  Gemm gemm; CUTLASS_OK(gemm.can_implement(args));
  cutlass::device_memory::allocation<uint8_t> ws(Gemm::get_workspace_size(args));
  CUTLASS_OK(gemm.initialize(args,ws.get(),stream));
  auto launch=[&] { CUTLASS_OK(gemm.run(stream)); };
  launch(); CUDA_OK(cudaStreamSynchronize(stream)); auto check=check_output(o,d.get());
  auto times=measure(launch,stream,o.iterations,o.repeats);
  cudaFuncAttributes attr; CUDA_OK(cudaFuncGetAttributes(&attr,cutlass::device_kernel<Kernel>));
  using Atom=typename ML::TiledMma::AtomShape_MNK;
  std::string extra="\"backend\":\"cutlass_mx\",\"tile\":[128,"+std::to_string(TN)+",128]"+
    ",\"schedule\":\""+(Pingpong ? "pingpong" : "cooperative")+"\""+
    ",\"mma_k\":"+std::to_string(int(size<2>(Atom{})))+
    ",\"stages\":"+std::to_string(ML::DispatchPolicy::Stages)+
    ",\"registers_per_thread\":"+std::to_string(attr.numRegs)+
    ",\"shared_bytes\":"+std::to_string(sizeof(typename Kernel::SharedStorage))+
    ",\"local_bytes\":"+std::to_string(attr.localSizeBytes)+
    ",\"smem_operand_bits\":["+std::to_string(cutlass::sizeof_bits<typename ML::SmemAllocTypeA>::value)+","+
    std::to_string(cutlass::sizeof_bits<typename ML::SmemAllocTypeB>::value)+"]";
  result(o,check,times,extra);
}

template<class T> void run_lt(Options const&o,cudaStream_t stream,cudaDataType_t type) {
  cutlass::device_memory::allocation<uint8_t> a,b;
  upload(a,input_bytes<T>(o.m,o.k,17,true)); upload(b,input_bytes<T>(o.n,o.k,29,true));
  cutlass::device_memory::allocation<cutlass::half_t> d(int64_t(o.m)*o.n);
  constexpr size_t ws_bytes=32*1024*1024;
  cutlass::device_memory::allocation<uint8_t> ws(ws_bytes);
  CUDA_OK(cudaMemset(d.get(),0,d.size()*sizeof(cutlass::half_t)));
  cublasLtHandle_t handle; BLAS_OK(cublasLtCreate(&handle));
  cublasLtMatmulDesc_t desc; BLAS_OK(cublasLtMatmulDescCreate(&desc,CUBLAS_COMPUTE_32F,CUDA_R_32F));
  cublasOperation_t op_a=CUBLAS_OP_T,op_b=CUBLAS_OP_N;
  BLAS_OK(cublasLtMatmulDescSetAttribute(desc,CUBLASLT_MATMUL_DESC_TRANSA,&op_a,sizeof(op_a)));
  BLAS_OK(cublasLtMatmulDescSetAttribute(desc,CUBLASLT_MATMUL_DESC_TRANSB,&op_b,sizeof(op_b)));
  cublasLtMatrixLayout_t al,bl,dl;
  BLAS_OK(cublasLtMatrixLayoutCreate(&al,type,o.k,o.m,o.k));
  BLAS_OK(cublasLtMatrixLayoutCreate(&bl,type,o.k,o.n,o.k));
  BLAS_OK(cublasLtMatrixLayoutCreate(&dl,CUDA_R_16F,o.m,o.n,o.m));
  cublasLtMatmulPreference_t pref; BLAS_OK(cublasLtMatmulPreferenceCreate(&pref));
  BLAS_OK(cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&ws_bytes,sizeof(ws_bytes)));
  // Default heuristics may choose FP16 intermediate split-K reductions even
  // with FP32 compute. Keep any split-K partial results in the compute type.
  uint32_t reduction_mask=CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE;
  BLAS_OK(cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK,
    &reduction_mask,sizeof(reduction_mask)));
  cublasLtMatmulHeuristicResult_t candidates[8]; int count=0;
  BLAS_OK(cublasLtMatmulAlgoGetHeuristic(handle,desc,al,bl,dl,dl,pref,8,candidates,&count));
  if(!count) { std::cerr << "No cuBLASLt algorithm\n"; std::exit(1); }
  float alpha=1,beta=0;
  auto launch_index=[&](int i) {
    BLAS_OK(cublasLtMatmul(handle,desc,&alpha,a.get(),al,b.get(),bl,&beta,d.get(),dl,d.get(),dl,
      &candidates[i].algo,ws.get(),ws_bytes,stream));
  };
  int best=-1; float best_time=1e30f; int checked_algos=0;
  for(int i=0;i<count;++i) if(candidates[i].state==CUBLAS_STATUS_SUCCESS) {
    auto t=measure([&]{launch_index(i);},stream,20,1,10); ++checked_algos;
    if(t[0]<best_time) { best=i; best_time=t[0]; }
  }
  if(best<0) { std::cerr << "No usable cuBLASLt algorithm\n"; std::exit(1); }
  launch_index(best); CUDA_OK(cudaStreamSynchronize(stream)); auto check=check_output(o,d.get());
  auto times=measure([&]{launch_index(best);},stream,o.iterations,o.repeats);
  int id=-1; size_t written=0;
  BLAS_OK(cublasLtMatmulAlgoConfigGetAttribute(&candidates[best].algo,CUBLASLT_ALGO_CONFIG_ID,&id,sizeof(id),&written));
  uint32_t reduction=0; int splitk=0;
  BLAS_OK(cublasLtMatmulAlgoConfigGetAttribute(&candidates[best].algo,CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,&reduction,sizeof(reduction),&written));
  BLAS_OK(cublasLtMatmulAlgoConfigGetAttribute(&candidates[best].algo,CUBLASLT_ALGO_CONFIG_SPLITK_NUM,&splitk,sizeof(splitk),&written));
  result(o,check,times,"\"backend\":\"cublasLt\",\"algo_id\":"+std::to_string(id)+
    ",\"heuristic_index\":"+std::to_string(best)+",\"candidates_timed\":"+std::to_string(checked_algos)+
    ",\"splitk\":"+std::to_string(splitk)+",\"reduction_scheme\":"+std::to_string(reduction)+
    ",\"workspace_bytes\":"+std::to_string(candidates[best].workspaceSize));
  BLAS_OK(cublasLtMatmulPreferenceDestroy(pref)); BLAS_OK(cublasLtMatrixLayoutDestroy(al));
  BLAS_OK(cublasLtMatrixLayoutDestroy(bl)); BLAS_OK(cublasLtMatrixLayoutDestroy(dl));
  BLAS_OK(cublasLtMatmulDescDestroy(desc)); BLAS_OK(cublasLtDestroy(handle));
}

template<class EA,class EB> void dispatch(Options const&o,cudaStream_t stream) {
#ifdef HYGEMM_TUNED_VARIANTS
  if(o.tile_n==16) run_cutlass<EA,EB,16>(o,stream);
  else if(o.tile_n==64) run_cutlass<EA,EB,64,true>(o,stream);
  else { std::cerr << "Tuned binary accepts tile-n=16 (cooperative) or 64 (pingpong)\n"; std::exit(1); }
#else
  if(o.tile_n==64) run_cutlass<EA,EB,64>(o,stream);
  else if(o.tile_n==128) run_cutlass<EA,EB,128>(o,stream);
  else { std::cerr << "Base binary accepts tile-n=64 or 128 (cooperative)\n"; std::exit(1); }
#endif
}

int main(int argc,char const**argv) {
  Options o(argc,argv); cudaDeviceProp p; CUDA_OK(cudaGetDeviceProperties(&p,0));
  if(p.major!=12) { std::cerr << "Requires SM12x\n"; return 1; }
  cudaStream_t stream; CUDA_OK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
  using F4=cutlass::mx_float4_t<cutlass::float_e2m1_t>;
  using F8=cutlass::mx_float8_t<cutlass::float_e4m3_t>;
  if(o.path=="w4a4") {
    dispatch<F4,F4>(o,stream);
  } else if(o.path=="w4a8") {
    dispatch<F4,F8>(o,stream);
  } else if(o.path=="w8a8_mx") {
    dispatch<F8,F8>(o,stream);
  } else if(o.path=="fp16") run_lt<cutlass::half_t>(o,stream,CUDA_R_16F);
  else if(o.path=="fp8_lt") run_lt<cutlass::float_e4m3_t>(o,stream,CUDA_R_8F_E4M3);
  else { std::cerr << "Unknown path\n"; return 1; }
  CUDA_OK(cudaStreamDestroy(stream));
}
