#include "output_channel_gemm.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>
#include <cublasLt.h>
#include "cute/tensor.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"

using namespace cute;
using namespace hygemm::oc;
#define CUDA(x) do { auto err=(x); if(err!=cudaSuccess) throw std::runtime_error(std::string(#x)+": "+cudaGetErrorString(err)); } while(0)
#define BLAS(x) do { auto err=(x); if(err!=CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::string(#x)+": "+std::to_string(int(err))); } while(0)
using Half=cutlass::half_t;
using SF=cutlass::float_ue8m0_t;
template<class T> using Buffer=cutlass::device_memory::allocation<T>;

struct Options {
  int outputs=4096,tokens=128,k=4096,high=-1,iterations=100,repeats=3,top=3;
  int low_id=-1,high_id=-1;
  double fraction=.10;
  std::string mapping="interleaved",policy="concurrent";
  bool full=false,verify_all=false,baseline=true;
  Options(int argc,char const**argv) {
    cutlass::CommandLine c(argc,argv);
    c.get_cmd_line_argument("out",outputs); c.get_cmd_line_argument("tokens",tokens); c.get_cmd_line_argument("k",k);
    c.get_cmd_line_argument("high",high); c.get_cmd_line_argument("fraction",fraction);
    c.get_cmd_line_argument("iterations",iterations); c.get_cmd_line_argument("repeats",repeats); c.get_cmd_line_argument("top",top);
    c.get_cmd_line_argument("low-id",low_id); c.get_cmd_line_argument("high-id",high_id);
    c.get_cmd_line_argument("mapping",mapping); c.get_cmd_line_argument("policy",policy);
    full=c.check_cmd_line_flag("full-check"); verify_all=c.check_cmd_line_flag("verify-all"); baseline=!c.check_cmd_line_flag("no-baseline");
    if(!std::isfinite(fraction) || fraction<0 || fraction>1) throw std::runtime_error("Invalid fraction");
    if(high==-1) high=int(std::llround(outputs*fraction));
    if(outputs<=0 || tokens<=0 || k<=0 || k%128 || high<0 || high>outputs || iterations<=0 || repeats<=0 || top<1 || top>8 ||
       low_id< -1 || low_id>=8 || high_id< -1 || high_id>=8 ||
       (mapping!="interleaved" && mapping!="random" && mapping!="contiguous") ||
       (policy!="serial" && policy!="concurrent")) throw std::runtime_error("Invalid options");
  }
};
uint32_t hash32(uint32_t x) { x^=x>>16; x*=0x7feb352du; x^=x>>15; x*=0x846ca68bu; return x^(x>>16); }
float raw_weight(int row,int k,bool fp8) {
  uint32_t h=hash32(uint32_t(row)*0x9e3779b9u+uint32_t(k)+17);
  constexpr float f4[]={-2,-1.5f,-1,-.5f,.5f,1,1.5f,2};
  return fp8 ? float(int(h%31)-15)/8 : f4[h&7];
}
float raw_activation(int token,int k) { return float(int(hash32(uint32_t(token)*0x85ebca6bu+k+29)%31)-15)/8; }
float scale(int row,int group,int seed) {
  constexpr float v[]={.5f,1,2};return v[hash32(uint32_t(row)*0x9e3779b9u+group+seed)%3];
}
template<class T> void upload(Buffer<T>& b,std::vector<T> const& v) {
  b.reset(v.size()); if(!v.empty()) CUDA(cudaMemcpy(b.get(),v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));
}
float median(std::vector<float> v) { std::sort(v.begin(),v.end());return v[v.size()/2]; }

struct Inputs {
  Options o;
  std::vector<int32_t> low,high;
  std::vector<bool> is_high;
  Buffer<uint8_t> w4,w8;
  Buffer<cutlass::float_e4m3_t> activation;
  Buffer<SF> sf4,sf8,sfa;
  Buffer<int32_t> ids4,ids8;
  Buffer<Half> guarded_output;
  static constexpr int guard=256;
  Half* output() { return guarded_output.get()+guard; }
  Inputs(Options const& options):o(options),is_high(o.outputs,false),guarded_output(int64_t(o.outputs)*o.tokens+2*guard) {
    if(o.mapping=="contiguous") for(int i=o.outputs-o.high;i<o.outputs;++i)is_high[i]=true;
    else for(int i=0;i<o.outputs;++i)
      is_high[i]=(int64_t(i+1)*o.high/o.outputs != int64_t(i)*o.high/o.outputs);
    for(int i=0;i<o.outputs;++i)(is_high[i]?high:low).push_back(i);
    if(o.mapping=="random") {
      // Random membership AND partition order exercise the general scatter contract.
      std::vector<int32_t> perm(o.outputs);std::iota(perm.begin(),perm.end(),0);
      std::mt19937 rng(20261024);std::shuffle(perm.begin(),perm.end(),rng);
      high.assign(perm.begin(),perm.begin()+o.high);low.assign(perm.begin()+o.high,perm.end());
      std::fill(is_high.begin(),is_high.end(),false);for(int x:high)is_high[x]=true;
    }
    upload(ids4,low);upload(ids8,high);
    pack_weights<cutlass::float_e2m1_t>(low,false,w4,sf4);
    pack_weights<cutlass::float_e4m3_t>(high,true,w8,sf8);
    std::vector<cutlass::float_e4m3_t> x(int64_t(o.tokens)*o.k);
    for(int t=0;t<o.tokens;++t)for(int k=0;k<o.k;++k)x[int64_t(t)*o.k+k]=cutlass::float_e4m3_t(raw_activation(t,k));
    upload(activation,x);
    using BS=cutlass::detail::Sm1xxBlockScaledConfig<32>;
    auto layout=BS::tile_atom_to_shape_SFB(make_shape(1,o.tokens,o.k,1));
    std::vector<SF> scales(cosize(layout),SF(1.0f));auto tensor=make_tensor(scales.data(),layout);
    for(int t=0;t<o.tokens;++t)for(int k=0;k<o.k;k+=32)tensor(t,k,0)=SF(scale(t,k/32,100));
    upload(sfa,scales);reset();
  }
  template<class T> void pack_weights(std::vector<int32_t> const& ids,bool fp8,Buffer<uint8_t>& w,Buffer<SF>& sf) {
    if(ids.empty())return;
    std::vector<uint8_t> bytes((int64_t(ids.size())*o.k*cutlass::sizeof_bits<T>::value+7)/8);
    auto ptr=cute::recast_ptr<T>(bytes.data());
    for(size_t r=0;r<ids.size();++r)for(int k=0;k<o.k;++k)ptr[int64_t(r)*o.k+k]=T(raw_weight(ids[r],k,fp8));
    upload(w,bytes);
    using BS=cutlass::detail::Sm1xxBlockScaledConfig<32>;
    auto layout=BS::tile_atom_to_shape_SFA(make_shape(int(ids.size()),o.tokens,o.k,1));
    std::vector<SF> scales(cosize(layout),SF(1.0f));auto tensor=make_tensor(scales.data(),layout);
    for(int r=0;r<int(ids.size());++r)for(int k=0;k<o.k;k+=32)tensor(r,k,0)=SF(scale(ids[r],k/32,88));
    upload(sf,scales);
  }
  DeviceProblem problem(bool fp8) {
    return {int(fp8?high.size():low.size()),o.tokens,o.k,o.outputs,
      fp8?w8.get():w4.get(),activation.get(),fp8?sf8.get():sf4.get(),sfa.get(),
      fp8?ids8.get():ids4.get(),output()};
  }
  void reset() {
    CUDA(cudaMemset(guarded_output.get(),0x5a,guarded_output.size()*sizeof(Half)));
    CUDA(cudaMemset(output(),0xff,int64_t(o.outputs)*o.tokens*sizeof(Half)));
    // Kernels use nonblocking streams; explicitly finish default-stream fills.
    CUDA(cudaDeviceSynchronize());
  }
  float reference(int out,int token) const {
    float sum=0;
    for(int k=0;k<o.k;++k)sum+=raw_weight(out,k,is_high[out])*scale(out,k/32,88)*raw_activation(token,k)*scale(token,k/32,100);
    return float(Half(sum));
  }
  int check() {
    std::vector<Half> host(guarded_output.size());CUDA(cudaMemcpy(host.data(),guarded_output.get(),host.size()*sizeof(Half),cudaMemcpyDeviceToHost));
    auto bits=reinterpret_cast<uint16_t const*>(host.data());
    for(int i=0;i<guard;++i)if(bits[i]!=0x5a5a || bits[host.size()-1-i]!=0x5a5a)throw std::runtime_error("Output guard overwritten");
    int64_t count=int64_t(o.outputs)*o.tokens;
    for(int64_t i=0;i<count;++i)if(!std::isfinite(float(host[guard+i])))throw std::runtime_error("Unwritten/nonfinite output at "+std::to_string(i));
    int checked=o.full?int(count):int(std::min<int64_t>(512,count));
    for(int i=0;i<checked;++i) {
      int64_t at=o.full?i:int64_t(hash32(i+3407))%count;
      if(!o.full && i==0)at=0;if(!o.full && i==1)at=count-1;
      // Explicitly sample both precision partitions, including their tail rows.
      if(!o.full && i>=2 && i<6) {
        auto const& ids=i<4?low:high;
        if(!ids.empty())at=int64_t(i%o.tokens)*o.outputs+(i%2?ids.back():ids.front());
      }
      float ref=reference(at%o.outputs,at/o.outputs),got=float(host[guard+at]);
      if(ref!=got)throw std::runtime_error("Mismatch at "+std::to_string(at)+": got "+std::to_string(got)+" expected "+std::to_string(ref));
    }
    return checked;
  }
};

class Graph {
  cudaGraph_t graph_{};cudaGraphExec_t exec_{};
 public:
  int kernels=0,nodes;
  Graph(std::function<void()> const& launch,cudaStream_t stream,int count):nodes(count) {
    launch();CUDA(cudaStreamSynchronize(stream));
    CUDA(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
    for(int i=0;i<count;++i)launch();
    CUDA(cudaStreamEndCapture(stream,&graph_));
    size_t n=0;CUDA(cudaGraphGetNodes(graph_,nullptr,&n));std::vector<cudaGraphNode_t> all(n);
    CUDA(cudaGraphGetNodes(graph_,all.data(),&n));
    for(auto node:all){cudaGraphNodeType type;CUDA(cudaGraphNodeGetType(node,&type));kernels+=(type==cudaGraphNodeTypeKernel);}
    CUDA(cudaGraphInstantiate(&exec_,graph_,nullptr,nullptr,0));
  }
  ~Graph(){if(exec_)cudaGraphExecDestroy(exec_);if(graph_)cudaGraphDestroy(graph_);}
  std::vector<float> measure(cudaStream_t stream,int repeats,int warm_ms) {
    auto t0=std::chrono::steady_clock::now();
    do {CUDA(cudaGraphLaunch(exec_,stream));CUDA(cudaStreamSynchronize(stream));}
    while(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count()<warm_ms);
    cudaEvent_t start,stop;CUDA(cudaEventCreate(&start));CUDA(cudaEventCreate(&stop));std::vector<float> out;
    for(int i=0;i<repeats;++i){CUDA(cudaEventRecord(start,stream));CUDA(cudaGraphLaunch(exec_,stream));
      CUDA(cudaEventRecord(stop,stream));CUDA(cudaEventSynchronize(stop));float ms;CUDA(cudaEventElapsedTime(&ms,start,stop));out.push_back(ms*1000/nodes);}
    CUDA(cudaEventDestroy(start));CUDA(cudaEventDestroy(stop));return out;
  }
};
struct Streams {
  cudaStream_t main,side;cudaEvent_t fork,join;
  Streams(){CUDA(cudaStreamCreateWithFlags(&main,cudaStreamNonBlocking));CUDA(cudaStreamCreateWithFlags(&side,cudaStreamNonBlocking));
    CUDA(cudaEventCreateWithFlags(&fork,cudaEventDisableTiming));CUDA(cudaEventCreateWithFlags(&join,cudaEventDisableTiming));}
  ~Streams(){cudaStreamDestroy(main);cudaStreamDestroy(side);cudaEventDestroy(fork);cudaEventDestroy(join);}
  void launch(Kernel* low,Kernel* high,bool concurrent) {
    if(concurrent && low && high){
      CUDA(cudaEventRecord(fork,main));CUDA(cudaStreamWaitEvent(side,fork,0));
      low->run(main);high->run(side);CUDA(cudaEventRecord(join,side));CUDA(cudaStreamWaitEvent(main,join,0));
    }else{if(low)low->run(main);if(high)high->run(main);}
  }
};
void print_info(KernelInfo x) {
  std::cout<<"{\"id\":"<<x.id<<",\"tile\":["<<x.tile_m<<','<<x.tile_n<<','<<x.tile_k<<"],\"stages\":"<<x.stages
    <<",\"registers\":"<<x.registers_per_thread<<",\"shared_bytes\":"<<x.shared_bytes<<",\"local_bytes\":"<<x.local_bytes
    <<",\"pingpong\":"<<(x.pingpong?"true":"false")<<'}';
}
void print_times(std::vector<float> const& t) {std::cout<<'[';for(size_t i=0;i<t.size();++i)std::cout<<(i?",":"")<<t[i];std::cout<<']';}

// Exact dequantized-input FP16 baseline. Heuristic search is outside timing.
float fp16_baseline(Inputs& d,Streams& streams) {
  auto const&o=d.o;std::vector<Half> w(int64_t(o.outputs)*o.k),x(int64_t(o.tokens)*o.k);
  for(int r=0;r<o.outputs;++r)for(int k=0;k<o.k;++k)w[int64_t(r)*o.k+k]=Half(raw_weight(r,k,d.is_high[r])*scale(r,k/32,88));
  for(int t=0;t<o.tokens;++t)for(int k=0;k<o.k;++k)x[int64_t(t)*o.k+k]=Half(raw_activation(t,k)*scale(t,k/32,100));
  Buffer<Half> dw,dx;upload(dw,w);upload(dx,x);constexpr size_t ws_size=32*1024*1024;Buffer<uint8_t> ws(ws_size);
  cublasLtHandle_t h;BLAS(cublasLtCreate(&h));cublasLtMatmulDesc_t desc;
  BLAS(cublasLtMatmulDescCreate(&desc,CUBLAS_COMPUTE_32F,CUDA_R_32F));cublasOperation_t op=CUBLAS_OP_T;
  BLAS(cublasLtMatmulDescSetAttribute(desc,CUBLASLT_MATMUL_DESC_TRANSA,&op,sizeof(op)));
  cublasLtMatrixLayout_t a,b,c;BLAS(cublasLtMatrixLayoutCreate(&a,CUDA_R_16F,o.k,o.outputs,o.k));
  BLAS(cublasLtMatrixLayoutCreate(&b,CUDA_R_16F,o.k,o.tokens,o.k));BLAS(cublasLtMatrixLayoutCreate(&c,CUDA_R_16F,o.outputs,o.tokens,o.outputs));
  cublasLtMatmulPreference_t pref;BLAS(cublasLtMatmulPreferenceCreate(&pref));
  BLAS(cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&ws_size,sizeof(ws_size)));
  uint32_t mask=CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE;
  BLAS(cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK,&mask,sizeof(mask)));
  cublasLtMatmulHeuristicResult_t algos[8];int n=0;BLAS(cublasLtMatmulAlgoGetHeuristic(h,desc,a,b,c,c,pref,8,algos,&n));
  if(!n)throw std::runtime_error("No FP16 algorithm");float alpha=1,beta=0;
  auto launch=[&](int id){BLAS(cublasLtMatmul(h,desc,&alpha,dw.get(),a,dx.get(),b,&beta,d.output(),c,d.output(),c,&algos[id].algo,ws.get(),ws_size,streams.main));};
  int best=-1;float best_us=1e30;
  for(int i=0;i<n;++i)if(algos[i].state==CUBLAS_STATUS_SUCCESS){Graph graph([&]{launch(i);},streams.main,20);
    float us=median(graph.measure(streams.main,3,5));if(us<best_us){best=i;best_us=us;}}
  if(best<0)throw std::runtime_error("No usable FP16 algorithm");
  d.reset();launch(best);CUDA(cudaStreamSynchronize(streams.main));d.check();
  Graph graph([&]{launch(best);},streams.main,o.iterations);auto times=graph.measure(streams.main,o.repeats,60);float us=median(times);
  std::cout<<"{\"kind\":\"fp16\",\"median_us\":"<<us<<",\"samples_us\":";print_times(times);std::cout<<"}\n";
  BLAS(cublasLtMatmulPreferenceDestroy(pref));BLAS(cublasLtMatrixLayoutDestroy(a));BLAS(cublasLtMatrixLayoutDestroy(b));BLAS(cublasLtMatrixLayoutDestroy(c));
  BLAS(cublasLtMatmulDescDestroy(desc));BLAS(cublasLtDestroy(h));return us;
}

#ifndef HYGEMM_OC_BENCH_UTILS_ONLY
int main(int argc,char const**argv) try {
  std::cout<<std::setprecision(9);Options o(argc,argv);cudaDeviceProp prop;CUDA(cudaGetDeviceProperties(&prop,0));
  if(prop.major!=12)throw std::runtime_error("SM12x required");Streams streams;Inputs inputs(o);
  std::vector<std::unique_ptr<Kernel>> low(8),high(8);
  std::vector<std::pair<float,int>> rank_low,rank_high;
  for(int id=0;id<8;++id) {
    if(!inputs.low.empty())low[id]=make_kernel(id,false,inputs.problem(false),streams.main);
    if(!inputs.high.empty())high[id]=make_kernel(id,true,inputs.problem(true),streams.main);
    if((low[id] && low[id]->info().id!=id) || (high[id] && high[id]->info().id!=id))
      throw std::runtime_error("Variant dispatch mismatch");
  }
  if(o.verify_all)for(int id=0;id<8;++id)for(int p=0;p<2;++p){
    if((p==0 && inputs.low.empty()) || (p==1 && inputs.high.empty()))continue;
    inputs.reset();streams.launch(low[p==0?id:0].get(),high[p==1?id:0].get(),p==1);
    CUDA(cudaStreamSynchronize(streams.main));inputs.check();
  }
  bool tune=(o.low_id<0 && !inputs.low.empty()) || (o.high_id<0 && !inputs.high.empty());
  if(tune)for(int id=0;id<8;++id)for(int p=0;p<2;++p){
    auto* kernel=p?high[id].get():low[id].get();if(!kernel)continue;
    Graph graph([&]{kernel->run(streams.main);},streams.main,32);auto times=graph.measure(streams.main,3,8);float us=median(times);
    (p?rank_high:rank_low).push_back({us,id});
    std::cout<<"{\"kind\":\"single_candidate\",\"path\":\""<<(p?"w8a8":"w4a8")<<"\",\"median_us\":"<<us<<",\"config\":";
    print_info(kernel->info());std::cout<<"}\n";
  }
  auto candidates=[&](auto rank,int fixed,bool empty){std::vector<int> ids;if(empty)return std::vector<int>{-1};
    if(fixed>=0)return std::vector<int>{fixed};std::sort(rank.begin(),rank.end());
    for(int i=0;i<std::min<int>(o.top,rank.size());++i)ids.push_back(rank[i].second);return ids;};
  auto lows=candidates(rank_low,o.low_id,inputs.low.empty()),highs=candidates(rank_high,o.high_id,inputs.high.empty());
  auto kernel=[&](bool hi,int id)->Kernel*{return id<0?nullptr:(hi?high[id].get():low[id].get());};
  int best_l=lows.front(),best_h=highs.front();bool best_concurrent=(o.policy=="concurrent");float best=1e30;
  if(tune)for(int lo:lows)for(int hi:highs)for(int policy=0;policy<(lo>=0 && hi>=0?2:1);++policy){
    Graph graph([&]{streams.launch(kernel(false,lo),kernel(true,hi),policy==1);},streams.main,32);
    auto times=graph.measure(streams.main,3,8);float us=median(times);
    std::cout<<"{\"kind\":\"pair_candidate\",\"low_id\":"<<lo<<",\"high_id\":"<<hi<<",\"policy\":\""<<(policy?"concurrent":"serial")
      <<"\",\"median_us\":"<<us<<",\"samples_us\":";print_times(times);std::cout<<"}\n";
    if(us<best){best=us;best_l=lo;best_h=hi;best_concurrent=policy==1;}
  }
  inputs.reset();streams.launch(kernel(false,best_l),kernel(true,best_h),best_concurrent);
  CUDA(cudaStreamSynchronize(streams.main));int checked=inputs.check();
  Graph selected([&]{streams.launch(kernel(false,best_l),kernel(true,best_h),best_concurrent);},streams.main,o.iterations);
  auto times=selected.measure(streams.main,o.repeats,80);float latency=median(times);
  int expected=(best_l>=0)+(best_h>=0);
  if(selected.kernels!=o.iterations*expected)throw std::runtime_error("Unexpected graph kernel count");
  float fp16=o.baseline?fp16_baseline(inputs,streams):0;
  std::cout<<"{\"kind\":\"result\",\"out\":"<<o.outputs<<",\"tokens\":"<<o.tokens<<",\"k\":"<<o.k<<",\"high\":"<<o.high
    <<",\"actual_fraction\":"<<double(o.high)/o.outputs<<",\"mapping\":\""<<o.mapping<<"\",\"low_id\":"<<best_l<<",\"high_id\":"<<best_h
    <<",\"policy\":\""<<(best_concurrent && expected==2?"concurrent":"serial")<<"\",\"median_us\":"<<latency<<",\"samples_us\":";print_times(times);
  std::cout<<",\"fp16_us\":"<<fp16<<",\"speedup_fp16\":"<<(fp16?fp16/latency:0)<<",\"tflops\":"<<2.0*o.outputs*o.tokens*o.k/latency/1e6
    <<",\"checked\":"<<checked<<",\"all_outputs_finite\":true,\"guards_passed\":true,\"full_check\":"<<(o.full?"true":"false")
    <<",\"verify_all\":"<<(o.verify_all?"true":"false")<<",\"kernels_per_request\":"<<expected<<",\"low_config\":";
  if(best_l>=0)print_info(kernel(false,best_l)->info());else std::cout<<"null";
  std::cout<<",\"high_config\":";if(best_h>=0)print_info(kernel(true,best_h)->info());else std::cout<<"null";std::cout<<"}"<<std::endl;
  return 0;
}catch(std::exception const&e){std::cerr<<"ERROR: "<<e.what()<<std::endl;return 1;}
#endif
