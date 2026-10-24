#define HYGEMM_OC_BENCH_UTILS_ONLY
#include "output_channel_bench.cu"
#include "output_channel_fused.h"

struct EagerSample { float device_us,submit_us,wall_us; };
EagerSample measure_eager(std::function<void()> const&launch,cudaStream_t stream,int count) {
  // Events include host-induced GPU idle gaps; this is deliberately NOT a
  // pure kernel-duration measurement. Also retain CPU enqueue and wall time.
  CUDA(cudaStreamSynchronize(stream));
  cudaEvent_t start,stop;CUDA(cudaEventCreate(&start));CUDA(cudaEventCreate(&stop));
  auto wall_start=std::chrono::steady_clock::now();CUDA(cudaEventRecord(start,stream));
  auto submit_start=std::chrono::steady_clock::now();
  for(int i=0;i<count;++i)launch();
  auto submit_end=std::chrono::steady_clock::now();CUDA(cudaEventRecord(stop,stream));CUDA(cudaEventSynchronize(stop));
  auto wall_end=std::chrono::steady_clock::now();float ms;CUDA(cudaEventElapsedTime(&ms,start,stop));
  CUDA(cudaEventDestroy(start));CUDA(cudaEventDestroy(stop));
  return {ms*1000/count,float(std::chrono::duration<double,std::micro>(submit_end-submit_start).count()/count),
    float(std::chrono::duration<double,std::micro>(wall_end-wall_start).count()/count)};
}

int main(int argc,char const**argv) try {
  std::cout<<std::setprecision(9);Options o(argc,argv);cutlass::CommandLine cmd(argc,argv);
  bool check_only=cmd.check_cmd_line_flag("check-only"),audit=cmd.check_cmd_line_flag("audit");
  bool require_switch=cmd.check_cmd_line_flag("require-switch");int fixed=-1;
  cmd.get_cmd_line_argument("fused-id",fixed);
  if(fixed< -1 || fixed>=8)throw std::runtime_error("Invalid fused ID");
  cudaDeviceProp prop;CUDA(cudaGetDeviceProperties(&prop,0));if(prop.major!=12)throw std::runtime_error("SM12x required");
  Streams streams;Inputs inputs(o);std::vector<std::unique_ptr<FusedKernel>> kernels(8);
  uint64_t switches_producer=0,switches_consumer=0;int total_checked=0;
  for(int id=0;id<8;++id) {
    if(fixed>=0 && id!=fixed)continue;
    kernels[id]=make_fused_kernel(id,inputs.problem(false),inputs.problem(true),streams.main);
    inputs.reset();kernels[id]->run(streams.main);CUDA(cudaStreamSynchronize(streams.main));total_checked+=inputs.check();
    if(audit) {
      Buffer<uint32_t> counters(6);CUDA(cudaMemset(counters.get(),0,24));CUDA(cudaDeviceSynchronize());
      auto diagnostic=make_fused_kernel(id,inputs.problem(false),inputs.problem(true),streams.main,counters.get());
      inputs.reset();diagnostic->run(streams.main);CUDA(cudaStreamSynchronize(streams.main));total_checked+=inputs.check();
      uint32_t counts[6];CUDA(cudaMemcpy(counts,counters.get(),24,cudaMemcpyDeviceToHost));
      auto meta=diagnostic->fused_info();auto info=meta.kernel;
      uint32_t low_tiles=((inputs.low.size()+127)/128)*((o.tokens+info.tile_n-1)/info.tile_n);
      uint32_t high_tiles=((inputs.high.size()+127)/128)*((o.tokens+info.tile_n-1)/info.tile_n);
      uint32_t k_tiles=(o.k+info.tile_k-1)/info.tile_k;
      if(counts[0]!=low_tiles || counts[1]!=high_tiles || counts[4]!=low_tiles*k_tiles || counts[5]!=high_tiles*k_tiles)
        throw std::runtime_error("Tile/stage audit count mismatch");
      switches_producer+=counts[2];switches_consumer+=counts[3];
      if(require_switch && (!counts[2] || !counts[3]))
        throw std::runtime_error("Variant "+std::to_string(id)+" did not execute a within-CTA switch");
      std::cout<<"{\"kind\":\"audit\",\"id\":"<<id<<",\"counts\":[";
      for(int i=0;i<6;++i)std::cout<<(i?",":"")<<counts[i];
      std::cout<<"],\"low_expected_bytes\":"<<meta.low_transaction_bytes<<",\"high_expected_bytes\":"<<meta.high_transaction_bytes
        <<",\"virtual_rows\":"<<meta.virtual_rows<<",\"config\":";print_info(info);std::cout<<"}\n";
    }
  }
  if(require_switch && (!switches_producer || !switches_consumer))throw std::runtime_error("No within-CTA precision switch observed");
  if(check_only) {
    std::cout<<"{\"kind\":\"validation\",\"out\":"<<o.outputs<<",\"tokens\":"<<o.tokens<<",\"k\":"<<o.k
      <<",\"high\":"<<o.high<<",\"checked\":"<<total_checked<<",\"producer_switches\":"<<switches_producer
      <<",\"consumer_switches\":"<<switches_consumer<<",\"passed\":true}"<<std::endl;return 0;
  }
  if((!inputs.low.empty() && o.low_id<0) || (!inputs.high.empty() && o.high_id<0))
    throw std::runtime_error("Provide saved dual --low-id/--high-id/--policy for a matched comparison");
  int best=fixed;float best_us=1e30f;
  for(int id=0;id<8;++id)if(kernels[id]) {
    Graph graph([&]{kernels[id]->run(streams.main);},streams.main,32);auto samples=graph.measure(streams.main,3,10);
    float us=median(samples);if(us<best_us){best=id;best_us=us;}
    std::cout<<"{\"kind\":\"fused_candidate\",\"median_us\":"<<us<<",\"samples_us\":";print_times(samples);
    std::cout<<",\"config\":";print_info(kernels[id]->info());std::cout<<"}\n";
  }
  auto low=inputs.low.empty()?nullptr:make_kernel(o.low_id,false,inputs.problem(false),streams.main);
  auto high=inputs.high.empty()?nullptr:make_kernel(o.high_id,true,inputs.problem(true),streams.main);
  auto dual=[&]{streams.launch(low.get(),high.get(),o.policy=="concurrent");};
  inputs.reset();dual();CUDA(cudaStreamSynchronize(streams.main));inputs.check();
  Graph fused_graph([&]{kernels[best]->run(streams.main);},streams.main,o.iterations);
  Graph dual_graph(dual,streams.main,o.iterations);
  int dual_count=int(bool(low))+int(bool(high));
  if(fused_graph.kernels!=o.iterations || dual_graph.kernels!=o.iterations*dual_count)
    throw std::runtime_error("Graph kernel node count mismatch");
  fused_graph.measure(streams.main,1,80);dual_graph.measure(streams.main,1,80);
  std::vector<float> fs,ds;
  for(int r=0;r<o.repeats;++r) {
    if(r%2==0){fs.push_back(fused_graph.measure(streams.main,1,0)[0]);ds.push_back(dual_graph.measure(streams.main,1,0)[0]);}
    else {ds.push_back(dual_graph.measure(streams.main,1,0)[0]);fs.push_back(fused_graph.measure(streams.main,1,0)[0]);}
  }
  if(cmd.check_cmd_line_flag("measure-eager")) {
    auto fused=[&]{kernels[best]->run(streams.main);};
    measure_eager(fused,streams.main,o.iterations);measure_eager(dual,streams.main,o.iterations);
    std::vector<float> fd,dd,fc,dc,fw,dw;
    for(int r=0;r<o.repeats;++r) {
      EagerSample f,d;
      if(r%2==0){f=measure_eager(fused,streams.main,o.iterations);d=measure_eager(dual,streams.main,o.iterations);}
      else {d=measure_eager(dual,streams.main,o.iterations);f=measure_eager(fused,streams.main,o.iterations);}
      fd.push_back(f.device_us);dd.push_back(d.device_us);fc.push_back(f.submit_us);dc.push_back(d.submit_us);
      fw.push_back(f.wall_us);dw.push_back(d.wall_us);
    }
    std::cout<<"{\"kind\":\"eager_comparison\",\"fused_id\":"<<best<<",\"fused_device_us\":"<<median(fd)
      <<",\"dual_device_us\":"<<median(dd)<<",\"fused_submit_us\":"<<median(fc)<<",\"dual_submit_us\":"<<median(dc)
      <<",\"fused_wall_us\":"<<median(fw)<<",\"dual_wall_us\":"<<median(dw)<<",\"wall_speedup\":"<<median(dw)/median(fw)
      <<",\"fused_wall_samples_us\":";print_times(fw);std::cout<<",\"dual_wall_samples_us\":";print_times(dw);
    std::cout<<",\"fused_device_samples_us\":";print_times(fd);std::cout<<",\"dual_device_samples_us\":";print_times(dd);
    std::cout<<",\"fused_submit_samples_us\":";print_times(fc);std::cout<<",\"dual_submit_samples_us\":";print_times(dc);std::cout<<"}\n";
  }
  // Recheck after repeated launches, with fresh NaN sentinels, for both paths.
  inputs.reset();kernels[best]->run(streams.main);CUDA(cudaStreamSynchronize(streams.main));inputs.check();
  inputs.reset();dual();CUDA(cudaStreamSynchronize(streams.main));inputs.check();
  auto info=kernels[best]->fused_info();
  std::cout<<"{\"kind\":\"comparison\",\"out\":"<<o.outputs<<",\"tokens\":"<<o.tokens<<",\"k\":"<<o.k<<",\"high\":"<<o.high
    <<",\"mapping\":\""<<o.mapping<<"\",\"fused_id\":"<<best<<",\"fused_us\":"<<median(fs)<<",\"dual_us\":"<<median(ds)
    <<",\"speedup_vs_dual\":"<<median(ds)/median(fs)<<",\"fused_samples_us\":";print_times(fs);
  std::cout<<",\"dual_samples_us\":";print_times(ds);
  std::cout<<",\"low_id\":"<<(low?o.low_id:-1)<<",\"high_id\":"<<(high?o.high_id:-1)<<",\"dual_policy\":\""<<o.policy
    <<"\",\"fused_kernel_nodes\":1,\"dual_kernel_nodes\":"<<dual_count<<",\"checked\":"<<total_checked
    <<",\"low_expected_bytes\":"<<info.low_transaction_bytes<<",\"high_expected_bytes\":"<<info.high_transaction_bytes
    <<",\"config\":";print_info(info.kernel);std::cout<<"}"<<std::endl;
  return 0;
}catch(std::exception const&e){std::cerr<<"ERROR: "<<e.what()<<std::endl;return 1;}
