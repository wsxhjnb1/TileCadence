#include <ATen/ATen.h>
#include <torch/library.h>
#include <vector>
namespace TORCH_EXTENSION_NAME {
void model_forward_out(const at::Tensor&,const at::Tensor&,const at::Tensor&,at::Tensor&,at::Tensor&,at::Tensor&,at::Tensor&,int64_t,int64_t,double,bool);
std::vector<int64_t> model_workspace_spec(const at::Tensor&,const at::Tensor&,const at::Tensor&,int64_t,int64_t,bool);
std::vector<int64_t> model_resources(const at::Tensor&,const at::Tensor&,const at::Tensor&,int64_t,int64_t,bool);
}
#define DECLARE_MODEL_LIBRARY(name,m) TORCH_LIBRARY(name,m)
DECLARE_MODEL_LIBRARY(TORCH_EXTENSION_NAME,m) {
    m.def("forward_out(Tensor q, Tensor k, Tensor v, Tensor(a!) out, Tensor(b!) lse, Tensor(c!) workspace_o, Tensor(d!) workspace_lse, int num_splits, int mapping, float scale, bool causal) -> ()",&TORCH_EXTENSION_NAME::model_forward_out);
    m.def("workspace_spec(Tensor q, Tensor k, Tensor v, int num_splits, int mapping, bool causal) -> int[]",&TORCH_EXTENSION_NAME::model_workspace_spec);
    m.def("resources(Tensor q, Tensor k, Tensor v, int num_splits, int mapping, bool causal) -> int[]",&TORCH_EXTENSION_NAME::model_resources);
}
