#include <ATen/ATen.h>
#include <torch/library.h>
#include <vector>

namespace TORCH_EXTENSION_NAME {
void decode_forward_out(const at::Tensor&,at::Tensor&,at::Tensor&,
    const at::Tensor&,const at::Tensor&,const at::Tensor&,at::Tensor&,at::Tensor&,
    at::Tensor&,at::Tensor&,int64_t,int64_t,bool);
std::vector<int64_t> decode_workspace_spec(const at::Tensor&,const at::Tensor&,
    const at::Tensor&,int64_t,int64_t,bool);
std::vector<int64_t> decode_resources(const at::Tensor&,const at::Tensor&,
    const at::Tensor&,int64_t,int64_t,bool);
}

#define DECLARE_DECODE_LIBRARY(name,m) TORCH_LIBRARY(name,m)
DECLARE_DECODE_LIBRARY(TORCH_EXTENSION_NAME,m) {
    m.def("forward_out(Tensor q, Tensor(a!) kcache, Tensor(b!) vcache, Tensor knew, Tensor vnew, "
          "Tensor old_lengths, Tensor(c!) out, Tensor(d!) lse, Tensor(e!) workspace_o, "
          "Tensor(f!) workspace_lse, int num_splits, int mapping, bool append) -> ()",
          &TORCH_EXTENSION_NAME::decode_forward_out);
    m.def("workspace_spec(Tensor q, Tensor kcache, Tensor old_lengths, int num_splits, int mapping, bool append) -> int[]",
          &TORCH_EXTENSION_NAME::decode_workspace_spec);
    m.def("resources(Tensor q, Tensor kcache, Tensor old_lengths, int num_splits, int mapping, bool append) -> int[]",
          &TORCH_EXTENSION_NAME::decode_resources);
}
