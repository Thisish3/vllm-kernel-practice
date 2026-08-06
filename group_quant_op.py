"""Register the Triton kernel as a PyTorch custom op.

This is the Python-level equivalent of what vLLM does in C++ with
STABLE_TORCH_LIBRARY / STABLE_TORCH_LIBRARY_IMPL
(see csrc/libtorch_stable/torch_bindings.cpp):

  1. declare a schema for the op            <-> ops.def("...")
  2. bind an implementation to it            <-> ops.impl("...", TORCH_BOX(&fn))

`torch.library.custom_op` does both steps in one decorator. The op ends
up callable as `torch.ops.practice.group_quant_int8(...)`, exactly like
vLLM's ops end up callable as `torch.ops._C.<name>(...)`.
"""

import torch

from group_quant_triton import group_quant_int8_triton


@torch.library.custom_op("practice::group_quant_int8", mutates_args=())
def group_quant_int8(x: torch.Tensor, group_size: int) -> list[torch.Tensor]:
    q, s = group_quant_int8_triton(x, group_size)
    return [q, s]


@group_quant_int8.register_fake
def _(x: torch.Tensor, group_size: int) -> list[torch.Tensor]:
    # Shape/dtype-only "fake" implementation so torch.compile / meta-device
    # tracing works without actually running the kernel. Mirrors the schema
    # string in vLLM's ops.def(...) declarations.
    num_tokens, hidden_size = x.shape
    num_groups = hidden_size // group_size
    q = x.new_empty(x.shape, dtype=torch.int8)
    s = x.new_empty((num_tokens, num_groups), dtype=torch.float32)
    return [q, s]


if __name__ == "__main__":
    x = torch.randn(4, 8, device="cuda", dtype=torch.float16)
    q, s = torch.ops.practice.group_quant_int8(x, group_size=4)
    print("q:", q)
    print("s:", s)
