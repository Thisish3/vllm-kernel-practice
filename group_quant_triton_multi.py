"""Multi-group-per-program variant of the Triton group-quant kernel.

The original kernel (group_quant_triton.py) launches one program per group.
vLLM's CUDA kernel instead lets one block handle several groups
(`groups_per_block`). This variant does the same in Triton: each program
loads a [GROUPS, group_size] tile, reduces along the group axis and writes
GROUPS scales.

Because x is row-major and groups are contiguous along hidden_size, group g
(counted across all tokens) starts at element g * group_size and its scale
lives at out_s[g] when out_s is [num_tokens, num_groups] row-major.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _group_quant_int8_multi_kernel(
    x_ptr,
    out_q_ptr,
    out_s_ptr,
    total_groups,
    group_size: tl.constexpr,
    GROUPS: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    g = pid * GROUPS + tl.arange(0, GROUPS)
    g_mask = g < total_groups
    offs = g[:, None] * group_size + tl.arange(0, group_size)[None, :]

    x = tl.load(x_ptr + offs, mask=g_mask[:, None], other=0.0).to(tl.float32)

    absmax = tl.max(tl.abs(x), axis=1)
    scale = absmax / 127.0
    scale = tl.where(scale == 0.0, 1.0, scale)

    q = tl.minimum(tl.maximum(x / scale[:, None], -127.0), 127.0)

    tl.store(out_q_ptr + offs, q.to(tl.int8), mask=g_mask[:, None])
    tl.store(out_s_ptr + g, scale, mask=g_mask)


def group_quant_int8_triton_multi(
    x: torch.Tensor, group_size: int, groups_per_program: int = 8
) -> tuple[torch.Tensor, torch.Tensor]:
    """Same contract as group_quant_int8_triton, GROUPS groups per program."""
    assert x.is_cuda and x.ndim == 2 and x.is_contiguous()
    num_tokens, hidden_size = x.shape
    assert hidden_size % group_size == 0
    # tl.arange needs power-of-two extents
    assert groups_per_program & (groups_per_program - 1) == 0

    num_groups = hidden_size // group_size
    total_groups = num_tokens * num_groups
    out_q = torch.empty_like(x, dtype=torch.int8)
    out_s = torch.empty((num_tokens, num_groups), dtype=torch.float32, device=x.device)

    grid = (triton.cdiv(total_groups, groups_per_program),)
    _group_quant_int8_multi_kernel[grid](
        x, out_q, out_s, total_groups,
        group_size=group_size, GROUPS=groups_per_program,
    )
    return out_q, out_s
