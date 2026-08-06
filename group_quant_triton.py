"""Per-group INT8 quantization kernel written in Triton.

Simplified re-implementation of the algorithm in vLLM's
csrc/libtorch_stable/quantization/w8a8/fp8/per_token_group_quant.cu
(per_token_group_quant_8bit_kernel), but using Triton's DSL instead of
raw CUDA. One Triton "program" (= one block) handles exactly one group.

For each group of `group_size` contiguous elements:
  1. load the group
  2. scale = max(|x|) / 127          (absmax quantization)
  3. store round(x / scale) as int8, and store `scale` separately
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _group_quant_int8_kernel(
    x_ptr,          # *input, shape [num_tokens, hidden_size], row-major
    out_q_ptr,      # *output int8, same shape as x
    out_s_ptr,      # *output fp32 scales, shape [num_tokens, num_groups]
    hidden_size,    # int
    group_size: tl.constexpr,
):
    # One program == one (token, group) pair.
    pid = tl.program_id(axis=0)
    num_groups = hidden_size // group_size
    token_id = pid // num_groups
    group_id = pid % num_groups

    offs = tl.arange(0, group_size)
    row_start = token_id * hidden_size + group_id * group_size

    x = tl.load(x_ptr + row_start + offs).to(tl.float32)

    absmax = tl.max(tl.abs(x), axis=0)
    scale = absmax / 127.0
    scale = tl.where(scale == 0.0, 1.0, scale)  # avoid div-by-zero on all-zero groups

    # vLLM's reference kernel does a plain truncating cast here (no
    # rounding) - `dst = DST_DTYPE(q)` in per_token_group_quant.cu - so we
    # match that instead of rounding to nearest.
    q = tl.minimum(tl.maximum(x / scale, -127.0), 127.0)

    tl.store(out_q_ptr + row_start + offs, q.to(tl.int8))
    tl.store(out_s_ptr + token_id * num_groups + group_id, scale)


def group_quant_int8_triton(
    x: torch.Tensor, group_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize the last dim of `x` in chunks of `group_size` to int8.

    Args:
        x: [num_tokens, hidden_size] floating point tensor on CUDA.
        group_size: must evenly divide hidden_size.

    Returns:
        (q, scales): q is int8 [num_tokens, hidden_size],
        scales is fp32 [num_tokens, hidden_size // group_size].
    """
    assert x.is_cuda, "Triton kernels need a CUDA tensor (use a Colab GPU runtime)"
    assert x.ndim == 2
    num_tokens, hidden_size = x.shape
    assert hidden_size % group_size == 0

    num_groups = hidden_size // group_size
    out_q = torch.empty_like(x, dtype=torch.int8)
    out_s = torch.empty((num_tokens, num_groups), dtype=torch.float32, device=x.device)

    grid = (num_tokens * num_groups,)
    _group_quant_int8_kernel[grid](
        x, out_q, out_s, hidden_size, group_size=group_size
    )
    return out_q, out_s
