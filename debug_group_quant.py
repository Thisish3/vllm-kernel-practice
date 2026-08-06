"""Minimal repro to see exactly where Triton vs PyTorch reference diverge."""

import torch

from group_quant_triton import group_quant_int8_triton
from test_group_quant import reference_group_quant_int8

torch.manual_seed(0)

# same shape/seed as the failing check_correctness() case
x = torch.randn(37, 256, device="cuda", dtype=torch.float16) * 5
group_size = 64

q_triton, s_triton = group_quant_int8_triton(x, group_size)
q_ref, s_ref = reference_group_quant_int8(x, group_size)

diff_mask = q_triton != q_ref
print("\nnum mismatched elements:", diff_mask.sum().item(), "/", q_triton.numel())

if diff_mask.any():
    idx = diff_mask.nonzero()
    for i, j in idx[:10]:
        i, j = i.item(), j.item()
        group = j // group_size
        print(
            f"[{i},{j}] x={x[i, j].item():.6f} "
            f"scale_triton={s_triton[i, group].item():.8f} "
            f"scale_ref={s_ref[i, group].item():.8f} "
            f"q_triton={q_triton[i, j].item()} q_ref={q_ref[i, j].item()} "
            f"raw_triton={x[i, j].item() / s_triton[i, group].item():.6f} "
            f"raw_ref={x[i, j].item() / s_ref[i, group].item():.6f}"
        )
