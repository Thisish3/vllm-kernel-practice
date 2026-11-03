"""Launch the three kernels compared under Nsight Compute, each once after a warm-up call.

    ncu --kernel-name regex:group_quant --launch-skip 0 ... python ncu_target.py

Order of launches: tri1_w4 (warm-up, measured), tri1_w1 (warm-up, measured), multi8 (warm-up, measured).
Input: fp16 [4096, 4096], group_size 128 (the 4096-token case of verify_rtx.py).
"""

import torch

from group_quant_triton import _group_quant_int8_kernel
from group_quant_triton_multi import group_quant_int8_triton_multi

H, G, N = 4096, 128, 4096


def tri1(x, nw):
    q = torch.empty_like(x, dtype=torch.int8)
    s = torch.empty((N, H // G), dtype=torch.float32, device=x.device)
    _group_quant_int8_kernel[(N * (H // G),)](x, q, s, H, group_size=G, num_warps=nw)


torch.manual_seed(0)
x = torch.randn(N, H, device="cuda", dtype=torch.float16) * 5
for f in (lambda: tri1(x, 4), lambda: tri1(x, 1), lambda: group_quant_int8_triton_multi(x, G, 8)):
    f()  # warm-up (JIT compile)
    torch.cuda.synchronize()
    f()  # measured
    torch.cuda.synchronize()
print("done")
