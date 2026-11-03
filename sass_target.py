"""Launch PyTorch's reference (whose x / scale runs in an at::native DivFunctor<float> kernel) and the
Triton one-group kernel once each, so Nsight Compute can dump their SASS:

    ncu --kernel-name regex:"DivFunctor|group_quant" --section SourceCounters -o sass python sass_target.py
    ncu -i sass.ncu-repz --page source --print-source sass --csv
"""
import torch

from group_quant_triton import group_quant_int8_triton
from verify_rtx import G, H, ref

x = torch.randn(16, H, device="cuda", dtype=torch.float16) * 5
ref(x, G)
group_quant_int8_triton(x, G)
torch.cuda.synchronize()
