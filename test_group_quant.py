"""Correctness check + benchmark for the Triton group-quant kernel.

Run this in a Colab GPU runtime (Runtime > Change runtime type > GPU):

    !pip install -q triton   # usually already present in Colab
    !python test_group_quant.py
"""

import torch

from group_quant_triton import group_quant_int8_triton


def reference_group_quant_int8(x: torch.Tensor, group_size: int):
    """Pure-PyTorch reference: same absmax-quantization math, no Triton."""
    num_tokens, hidden_size = x.shape
    num_groups = hidden_size // group_size
    xg = x.reshape(num_tokens, num_groups, group_size).float()

    absmax = xg.abs().amax(dim=-1)
    scale = (absmax / 127.0).clamp_min(1e-12)

    # truncating cast (no rounding), matching vLLM's `DST_DTYPE(q)` cast
    q = (xg / scale.unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
    return q.reshape(num_tokens, hidden_size), scale


def check_correctness():
    torch.manual_seed(0)
    x = torch.randn(37, 256, device="cuda", dtype=torch.float16) * 5
    group_size = 64

    q_triton, s_triton = group_quant_int8_triton(x, group_size)
    q_ref, s_ref = reference_group_quant_int8(x, group_size)

    # Quantized ints should match exactly (same rounding rule); scales
    # should match up to fp32 rounding.
    assert torch.equal(q_triton, q_ref), "int8 outputs diverge from reference"
    torch.testing.assert_close(s_triton, s_ref, rtol=1e-4, atol=1e-6)
    print("correctness OK: Triton kernel matches PyTorch reference")


def benchmark():
    import triton

    x = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    group_size = 128

    ms_triton = triton.testing.do_bench(
        lambda: group_quant_int8_triton(x, group_size)
    )
    ms_ref = triton.testing.do_bench(
        lambda: reference_group_quant_int8(x, group_size)
    )
    print(f"Triton kernel : {ms_triton:.4f} ms")
    print(f"PyTorch ref   : {ms_ref:.4f} ms")
    print(f"speedup       : {ms_ref / ms_triton:.2f}x")


if __name__ == "__main__":
    check_correctness()
    benchmark()
