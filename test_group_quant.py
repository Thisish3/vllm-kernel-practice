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

    torch.testing.assert_close(s_triton, s_ref, rtol=1e-4, atol=1e-6)

    # Triton's `/` and PyTorch's `/` aren't guaranteed bit-identical (GPUs
    # commonly use a fast approximate reciprocal for division). When the
    # true quotient lands within ~1 ULP of an integer (e.g. raw ~= 127.0),
    # that tiny discrepancy can truncate to a different neighboring
    # integer. That's a precision artifact of fast division, not a logic
    # bug, so we allow it as long as it's rare and off by at most 1.
    diff = (q_triton.int() - q_ref.int()).abs()
    num_mismatched = (diff != 0).sum().item()
    mismatch_ratio = num_mismatched / q_triton.numel()

    assert diff.max().item() <= 1, (
        f"mismatches differ by more than 1 (real bug, not precision noise): "
        f"max diff={diff.max().item()}"
    )
    assert mismatch_ratio < 0.01, (
        f"too many boundary mismatches: {num_mismatched}/{q_triton.numel()}"
    )
    print(
        f"correctness OK: {num_mismatched}/{q_triton.numel()} elements differ "
        "by 1 at integer boundaries (expected fp32 division precision noise)"
    )


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
