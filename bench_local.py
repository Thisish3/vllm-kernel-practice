"""Local GPU benchmark: PyTorch reference vs. our Triton kernels vs. vLLM's CUDA kernel.

Runs inside the vllm/vllm-openai image (torch, triton and vllm._C available):

    python bench_local.py            # writes results/bench_<gpu>.json

Effective bandwidth counts the bytes the op must move at minimum:
fp16 input read + int8 output write + fp32 scales write.
"""

import json
import os
import statistics

import torch
import triton
from torch.profiler import ProfilerActivity, profile

from group_quant_triton import group_quant_int8_triton
from group_quant_triton_multi import group_quant_int8_triton_multi
from test_group_quant import reference_group_quant_int8

HIDDEN = 4096
GROUP = 128
TOKENS = [16, 256, 4096]
MULTI = [2, 4, 8, 16]
REPS = 5


def bench_ms(fn):
    # median of REPS do_bench medians (each do_bench already repeats internally)
    return statistics.median(triton.testing.do_bench(fn) for _ in range(REPS))


def min_bytes(tokens, hidden, group):
    return tokens * hidden * (2 + 1) + tokens * (hidden // group) * 4


def kernel_count(fn):
    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return sum(1 for e in prof.events() if e.device_type.name == "CUDA")


def compare(a, b):
    (qa, sa), (qb, sb) = a, b
    d = (qa.int() - qb.int()).abs()
    return {
        "q_mismatch": int((d != 0).sum()),
        "q_max_diff": int(d.max()),
        "numel": qa.numel(),
        "scale_max_rel_diff": float(((sa - sb).abs() / sb.abs().clamp_min(1e-12)).max()),
    }


def main():
    from vllm.model_executor.layers.quantization.utils.int8_utils import (
        per_token_group_quant_int8,
    )
    import vllm

    torch.manual_seed(0)
    impls = {
        "pytorch_ref": lambda x, g: reference_group_quant_int8(x, g),
        "triton_1group": lambda x, g: group_quant_int8_triton(x, g),
        **{
            f"triton_{m}groups": (lambda m: lambda x, g: group_quant_int8_triton_multi(x, g, m))(m)
            for m in MULTI
        },
        "vllm_cuda": lambda x, g: per_token_group_quant_int8(x, g),
    }

    out = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "vllm": vllm.__version__,
        "hidden": HIDDEN,
        "group_size": GROUP,
        "speed": [],
    }

    for t in TOKENS:
        x = torch.randn(t, HIDDEN, device="cuda", dtype=torch.float16)
        nbytes = min_bytes(t, HIDDEN, GROUP)
        row = {"tokens": t, "min_bytes": nbytes, "impls": {}}
        for name, fn in impls.items():
            ms = bench_ms(lambda: fn(x, GROUP))
            row["impls"][name] = {
                "us": round(ms * 1e3, 2),
                "gbps": round(nbytes / (ms * 1e-3) / 1e9, 1),
            }
        out["speed"].append(row)
        print(json.dumps(row), flush=True)

    x = torch.randn(4096, HIDDEN, device="cuda", dtype=torch.float16) * 5
    out["kernel_launches"] = {
        name: kernel_count(lambda fn=fn: fn(x, GROUP)) for name, fn in impls.items()
    }

    ref = reference_group_quant_int8(x, GROUP)
    out["accuracy_vs_pytorch_ref"] = {
        name: compare(fn(x, GROUP), ref)
        for name, fn in impls.items()
        if name != "pytorch_ref"
    }

    sweep = []
    for gs in [32, 64, 128, 256, 512, 1024, 2048, 4096]:
        q, s = group_quant_int8_triton_multi(x, gs, 8 if gs <= 1024 else 1)
        x_hat = (q.reshape(4096, -1, gs).float() * s.unsqueeze(-1)).reshape_as(x)
        err = (x.float() - x_hat).abs()
        sweep.append({
            "group_size": gs,
            "mean_abs_err": round(err.mean().item(), 5),
            "mean_rel_err_pct": round((err / (x.float().abs() + 1e-6)).mean().item() * 100, 3),
            "scale_kb": round(s.numel() * 4 / 1024, 1),
            "scale_overhead_pct_of_int8": round(4 / gs * 100, 3),
        })
    out["group_size_sweep"] = sweep

    os.makedirs("results", exist_ok=True)
    tag = out["gpu"].replace("NVIDIA GeForce ", "").replace(" ", "_").lower()
    with open(f"results/bench_{tag}.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    keys = ("kernel_launches", "accuracy_vs_pytorch_ref", "group_size_sweep")
    print(json.dumps({k: out[k] for k in keys}, indent=2))


if __name__ == "__main__":
    main()
