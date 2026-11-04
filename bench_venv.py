"""Run bench_local.py unchanged N times in the WSL venv and aggregate the results.

    /root/venv-vllm0112/bin/python bench_venv.py

bench_local.py writes results/bench_<gpu>.json relative to the working directory,
so each run gets its own temp working directory. That keeps the committed
results/bench_rtx_3060_ti.json untouched.

Per-cell value = median over the N script runs. Each run's cell is itself
bench_local.py's median of 5 do_bench calls (do_bench default return_mode, mean).

The 1GiB copy repeats the method behind the earlier 349GB/s figure: fp32 1GiB
tensors, 5 warmup copies, then 20 b.copy_(a) timed with perf_counter, counting
2 * 1GiB (read + write). The earlier script used 3 repeats; this one uses 5.

Writes results/bench_rtx_3060_ti_venv.json.
"""

import json
import os
import statistics
import subprocess
import sys
import tempfile
import time

import torch
import triton

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

RUNS = 5
SPEC_GBPS = 448
OUT = os.path.join(HERE, "results", "bench_rtx_3060_ti_venv.json")


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()


def script_runs():
    runs = []
    for i in range(RUNS):
        with tempfile.TemporaryDirectory() as cwd:
            t = time.time()
            subprocess.run([sys.executable, os.path.join(HERE, "bench_local.py")],
                           cwd=cwd, check=True, stdout=subprocess.DEVNULL)
            with open(os.path.join(cwd, "results", "bench_rtx_3060_ti.json")) as f:
                runs.append(json.load(f))
        print(f"run {i + 1}/{RUNS} done in {time.time() - t:.0f}s", flush=True)
    return runs


def speed_table(runs):
    table = []
    for i, row in enumerate(runs[0]["speed"]):
        nbytes = row["min_bytes"]
        impls = {}
        for name in row["impls"]:
            us = [r["speed"][i]["impls"][name]["us"] for r in runs]
            med = statistics.median(us)
            impls[name] = {
                "us_runs": us,
                "us_median": med,
                "gbps_median": round(nbytes / (med * 1e-6) / 1e9, 1),
                "pct_of_spec": round(nbytes / (med * 1e-6) / 1e9 / SPEC_GBPS * 100, 1),
            }
        cuda = impls["vllm_cuda"]["us_median"]
        for v in impls.values():
            v["time_vs_vllm_cuda"] = round(v["us_median"] / cuda, 3)
        table.append({"tokens": row["tokens"], "min_bytes": nbytes, "impls": impls})
    return table


def copy_1gib():
    gib = 1024**3
    a = torch.empty(gib // 4, dtype=torch.float32, device="cuda")
    b = torch.empty_like(a)
    for _ in range(5):
        b.copy_(a)
    torch.cuda.synchronize()
    runs = []
    for _ in range(RUNS):
        t = time.perf_counter()
        for _ in range(20):
            b.copy_(a)
        torch.cuda.synchronize()
        runs.append(2 * gib / ((time.perf_counter() - t) / 20) / 1e9)
    del a, b
    torch.cuda.empty_cache()
    med = statistics.median(runs)
    return {"gbps_runs": [round(x, 1) for x in runs], "gbps_median": round(med, 1),
            "pct_of_spec": round(med / SPEC_GBPS * 100, 1)}


def accuracy():
    from group_quant_triton_multi import group_quant_int8_triton_multi
    from test_group_quant import reference_group_quant_int8
    from vllm.model_executor.layers.quantization.utils.int8_utils import (
        per_token_group_quant_int8,
    )

    group = 128
    torch.manual_seed(0)
    x = torch.randn(4096, 4096, device="cuda", dtype=torch.float16) * 5
    rq, rs = reference_group_quant_int8(x, group)

    # direct op call, so the result does not depend on the wrapper's dispatch
    cq = torch.empty_like(x, dtype=torch.int8)
    cs = torch.empty(4096, 4096 // group, device="cuda", dtype=torch.float32)
    torch.ops._C.per_token_group_quant_int8(x, cq, cs, group, 1e-10, -128.0, 127.0)
    wq, ws = per_token_group_quant_int8(x, group)

    def diff(q, s):
        d = (q.int() - rq.int()).abs()
        return {
            "numel": q.numel(),
            "q_mismatch": int((d != 0).sum()),
            "q_max_diff": int(d.max()),
            "scale_numel": s.numel(),
            "scale_not_equal": int((s != rs).sum()),
            "scale_max_rel_diff": float(((s - rs).abs() / rs.abs().clamp_min(1e-12)).max()),
        }

    return {
        "input": "torch.manual_seed(0); randn(4096, 4096, fp16) * 5; group_size 128",
        "triton_8groups": diff(*group_quant_int8_triton_multi(x, group, 8)),
        "vllm_cuda_op": diff(cq, cs),
        "op_equals_wrapper": bool(torch.equal(cq, wq) and torch.equal(cs, ws)),
    }


def main():
    import vllm

    env = {
        "nvidia_smi": sh("nvidia-smi --query-gpu=name,driver_version,memory.total,"
                         "clocks.max.sm,clocks.max.mem,power.limit --format=csv,noheader"),
        "cuda_from_nvidia_smi": sh("nvidia-smi | grep -o 'CUDA Version: [0-9.]*'"),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "triton": triton.__version__,
        "vllm": vllm.__version__,
        "python": sys.version.split()[0],
        "kernel": sh("uname -r"),
        "repo_commit": sh(f"git -c safe.directory='*' -C {HERE} rev-parse HEAD"),
    }
    runs = script_runs()
    out = {
        "env": env,
        "method": {
            "script": "bench_local.py, unchanged, one fresh process per run",
            "runs": RUNS,
            "cell": "median over runs of bench_local.py's median of 5 do_bench means",
            "spec_gbps": SPEC_GBPS,
            "copy": "fp32 1GiB, 5 warmup, 20 copies per repeat, 5 repeats, 2*GiB/time",
        },
        "speed": speed_table(runs),
        "copy_1GiB": copy_1gib(),
        "accuracy_4096x4096": accuracy(),
        "bench_local_accuracy_per_run": [r["accuracy_vs_pytorch_ref"] for r in runs],
        "kernel_launches_per_run": [r["kernel_launches"] for r in runs],
        "group_size_sweep_run1": runs[0]["group_size_sweep"],
        "raw_runs_speed": [r["speed"] for r in runs],
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: out[k] for k in ("env", "copy_1GiB", "accuracy_4096x4096")}, indent=2))
    for row in out["speed"]:
        print(row["tokens"], {k: (v["us_median"], v["gbps_median"]) for k, v in row["impls"].items()})


if __name__ == "__main__":
    main()
