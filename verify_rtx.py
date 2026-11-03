"""Section 2 of verify_claims_colab.ipynb (cells 9-10) as a script, run on a local GPU.

    python verify_rtx.py            # writes results/verify_<gpu>.json and results/ptx/*.ptx

Additions over the notebook (the notebook's own code is kept as is):
  - every one of the 5 do_bench medians is saved, not only their median
  - the one-group (num_warps=4) and multi8 kernels are also timed the way
    bench_local.py does (do_bench default return_mode, i.e. mean) so the numbers
    can be put next to results/bench_rtx_3060_ti.json
  - all mismatching elements are checked, not only the first 5
  - kernel count per call with torch.profiler (same method as bench_local.py)
  - the compiled PTX of _group_quant_int8_kernel (num_warps 1/2/4) and of the
    multi8 kernel is dumped, and the lines that do the x / scale division are saved
"""

import json
import os
import re
import statistics

import torch
import triton
import triton.language as tl
from torch.profiler import ProfilerActivity, profile

from group_quant_triton import _group_quant_int8_kernel, group_quant_int8_triton
from group_quant_triton_multi import _group_quant_int8_multi_kernel, group_quant_int8_triton_multi

REPS = 5
H, G = 4096, 128


# ---- notebook cell 9 (unchanged except that bench() also returns the 5 values) ----
def ref(x, g):
    n, h = x.shape; xg = x.reshape(n, h // g, g).float(); s = (xg.abs().amax(-1) / 127.0).clamp_min(1e-12)
    return (xg / s.unsqueeze(-1)).clamp(-127, 127).to(torch.int8).reshape(n, h), s


def tri1(x, g, nw):
    n, h = x.shape; q = torch.empty_like(x, dtype=torch.int8); s = torch.empty((n, h // g), dtype=torch.float32, device=x.device)
    _group_quant_int8_kernel[(n * (h // g),)](x, q, s, h, group_size=g, num_warps=nw); return q, s


def bench_runs(fn, **kw):
    return [triton.testing.do_bench(fn, **kw) * 1000 for _ in range(REPS)]


def gbps(n, us):
    return (n * H * 2 + n * H + n * (H // G) * 4) / (us * 1e-6) / 1e9


def entry(runs):
    return {"median_us": round(statistics.median(runs), 2), "runs_us": [round(v, 2) for v in runs]}


def kernel_count(fn):  # same as bench_local.py
    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    names = [e.name for e in prof.events() if e.device_type.name == "CUDA"]
    return len(names), names


# x / scale computed by the same Triton expression, stored as fp32 instead of int8
@triton.jit
def _div_only_kernel(x_ptr, s_ptr, out_ptr, group_size: tl.constexpr, PRECISE: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, group_size)
    x = tl.load(x_ptr + pid * group_size + offs).to(tl.float32)
    scale = tl.load(s_ptr + pid)
    if PRECISE:
        y = tl.math.div_rn(x, scale)
    else:
        y = x / scale
    tl.store(out_ptr + pid * group_size + offs, y)


def div_only(x, s, precise):
    out = torch.empty(x.shape, dtype=torch.float32, device=x.device)
    k = _div_only_kernel[(s.numel(),)](x, s, out, group_size=G, PRECISE=precise)
    return out, k


def div_lines(ptx):
    return [l.strip() for l in ptx.splitlines() if re.search(r"\b(div|rcp)\.", l)]


def main():
    dev = torch.cuda.get_device_name()
    tag = dev.replace("NVIDIA GeForce ", "").replace(" ", "_").lower()
    os.makedirs("results/ptx", exist_ok=True)
    out = {
        "gpu": dev,
        "torch": torch.__version__, "torch_cuda": torch.version.cuda, "triton": triton.__version__,
        "hidden": H, "group_size": G,
        "unit": "us; median_us = median of runs_us, each run is one triton.testing.do_bench(return_mode='median')",
        "note": "multiN uses default num_warps=4; tri1_wK is the one-group kernel with num_warps=K",
    }

    x = torch.empty(256 * 1024 * 1024 // 2, dtype=torch.float16, device="cuda"); y = torch.empty_like(x)
    cp = bench_runs(lambda: y.copy_(x), return_mode="median")
    out["copy_512MiB_GBps"] = {"median": round(2 * x.numel() * 2 / (statistics.median(cp) * 1e-6) / 1e9, 1),
                               "runs": [round(2 * x.numel() * 2 / (v * 1e-6) / 1e9, 1) for v in cp]}
    print("copy 512MiB GB/s", out["copy_512MiB_GBps"])
    del x, y

    rows = []
    for n in (16, 256, 4096):
        x = torch.randn(n, H, device="cuda", dtype=torch.float16) * 5
        r = {"tokens": n, "pytorch_ref": entry(bench_runs(lambda: ref(x, G), return_mode="median"))}
        for nw in (1, 2, 4):
            r[f"tri1_w{nw}"] = entry(bench_runs(lambda: tri1(x, G, nw), return_mode="median"))
        for k in (1, 2, 4, 8, 16):
            r[f"multi{k}"] = entry(bench_runs(lambda: group_quant_int8_triton_multi(x, G, k), return_mode="median"))
        if n == 4096:
            r["gbps_from_median"] = {k: round(gbps(n, v["median_us"]), 1) for k, v in r.items() if isinstance(v, dict)}
            # bench_local.py's method: triton.testing.do_bench default return_mode ("mean")
            r["bench_local_method"] = {
                "pytorch_ref": entry(bench_runs(lambda: ref(x, G))),
                "triton_1group(=tri1_w4)": entry(bench_runs(lambda: group_quant_int8_triton(x, G))),
                "triton_8groups(=multi8)": entry(bench_runs(lambda: group_quant_int8_triton_multi(x, G, 8))),
            }
        rows.append(r)
        print(n, " ".join(f"{k}={v['median_us']:.1f}us" for k, v in r.items() if isinstance(v, dict) and "median_us" in v))
    out["speed"] = rows

    # ---- notebook cell 10 ----
    torch.manual_seed(0)
    x = torch.randn(4096, H, device="cuda", dtype=torch.float16) * 5
    q1, s1 = group_quant_int8_triton(x, G); qm, sm = group_quant_int8_triton_multi(x, G, 8); qr, sr = ref(x, G)
    d = (q1.int() - qr.int()).abs()
    raw = (x.float().reshape(4096, -1, G) / s1.unsqueeze(-1)).reshape(4096, H)
    mism = d != 0
    gmax = x.float().abs().reshape(4096, -1, G).amax(-1, keepdim=True).expand(-1, -1, G).reshape(4096, H)
    is_gmax = x.float().abs() == gmax
    acc = {
        "mismatches_vs_pytorch": int(mism.sum()), "elements": d.numel(), "max_diff": int(d.max()),
        "scale_equal": torch.equal(s1, sr),
        "multi8_bitwise_equal_single": torch.equal(q1, qm) and torch.equal(s1, sm),
        # checks over ALL mismatching elements
        "mismatch_all_torch_quotient_abs_eq_127": bool((raw[mism].abs() == 127.0).all()),
        "mismatch_all_triton_abs_eq_126": bool((q1[mism].int().abs() == 126).all()),
        "mismatch_all_ref_abs_eq_127": bool((qr[mism].int().abs() == 127).all()),
        "mismatch_all_group_absmax_element": bool(is_gmax[mism].all()),
        "elements_with_torch_quotient_abs_eq_127": int((raw.abs() == 127.0).sum()),
        "first5": [],
    }
    # x / scale as Triton computes it (same expression, fp32 output), default and div_rn
    tq, kdiv = div_only(x, s1, False)
    tq_rn, kdiv_rn = div_only(x, s1, True)
    acc["triton_quotient_at_mismatch_unique"] = sorted({float(v) for v in tq[mism].abs().unique().tolist()})
    acc["triton_quotient_bitwise_equal_torch_all"] = torch.equal(tq, raw)
    acc["triton_quotient_differs_from_torch_count"] = int((tq != raw).sum())
    acc["triton_div_rn_quotient_bitwise_equal_torch_all"] = torch.equal(tq_rn, raw)
    acc["triton_div_rn_differs_from_torch_count"] = int((tq_rn != raw).sum())
    for i, j in mism.nonzero()[:5].tolist():
        acc["first5"].append({"i": i, "j": j, "x": float(x[i, j]), "scale": float(s1[i, j // G]),
                              "x_over_scale_torch": "%.6f" % raw[i, j].item(),
                              "x_over_scale_triton": "%.6f" % tq[i, j].item(),
                              "x_over_scale_triton_hex": float(tq[i, j]).hex(),
                              "q_triton": int(q1[i, j]), "q_ref": int(qr[i, j])})
    out["accuracy_4096x4096"] = acc
    print(json.dumps(acc, indent=1))

    # ---- kernel count (torch.profiler) ----
    kc = {}
    for name, fn in (("pytorch_ref", lambda: ref(x, G)), ("triton_1group", lambda: group_quant_int8_triton(x, G)),
                     ("triton_multi8", lambda: group_quant_int8_triton_multi(x, G, 8))):
        c, names = kernel_count(fn)
        kc[name] = {"count": c, "kernels": names}
    out["kernel_launches"] = kc
    print({k: v["count"] for k, v in kc.items()})

    # ---- PTX ----
    ptx = {}
    xs = torch.randn(16, H, device="cuda", dtype=torch.float16)
    q = torch.empty_like(xs, dtype=torch.int8); s = torch.empty((16, H // G), dtype=torch.float32, device="cuda")
    for nw in (1, 2, 4):
        k = _group_quant_int8_kernel[(16 * (H // G),)](xs, q, s, H, group_size=G, num_warps=nw)
        fn = f"results/ptx/group_quant_int8_kernel_w{nw}.ptx"
        open(fn, "w").write(k.asm["ptx"])
        ptx[f"tri1_w{nw}"] = {"file": fn, "div_lines": div_lines(k.asm["ptx"]),
                              "n_regs": k.n_regs, "n_spills": k.n_spills}
    k = _group_quant_int8_multi_kernel[(triton.cdiv(16 * (H // G), 8),)](xs, q, s, 16 * (H // G), group_size=G, GROUPS=8)
    open("results/ptx/group_quant_int8_multi8_kernel.ptx", "w").write(k.asm["ptx"])
    ptx["multi8"] = {"file": "results/ptx/group_quant_int8_multi8_kernel.ptx", "div_lines": div_lines(k.asm["ptx"]),
                     "n_regs": k.n_regs, "n_spills": k.n_spills}
    ptx["div_only_default"] = {"div_lines": div_lines(kdiv.asm["ptx"])}
    ptx["div_only_div_rn"] = {"div_lines": div_lines(kdiv_rn.asm["ptx"])}
    out["ptx"] = ptx
    print(json.dumps(ptx, indent=1))

    fn = f"results/verify_{tag}.json"
    json.dump(out, open(fn, "w"), indent=1)
    print("wrote", fn)


if __name__ == "__main__":
    main()
