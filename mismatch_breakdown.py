"""Break down the Triton-vs-PyTorch int8 mismatches of verify_rtx.py by (torch quotient, Triton quotient, q values).

    python mismatch_breakdown.py   -> results/mismatch_breakdown_<gpu>.json

Same input as verify_rtx.py (seed 0, fp16 [4096, 4096] * 5, group 128).
"Triton quotient" is x / scale computed by a Triton kernel with the same expression (div.full.f32 in PTX);
it is checked to reproduce q of the real kernel exactly.
"""
import collections
import json

import torch

from group_quant_triton import group_quant_int8_triton
from verify_rtx import G, H, div_only, ref


def main():
    torch.manual_seed(0)
    x = torch.randn(4096, H, device="cuda", dtype=torch.float16) * 5
    q1, s1 = group_quant_int8_triton(x, G)
    qr, sr = ref(x, G)
    raw = (x.float().reshape(4096, -1, G) / s1.unsqueeze(-1)).reshape(4096, H)
    tq, _ = div_only(x, s1, False)
    # does trunc(clamp(Triton quotient)) reproduce the real kernel's q everywhere?
    q_from_tq = tq.clamp(-127, 127).to(torch.int8)
    mism = q1 != qr
    cats = collections.Counter()
    for t, r, a, b in zip(raw[mism].abs().tolist(), tq[mism].abs().tolist(),
                          qr[mism].int().abs().tolist(), q1[mism].int().abs().tolist()):
        cats[f"torch|x/s|={t!r} triton|x/s|={r!r} |q_ref|={a} |q_triton|={b}"] += 1
    # all elements whose quotient lands exactly on an integer in one implementation but not the other
    tq_int = tq == tq.trunc()
    raw_int = raw == raw.trunc()
    out = {
        "gpu": torch.cuda.get_device_name(),
        "q_from_triton_quotient_equals_kernel_q": torch.equal(q_from_tq, q1),
        "mismatches": int(mism.sum()),
        "categories": dict(cats.most_common()),
        "elements_quotient_differs": int((tq != raw).sum()),
        "elements_quotient_differs_and_one_is_integer": int(((tq != raw) & (tq_int | raw_int)).sum()),
        "max_abs_quotient_diff": float((tq - raw).abs().max()),
    }
    print(json.dumps(out, indent=1))
    tag = out["gpu"].replace("NVIDIA GeForce ", "").replace(" ", "_").lower()
    json.dump(out, open(f"results/mismatch_breakdown_{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
