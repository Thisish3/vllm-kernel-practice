#!/bin/bash
# Dump SASS of PyTorch's DivFunctor<float> kernel and the Triton one-group kernel with Nsight Compute.
# Writes results/sass_div.txt (only the division-related lines + kernel names). The report stays outside the repo.
NCU=${NCU:-/opt/nvidia/nsight-compute/2026.3.1/ncu}
OUT=${OUT:-/root/ncu}
mkdir -p $OUT
$NCU --kernel-name-base demangled --kernel-name regex:'DivFunctor<float|group_quant' --section SourceCounters -f -o $OUT/sass python sass_target.py
$NCU -i $(ls $OUT/sass.ncu-rep*) --page source --print-source sass --csv > $OUT/sass.csv
# Triton one-group kernel separately (the demangled regex above only caught the PyTorch kernel)
$NCU --kernel-name regex:group_quant --section SourceCounters -f -o $OUT/sass_triton python sass_target.py
$NCU -i $(ls $OUT/sass_triton.ncu-rep*) --page source --print-source sass --csv > $OUT/sass_triton.csv
python3 - "$OUT" > results/sass_div.txt <<'PY'
import csv, re, sys
out = sys.argv[1]
for f in ("sass.csv", "sass_triton.csv"):
    rows = list(csv.reader(open(f"{out}/{f}")))
    print("=== kernel:", rows[0][1])
    src = [r[1].strip() for r in rows[2:] if len(r) > 1]
    hits = [i for i, s in enumerate(src) if re.match(r"(@\S+\s+)?(MUFU|FCHK)", s)]
    shown = set()
    for i in hits:
        for j in range(max(0, i - 2), min(len(src), i + 9)):
            if j not in shown:
                print(f"{j:4d}  {src[j]}"); shown.add(j)
        print("  ...")
    print("total SASS lines:", len(src), " MUFU.RCP:", sum("MUFU.RCP" in s for s in src),
          " FCHK:", sum(s.lstrip("@!P0123456789 ").startswith("FCHK") for s in src),
          " CALL:", sum("CALL" in s for s in src))
PY
