#!/bin/bash
# Nsight Compute on the three kernels launched by ncu_target.py (each launched twice; the 2nd launch of each is the measured one).
#   bash ncu_run.sh    (inside the venv with vllm 0.11.2 / torch 2.9.0 / triton 3.5.0)
# Writes results/ncu_metrics.csv. The .ncu-rep stays outside the repo.
NCU=${NCU:-/opt/nvidia/nsight-compute/2026.3.1/ncu}
OUT=${OUT:-/root/ncu}
mkdir -p $OUT
METRICS=gpu__time_duration.sum,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes.sum.per_second,dram__bytes_read.sum,dram__bytes_write.sum,\
sm__warps_active.avg.pct_of_peak_sustained_active,sm__maximum_warps_per_active_cycle_pct,\
smsp__inst_executed.sum,smsp__thread_inst_executed.sum,\
launch__grid_size,launch__block_size,launch__registers_per_thread,\
launch__occupancy_limit_blocks,launch__occupancy_limit_warps,launch__occupancy_limit_registers,launch__occupancy_limit_shared_mem,\
smsp__sass_inst_executed_op_global_ld.sum,smsp__sass_inst_executed_op_global_st.sum,\
l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_requests_pipe_lsu_mem_global_op_st.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,smsp__cycles_active.avg,sm__cycles_elapsed.avg
$NCU --version | tail -1
$NCU --kernel-name regex:group_quant --metrics $METRICS -f -o $OUT/group_quant python ncu_target.py
$NCU -i $(ls $OUT/group_quant.ncu-rep*) --csv --page raw > results/ncu_metrics.csv
