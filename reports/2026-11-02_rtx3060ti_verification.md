# RTX 3060 Ti 재측정·원인 확인 보고 (2026-11-02 ~ 11-03, Windows 데스크톱 세션)

Mac 쪽 세션에 넘기기 위한 보고입니다. 측정값과 도구 출력에서 직접 확인한 사실만 적었습니다. 원인을 확인하지 못한 것은 그렇게 적었습니다.
같은 파일이 `vllm-kernel-practice/reports/`와 `vllm-serving-tuning-practice/reports/`에 있습니다.

## 커밋
| 저장소 | 커밋 | 상태 |
|---|---|---|
| Thisish3/vllm-kernel-practice | `188ddcd` Cross-check the T4 group-quant findings on RTX 3060 Ti | push 완료 |
| Thisish3/vllm-serving-tuning-practice | `6422fd7` Re-measure experiment 1 on RTX 3060 Ti and add n-gram spec decode | push 완료 |

## 공통 환경
- GPU: NVIDIA GeForce RTX 3060 Ti 8GB, 드라이버 581.80 (nvidia-smi 표시 CUDA Version 13.0)
- CPU i5-10400, RAM 16GB (WSL에 7.7GB), Windows 10 + WSL2 Ubuntu 22.04.2, 커널 6.6.87.2-microsoft-standard-WSL2
- Python 환경: WSL의 pip venv `/root/venv-vllm0112` — vllm 0.11.2, torch 2.9.0+cu128, triton 3.5.0
- **지시와 다른 점**: `vllm/vllm-openai:v0.11.2` Docker 이미지를 쓰지 못했습니다. Docker Desktop에는 처음부터 없었고, WSL k3s containerd에서도 지워진 상태였습니다(k3s는 켜면 재시작 루프). 이미지가 압축만 14.0GB인데 C: 여유가 28GB라 다시 받지 못했습니다. 이전 측정의 이미지는 torch 2.9.0+cu129, 이번 venv는 cu128입니다.
- Nsight Compute 2026.3.1 (2023.2.2는 WSL 미지원 에러), Nsight Systems 2023.2.3 / 2025.6.3 / 2026.3.2 (모두 GPU 커널 미기록, 아래 C 참고)
- 모델 revision: TinyLlama/TinyLlama-1.1B-Chat-v1.0 `fe8a4ea1ffedaf415f4da2f062534de366a451e6`, Qwen/Qwen2.5-1.5B-Instruct `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`

---

## 작업 1 — Triton 양자화 커널 교차 검증 (vllm-kernel-practice)

`verify_claims_colab.ipynb` 2절(셀 9·10)을 `verify_rtx.py`로 옮겨 실행했습니다. 결과는 `results/verify_rtx_3060_ti.json`에 있습니다.
시간은 `triton.testing.do_bench(return_mode='median')`을 5회 돌린 값의 중앙값입니다(µs).

### (a) 4096토큰, fp16 [4096, 4096], group_size 128
| 구현 | T4 | 3060 Ti 중앙값 | 3060 Ti 5회 | 실효 대역폭 |
|---|---|---|---|---|
| PyTorch 순정 | 2,811.9 | 1,632.3 | 1635.33, 1634.82, 1632.26, 1632.26, 1632.26 | 31.2GB/s |
| 그룹 1개, 4 warp | 485.0 | 266.2 | 266.24 ×5 | 191.0GB/s |
| 그룹 1개, 2 warp | 254.0 | 187.4 | 187.39 ×5 | 271.4GB/s |
| 그룹 1개, 1 warp | 229.2 | 170.8 | 170.74, 170.82, 170.80, 171.01, 170.75 | 297.8GB/s |
| multi1 | 422.2 | 241.7 | 241.66 ×5 | 210.4GB/s |
| multi2 | 236.4 | 144.2 | 143.71, 144.16, 144.38, 144.38, 144.22 | 352.6GB/s |
| multi4 | 205.8 | 126.0 | 125.95 ×5 | 403.8GB/s |
| multi8 | 202.8 | 126.0 | 125.95 ×5 | 403.8GB/s |
| multi16 | 214.2 | 126.0 | 125.95, 125.86, 125.95, 125.95, 125.95 | 403.8GB/s |

- 4 warp > 2 warp > 1 warp > multi8 순서는 T4와 같습니다. 4 → 1 warp 배율은 T4 2.12배, 3060 Ti 1.56배입니다.
- do_bench가 5회 모두 같은 값을 낸 칸이 여럿입니다.
- 512MiB 복사: 414.3GB/s (5회 413.8, 414.3, 414.3, 414.5, 413.8)
- 16토큰 중앙값(µs): PyTorch 31.74, w1 4.19, w2 4.32, w4 5.12, multi1 5.12, multi2 4.14, multi4 4.1, multi8 4.1, multi16 4.1
- 256토큰 중앙값(µs): PyTorch 128.67, w1 14.34, w2 15.36, w4 20.48, multi1 19.22, multi2 13.31, multi4 11.3, multi8 11.94, multi16 11.26

### 기존 `results/bench_rtx_3060_ti.json`과 비교 (같은 GPU, bench_local.py와 같은 측정법 = do_bench 기본값 mean, 5회 중앙값)
| 4096토큰 | 기존 값 (이미지 cu129) | 이번 값 (venv cu128) | 이번 5회 |
|---|---|---|---|
| PyTorch 순정 | 1,914.2 | 1,653.4 | 1653.36, 1677.06, 1635.43, 1635.44, 1672.66 |
| 그룹 1개 (=4 warp) | 319.5 | 271.4 | 271.36, 271.49, 271.18, 273.03, 269.16 |
| 그룹 8개 | 140.7 | 127.2 | 127.19, 127.30, 126.71, 126.94, 127.79 |
| 배율 | 2.27배 | 2.13배 | |
| 그룹 8개 대역폭 (사양 448 대비) | 361.4GB/s (81%) | 399.8GB/s (89%) | |

차이가 왜 생겼는지는 확인하지 못했습니다. 달라진 것은 컨테이너 → venv, cu129 → cu128, 측정 시점입니다.

### (b) PyTorch 참조 구현과의 차이 (seed 0, randn*5)
- 다른 원소 21,654개 / 16,777,216개, 최대 차이 1. T4는 21,743개, 기존 로컬은 21,690개였습니다.
- scale 동일. multi8과 single은 비트 단위로 같습니다.
- 21,654개를 전부 분류했습니다(`mismatch_breakdown.py`). 모두 그룹 최댓값 원소였고 경우는 두 가지였습니다.
  - PyTorch x/scale ±127.0, Triton ±126.99999237 → q는 127 대 126: **18,039개**
  - PyTorch x/scale ±126.99999237, Triton ±127.0 → q는 126 대 127: **3,615개**
- 따라서 "PyTorch가 정확히 ±127.000000이고 Triton은 ±126"이라는 설명은 18,039개에만 맞습니다. T4 노트북은 차이 원소를 앞의 5개만 출력했습니다.
- 전체에서 두 구현의 x/scale이 비트 단위로 다른 원소는 5,296,989개, 최대 차이는 1.53e-5였습니다.

### (c) torch.profiler 커널 수
PyTorch 순정 8개, Triton 그룹 1개 버전 1개, multi8 1개로 T4와 같습니다.

### 원인 확인 A — 나눗셈이 어떤 명령으로 컴파일됐나
PTX(num_warps=4)입니다. `.loc` 주석으로 소스 줄을 확인했습니다.
```
	mov.b32 	%r42, 0f42FE0000;                       // 127.0
	.loc	1 39 21     // group_quant_triton.py:39:21   scale = absmax / 127.0
	div.full.f32 	%r43, %r41, %r42;
	.loc	1 45 34     // group_quant_triton.py:45:34   x / scale
	div.full.f32 	%r44, %r23, %r7;
```
w1, w2, multi8에서도 `x/scale`은 모두 `div.full.f32`였습니다(스레드당 원소 수만큼 반복). PTX 파일은 `results/ptx/`에 있습니다.

SASS(ncu source 페이지, `results/sass_div.txt`):
```
Triton _group_quant_int8_kernel (w4)
  72  FMUL R10, R10, 0.0078740157186985015869   # 상수 = 1/127
  81  MUFU.RCP R4, R4
  88  FMUL R5, R4, R5
  89  FMNMX R5, R5, -127, !PT
  90  FMNMX R5, R5, 127, PT
  91  F2I.S16.TRUNC.NTZ R5, R5
PyTorch elementwise_kernel<..., DivFunctor<float>>
 269  MUFU.RCP R3, R9
 270  FCHK P0, R0, R9
 271  FFMA R4, -R9, R3, 1
 272  FFMA R3, R3, R4, R3
 273  FFMA R4, R0, R3, RZ
 274  FFMA R2, -R9, R4, R0
 275  FFMA R3, R3, R2, R4
 276  @!P0  BRA 0x500e9af90
      (FCHK에 걸리면 CALL.REL.NOINC로 별도 경로)
```
- Triton SASS(112줄)에는 FCHK와 CALL이 0개, PyTorch 나눗셈 커널에는 각각 2개 있었습니다.
- Triton 나눗셈만 `tl.math.div_rn`(PTX `div.rn.ftz.f32`)으로 바꾸면 16,777,216개 전부의 x/scale이 PyTorch와 비트 단위로 같아졌습니다.
- scale 계산은 Triton SASS에서 상수 곱(FMUL 1/127)이었고, PyTorch 쪽 커널 목록에는 MulFunctor가 있습니다. scale 값은 두 구현이 같았습니다.
- CUDA 12.2 cuobjdump로 libtorch_cuda.so를 읽으려 했지만 `cuobjdump fatal : Support for 'sm_2' has been removed.`로 실패했습니다. 그래서 PyTorch SASS는 ncu로 뽑았습니다.

### 원인 확인 B — Nsight Compute (4096토큰, 각 커널은 워밍업 뒤 두 번째 실행)
- ncu 2023.2.2: `==ERROR== Profiling is not supported on device 0 as it uses the Windows Subsystem for Linux (WSL).`
- ncu 2026.3.1: 처음엔 `ERR_NVGPUCTRPERM`. 사용자가 NVIDIA 제어판에서 "모든 사용자에게 GPU 성능 카운터 액세스 허용"을 켠 뒤 `Failed to prepare kernel for profiling / Unknown error on device 0`가 났고, `wsl --shutdown` 후에 정상 동작했습니다.

| 지표 | w4 | w1 | multi8 |
|---|---|---|---|
| grid × block | 131072 × 128 | 131072 × 32 | 16384 × 128 |
| 스레드당 원소 수 (코드 계산) | 1 | 4 | 8 |
| 시간 (ncu, 대표 수치 아님) | 304.4µs | 190.8µs | 123.6µs |
| DRAM % / GB/s | 39.0% / 169.5 | 61.6% / 267.9 | 94.8% / 411.9 |
| DRAM 읽기 / 쓰기 | 34.0 / 17.6MB | 33.6 / 17.5MB | 33.6 / 17.3MB |
| achieved occupancy | 88.0% | 28.0% | 94.5% |
| theoretical occupancy | 100% | 33.3% (SM당 블록 16개 한도) | 100% |
| 실행 명령 수 (warp) | 49,545,216 | 14,811,136 | 8,093,696 |
| 실행 명령 수 (thread) | 1,577,320,448 | 465,829,888 | 256,638,976 |
| global ld / st 명령 (warp) | 524,288 / 655,360 | 131,072 / 262,144 | 65,536 / 81,920 |
| SM throughput | 77.7% | 30.2% | 26.9% |
| 레지스터/스레드 | 17 | 18 | 20 |

- **설명되는 것**: w4 → w1에서 DRAM 바이트는 거의 같고, 실행 명령 수는 3.35배 줄었습니다(load 명령 4배, store 명령 2.5배). SM throughput은 77.7 → 30.2%, DRAM은 39.0 → 61.6%가 됐습니다. occupancy는 88 → 28%로 떨어졌는데도 빨라졌으므로 occupancy로는 설명되지 않습니다.
- **설명되지 않는 것**: w4의 SM throughput 77.7%가 SM 안의 어느 유닛에서 나온 것인지는 재지 않았습니다. 명령 수는 3.35배 줄었는데 시간은 1.6배만 줄었습니다. w1 → multi8에서는 명령 수·occupancy·DRAM이 함께 바뀌어 어느 쪽이 시간 차이를 만들었는지 나눌 수 없습니다.

---

## 작업 2 — n-gram speculative decoding (vllm-serving-tuning-practice/spec_decode)

- 모델은 Qwen2.5-1.5B-Instruct fp16, `max_model_len=4096`, `gpu_memory_utilization=0.8`, 채팅 템플릿을 적용했습니다.
- `temperature=0`, `max_tokens=min_tokens=256`, 배치 1, 프롬프트마다 5회 반복했습니다.
- spec 설정: `speculative_config={"method":"ngram","num_speculative_tokens":4,"prompt_lookup_max":4,"prompt_lookup_min":2}`
- 설정 키와 지표 이름은 설치된 소스에서 직접 확인했습니다.
  - `vllm/config/speculative.py` 44·61·67·107·110행
  - `vllm/v1/spec_decode/metrics.py` 154·163·172행 (`vllm:spec_decode_num_drafts`, `_num_draft_tokens`, `_num_accepted_tokens`)
  - `vllm/entrypoints/llm.py` 228–229행: 지정하지 않으면 `disable_log_stats=True`라 `False`로 지정했습니다.
- tok/s = 256 ÷ generate 벽시계 시간(prefill 포함)

| 워크로드 | 프롬프트 토큰 | off 중앙값 | off 5회 | ngram 중앙값 | ngram 5회 | 배율 | ms/token | 수락률 | greedy 동일 |
|---|---|---|---|---|---|---|---|---|---|
| 코드 복사 | 838 | 98.3 | 95.6, 98.1, 98.3, 98.7, 98.7 | 396.5 | 354.7, 396.5, 399.8, 392.9, 404.9 | 4.03배 | 10.17 → 2.52 | 96.2% (1010/1050) | 예 |
| 대화 1 | 43 | 98.8 | 98.8, 99.2, 98.7, 98.7, 98.9 | 93.8 | 93.8, 93.1, 93.8, 93.9, 93.6 | 0.95배 | 10.13 → 10.67 | 14.1% (45/320) | 예 |
| 대화 2 | 42 | 98.7 | 98.7, 98.8, 98.7, 98.7, 98.8 | 103.3 | 103.3, 103.0, 103.4, 103.4, 103.1 | 1.05배 | 10.13 → 9.68 | 21.5% (155/720) | 예 |
| 대화 3 | 44 | 98.6 | 98.5, 98.6, 98.7, 98.6, 98.8 | 95.4 | 95.4, 95.5, 95.3, 95.3, 95.4 | 0.97배 | 10.14 → 10.49 | 15.5% (65/420) | 예 |

- **`VLLM_ENABLE_V1_MULTIPROCESSING=0`에서는 spec decode가 동작하지 않았습니다.**
  - 이 값으로 처음 쟀을 때 ngram이 더 느렸고(복사 99.8 → 90.4 tok/s), 수락 카운터가 0이었습니다.
  - `NgramProposer.propose`를 감싸 세어 보니(`debug_metrics.py`) 256번 호출돼 매번 draft 4개를 돌려줬는데, `spec_decode_num_drafts`는 0이었습니다.
  - 기본값(1)에서는 draft 210개 중 202개가 수락됐습니다. 원인은 확인하지 못했습니다. 위 표는 기본값(1)으로 잰 값입니다.
- spec decode를 켜면 `min_p, logit_bias, and min_tokens parameters won't currently work with speculative decoding enabled.` 경고가 나옵니다(`vllm/v1/sample/logits_processor/__init__.py` 205행). 이번 측정에서는 ngram 출력도 매번 256토큰이었고 off와 같았습니다.

---

## 작업 3 — Continuous batching 재측정 (vllm-serving-tuning-practice/continuous_batching)

- 노트북 실험 1 코드를 그대로 썼습니다(`VLLM_ENABLE_V1_MULTIPROCESSING=0`, `gpu_memory_utilization=0.85`, `max_model_len=2048`, prefix caching 켬, `max_tokens=min_tokens=128`).
- n 루프를 한 프로세스에서 5라운드 반복했습니다. Colab용 libcudart 미리 로드 셀은 넣지 않았습니다.
- TinyLlama가 이 GPU에서는 bf16 커널로 실행됐습니다. T4 실행의 dtype은 기록이 없습니다.

| n | 시간 중앙값 | 시간 5회 | tok/s 중앙값 | tok/s 5회 | 시간/n |
|---|---|---|---|---|---|
| 1 | 0.966s | 0.979, 0.968, 0.962, 0.963, 0.966 | 132.5 | 130.8, 132.2, 133.0, 133.0, 132.5 | 0.966 |
| 2 | 0.970s | 0.971, 0.973, 0.969, 0.967, 0.970 | 264.0 | 263.8, 263.1, 264.3, 264.8, 264.0 | 0.485 |
| 4 | 0.976s | 0.976, 0.972, 0.973, 0.976, 0.979 | 524.8 | 524.6, 526.8, 526.0, 524.8, 523.1 | 0.244 |
| 8 | 1.018s | 1.022, 1.017, 1.017, 1.018, 1.021 | 1006.2 | 1002.0, 1006.5, 1006.7, 1006.2, 1003.2 | 0.127 |
| 16 | 1.074s | 1.075, 1.091, 1.074, 1.071, 1.073 | 1907.1 | 1905.8, 1877.1, 1907.1, 1911.6, 1907.8 | 0.067 |
| 30 | 1.353s | 1.360, 1.353, 1.352, 1.366, 1.352 | 2838.0 | 2824.3, 2838.0, 2841.3, 2810.7, 2839.4 | 0.045 |

T4와 비교하면 n=1 처리량은 63.2 → 132.5, n=30은 1,696.6 → 2,838.0 tok/s입니다. n=1 → 30 처리량 배율은 26.8배 → 21.4배, 총 시간 증가는 1.11배 → 1.40배입니다.

---

## 원인 확인 C — Nsight Systems 대신 torch.profiler

- **Nsight Systems는 실패했습니다.** 2023.2.3, 2025.6.3, 2026.3.2 모두 CUDA API 호출만 기록하고 GPU 커널은 기록하지 않았습니다. 행렬곱 10번짜리 최소 예제도 `SKIPPED: mm.sqlite does not contain CUDA kernel data.`였고, 진단 메시지는 `CUDA hardware tracing is not supported on this system. A legacy (software instrumented) trace was collected instead.`였습니다. 원인은 확인하지 못했습니다.
- 대신 vLLM 내장 torch.profiler(`VLLM_TORCH_PROFILER_DIR`, `llm.start_profile()`)로 측정한 generate 한 번을 잡았습니다. 시간은 원인 확인용이며 대표 수치가 아닙니다.
- 스텝 경계는 `Memcpy DtoH`입니다. DtoH 수가 스텝 수와 일치했습니다(128 / 128 / 256 / 54). 분석은 `prof_steps.py`로 했습니다.
- 작업 2 대상은 복사 프롬프트입니다. 작업 3은 n=1과 n=30을 잡았습니다(지시문이 "n=1과 …"에서 잘려 있어 n=30을 골랐습니다).

| | 작업 3 n=1 | 작업 3 n=30 | 작업 2 off | 작업 2 ngram |
|---|---|---|---|---|
| 스텝 수 (첫 스텝 prefill) | 128 | 128 | 256 | 54 |
| 디코드 스텝당 커널 수 | 299 | 299 | 405 | 300 |
| 디코드 스텝 GPU busy (중앙값) | 6.03ms | 8.02ms | 8.52ms | 9.85ms |
| 스텝 안 GPU 유휴 (중앙값) | 0.49ms | 1.20ms | 0.54ms | 0.22ms |
| 스텝 사이 GPU 유휴 (중앙값) | 1.26ms | 1.44ms | 1.58ms | 2.15ms |
| 프로파일러 켠 generate 시간 | 1.024s | 1.436s | 2.775s | 0.673s |

커널 시간 상위 5개(커널 시간 합 대비):
- **n=1**: gemv2T bf16 (4,4) 45.4%, cutlass wmma bf16 16x16_128x2 24.0%, gemv2T bf16 (2,4) 20.1%, flash_fwd_splitkv 3.3%, flash_fwd_splitkv_combine 1.6%
- **n=30**: cutlass tensorop bf16 gemm_relu 128x64_32x6 36.5%, ampere bf16 gemm 128x64 stages_64x4 25.6%, ampere bf16 gemm 128x64 stages_32x6 14.2%, flash_fwd_splitkv 8.2%, ampere bf16 gemm 64x64 sliced1x2 6.8%
- **spec off**: gemv2T fp16 56.0%, cutlass wmma f16 16x16_128x2 27.7%, gemv2T fp16 (variant) 5.8%, flash_fwd_splitkv 3.1%, combine 2.5%
- **spec ngram**: cutlass tensorop f16 gemm_relu 64x64_32x10 40.6%, ampere fp16 gemm 128x64 21.5%, flash_fwd_splitkv 13.0%, ampere fp16 gemm 64x64 stages_64x6 11.2% (53회), ampere fp16 gemm 64x64 sliced1x2 relu 5.7%

기타 사실: vLLM이 `Using 'pin_memory=False' as WSL is detected. This may slow down the performance.` 경고를 냈고, 트레이스의 HtoD 복사는 모두 `Pageable -> Device`였습니다.

---

## 기존 README/블로그 주장 중 이번 측정과 맞지 않는 것
1. 커널 README "Colab T4 검증": "다른 원소는 PyTorch x/scale이 정확히 ±127.000000, Triton은 ±126" → 3060 Ti에서는 21,654개 중 18,039개만 해당하고, 3,615개는 반대 방향입니다.
2. `bench_rtx_3060_ti.json` / README 표: 그룹 1개 319.5µs → 그룹 8개 140.7µs, 2.27배, 81% → 이번은 271.4 → 127.2µs, 2.13배, 89%입니다(원인 미확인).
3. README "1GiB 복사 349GB/s" → 이번 512MiB 복사는 414.3GB/s입니다. 측정법이 달라 직접 비교는 안 됩니다.
4. README 핵심 발견 1: "스레드당 원소가 1개뿐이었던 것이 주된 원인" → ncu로는 같은 바이트에서 명령 수가 3.35배 줄었다는 것까지만 확인했습니다. 그것이 시간 차이의 원인인지는 단정할 수 없습니다.
5. T4 절 "warp 4 → 1에서 2.1배" → 3060 Ti에서는 1.56배입니다(같은 방향).
6. 서빙 README 실험 1: "처리량 약 26.8배, 총 시간 1.1배" → 3060 Ti에서는 21.4배, 1.40배입니다.
7. 불일치 원소 수: T4 21,743, 기존 로컬 21,690, 이번 21,654.

## 만든/바꾼 파일
- **vllm-kernel-practice**
  - 스크립트: `verify_rtx.py`, `mismatch_breakdown.py`, `ncu_target.py`, `ncu_run.sh`, `sass_target.py`, `sass_run.sh`
  - 결과: `results/verify_rtx_3060_ti.json`, `results/mismatch_breakdown_rtx_3060_ti.json`, `results/ncu_metrics.csv`, `results/sass_div.txt`, `results/ptx/*.ptx`
  - 수정: `README.md`(RTX 3060 Ti 교차 검증 절), `.gitignore`
- **vllm-serving-tuning-practice**
  - 작업 3: `continuous_batching/exp1.py`, `continuous_batching/results/*.json`
  - 작업 2: `spec_decode/spec.py`, `spec_decode/debug_metrics.py`, `spec_decode/copy_source.py.txt`, `spec_decode/README.md`, `spec_decode/results/*.json`
  - 공통: `prof_steps.py`
  - 수정: `README.md`(실험 1 재측정 표, 실험 6), `.gitignore`
- `.ncu-repz`, `.nsys-rep`, torch 트레이스는 커밋하지 않았습니다(WSL `/root/ncu`, `/root/nsys`, `/root/prof`).

## 실패했거나 못 한 것
- **Docker 이미지 미사용**: 위 "공통 환경" 참고. 이 과정에서 중단한 docker pull이 Docker Desktop 쪽에서 계속 진행돼 C:가 가득 찼습니다. 받은 레이어를 정리하고 vhdx를 압축해 복구했습니다.
- **Nsight Systems**: GPU 커널 미기록(위 에러 원문). torch.profiler로 대체했습니다.
- **CUDA 12.2 cuobjdump**: `Support for 'sm_2' has been removed`로 실패해 ncu SASS로 대체했습니다.
