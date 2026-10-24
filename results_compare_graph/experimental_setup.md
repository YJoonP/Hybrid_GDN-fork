# Experimental Setup — DFlash GDN 롤백 비교 (CUDA graph 엔진)

## 1. 하드웨어·소프트웨어

| 항목 | 값 |
|---|---|
| GPU | NVIDIA GeForce RTX 3090 1장 (24 GB, compute capability 8.6). 모든 셀을 같은 GPU(GPU 0)에서 측정 |
| 드라이버 / CUDA | 575.57.08 / CUDA runtime 12.8 (torch 빌드), nvcc 12.6 |
| Python / PyTorch | 3.11.17 / 2.11.0+cu128 |
| Transformers | 5.20.2 |
| Triton | 3.6.0 |
| flash-linear-attention | 0.5.2 (AR decode의 GDN 재귀 및 State 검사 기준식) |
| DFlash 코드 | z-lab/dflash 커밋 `07ebd93` (`pip install -e --no-deps`) |
| attention 구현 | torch SDPA, 명시적 boolean mask. prefill은 HF sdpa |

GPU 클럭 고정 등의 별도 제어는 하지 않았습니다.

## 2. 모델

| 역할 | 모델 | 구성 |
|---|---|---|
| Target | `Qwen/Qwen3.5-4B` (bf16) | 32층 = Gated DeltaNet(GDN) 24층 + full attention 8층. GDN State: 층마다 32 head × 128 × 128 fp32 (시퀀스당 2 MiB/층, 24층 48 MiB). conv State: 8192 채널 × 3, bf16 |
| Drafter | `z-lab/Qwen3.5-4B-DFlash` (bf16) | 6층(sliding-window 5 + full 1), block size 16 (anchor 1 + draft 15), mask token 248077. target 8개 층(1, 5, …, 29)의 hidden state를 문맥으로 사용. 임베딩·lm_head는 target과 공유 |

## 3. 데이터와 생성 설정

| 항목 | 값 |
|---|---|
| 데이터 | GSM8K test(1,319문항)에서 비복원 추출한 256문항, `random.Random(42).sample` (`data/gsm8k_256_indices.json`) |
| 프롬프트 | zero-shot, chat template 1턴, `enable_thinking=False`. 길이 38~174토큰(중앙값 69.5) |
| 생성 | 그리디(temperature 0), 시퀀스당 새 토큰 128개, EOS 무시 |
| decode 토큰 수 | 첫 토큰은 prefill이 생성하므로 decode가 만든 127개만 셈. run당 256 × 127 = 32,512토큰 |

## 4. 배치 방식

- **continuous batching:** 시퀀스 슬롯 수 = batch 크기(1 / 8 / 64 / 128). 슬롯이 비면 대기열의 다음 문항을 바로 넣고, 끝난 시퀀스는 즉시 빼서 나머지 행을 앞으로 당깁니다. 256문항을 모두 끝내는 것이 한 run입니다.
- **prefill:** 새 시퀀스마다 HF 기본 forward(DynamicCache, batch 1)로 프롬프트를 처리한 뒤 엔진 버퍼(KV, GDN State, conv State)에 복사합니다. prefill 시간은 처리량에서 빼고 따로 기록합니다.

## 5. 엔진: CUDA graph 실행

- **녹화 단위:** decode 한 스텝 전체를 CUDA graph 하나로 녹화합니다.
  - DFlash: 초안 → (confidence·저장 위치 계산) → 검증 → 저장 → 수락·토큰 기록 → 복원 → 길이 갱신
  - AR: forward → 토큰 기록 → 길이 갱신
- **batch 구간(bucket):** 1, 2, 4, …, batch 크기마다 graph를 하나씩 녹화합니다(예: batch 8 → {1, 2, 4, 8}). 살아 있는 시퀀스 수 n에 대해 n 이상인 가장 작은 graph를 재생하고, 남는 행은 빈 행으로 둡니다. 모든 graph는 메모리 pool 하나를 공유합니다. 녹화 전에 임시 State로 2회 미리 실행하고, 녹화가 끝나면 상태를 초기화합니다.
- **고정 모양:**
  - target의 full attention은 항상 버퍼 전체 길이(max_len = 최대 프롬프트 174 + 128 + 2×16 + 1 = 335)에 대해 계산하고 mask로 가립니다.
  - drafter 문맥은 항상 최근 16개 위치입니다. 프롬프트의 drafter 문맥(K/V)은 시퀀스를 넣을 때 미리 기록하고, 이 시간은 decode 시간의 draft 구간에 포함합니다.
- **GDN 계산:**
  - 검증: 자체 Triton 커널(`_gdn_verify_kernel`, fp32 State, fla fused_recurrent와 같은 연산 순서). 확정 State S_0은 읽기만 합니다.
  - 복원 재계산: `_gdn_replay_kernel`
  - AR decode: fla `fused_recurrent_gated_delta_rule`
- **호스트 동기화:** 스텝마다 graph 재생 뒤 확정 토큰 수를 한 번 읽습니다.

## 6. 비교 대상

| 방식 | 검증 중 저장 | 수락 뒤 복원 | 설정 |
|---|---|---|---|
| AR | – | – | 스텝당 1토큰, 같은 엔진·같은 배치 방식 |
| fullstate | 16개 위치 전부의 State | 수락 위치 저장본 복사 | |
| SpecLA | 없음 (검증의 k, v, g = log α, β와 conv 입력 창을 보관) | S_0에서 수락 토큰만큼 재계산 | 논문 5.1절 방식, 지연 반영 없음 |
| modeal | 최빈 수락 길이 위치 m 1칸 | 복사, 또는 m·S_0에서 재계산 | 수락 길이 히스토그램은 run마다 초기화, 시작 m = 8, 동률이면 작은 값 |
| conf-K A ×8 | confidence로 고른 위치 + 마지막 위치 | 가장 가까운 저장본에서 재계산 | c_k = 드래프트 softmax의 선택 토큰 확률. top-K(K = 1, 2, 4): P(j) = c_1…c_j(1 − c_{j+1}) 상위 K. θ ∈ {0.1, 0.3, 0.5, 0.7, 0.9}: 앞에서부터 연속으로 c_i ≥ θ인 개수 |
| conf-K B ×8 | A와 같음 | A와 같음 | c_k = sigmoid(MLP([drafter hidden ; 앞 토큰 target 임베딩])), MLP 5120→256→1 (`confk-head-B/heads/head_mlp.pt`). head 계산은 draft 구간에 포함 |
| oracle | 실제 확정 위치 1칸 | 저장본 복사 | 정답 = 같은 엔진·같은 batch의 SpecLA run 1 기록(시퀀스 순서 번호 기준). 정답 전송 시간은 측정에서 제외 |

conv State는 모든 방식이 검증의 conv 입력 창에서 확정 위치 직전 3개를 모아 복원합니다. attention KV는 확정 길이만 갱신합니다(뒤쪽은 mask 후 덮어씀).

## 7. 측정 절차

1. **Phase 0:** 환경 기록(`env.json`).
2. **Phase 3:** 정확성 검사(8절). 통과해야 측정을 진행합니다.
3. **Phase 4:** batch 1 / 8 / 64 / 128 각각
   - 메모리 할당과 graph 녹화. 여기서 메모리가 부족하면 그 셀은 OOM으로 기록합니다.
   - warmup: 첫 batch 크기만큼 문항을 넣고 decode 5스텝 실행 후 초기화. 측정에서 제외합니다.
   - 256문항 전체 run을 3회 반복합니다.
- AR 셀은 fullstate 결과 폴더에서 한 번 측정하고, 환경 기록이 같은 나머지 방식은 이를 재사용합니다.
- 각 방식(결과 폴더)은 별도 프로세스로 GPU 0에서 순서대로 실행했습니다(동시 실행 없음).

## 8. 정확성 검사 (Phase 3)

- **State 검사:** 10문항(seed 0), batch 1. 한 스텝 뒤 복원한 GDN recurrent State·conv State를, 같은 검증 입력으로 스텝 전 State에서 fla fused_recurrent로 다시 계산한 값과 비교합니다. **최대 절대 오차가 정확히 0이어야 통과**합니다.
- **출력 검사:** 앞 50문항, batch 1. DFlash와 AR의 출력 128토큰을 비교합니다. 불일치가 있으면, 처음 갈라지는 지점마다 target의 1·2위 logit 차이가 top-1 크기 기준 bf16 2 ulp 이내(동률)일 때만 통과로 봅니다(사용자 승인 기준).
- **결과:** 모든 방식이 State 오차 0, 출력 36/50 일치였습니다. 불일치 14개는 모두 동률 기준 안이었습니다.
- **추가 확인:** 같은 batch 안에서 방식·반복이 달라도 60번의 DFlash run 출력이 256문항 모두 같았습니다.
- **conf-K B 연결 확인:** head 재구현이 패키지 기대 출력과 오차 0이었습니다. GSM8K test 1·4번 문제로 엔진 연결을 확인했습니다(drafter hidden 코사인 ≥ 0.998, 다른 초안 토큰은 모두 동률 지점).

## 9. 지표

| 지표 | 정의 |
|---|---|
| decode 처리량 (tok/s) | run의 decode 토큰 수(32,512) ÷ decode 시간 |
| decode 시간 | 스텝마다 graph 재생 전후 동기화 사이의 실제 경과 시간 합 + 끝난 시퀀스 제거 시간 + (graph) 프롬프트 drafter 문맥 기록 시간. prefill 제외 |
| speedup | 같은 엔진·같은 batch의 AR 처리량 대비 배수 |
| τ | decode 토큰 ÷ (시퀀스별 스텝 수의 합). 스텝당 확정 토큰 수 |
| TPOT (ms) | 시퀀스별 (decode 시간 ÷ decode 토큰)의 평균 |
| 구간 시간 | graph 안에 넣은 CUDA 이벤트로 잰 GPU 시간: draft / verify / save / accept / restore / KV truncate. host gap = 스텝 경과 시간 − 구간 합 |
| peak 메모리 | `torch.cuda.max_memory_allocated` |
| 저장본 크기 | 롤백용으로 할당한 State 버퍼 크기 (OOM 셀은 필요량을 계산) |
| 저장본 적중률 (conf-K, oracle) | 복원이 저장본 복사만으로 끝난 행·스텝의 비율. 그 외 S_0부터 재계산 비율, 행당 재계산 토큰 수도 기록 |

표의 값은 모두 3회 평균이고, ± 는 3회의 표준편차입니다(모든 셀이 평균의 1% 미만).

## 10. 주의·한계

- attention을 버퍼 전체 길이로 계산하므로 bf16 덧셈 순서가 eager 엔진과 다릅니다. 그래서 eager와 graph의 출력은 75~84%만 같습니다. 같은 엔진 안에서는 방식과 무관하게 같습니다. 같은 이유로 큰 batch의 AR은 eager보다 8~10% 느립니다.
- conf-K 예측 통계는 방법 A 전체와 방법 B top-1에서 run 2가 0으로 기록됐습니다. graph가 녹화 때의 텐서 주소에 계속 쓰는데, 통계 초기화가 새 텐서를 만들어서 생긴 문제입니다. 이후 수정했고, 해당 셀은 값이 있는 run(1, 3; 서로 정확히 같음)만 씁니다. 처리량과 출력에는 영향이 없습니다.
- oracle의 첫 측정은 정답 기록 연결 오류(문제 번호 대 순서 번호)로 무효 처리했습니다(`results_superseded/`). 표의 값은 고친 뒤 다시 잰 결과입니다(적중 100%).
- 데이터는 GSM8K 짧은 문맥, 128토큰 한 조건뿐입니다.

## 11. 재현

```bash
# 방식 하나 (예: SpecLA), GPU 0
GPU=0 src/run_gpu.sh logs/graph/specla.log .venv/bin/python run_all.py --engine graph --rollback specla \
  --phases 0,3,4 --accept-tie-mismatch --reuse-ar-from results_graph_fullstate
# conf-K 변형:  --rollback confk  --select topk --k 2   또는  --select thr --thr 0.5
# conf-K B:     --rollback confkb (+ 같은 선택 옵션)
# oracle:       --rollback oracle --oracle-from results_graph_specla
# 분석
.venv/bin/python analyze_all.py --engine graph && .venv/bin/python output_identity.py
```

결과는 `results_graph_<방식>/`(`env.json`, `state_check.json`, `output_check.json`, `cells/`, `raw/`), 비교 표와 그림은 `results_compare_graph/`에 있습니다.
