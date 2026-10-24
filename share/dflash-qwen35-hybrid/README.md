# DFlash for Qwen3.5 (Hybrid GDN) — Transformers 구현 + batch-64 베이스라인

Qwen3.5-4B(Gated DeltaNet 24층 + Full attention 8층)를 Target으로, 공개 드래프트 [`z-lab/Qwen3.5-4B-DFlash`](https://huggingface.co/z-lab/Qwen3.5-4B-DFlash)를 써서 **DFlash speculative decoding**을 HF Transformers에서 돌리는 코드와 Spec-Bench Math 베이스라인입니다.

- batch 1: 공식 [`z-lab/dflash`](https://github.com/z-lab/dflash) 루프에 **GDN State 롤백**을 추가
- 정적 batch(예: 64): 직접 만든 배치 엔진
- 디코딩: 그리디(temperature 0)만 지원

## 왜 공식 코드를 그대로 쓰지 않았나

공식 Transformers 백엔드는 지원 모델 목록에 Qwen3.5가 없습니다. 그대로 돌려 보면 결과가 틀립니다. 거절된 드래프트 토큰을 되돌릴 때 `DynamicCache.crop()`을 쓰는데, 이 함수는 Attention KV와 GDN conv state만 자르고 **GDN recurrent state는 되돌리지 않습니다.** 그래서 거절된 토큰이 State에 그대로 남고, Target 분포가 보존되지 않습니다(lossy).

teacher-forced 검사로 확인했습니다. 생성된 각 토큰이 같은 prefix에 대한 Target의 argmax인지 보고, 아니라면 logit 차이를 잽니다. GSM8K 5문항 × 512토큰, Qwen3.5-4B 결과입니다.

| 디코딩 | argmax와 다른 토큰 | 그 지점의 최대 logit 차이 |
|---|---|---|
| AR (기준) | 16 / 2560 | 0.125 |
| DFlash + GDN 롤백 (이 코드) | 20 / 2560 | 0.125 |
| DFlash, 공식 Transformers 경로 (롤백 없음) | 88 / 2560 | **19.5** |

logit 차이 0.125 이하는 bf16 동점(tie)이 뒤집힌 수준입니다. AR 자체도 prefill 경로와 이 정도로 어긋납니다. 19.5는 명백히 틀린 토큰입니다.

## 구성

| 파일 | 내용 |
|---|---|
| `src/dflash_qwen35.py` | batch-1 DFlash (공식 `dflash_generate` + GDN 롤백). 검증 블록의 위치별 State를 저장하는 Triton 커널(fla `fused_recurrent`와 비트 단위 일치) |
| `src/dflash_batch.py` | 정적 batch 엔진 `BatchDFlash` (AR은 `block_size=1`로 같은 엔진에서 실행) |
| `src/dflash_batch_bench.py` | Spec-Bench batch 벤치마크 (AR vs DFlash, teacher-forced 검사 옵션) |
| `src/dflash_bench.py`, `src/dflash_summary.py` | batch-1 벤치마크와 요약 |
| `src/dflash_check.py` | batch-1 정확성 검사 (위 표) |
| `scripts/setup.sh` | venv, 패키지, DFlash 코드(커밋 고정), 모델, Spec-Bench 데이터 준비 |
| `scripts/run_specbench_math_batch64.sh` | 아래 결과 재현 |
| `scripts/check_batch1.sh` | 위 정확성 표 재현 |
| `results/` | 아래 결과의 원본 (batch별 요약 JSON, 문항별 출력·수락 길이 JSONL) |

## 설치와 실행

테스트 환경은 Python 3.11, RTX 3090 24GB(sm86), NVIDIA driver 575(CUDA 12.9)입니다. 패키지 버전은 `requirements.txt`에 고정돼 있습니다(torch 2.11.0+cu128, transformers 5.20.2, flash-linear-attention 0.5.2).

```bash
bash scripts/setup.sh                       # 모델 약 10GB 다운로드 포함
bash scripts/run_specbench_math_batch64.sh  # 약 7분 (RTX 3090)
bash scripts/check_batch1.sh                # 약 3분
GPU=1 bash scripts/run_specbench_math_batch64.sh   # 다른 GPU를 쓰려면
```

`dflash_batch_bench.py`의 주요 옵션:
- `--batch-size`
- `--max-new-tokens`
- `--block-size` (기본값은 드래프트 설정의 16)
- `--category` (Spec-Bench 카테고리, 기본 `math_reasoning`)
- `--reasoning off` (thinking 끄기)
- `--check N`

첫 실행에는 Triton 커널 컴파일 시간이 들어가므로, 스크립트가 작은 batch로 warm-up을 먼저 돌립니다.

## 결과: Spec-Bench Math, batch 64

설정:
- **데이터**: Spec-Bench `math_reasoning` 80문항(GSM8K test에서 추출)
- **디코딩**: 그리디, `max_new_tokens=1024`, chat template 기본값(thinking 켜짐)
- **모델**: Qwen3.5-4B bf16 + Qwen3.5-4B-DFlash bf16(block 16)
- **배치**: 정적 batch 64로 2회 실행. 두 번째 batch는 실제 16문항에 앞 문항 48개를 채움용으로 더했습니다. 채움 문항도 디코딩과 시간 측정에는 포함하지만, 문항별 지표에서는 제외합니다.
- **하드웨어**: RTX 3090 한 장, HF eager(CUDA graph 없음)

| | AR (block 1) | DFlash (block 16) |
|---|---|---|
| Decode 처리량 | 665.1 tok/s | **945.6 tok/s (1.42×)** |
| End-to-end 처리량 (prefill 포함) | 636.2 tok/s | 879.0 tok/s (1.38×) |
| #MAT (총 토큰 / 검증 스텝) | 1 | **7.18** |
| τ (문항별 평균 수락 길이의 평균) | – | 7.43 |
| batch당 디코딩 스텝 | 1023 | 235 / 227 |
| 평균 활성 batch 크기 | 59 | 37 |
| Peak GPU 메모리 | 18.0 GB | 22.2 GB |
| DFlash 시간 비중 (드래프트 / 검증) | – | 33% / 67% |
| 평균 출력 길이, 1024 상한 도달 | 952, 54/80 | 945, 50/80 |

스텝당 수락 길이 분포(드래프트 토큰 + bonus 1개):

```
1:9.7%  2:13.9%  3:10.1%  4:7.8%  5:6.8%  6:6.1%  7:5.6%  8:4.9%
9:4.1%  10:3.7%  11:3.4%  12:2.8%  13:2.5%  14:2.3%  15:1.9%  16:14.6%
```

batch 64에서도 정확성을 확인했습니다. batch·모드마다 8개 출력을 teacher-forced로 검사한 결과, argmax와 다른 토큰이 AR 52 / 15,329, DFlash 44 / 14,343이었고 최대 logit 차이는 둘 다 0.125였습니다.

## 결과를 읽을 때 주의할 점

- **batch 64에서 speedup이 작은 이유**: 검증 한 스텝이 64 × 16 = 1,024토큰을 처리하는데, 3090에서는 이것이 연산량 병목입니다. 참고로 Qwen3.5-9B-DFlash 모델 카드는 B200·SGLang·동시성 32·block 8 조건에서 GSM8K 2.35×를 보고합니다. 이 수치보다 작게 나오는 것이 정상입니다.
- **정적 batch**: 끝난 시퀀스는 batch에서 빠지고 새 요청으로 채워지지 않습니다. DFlash는 시퀀스마다 진행 속도가 달라서 batch가 더 빨리 비고(평균 활성 37 대 59), 이것도 speedup을 낮춥니다. continuous batching 서버에서는 다르게 나옵니다.
- **HF eager 실행**: CUDA graph와 torch.compile을 쓰지 않아 CPU 오버헤드가 포함됩니다. 절대 처리량은 서빙 엔진보다 낮고, 머신 부하에 따라 달라집니다. 이 결과도 다른 사용자가 함께 쓰는 서버에서 측정했습니다. 하드웨어와 무관하게 비교할 수 있는 지표는 #MAT/τ입니다.
- **출력 길이**: thinking이 켜져 있어 출력 대부분이 1024토큰 상한에서 잘립니다. 이 벤치마크는 속도 측정용이고, 정답률을 측정하지는 않습니다.
- **드래프트 비용**: DFlash 시간의 1/3이 드래프트입니다. 대부분 15개 위치 × batch 전체에 대해 248K 어휘 전체의 logit을 계산하는 비용입니다.
- **미지원 사항**: temperature > 0 샘플링은 batch 엔진에 없습니다. 9B Target은 24GB 한 장에서 batch 64가 메모리에 들어가지 않습니다(가중치 20.5GB에 GDN State 3.2GB가 더해짐).

## 구현 요약 (batch 엔진)

- **Attention KV**: 시퀀스별 정적 버퍼에 절대 위치로 씁니다. 시퀀스마다 커밋된 길이 `start`를 따로 두고, 검증 블록에서 거절된 KV는 마스크로 가린 뒤 다음 스텝에서 덮어씁니다.
- **GDN 롤백**: 검증 패스는 커밋된 State에서 출력만 계산하고 State는 바꾸지 않습니다. 대신 블록의 k, v, g, β와 conv 입력을 저장해 둡니다. 수락 길이가 정해지면 `gdn_commit` 커널이 시퀀스마다 수락된 토큰만 재귀를 다시 돌려 State를 갱신합니다. conv state는 수락 위치에서 gather합니다. 이렇게 하면 위치별 State를 모두 저장하는 방식(4B, batch 64 기준 약 50GB) 대신 수백 MB면 됩니다.
- **드래프트**: 공식 `DFlashDraftModel`의 가중치를 그대로 씁니다. 구조도 같습니다: 컨텍스트 KV 캐시, sliding 층은 블록 내 causal, full 층은 블록 내 양방향. 시퀀스마다 새 컨텍스트 개수가 달라 padding과 마스크로 처리했습니다.
- **Prefill**: 프롬프트마다 HF 원본 forward를 돌리고, 그 캐시를 batch 버퍼로 복사합니다.

## 출처

- DFlash: Chen et al., *DFlash: Block Diffusion for Flash Speculative Decoding*, arXiv:2602.06036. 코드는 [z-lab/dflash](https://github.com/z-lab/dflash) (커밋 `07ebd93`)
- Spec-Bench: Xia et al., 2024. [hemingkx/Spec-Bench](https://github.com/hemingkx/Spec-Bench)
- GDN 커널: [flash-linear-attention](https://github.com/fla-org/flash-linear-attention)의 `fused_recurrent_gated_delta_rule`을 바탕으로 수정
