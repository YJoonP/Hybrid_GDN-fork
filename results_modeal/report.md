# DFlash-b16 (SpecLA 롤백) — Qwen3.5-4B, RTX 3090

**상태: Phase 3(정확성 검증)에서 중단. Phase 4(본 실행)와 전체 State 저장 방식과의 속도 비교는 수행하지 않았습니다.**
원인과 근거는 전체 State 저장 쪽과 같습니다. `results_fullstate/report.md` §3을 참고하세요.

## SpecLA 롤백 정의 (이 구현)

- **기록**: 매 검증 스텝에서 시퀀스별 수락 길이 AL(= anchor + 수락된 draft 수 = 이번 스텝에 확정된 토큰 수)을 온라인 히스토그램에 더합니다. 지금까지의 최빈값을 m이라 합니다.
- **저장**: 검증 패스는 recurrent·conv state를 **위치 m에서만 1개** 저장합니다(전체 State 방식은 16개).
- **복원**:
  - AL = m: 저장본을 복사합니다.
  - AL > m: 저장본에서 출발해 블록의 [m, AL) 토큰을 재계산합니다.
  - AL < m: 검증 중 수정되지 않은 원래 State S_0에서 [0, AL) 토큰을 재계산합니다.
  - 재계산은 검증 패스가 이미 만든 k, v, g, β를 재사용하는 Triton 커널(`gdn_replay`)로 하고, 연산 순서는 fla `fused_recurrent`와 같습니다.
  - conv state는 AL ≠ m이면 검증 패스의 conv 입력에서 창을 가져옵니다.
- **snapshot 버퍼**: 층당 1개 × batch. batch 64에서 3.3 GB, 128에서 6.6 GB입니다(전체 State 방식은 52.7 GB, 105 GB).

## Phase 3 결과

- **State 검증 통과.**
  - 복원 recurrent state는 같은 입력으로 재계산한 값과 오차 **0**이고, conv state도 오차 0입니다.
  - AR 기준 최대 상대오차는 recurrent 0.041, conv 0.067로 잡음 바닥과 같습니다.
- **출력 검증 39/50 일치로 불합격.** 불일치 11건은 모두 logit 1·2위 차이 0.0 또는 0.125인 동률 지점입니다.
- 이 50문항의 출력은 전체 State 저장 방식과 **50/50 동일**합니다.

## 결정 기록 (SpecLA 고유)

1. **히스토그램 범위**: run마다 초기화하고, 과거 스텝만 씁니다(인과적). warmup과 보정 스텝은 포함하지 않습니다.
2. **첫 관측 전 m**: block/2 = 8로 둡니다. 동률이면 작은 AL을 고릅니다.
3. **히스토그램에 넣는 AL**: max_new_tokens 상한으로 잘리기 전 값(acc + 1)을 씁니다. 복원에는 잘린 확정 수를 씁니다.
4. **m의 공유 범위**: 시퀀스별이 아니라 batch 전체가 같은 m을 씁니다.
5. **AR 셀**: SpecLA 실행에서는 `--reuse-ar-from results_fullstate`로, env.json이 같을 때 전체 State 쪽 AR 셀을 재사용하도록 구현했습니다(재사용 시 셀 JSON에 `reused_from` 기록). 아직 AR 셀이 없어 재사용은 일어나지 않았습니다.

나머지 결정(블록 정의, bonus 처리, KV truncate, 토큰 계산, warmup, 시간 측정 방식)은 `results_fullstate/report.md` §4와 같습니다.
