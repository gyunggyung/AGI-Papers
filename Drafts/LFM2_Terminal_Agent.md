---
id: LFM2_Terminal_Agent
category: Agents
title: "LFM2-24B Terminal Agent: FP8 LoRA SFT + RL Strategy"
---
# LFM2-24B 터미널 에이전트: 소수 정예 SFT + RL 전략

> **H100 1대에서 FP8 LoRA로 Terminal-Corpus 4,500개 SFT(1에폭) → 실환경 RL(GRPO/PPO)로 자가 치유형 터미널 에이전트 구축. 활성 파라미터 2.3B의 미친 속도.**

## 1. 왜 LFM2-24B-A2B인가?

*   **24B 전체 / 2.3B 활성 (MoE)**: CPU에서도 초당 112토큰 (AMD CPU 기준).
*   **하이브리드 아키텍처**: 30개 Conv 블록 + 10개 GQA 어텐션, 40 레이어.
*   **32K 컨텍스트**: Terminal-Corpus의 궤적 데이터(평균 17K 토큰)와 정확히 부합.
*   **네이티브 Function Calling**: 에이전트 파이프라인에 즉시 활용.
*   H100 80GB VRAM: FP8 로드 시 ~24GB, LoRA+KV Cache 포함해도 여유.

## 2. SFT 전략: 소수 정예 1에폭

### 데이터 샘플링
Terminal-Corpus의 9개 도메인에서 도메인당 **500개**, 총 **~4,500개** 정밀 추출.

### 왜 적은 데이터 + 1에폭인가?
*   **목적은 포맷 강제화만**: `analysis → plan → commands` JSON 구조 안착.
*   과적합 방지: 26만 개의 0.1에폭보다, 4,500개의 1에폭이 포맷 학습에 확실.
*   RL의 탐험(Exploration) 본능 유지: SFT에서 정답을 외우면 RL에서 경직.

### 학습 세팅
*   **FP8 LoRA** (QLoRA 아님): H100의 네이티브 FP8 텐서코어로 최대 3배 빠름.
*   **소요 시간**: H100 1대에서 밥 한 끼 수준.

## 3. RL 전략: 실환경 강화학습

### 왜 RL이 터미널에 찰떡인가?
*   **명확한 보상**: Exit Code 0 (성공) = Reward +1, 에러 = -1.
*   **RLVR**: 컴파일/테스트 통과 여부로 자동 검증 → 인간 개입 불필요.

### 파이프라인
1.  SFT 완료 모델을 Docker 샌드박스에 배치.
2.  GRPO/PPO로 LoRA 가중치만 업데이트.
3.  활성 2.3B 덕분에 Rollout(추론) 속도가 미친 듯이 빠름.

## 4. 비용 추산

| 단계 | 장비 | 데이터 | 시간 | 비용 |
|---|---|---|---|---|
| SFT | H100 1대 | 4,500개 × 1에폭 | ~2시간 | ~$6 |
| RL | H100 1대 | 실환경 루프 | ~48시간 | ~$144 |
| **총합** | | | | **~$150** |

---

## 🔗 References
*   **LFM2-24B**: [LiquidAI/LFM2-24B-A2B](https://huggingface.co/LiquidAI/LFM2-24B-A2B)
*   **Terminal-Corpus**: [nvidia/nemotron-terminal](https://huggingface.co/collections/nvidia/nemotron-terminal)
*   **FlashOptim**: 학습 시 옵티마이저 메모리 36% 절감 적용 권장.
