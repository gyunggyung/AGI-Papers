---
id: Search_R1_Plus_Plus
category: Post_Training
title: "Search-R1++: How to Train Your Deep Research Agent"
---
# Search-R1++: 딥 리서치 에이전트 훈련의 교과서

> **"생각을 많이 할수록 성능이 올라간다"는 통념을 뒤집다. Fast Thinking + F1+ 보상 + REINFORCE의 삼위일체로 기존 방법을 압도.**

## 1. 프롬프트 템플릿: 생각은 짧을수록 좋다

*   **Slow Thinking** (기존): `<think>` 태그로 명시적 추론 강제 → 불필요한 추론 길이를 늘리는 꼼수로 학습 붕괴.
*   **Fast Thinking** (제안): 추론 과정 생략, `<search>`와 `<answer>` 결정만 직접 출력.
*   **결과 (Qwen2.5-7B)**: Slow 0.403 → Fast **0.422**.

## 2. 보상 함수: 답변 회피(Answer Avoidance) 문제 해결

| 보상 | Qwen2.5-7B | Qwen2.5-3B |
|---|---|---|
| F1 (순수) | 0.391 | 0.288 |
| EM | 0.422 | 0.297 |
| **F1+ (제안)** | **0.429** | **0.321** |

*   **F1+**: 검색이나 답변을 아예 안 하면 감점하는 **행동 수준 페널티** 추가.
*   수식: `R_F1+ = R_F1 − α·I[a_s = 0] − β·I[a_a = 0]`

## 3. 정책 최적화: 단순한 것이 최고다

| 알고리즘 | Qwen2.5-7B (정확도/검색 횟수) | 특징 |
|---|---|---|
| GRPO | 0.433 / 1.44 | 분산 과다로 가장 불안정, 자주 붕괴 |
| PPO | 0.422 / 1.97 | Critic 편향으로 불필요한 검색 반복 |
| **REINFORCE** | **0.437 / 1.35** | 외부 베이스라인 없이 가장 효율적 경로 학습 |

## 4. 최종 결과: Search-R1++

*   **설정**: Fast Thinking + F1+ 보상 + REINFORCE.
*   **Qwen2.5-7B**: Search-R1 0.403 → Search-R1++ **0.442** (+3.9%).
*   **Qwen2.5-3B**: Search-R1 0.289 → Search-R1++ **0.331** (+4.2%).

---

## 🔗 References
*   **Paper**: How to Train Your Deep Research Agent?
