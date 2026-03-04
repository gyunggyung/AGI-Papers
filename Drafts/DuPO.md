---
id: DuPO
category: Post_Training
title: "DuPO: Dual Preference Optimization for LLM Self-Verification"
---
# DuPO: 검증 불가능한 영역에서의 자기 검증

> **번역처럼 정답이 없는 영역에서, 역번역(Back-Translation)을 보상으로 활용하여 외부 데이터 없이 모델이 스스로 정합성을 학습하는 Self-Supervised RL.**

## 1. 기존의 한계

코딩이나 수학은 컴파일러나 정답이라는 명확한 검증 도구(Verifier)가 있어 RLVR(Reinforcement Learning with Verifiable Rewards) 적용이 쉽습니다. 반면 번역은 정답이 모호합니다. 모델이 그럴듯한 오역을 해도 이를 잡아내 패널티를 부여할 방법이 마땅치 않았습니다.

## 2. DuPO의 해법: 일반화된 쌍대성

DuPO는 입력과 출력의 관계를 역으로 이용하는 '일반화된 쌍대성(Generalized Duality)'으로 이 난제를 돌파합니다.

*   **Forward**: 원문(x) → 번역문(y)
*   **Backward (Dual)**: 번역문(y) → 원문 복원(x')
*   **보상(Reward)**: "네가 번역한 문장(y)이 정확하다면, 역번역 시 원문(x)의 의미가 보존되어야 한다." 이 복원 정확도가 보상이 됩니다.

## 3. 성과 (Seed-X-7B)

*   **데이터 효율성**: 언어당 약 1,000개 프롬프트 + 검증용 약 7,000개로 성능 향상.
*   **성능**: 28개 언어, 756개 번역 방향에서 평균 COMET 점수 +2.13 상승.
*   **비교**: 7B 모델이 GPT-4o, DeepSeek-R1과 동등한 번역 품질.
*   **한계**: PPO(인간 선호 데이터 사용) 대비 약 0.7% 낮지만, 비용 대비 PPO 성능의 99% 이상을 따라잡음.

## 4. 시사점

르쿤의 "에너지 최소화를 통한 자기 검증" 비전을 현재 LLM 구조 안에서 가장 현실적으로 구현한 연구입니다. GPU Poor 환경에서 작은 모델의 번역 성능을 끌어올리는 핵심 전략이 될 수 있습니다.

---

## 🔗 References
*   **Paper (DuPO)**: [DuPO: Enabling Reliable LLM Self-Verification via Dual Preference Optimization](https://arxiv.org/abs/2508.14460)
*   **Paper (Seed-X)**: [Seed-X-7B](https://arxiv.org/abs/2507.13618)
*   **GitHub**: [ByteDance-Seed/Seed-X-7B](https://github.com/ByteDance-Seed/Seed-X-7B)
