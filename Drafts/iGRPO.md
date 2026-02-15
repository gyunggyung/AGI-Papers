---
id: iGRPO
category: Pre-training & Post-training
title: iGRPO: Iterative GRPO
---
# iGRPO: 스스로 비평하며 성장하는 추론

> **"한 번에 정답을 맞히는 천재는 없다. 고쳐 쓰면서 성장하는 범재가 있을 뿐." GRPO의 진화형, Iterative GRPO.**

## 1. 개요

*   **제목**: iGRPO: Self-Feedback-Driven LLM Reasoning
*   **문제 의식**: 기존의 GRPO(Group Relative Policy Optimization)는 여러 개의 답을 생성하고 그중 좋은 것에 보상을 줍니다. 하지만 이는 **"이미 생성된 답 중에서 고르는"** 수준에 머뭅니다.
*   **제안**: 모델이 초안(Draft)을 작성하고, 스스로 비평한 뒤, 이를 반영해 **다시 작성(Refining)**하는 과정을 반복(Iterative)하며 학습하는 **iGRPO**를 제안합니다.

---

## 2. 핵심 알고리즘: Two-Stage Process

iGRPO는 학습 과정을 두 단계로 쪼갭니다.

### Stage 1: Exploration
*   모델이 문제에 대해 여러 개의 탐색적 초안(Exploratory Draft)을 생성합니다.
*   보상 모델(Reward Model)이 이 중 가장 잠재력 있는 초안을 선택합니다.

### Stage 2: Self-Reflection & Refining
*   선택된 초안을 모델에게 **프롬프트(피드백)**로 다시 넣어줍니다.
*   모델은 자신의 이전 답변을 보고 "여기서 논리적 비약이 있었네"라고 깨달으며 더 나은 최종 답변을 생성합니다.
*   이 최종 답변에 대해 GRPO 업데이트를 수행합니다.

---

## 3. 실험 결과

*   **SOTA 달성**: AIME24(85.62%), AIME25(79.64%) 등 초고난도 수학 벤치마크에서 Nemotron-7B 기반 모델로 신기록을 달성했습니다.
*   **효율성**: PPO처럼 별도의 Critic 모델을 학습시킬 필요 없이(Value-Free), GRPO의 장점인 낮은 메모리 사용량을 유지하면서 성능은 훨씬 높습니다.

---

## 4. 결론

iGRPO는 **In-Context Learning의 원리를 RL에 적용**한 성공 사례입니다.
인간도 글을 쓸 때 초고를 쓰고 퇴고를 합니다. LLM 추론 학습도 이제 "One-shot 정답"이 아니라, **"Iterative Refinement(반복적 정제)"**의 사이클을 내재화하는 방향으로 가고 있습니다.
