---
id: Nested_Learning
category: Architecture
title: Nested Learning: The Illusion of Deep Learning
---
# Nested Learning: 딥러닝은 '깊이'가 아니라 '중첩'이다

> **구글 딥마인드의 도발적인 제안. "우리가 알던 딥러닝은 틀렸다. 지능은 층을 쌓는 것이 아니라, 시간을 겹치는 것에서 나온다."**

## 1. 논문 개요

*   **제목**: Nested Learning: The Illusion of Deep Learning Architectures
*   **저자**: Google Research
*   **핵심 주장**: 기계 학습 모델을 단일 과정으로 보지 말고, 서로 다른 시간척도(Time-scale)를 가진 **여러 최적화 문제의 중첩(Nesting)**으로 봐야 합니다. 인간의 뇌가 단기 기억과 장기 기억을 각기 다른 속도로 처리하듯, AI도 구조적/알고리즘적 레벨을 분리해야 합니다.

---

## 2. Nested Learning의 철학

### The Illusion of Depth
우리는 흔히 신경망의 레이어를 깊게 쌓으면(Deep) 지능이 생긴다고 믿습니다. 하지만 이 논문은 그 깊이가 **"서로 다른 속도로 도는 루프들의 집합"**일 때 진정한 의미가 있다고 말합니다.

### Multi-level Optimization
시스템을 다음과 같이 계층화합니다.
1.  **Inner Loop (Fast)**: 입력 데이터에 즉각 반응하는 빠른 적응 (예: In-context Learning, Activation).
2.  **Outer Loop (Slow)**: 오랜 시간에 걸쳐 천천히 변하는 지식 (예: Weights, Hyper-parameters).

이 구조는 **Catastrophic Forgetting (파국적 망각)**을 해결하는 열쇠가 됩니다. 빠른 루프는 새로운 정보를 흡수하고, 느린 루프는 과거의 중요한 지식을 보존합니다.

---

## 3. Hope Architecture

이 철학을 증명하기 위해 **Hope**라는 아키텍처를 제안했습니다.
*   **특징**: Titans 아키텍처의 변형으로, **Continuum Memory System (CMS)**을 탑재했습니다. 이는 기억을 단기/장기로 딱 자르는 것이 아니라, 다양한 업데이트 속도를 가진 메모리 모듈들을 연속체(Continuum)로 배치한 것입니다.
*   **성능**: 기존 Transformer나 최신 RNN 모델들보다 Long-Context 처리 능력과 Few-shot 일반화 능력이 뛰어남을 입증했습니다.

---

## 4. 결론 및 Insight

이 논문은 AI 아키텍처 설계의 패러다임을 **"공간(Layer Depth)"에서 "시간(Time Scale)"으로** 전환하자고 제안합니다.
AGI로 가는 길은 단순히 모델을 크게 키우는 것이 아니라, 인간의 뇌처럼 **"기억을 어떻게 관리하고 갱신할 것인가"**에 대한 구조적 해답을 찾는 과정일지도 모릅니다.
