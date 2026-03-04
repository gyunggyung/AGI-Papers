---
id: mHC
category: Architecture
title: "mHC: Manifold-Constrained Hyper-Connections"
---
# mHC: ResNet의 10년 묵은 한계를 깨다

> **딥시크가 1967년 수학(Sinkhorn-Knopp)으로 ResNet(잔차 연결)의 근본적 한계인 신호 폭발 문제를 해결. 3,000배 → 1.6배로 억제.**

## 1. 배경: Hyper-Connections의 등장과 문제

지난 10년간 Transformer의 정보 흐름은 1차선 도로(C 차원)였습니다. 최근 연구들은 이 도로를 n배(n × C) 넓히면 성능이 좋아진다는 것을 발견했습니다(Hyper-Connections).

*   **문제 (Signal Explosion)**: 도로만 넓혔더니 사고가 터졌습니다. 레이어 당 5%의 미세한 신호 증폭이 60개 레이어를 거치면, 최대 **3,000배**까지 폭발합니다. 학습이 불가능합니다.

## 2. 해결: 60년 전 수학의 부활

DeepSeek은 Sinkhorn-Knopp 알고리즘(1967)으로 이 문제를 해결합니다.

*   **Doubly Stochastic Matrix**: 연결 행렬의 행의 합과 열의 합이 모두 1이 되도록 강제합니다.
*   **비유**: 차선 변경은 허용하되, 차량 복제는 금지. 들어온 차량 수 = 나가는 차량 수.
*   **결과**: 3,000배 폭발 → **1.6배** 수준으로 억제. 수학적 제약 하나로 학습 안정성 확보.

## 3. 시스템 최적화: 현실을 벗어나지 않는 공학

도로를 4배 넓혔으니, 메모리도 4배 필요합니다. VRAM이 터지지 않도록:

*   **Kernel Fusion (TileLang)**: 여러 연산을 하나의 커널로 합쳐 메모리 대역폭 낭비 감소.
*   **Selective Recomputing**: 중간값을 저장하지 않고, 역전파 때 그때그때 다시 계산. 메모리를 아끼기 위해 연산량을 희생.
*   **Trade-off**: 학습 시간 약 6.7% 증가, 하지만 성능과 안정성 확보.

## 4. 패러다임 전환: Micro → Macro Design

Attention이나 FFN 같은 블록 내부(Micro Design) 최적화에서, 블록 간 연결 배선(Macro Design) 최적화로의 관점 전환입니다.

---

## 🔗 References
*   **Paper**: [mHC: Manifold-Constrained Hyper-Connections](https://arxiv.org/pdf/2512.24880)
