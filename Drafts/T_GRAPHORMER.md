---
id: T_GRAPHORMER
category: Architecture
title: "T-GRAPHORMER: Transformers for Spatiotemporal Forecasting"
---
# T-GRAPHORMER: 시공간 예측을 위한 트랜스포머

> **텍스트에 갇혀 있던 트랜스포머를 물리적 세계로 끌어내, 교통 흐름 같은 복잡한 시공간 역학을 분석하고 예측한다.**

## 1. 기존의 한계

언어 모델은 일차원적인 텍스트의 순차적 처리에만 갇혀 있었습니다. 교통 흐름, 날씨 패턴 등 공간과 시간이 얽힌 복잡한 데이터를 다루기 위해서는 전용 GNN(Graph Neural Network)에 의존해야 했습니다.

## 2. 새로운 접근: 시간 부호화 + 트랜스포머

T-GRAPHORMER는 트랜스포머에 시간 부호화(Temporal Encoding)를 결합하여 시공간 데이터를 직접 학습합니다.

*   **시간적 그래프 구조**: 도로 네트워크와 같은 공간적 관계를 그래프로 표현하고, 시간 축에 따른 변화를 동시에 포착합니다.
*   **비선형 역학 모델링**: 교통 흐름, 기상 현상 등 비선형적인 시공간 패턴을 트랜스포머의 장거리 의존성 학습 능력으로 예측합니다.
*   **범용성**: 텍스트→시공간으로 트랜스포머의 영역을 확장하는 유의미한 시도입니다.

## 3. 시사점

AI의 뇌(트랜스포머)가 텍스트를 넘어 물리적 세계의 역학을 이해하기 시작했다는 신호탄입니다.

---

## 🔗 References
*   **Paper**: [T-GRAPHORMER: Using Transformers for Spatiotemporal Forecasting](https://arxiv.org/abs/2501.13274)
