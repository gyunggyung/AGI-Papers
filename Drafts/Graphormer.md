---
id: Graphormer
category: Architecture
title: "Do Transformers Really Perform Bad for Graph Representation?"
---
# Graphormer: 트랜스포머의 그래프 정복

> **분자 구조, 소셜 네트워크 같은 그래프 데이터도 트랜스포머가 먹을 수 있다. 중심성과 공간 부호화가 핵심.**

## 1. 기존의 한계

화학 분자 구조나 소셜 네트워크 같은 그래프 데이터는 전용 신경망(GNN: Graph Neural Network)의 전유물로 여겨졌습니다. 트랜스포머는 순차적 데이터에는 강하지만, 비정형적인 그래프에는 성능이 떨어진다는 것이 기존 통념이었습니다.

## 2. 새로운 접근: 구조적 부호화

Graphormer는 두 가지 핵심 부호화를 통해 트랜스포머에 그래프 구조 정보를 주입합니다.

*   **중심성 부호화 (Centrality Encoding)**: 각 노드의 "중요도(차수)"를 어텐션 계산에 반영하여, 허브 노드와 잎 노드를 구분합니다.
*   **공간 부호화 (Spatial Encoding)**: 두 노드 사이의 최단 경로 거리를 어텐션 바이어스로 활용하여, 그래프의 위상적 구조를 인코딩합니다.
*   **성과**: OGB-LSC 벤치마크에서 1위를 달성하며 GNN의 영역을 트랜스포머가 침범할 수 있음을 증명했습니다.

## 3. 시사점

"트랜스포머는 텍스트에만 잘 맞는다"는 고정관념을 깨고, 범용적인 패턴 인식 엔진으로서의 가능성을 확인한 연구입니다.

---

## 🔗 References
*   **Paper**: [Do Transformers Really Perform Bad for Graph Representation?](https://arxiv.org/abs/2106.05234)
