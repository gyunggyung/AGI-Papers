---
id: RadixMLP
category: Architecture
title: "RadixMLP: Intra-batch Deduplication for Causal Transformers"
---
# RadixMLP: Intra-batch Deduplication

> **동일한 시스템 프롬프트를 매번 재계산하는 비효율을 트리 구조로 제거하여, 대규모 서빙의 추론 병목을 알고리즘 단에서 파괴.**

## 1. 기존의 한계

동일한 시스템 프롬프트를 가진 수천 개의 요청이 들어와도, 모델은 앞부분의 똑같은 연산을 매번 반복했습니다. KV 캐시가 이를 줄여주지만, MLP 계층은 여전히 중복 연산이 발생합니다.

## 2. 새로운 접근: 접두사 중복 제거

RadixMLP는 공유되는 접두사 프롬프트의 연산을 트리(Radix Tree) 구조로 압축하여 중복을 원천 제거합니다.

*   **배치 내 중복 제거(Intra-batch Deduplication)**: 같은 배치 안에서 공통된 접두사를 자동으로 인식하고, 한 번만 계산합니다.
*   **MLP 최적화**: Attention뿐 아니라 MLP 계층에서도 중복 연산을 제거하여 전체 추론 비용을 절감합니다.
*   **알고리즘적 해결**: 하드웨어 변경 없이, 순전히 소프트웨어 알고리즘으로 병목을 돌파합니다.

## 3. 시사점

SGLang의 RadixAttention과 결이 같은 "재사용의 미학"이며, 대규모 API 서빙 환경에서 즉시 적용 가능한 실용적 최적화입니다.

---

## 🔗 References
*   **Paper**: [RadixMLP - Intra-batch Deduplication for Causal Transformers](https://arxiv.org/abs/2601.15013)
