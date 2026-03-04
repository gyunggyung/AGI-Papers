---
id: Qwen3_5
category: Architecture
title: "Qwen3.5: Towards Native Multimodal Agents"
---
# Qwen3.5: 네이티브 멀티모달 에이전트

> **델타넷 기반 선형 주의집중 + 희소 전문가 혼합의 하이브리드 구조로, 화면 픽셀을 직접 제어하는 행동 지능과 추론 속도를 동시에 확보.**

## 1. 기존의 한계

무거운 기존의 디코더-온리 구조(Dense Transformer)는 다중모달(텍스트+이미지+비디오)과 에이전트 기능을 동시에 수행하면 연산량이 폭발하여 속도가 현저히 떨어졌습니다.

## 2. 새로운 접근: 하이브리드 아키텍처

Qwen3.5는 근본적으로 다른 구조를 채택했습니다.

*   **델타넷(DeltaNet)**: 순환 신경망(RNN)의 진화형으로, O(N) 선형 시간 복잡도를 가진 선형 주의집중(Linear Attention) 메커니즘입니다.
*   **희소 전문가 혼합(Sparse MoE)**: 필요한 전문가만 활성화하여 연산량을 줄이면서도 모델의 총 지식 용량은 유지합니다.
*   **네이티브 멀티모달**: 텍스트를 넘어 화면의 픽셀을 직접 인식하고 제어하는 행동 기반 에이전트로 설계되었습니다.
*   **성과**: 추론 속도와 멀티모달 성능을 동시에 잡는 압도적인 결과를 보여줍니다.

## 3. 시사점

LLM의 아키텍처가 Pure Transformer에서 하이브리드(RNN+Attention+MoE)로 전환되는 트렌드의 결정적 증거입니다.

---

## 🔗 References
*   **Blog**: [Qwen3.5: Towards Native Multimodal Agents](https://qwen.ai/blog?id=qwen3.5)
