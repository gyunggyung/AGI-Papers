---
id: Molmo2
category: Architecture
title: "Molmo2: Open Vision-Language Models with Grounding"
---
# Molmo2: 행동 기반 시각 지능

> **화면 내 특정 객체를 픽셀 단위로 정확히 지목하고 추적하는 진정한 의미의 행동 기반 시각 지능을, 완전한 개방형 가중치로 공개.**

## 1. 기존의 한계

기존 시각-언어 모델(VLM)들은 이미지나 비디오의 맥락을 두루뭉술한 텍스트로 묘사하는 데 그쳤습니다. "이 사진에 고양이가 있습니다"라고 말할 수는 있어도, 고양이가 화면의 정확히 어디에 있는지(Grounding)를 지목하지 못했습니다.

## 2. 새로운 접근: 픽셀 수준 그라운딩

Molmo2는 시각적 이해를 넘어, 화면에서의 정확한 위치 지목(Pointing)과 추적(Tracking)을 지원합니다.

*   **픽셀 레벨 그라운딩**: "저 빨간 차를 가리켜봐"라는 지시에 정확한 바운딩 박스나 포인트를 반환합니다.
*   **비디오 이해**: 정지 이미지뿐 아니라 비디오에서도 객체를 추적하고 이해합니다.
*   **완전 개방형**: 모델 가중치와 데이터를 완전히 공개하여, 누구나 에이전트의 "눈"으로 활용할 수 있습니다.

## 3. 시사점

에이전트가 화면을 "보고" 행동하는 Computer Use의 핵심 부품으로 활용 가능한 오픈소스 VLM입니다.

---

## 🔗 References
*   **Paper**: [Molmo2: Open Weights and Data for Vision-Language Models with Video Understanding and Grounding](https://arxiv.org/abs/2601.10611)
