---
id: ASA_Tool_Calling
category: Agents
title: ASA: Training-Free Tool Calling
---
# ASA: Activation Steering Adapter

> **"게으른 에이전트(Lazy Agent)"를 깨우는 가장 가벼운 방법. 학습 없이(Training-Free), 단 20KB의 벡터 주입만으로 도구 호출 성능을 2배로 만듭니다.**

## 1. 논문 개요

*   **제목**: Activation Steering Adapter (ASA) for LLM Tool Calling
*   **핵심 문제**: LLM은 문맥상 도구를 써야 한다는 것을 **알면서도(Representation)**, 실제로 도구 호출 토큰을 생성하는 **행동(Behavior)으로 옮기지 않는** 경우가 많습니다. 이를 'Representation-Behavior Gap'이라고 합니다.
*   **해결책**: 모델의 파라미터를 건드리지 않고(Training-Free), 추론 시점에 내부 활성화(Activation) 값에 특정 벡터를 더해주어 강제로 도구 사용 모드를 켜는 **ASA**를 제안합니다.

---

## 2. ASA의 작동 원리

ASA는 크게 두 단계로 작동하는 **Inference-time Controller**입니다.

### 1) Steering Vector (운전대 돌리기)
*   도구 사용이 필요한 예제와 필요 없는 예제 사이의 **내부 활성화 값(Inner State) 차이**를 계산합니다.
*   이 차이만큼을 벡터(Steering Vector)로 만들어, 추론 시점에 모델의 히든 스테이트에 더해줍니다. 이는 마치 모델의 뇌에 "지금은 도구를 쓸 때야!"라는 전기 신호를 주는 것과 같습니다.

### 2) Gate & Router (똑똑한 스위치)
*   무조건 벡터를 주입하면 도구가 필요 없는 상황에서도 도구를 호출하려 할(False Positive) 위험이 있습니다.
*   ASA는 경량화된 **Router**와 **Probe**를 사용하여, 현재 입력이 도구 호출이 필요한 상황인지 실시간으로 판단하고 벡터 주입 여부를 결정합니다.

---

## 3. 실험 결과

*   **성능 향상**: Qwen2.5-1.5B 모델 기준, F1 Score가 **0.18에서 0.50으로 2배 이상 향상**되었습니다.
*   **오탐(False Positive) 감소**: 기존의 프롬프트 엔지니어링이나 파인튜닝 방식은 도구 호출을 늘리면 오탐도 같이 늘어나는 경향이 있었지만, ASA는 오탐률을 **0.15에서 0.05로 획기적으로 낮췄습니다.**
*   **극강의 효율성**: LoRA 같은 파인튜닝 방식은 수십 MB의 저장 공간과 학습 비용이 들지만, ASA는 **단 20KB**의 저장 공간만 차지하며 별도의 학습 과정이 필요 없습니다.

---

## 4. 결론 및 제언

ASA는 **"모델을 재학습시키지 않고 행동을 교정한다"**는 측면에서 매우 실용적입니다. API가 자주 바뀌거나 새로운 도구가 추가될 때마다 모델을 다시 튜닝하는 것은 비용이 많이 듭니다. ASA는 이러한 MLOps의 고통을 덜어줄 수 있는 강력한 대안입니다.

특히 **On-Device AI**나 **Local LLM** 환경처럼 자원이 제한적인 곳에서 에이전트를 구동할 때, 거대 모델 부럽지 않은 도구 사용 능력을 보여줄 수 있는 핵심 기술이 될 것입니다.

---

## 5. Tech Note
*   **구현 난이도**: 중간. PyTorch Hook을 이용해 모델 내부 활성화 값을 가로채고 수정하는 코드가 필요합니다.
*   **활용**: `transformers` 라이브러리의 `forward_hook`을 활용하여 자신만의 ASA 어댑터를 만들어볼 수 있습니다.
