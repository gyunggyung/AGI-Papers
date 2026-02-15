---
id: QED_Nano
category: Pre-training & Post-training
title: QED-Nano: Tiny Model, Big Proofs
---
# QED-Nano: 다윗이 골리앗을 이기는 법

> **"모델 크기가 깡패다"는 법칙이 깨졌다. 수학 증명에서 4B 모델이 100B 모델을 압도한 비결.**

## 1. 개요

*   **모델명**: QED-Nano
*   **스펙**: 4 Billion Parameters (Qwen 기반)
*   **성과**: IMO-ProofBench(국제수학올림피아드 증명)에서 **4B 모델** 주제에, GPT-4나 Gemini Ultra 같은 수천억 파라미터 모델들과 대등하거나 더 뛰어난 증명 능력을 보여주었습니다.

---

## 2. 승리의 비결 (Secret Science)

어떻게 4B 모델이 100B 모델을 이겼을까요? 핵심은 **"선택과 집중"** 그리고 **"도구"**입니다.

### 1) Agent Scaffold (백지장도 맞들면 낫다)
QED-Nano는 혼자 일하지 않습니다. 강력한 **Agent Scaffold** 안에서 동작합니다.
*   **Massive Inference**: 문제 하나를 풀기 위해 추론 시점에 **100만(1M) 토큰** 이상을 쏟아붓습니다. 
*   가설 생성 → 검증(Lean/Coq 같은 툴 사용) → 수정 → 재시도. 이 과정을 수없이 반복합니다.

### 2) Domain-Specific RL (Rubrics as Rewards)
범용적인 지식은 과감히 버리고, 수학 증명에만 특화된 강화학습을 수행했습니다.
*   특히 **"Rubrics as Rewards"** 기법을 사용했습니다. 단순히 답이 맞았냐 틀렸냐(Binary)가 아니라, 증명 과정의 **부분 점수(Rubric)**를 세밀하게 보상으로 주어 학습 효율을 극대화했습니다.

---

## 3. 시사점

QED-Nano는 **Small Language Model (sLLM)**의 미래를 보여줍니다.
모든 것을 다 아는 백과사전형 거대 모델(Generalist)도 필요하지만, 특정 도메인(수학, 코딩, 법률)에서는 **"작지만 툴을 잘 쓰는 전문가(Specialist)"**가 훨씬 효율적이고 강력할 수 있습니다.

Open-Yaongi 프로젝트가 지향해야 할 방향성이 바로 이것 아닐까요?
**"작게 만들고, 깊게 가르치고(RL), 도구를 쥐여줘라."**
