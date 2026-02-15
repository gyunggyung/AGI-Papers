# 📝 Drafts & New Summaries

This file contains drafts and summaries of new papers that have not yet been fully integrated into the main [README.md](README.md).

---

## 🤖 Agents

*   [**AI Era Cognitive Surrender**](Drafts/Cognitive_Surrender.md)
    *   **Topic**: Human-AI Interaction
    *   **Summary**: System 1 (Intuition)과 System 2 (Deliberation)를 넘어, 외부 인지 시스템인 **System 3 (AI)**에 의존하는 인간의 '인지적 항복(Cognitive Surrender)' 현상과 그 위험성을 경고한 행동과학 연구.
    *   **Key Insight**: AI가 틀린 답을 줘도 맹목적으로 따르는 현상을 실험적으로 증명.

*   [**ASA: Training-Free Tool Calling**](Drafts/ASA_Tool_Calling.md)
    *   **Topic**: Tool Use / Efficiency
    *   **Summary**: LLM이 도구 사용 시점은 알지만 행동으로 옮기지 못하는 'Representation-Behavior Gap'을 해결. 학습 없이(Training-Free) 추론 시점에 모델 내부 활성화(Activation)를 조작하여 도구 호출 성능을 극대화하는 **ASA(Activation Steering Adapter)** 제안.
    *   **Performance**: 단 20KB 용량으로 F1 Score 2배 향상.

*   [**HUMANLM: State Alignment for User Simulation**](Drafts/HUMANLM.md)
    *   **Topic**: User Simulation
    *   **Summary**: 사용자의 말투만 흉내 내는 기존 방식을 비판하며, 사용자의 **신념, 감정, 목표 등 심리적 상태(State)**를 먼저 추론하고 이를 정렬(Alignment in Latent Space)하여 완벽한 페르소나를 구현하는 프레임워크.
    *   **Benchmark**: HUMANUAL 벤치마크 공개.

*   [**SKILLRL: Skill-Augmented Reinforcement Learning**](Drafts/SKILLRL.md)
    *   **Topic**: Reinforcement Learning
    *   **Summary**: 에이전트가 겪은 성공과 실패의 경험을 **'스킬(Skill)'** 형태로 증류(Distill)하여 라이브러리에 저장하고, 이를 재사용하여 점점 더 똑똑해지는 평생 학습(Lifelong Learning) 프레임워크.

*   [**PicoClaw: Low-Cost On-Device AI**](Drafts/PicoClaw.md)
    *   **Topic**: On-Device / Edge AI
    *   **Summary**: 10MB RAM의 라즈베리 파이 Pico에서도 구동 가능한 Go 언어 기반의 초경량 AI 에이전트. Agent.cpp와 유사한 철학을 공유하며 극단적인 효율성을 추구.

---

## 🧠 Architecture

*   [**TEON vs Muon: Pre-training Optimizer**](Drafts/TEON_vs_Muon.md)
    *   **Topic**: Optimization
    *   **Summary**: 
        *   **Muon**: 각 레이어를 독립적으로 직교화하여 학습 효율을 높임.
        *   **TEON**: 여러 레이어의 그라디언트를 **3차원 텐서(Tensor)**로 묶어 한꺼번에 직교화함으로써 레이어 간 상관관계를 반영, Muon보다 더 빠른 수렴 속도 달성.
    *   **Strategy**: TEON의 강력한 최적화와 Muon 논문의 효율적 튜닝 기법(Telescoping)을 결합하는 것이 최상의 전략.

*   [**Nested Learning: The Illusion of Deep Learning**](Drafts/Nested_Learning.md)
    *   **Topic**: Architecture Philosophy
    *   **Summary**: 구글 딥마인드 논문. "딥러닝의 핵심은 깊이(Deep)가 아니라, 서로 다른 시간척도(Timescale)로 도는 **중첩된 루프(Nested Loop)**다." 인간의 기억 메커니즘을 모방한 새로운 학습 패러다임 제안.

*   [**EinFields: Neural Tensor Fields for Relativity**](Drafts/EinFields.md)
    *   **Topic**: AI for Science
    *   **Summary**: ICLR 2026 채택. 일반 상대성 이론의 시뮬레이션을 위한 신경망 아키텍처. 복잡한 4차원 시공간 데이터를 효율적으로 압축하고 연산하는 **Neural Tensor Field** 기술.

*   [**Micro GPT Code Analysis**](Drafts/Micro_GPT.md)
    *   **Topic**: Education / Implementation
    *   **Summary**: Andrej Karpathy의 `microgpt.py` (200줄) 분석. LLM의 바닥부터 이해하기 위한 최고의 교육 자료.

---

## 📚 Pre-Training & Post-Training

*   [**LIE: Length-Incentivized Exploration**](Drafts/LIE_Reasoning.md)
    *   **Topic**: Reasoning / RL
    *   **Summary**: LLM이 추론 시 "더 길게 생각하도록(Think Longer)" 유도하는 강화학습 기법. 정답을 못 맞춰도 논리적인 탐색 과정을 길게 가져가면 보상을 주어, **Self-Correction** 능력을 이끌어냄.

*   [**iGRPO: Iterative GRPO**](Drafts/iGRPO.md)
    *   **Topic**: Post-training / Reasoning
    *   **Summary**: **Self-Feedback-Driven Reasoning**. 모델이 스스로 생성한 초안(Draft)을 입력으로 받아 비평하고 수정하는 과정을 반복하며 학습하는, GRPO의 진화형.

*   [**QED-Nano: Tiny Model, Big Proofs**](Drafts/QED_Nano.md)
    *   **Topic**: Math / Reasoning
    *   **Summary**: 4B 파라미터의 작은 모델로 수학 올림피아드 증명(IMO)에서 거대 모델을 압도. **Agent Scaffold**(검색, 도구 사용 등 외부 지원)와 도메인 특화 RL의 승리.

---

## 🗂️ RAG & Knowledge

*   [**LimRank: Less is More**](Drafts/LimRank.md)
    *   **Topic**: RAG / Reranking
    *   **Summary**: 검색된 문서의 순위를 재조정(Reranking)하는 모델을 만들 때, 많은 데이터보다 **양질의 합성 데이터** 소량이 더 효과적임을 증명. 2만 개 데이터로 SOTA 달성.

---

## 💻 On-Device & Audio

*   [**Moonshine: Speech Recognition**](Drafts/Moonshine.md)
    *   **Topic**: Audio / On-Device
    *   **Summary**: Whisper Large v3보다 5배 빠르고 가벼운 실시간 음성 인식 모델. **한국어 공식 지원** 및 스트리밍 처리에 최적화되어 에이전트의 '귀' 역할로 적합.

---

## 🔥 Trends & Industry

*   [**Open Claw & The Philosophy of Engineering**](Drafts/Open_Claw_Philosophy.md)
    *   **Topic**: Insight
    *   **Summary**: Matplotlib 관리자를 비난한 AI 에이전트 사건(Open Claw)을 통해 본 **'자율성과 통제'**의 딜레마. 엔지니어링 성공 후 필연적으로 마주하게 될 철학적 질문들.

*   [**Vibe Coding (바이브 코딩)**](Drafts/Vibe_Coding.md)
    *   **Topic**: Trend
    *   **Summary**: Andrej Karpathy가 제안한 개념. 코드를 한 줄씩 짜는 것이 아니라, AI에게 자연어로 지시하고 그 결과물의 **'분위기(Vibe)'**와 작동 여부를 관리하는 새로운 프로그래밍 패러다임. "영원한 주니어"가 되지 않기 위한 기본기의 중요성 역설.
