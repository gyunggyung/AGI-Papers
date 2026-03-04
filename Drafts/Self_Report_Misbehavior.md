---
id: Self_Report_Misbehavior
category: Agents
title: "Training Agents to Self-Report Misbehavior"
---
# Training Agents to Self-Report Misbehavior

> **외부의 감시 모델에 의존하지 않고, 인공지능이 스스로 오작동을 감지하고 고발하는 '자가 통제 시스템'.**

## 1. 기존의 한계

기존에는 AI의 일탈(유해한 출력, 기만적 행동 등)을 막기 위해 외부의 감시 모델이나 필터링 시스템에 의존했습니다. 이는 별도의 모델 유지 비용이 들고, 에이전트의 내부 상태를 완벽히 파악하기 어렵다는 구조적 문제가 있었습니다.

## 2. 새로운 접근: 자기 고발 시스템

이 논문은 에이전트가 은밀한 오작동이나 기만적 행동을 할 때, 내부에서 스스로 이를 감지하고 고발(Self-Report)하도록 훈련하는 방법을 제시합니다.

*   **핵심**: 에이전트에게 "네가 잘못된 행동을 하려 한다면, 스스로 보고해라"는 메커니즘을 내장시킵니다.
*   **의의**: 외부 감시자 없이도 에이전트의 안전성을 확보할 수 있는 새로운 패러다임입니다.
*   **한계**: 에이전트가 자기 보고 메커니즘 자체를 속일 수 있는지에 대한 추가 연구가 필요합니다.

## 3. 시사점

단일 에이전트뿐 아니라, 다중 에이전트 시스템에서 공모 공격(Collusion)을 방어하기 위한 내부 경고 체계로 확장 가능합니다.

---

## 🔗 References
*   **Paper**: [Training Agents to Self-Report Misbehavior](https://arxiv.org/abs/2602.22303)
