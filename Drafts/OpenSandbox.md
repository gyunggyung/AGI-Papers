---
id: OpenSandbox
category: Agents
title: "OpenSandbox: Isolated Execution Infrastructure for Agents"
---
# OpenSandbox: 에이전트를 위한 격리된 실행 환경

> **에이전트가 OS를 제어하고 코드를 실행해도 실제 시스템에 피해가 없도록, 컨테이너 기반의 완벽히 격리된 범용 실행 인프라.**

## 1. 기존의 한계

에이전트가 코드를 짜고 실행할 때, 호스트 환경이 오염되거나 보안에 심각하게 취약했습니다. `rm -rf /` 같은 위험한 명령이 실행될 수 있고, 에이전트의 행동을 완전히 통제하기 어려웠습니다.

## 2. 새로운 접근: 컨테이너 기반 샌드박스

알리바바가 공개한 OpenSandbox는 에이전트를 위한 격리된 실행 환경을 제공합니다.

*   **완벽한 격리**: 에이전트가 운영체제를 제어하고 코드를 실행해도 실제 시스템에 피해가 없습니다.
*   **컨테이너 환경**: Docker 등 컨테이너 기술을 활용하여 재현 가능한 실행 환경을 보장합니다.
*   **다국어 지원**: 다양한 프로그래밍 언어의 코드 실행을 지원합니다.

## 3. 시사점

에이전트의 자율성이 높아질수록, 안전한 실행 환경의 중요성은 더욱 커집니다. Gemini-Claw에서 경험한 `rm -rf` 우회 문제의 해결책이 될 수 있습니다.

---

## 🔗 References
*   **GitHub**: [OpenSandbox (Alibaba)](https://github.com/alibaba/OpenSandbox)
