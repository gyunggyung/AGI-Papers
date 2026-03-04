---
id: AgentCgroup
category: Agents
title: "AgentCgroup: OS Resource Control for AI Agents"
---
# AgentCgroup: AI 에이전트의 OS 리소스 제어

> **다중 테넌트 클라우드에서 AI 코딩 에이전트의 메모리 피크가 15.4배 폭발. 커널 내부(eBPF)에서 마이크로초 단위로 반응하는 의도 기반 리소스 컨트롤러.**

## 1. AI 코딩 에이전트의 리소스 특성

144개의 SWE-rebench 작업을 Claude Haiku 4.5, GLM-4.7-Flash로 실행하여 분석한 결과:

*   전체 지연 시간의 **56–74%가 OS 수준 실행**(툴 실행, 컨테이너 초기화)에서 발생.
*   다중 테넌트 환경의 병목은 CPU가 아니라 **메모리**.
*   안정적 기본 사용량(~185MB) 위에 **최대 15.4배의 극단적 메모리 버스트** 발생.
*   동일 작업 반복 시 실행 시간 1.8배 차이, 작업 간 수요 **최대 20배 차이**.

## 2. 기존 리소스 관리의 3가지 불일치

| 문제 | 설명 |
|---|---|
| **세분성 불일치** | 리소스 수요는 Tool-call 단위로 급변하지만, 정책은 컨테이너 단위 |
| **반응성 불일치** | 1~2초 내 폭증하지만, 기존 시스템의 반응 속도는 밀리초~분 단위 |
| **적응성 불일치** | 매번 실행 경로가 달라 과거 기록 기반 예측 불가, OOM 시 LLM 컨텍스트 유실 |

## 3. AgentCgroup 설계

*   **세밀한 리소스 도메인 분리**: 툴 호출 경계마다 임시 하위 cgroup v2 생성.
*   **커널 내부 초고속 제어 (eBPF)**: CPU는 `sched_ext`, 메모리는 `memcg_bpf_ops` 훅으로 마이크로초 단위 스로틀링.
*   **의도 기반 동적 조율**: 메모리 부족 시 자연어 피드백으로 에이전트에게 더 적은 리소스를 쓰도록 유도.

## 4. 평가 결과

*   **생존율**: 기존 66% → AgentCgroup(BPF) **100%**.
*   **P95 할당 지연 시간**: **29% 감소**.

---

## 🔗 References
*   **Paper**: AgentCgroup: Understanding and Controlling OS Resources of AI Agents
