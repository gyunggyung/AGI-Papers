---
id: PicoClaw
category: On-Device
title: PicoClaw: Low-Cost On-Device AI
---
# PicoClaw: $10 컴퓨터에서 돌아가는 에이전트

> **"AI를 돌리기 위해 H100이 필요하다는 편견을 버려라." 10MB 램으로 구동되는 초경량 엣지 에이전트의 미학.**

## 1. 프로젝트 개요

*   **프로젝트명**: PicoClaw
*   **개발 철학**: Extreme Efficiency & Minimalism.
*   **목표**: 라즈베리 파이 Pico W($6)나 Zero($10), 128MB RAM을 가진 임베디드 리눅스 보드에서도 **실용적인 AI 에이전트**를 구동하는 것.
*   **구현 언어**: Go (Golang)

---

## 2. 왜 PicoClaw인가?

오늘날 AI 트렌드는 "더 크게, 더 많이"입니다. 하지만 PicoClaw는 정반대의 길을 갑니다.

### 압도적인 효율성
*   **Memory Footprint**: 실행 시 10MB 미만의 RAM을 사용합니다. Python 기반 에이전트들이 기본적으로 수백 MB를 먹는 것과 대조적입니다.
*   **Boot Time**: 0.6GHz 싱글코어 프로세서에서도 1초 이내에 부팅됩니다.
*   **Single Binary**: 의존성 지옥(Dependency Hell) 없이 단 하나의 실행 파일로 어디서든 돌아갑니다.

### AI-Bootstrapped Development
이 프로젝트의 흥미로운 점은 개발 과정입니다. 개발자는 아키텍처 설계와 고수준의 지시만 내리고, **코드의 95%를 AI 에이전트가 직접 작성**했습니다. 이는 "AI로 만든 AI"라는 점에서도 상징적입니다.

---

## 3. 주요 기능 및 아키텍처

### Hybrid Intelligence
작은 기기에서 어떻게 똑똑한 에이전트가 가능할까요? **하이브리드 전략**을 씁니다.
*   **Brain (Cloud/Network)**: 복잡한 추론, 자연어 이해는 OpenAI API나 로컬 네트워크의 강한 모델(홈 서버)에 위임합니다.
*   **Body (Edge)**: 도구 실행, 센서 제어, 간단한 규칙 기반 판단은 기기 내에서 로컬로 처리합니다.

### Connectivity
텔레그램, 디스코드, QQ 등 주요 메신저와 연동되어, 사용자는 채팅 앱만 있으면 언제 어디서든 자신의 '작은 비서'에게 명령을 내릴 수 있습니다. 집안의 IoT 기기를 제어하거나, 서버 상태를 모니터링하는 데 최적화되어 있습니다.

---

## 4. 결론: 엣지 AI의 미래

PicoClaw는 **Agent.cpp**나 **Tiny MoA**와 같은 철학을 공유합니다. 모든 것이 클라우드에서 처리될 수는 없습니다. 프라이버시, 지연 시간(Latency), 비용 문제를 해결하려면 결국 **똑똑한 단말(Edge)**이 필요합니다.

PicoClaw는 "남는 라즈베리 파이 하나로 나만의 자비스를 만들 수 있다"는 가능성을 보여줍니다. 거대 모델 경쟁 속에서 이런 **경량화, 최적화 기술**이야말로 엔지니어링의 정수(Essence)가 아닐까요?

## Links
*   [GitHub Repository](https://github.com/Sipeed/PicoClaw)
