---
id: Agentic_File_System
category: Agents
title: "Everything is Context: Agentic File System"
---
# Everything is Context: Agentic File System

> **유닉스의 "Everything is a file" 철학을 에이전트에 이식하여, 지식·도구·기억을 파일 시스템처럼 마운트하고 관리하는 진정한 운영체제 아키텍처.**

## 1. 기존의 한계

기존 에이전트들은 프롬프트 길이에 의존하거나, 파편화된 검색 증강(RAG)으로 단기 기억만 얕게 활용했습니다. 컨텍스트 윈도우를 넘어서는 지식은 접근 자체가 불가능했습니다.

## 2. 새로운 접근: Context Engineering

이 논문은 컨텍스트 엔지니어링(Context Engineering)이라는 새로운 관점에서, 에이전트의 모든 상태를 유닉스 파일 시스템으로 추상화합니다.

*   **마운트(Mount)**: 외부 지식 베이스, API 도구, 과거 대화 기억을 파일 시스템의 디렉토리처럼 마운트합니다.
*   **영속성**: 세션이 끝나도 지식이 보존되는 영구적 기억 구조를 제공합니다.
*   **모듈성**: 필요에 따라 지식 모듈을 탈착(mount/unmount)할 수 있어, 에이전트의 역할을 동적으로 변경 가능합니다.

## 3. 시사점

에이전트를 단순한 챗봇이 아니라, 자체적인 파일 시스템과 메모리를 가진 독립적인 운영체제로 진화시키는 비전을 제시합니다.

---

## 🔗 References
*   **Paper**: [Everything is Context: Agentic File System Abstraction for Context Engineering](https://arxiv.org/abs/2512.05470)
