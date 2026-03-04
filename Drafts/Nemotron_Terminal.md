---
id: Nemotron_Terminal
category: Agents
title: "Nemotron-Terminal: Scaling LLM Terminal Capabilities"
---
# Nemotron-Terminal: 32B가 480B를 이기는 터미널 데이터 공학

> **모델 크기보다 고품질 터미널 궤적 데이터가 중요하다. 32B SFT 모델이 Terminal-Bench 2.0에서 480B Qwen3-Coder를 능가(27.4%).**

## 1. 문제: 터미널 에이전트 훈련 데이터의 부재

*   최첨단 터미널 에이전트(Claude Code, Codex CLI)의 데이터 전략은 영업 비밀.
*   터미널 궤적(Trajectory) 수집은 Docker 환경 구축 + 다중 턴 상호작용 필요 → 고비용.

## 2. 해결: Terminal-Task-Gen 파이프라인

### Dataset Adaptation
기존 수학·코드·SWE 프롬프트 데이터를 Terminal-Bench 포맷으로 변환.

### Synthetic Task Generation
*   교사 모델: **DeepSeek-V3.2**로 9개 도메인(데이터 과학, 보안, 시스템 관리 등) 태스크 합성.
*   **도메인별 도커 이미지 9개**를 돌려 사용 → 비용·속도 효율 극대화.

## 3. 학습 디테일 (Full Fine-Tuning, SFT Only)

*   **RL 미적용** — 향후 연구 과제로만 남김.
*   **방식**: Full Fine-Tuning (LoRA 아님), veRL 프레임워크.
*   **데이터**: Terminal-Corpus **264,207개**, 궤적당 평균 17,363 토큰.

| 모델 | GPU 수 | 배치 | LR | 에폭 | 최대 시퀀스 |
|---|---|---|---|---|---|
| 8B, 14B | 32대 (4노드×8) | 128 | 2e-5 | 2 | 32K |
| 32B | 128대 (16노드×8) | 128 | 2e-5 | 2 | 32K |

## 4. 결과

*   **Nemotron-Terminal-32B**: Terminal-Bench 2.0에서 **27.4%** — 480B Qwen3-Coder 능가.
*   **핵심 교훈**: 모델 크기 < 고품질 터미널 궤적 데이터.

## 5. 공개 자료
*   **데이터셋+모델**: [nvidia/nemotron-terminal (HuggingFace Collection)](https://huggingface.co/collections/nvidia/nemotron-terminal)

---

## 🔗 References
*   **Paper**: On Data Engineering for Scaling LLM Terminal Capabilities
*   **HuggingFace**: [nvidia/nemotron-terminal](https://huggingface.co/collections/nvidia/nemotron-terminal)
