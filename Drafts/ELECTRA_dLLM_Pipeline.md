---
id: ELECTRA_dLLM_Pipeline
category: Agents
title: "Sniper-Router-Surgeon: ELECTRA + dLLM Coding Agent"
---
# Sniper-Router-Surgeon: ELECTRA + dLLM 코딩 에이전트

> **ELECTRA가 버그 토큰을 저격(Sniper)하고, dLLM(확산 언어 모델)이 양방향 문맥으로 수술(Surgeon)하며, RLVR이 검증하는 자가 치유형 코딩 에이전트 아키텍처.**

## 1. 아키텍처 개요

| 단계 | 역할 | 모델 | 비유 |
|---|---|---|---|
| **Sniper** | 에러 확률이 높은 토큰(FAKE) 특정 | ELECTRA-Large (335M) | X-ray 진단 |
| **Router** | 마킹된 토큰을 [MASK] 좌표로 변환 | Logic Layer | 수술 좌표 전달 |
| **Surgeon** | 양방향 문맥 참조로 병렬 수정 | GLM-4.7-Flash dLLM (30B-A3B) | 외과 수술 |
| **Verifier** | 샌드박스에서 실행·검증 (RLVR) | Compiler/Tester | 조직 검사 |

## 2. 왜 GLM-4.7-Flash인가?

*   **SWE-bench Verified 73.8%** (SOTA급) — Qwen3.5-35B의 22%를 3배 이상 상회.
*   **MLA(Multi-Latent Attention)** 구조로 dLLM의 양방향 노이즈 제거에 최적.
*   **30B-A3B MoE**: 토큰당 3B만 활성화, 16GB VRAM에서도 동작 가능.

## 3. Sniper (ELECTRA) 학습 전략

*   **베이스**: `google/electra-large-discriminator` Post-training.
*   **GLiNER2 방식 적용**: 단순 이진 분류가 아닌, Schema-Driven Interface로 에러 종류를 동적 식별.
*   **데이터**: [alexjercan/bugnet](https://huggingface.co/datasets/alexjercan/bugnet), [JetBrains-Research/diff-xyz](https://huggingface.co/datasets/JetBrains-Research/diff-xyz).
*   **비용**: H100 1대, 약 $30~$45 (5~6만 원).

## 4. Surgeon (dLLM) 전환: LLaDA 2.0 방식

WSD(Warmup-Stable-Decay) 스케줄링으로 AR → dLLM 전환:
1.  **Warmup**: Block size 1(AR) → 4096(Full-seq)으로 점진적 확대.
2.  **Stable**: 4096 고정으로 대규모 코드 데이터 학습.
3.  **Decay**: Block size 32로 축소 → BD3LM 구조, AR급 추론 속도.

## 5. RLVR 무한 루프

```
실패의 자산화:
RLVR Fail → 에러 로그 → Sniper에게 힌트 → dLLM 재수술 → 재검증
확률이 0이 아닌 한, 시도 횟수(n→∞)로 성공 확률은 100%로 수렴.
```

*   **예상 총비용**: H100 4대 클러스터 24시간 = 약 $287.
*   **FlashOptim 적용**: 옵티마이저 메모리 16B → 7B/파라미터로 36% 절감.

---

## 🔗 References
*   **LLaDA 2.0**: [Scaling Up Diffusion Language Models to 100B](https://arxiv.org/abs/2502.09992)
*   **dLLM Framework**: [Simple Diffusion Language Modeling](https://arxiv.org/abs/2602.22661)
*   **Drifting Models**: [Generative Modeling via Drifting](https://arxiv.org/abs/2501.14478)
*   **GLM-4.7-Flash**: [zai-org/GLM-4.7-Flash](https://huggingface.co/zai-org/GLM-4.7-Flash)
*   **ELECTRA**: [google/electra-large-discriminator](https://huggingface.co/google/electra-large-discriminator)
