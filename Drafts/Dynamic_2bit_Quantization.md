---
id: Dynamic_2bit_Quantization
category: On_Device
title: "Dynamic 2-bit Quantization for Edge LLM Deployment"
---
# 2비트 다이내믹 양자화: 16GB RAM에서 35B를 돌리는 법

> **Unsloth Dynamic 2.0의 혼합 정밀도 양자화로, MoE 모델(35B-A3B)을 12GB GGUF로 압축하여 일반 노트북 CPU에서 실행. 핵심은 Imatrix 캘리브레이션 데이터의 도메인 혼합.**

## 1. 왜 2비트가 필요한가?

16GB RAM 환경에서:
*   OS 기본 점유: ~3–4GB → 가용 RAM: ~12GB.
*   35B MoE(활성 3B) 모델을 4비트로 양자화해도 **15–17GB** → 불가능.
*   **2비트(IQ2)**: ~12GB → 16GB RAM에 턱걸이 가능.
*   KV Cache 여유 공간 필요 → 가벼울수록 유리.

## 2. Unsloth Dynamic 2.0의 핵심

전체를 무식하게 2비트로 통일하는 것이 아니라, **Imatrix(중요도 행렬)** 기반 혼합 정밀도:

*   **중요 레이어 (Attention)**: 4~6비트로 정밀 유지 → 추론 능력 보존.
*   **덜 중요 레이어 (MoE FFN)**: 1.5~2비트로 극단 압축 → 용량 절감.
*   결과: 전체 용량은 2비트급이지만 지능은 일반 2비트보다 훨씬 우수.

## 3. 커스텀 캘리브레이션 레시피 (150만 토큰)

| 비율 | 데이터 | 목적 |
|---|---|---|
| 50% | Bartowski groups_merged.txt (위키, 수학, 코드) | 범용 추론력 방어 |
| 35% | 고품질 한국어 (위키백과/뉴스) | 한국어 가중치 보존 |
| 15% | Nemotron-Terminal-Corpus | 터미널 에이전트 포맷 방어 |

## 4. FP8 vs 4-bit (H100 환경)

*   LoRA/GRPO 학습 시 FP8이 4-bit보다 **최대 3배 빠름** (Weight Sharing 기술).
*   양자화 원본은 **bf16**으로 올려야 정보 손실 최소화.

## 5. LFM2-24B-A2B vs Qwen3.5-35B-A3B

16GB RAM CPU 환경에서는 **LFM2-24B-A2B 압승**:
*   2비트 시 ~7–8GB → **4–5GB 여유 공간**으로 긴 컨텍스트 처리 가능.
*   Liquid 아키텍처: 내부 상태 압축으로 KV Cache 메모리 절감.
*   활성 파라미터 2.3B < Qwen 3B → CPU 연산 속도 우위.

---

## 🔗 References
*   **Unsloth Dynamic 2.0**: [unsloth.ai/docs](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs)
*   **Qwen3.5 GGUF**: [unsloth/Qwen3.5-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF)
*   **LFM2-24B**: [LiquidAI/LFM2-24B-A2B](https://huggingface.co/LiquidAI/LFM2-24B-A2B)
