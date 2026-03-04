---
id: FlashOptim
category: Architecture
title: "FlashOptim: Memory-Efficient Deep Learning Optimizer"
---
# FlashOptim: 옵티마이저 메모리를 반토막 내다

> **모델 품질 100% 유지하면서, AdamW의 파라미터당 메모리를 16바이트 → 7바이트(5바이트)로 절반 이하로 축소. 성능 손실 제로(Zero Degradation).**

## 1. 문제: 메모리의 벽

AdamW 옵티마이저는 파라미터 1개당 **16바이트**의 메모리를 차지:
*   마스터 가중치: 4B (FP32)
*   기울기(Gradient): 4B
*   모멘텀(Momentum): 4B
*   분산(Variance): 4B

→ 7B 모델 학습 시 파라미터 관련 메모리만 **최소 112GB** 필요.

## 2. 핵심 기술 2가지

### Weight Splitting (마스터 가중치 쪼개기)
*   FP32(4B) → **BF16(2B) + INT8 에러 보정값(1B)** = 3B.
*   ULP(Unit in the Last Place) 스케일링으로 24비트급 정밀도 확보.
*   비유: 세계 지도(FP32) 대신 전국 지도(BF16) + 동네 확대경(INT8).

### Companding (옵티마이저 상태 양자화)
*   **모멘텀**: `φ_m(x) = 2x/(1+|x|)` 함수로 극단값을 중심부로 밀어넣은 뒤 INT8 압축.
*   **분산**: `φ_v(x) = √x` 제곱근으로 크기 축소 후 UINT8 압축.
*   단순 선형 양자화는 학습 발산(divergence) → Companding으로 해결.

## 3. 결과 (숫자로 증명)

| 지표 | 기존 AdamW | FlashOptim |
|---|---|---|
| 파라미터당 메모리 | 16B | **7B** (기울기 해제 시 5B) |
| Llama-3.1-8B 최대 메모리 | 175 GiB | **113 GiB** (−36%) |
| 체크포인트 크기 (7B) | 84GB | **35GB** |
| 품질 손실 | — | **Zero Degradation** |
| 속도 | 12.5ms/step | **11.5ms/step** (오히려 향상) |

*   커스텀 Triton 커널로 압축/해제 과정을 단일 연산으로 융합(Fusion).
*   ResNet-50, GPT-2, Llama-3.1 등에서 **정확도·Loss 동일** 확인.

---

## 🔗 References
*   **Paper**: FlashOptim: Memory-Efficient Deep Learning Training
