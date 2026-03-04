---
id: TP_vs_DeviceMap
category: Architecture
title: "Multi-GPU: Tensor Parallelism vs Device Map"
---

# Multi-GPU: Tensor Parallelism vs Device Map

## 핵심 요약

HuggingFace 블로그 ([ariG23498/tp-vs-dm](https://huggingface.co/blog/ariG23498/tp-vs-dm)). Transformers 라이브러리에서 멀티 GPU를 사용하는 두 가지 접근법의 차이와 사용법. 추론 목적이냐 속도 목적이냐에 따라 선택이 갈린다.

## 사전 설정: GPU 지정

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"
# cuda:0 → 물리적 GPU 3, cuda:1 → 물리적 GPU 4
```

## device_map="auto" (메모리 분산)

### 작동 원리
- 모델 레이어를 여러 GPU에 **순차적**으로 나누어 적재
- GPU 0 계산 완료 → GPU 1 이어서 계산 (Pipeline 방식)

### 사용법
```python
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    dtype="auto",
    device_map="auto",  # 핵심
)
```

### 특징
| 항목 | 설명 |
|------|------|
| 설정 난이도 | ⭐ 매우 쉬움 |
| 속도 향상 | ❌ 없음 (순차 실행) |
| 용도 | 큰 모델을 일단 올리기 |
| 학습 | 비효율적/불가 |

## tp_plan="auto" (Tensor Parallelism)

### 작동 원리
- 행렬 연산 **자체**를 쪼개 여러 GPU가 **동시에** 계산
- 진정한 병렬 연산

### 사용법
```bash
torchrun --nproc_per_node=2 run_model.py
```
```python
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    dtype="auto",
    tp_plan="auto",  # 핵심
)
```

### 특징
| 항목 | 설명 |
|------|------|
| 설정 난이도 | ⭐⭐⭐ 분산 환경 필요 |
| 속도 향상 | ✅ 실질적 향상 |
| 용도 | 빠른 추론/학습 |
| 지원 | 모든 모델이 지원하지는 않음 |

## 비교 요약

| 특징 | Device Map | Tensor Parallelism |
|------|-----------|-------------------|
| 설정 | 쉬움 | 복잡 (torchrun 필요) |
| 속도 | 순차 실행 | **병렬 실행** |
| 메모리 분산 | ✅ | ✅ |
| 학습 가능 | ❌ | ✅ |
| 추천 상황 | "일단 돌려보자" | "빠르게 서빙하자" |

## 결정 기준

- **device_map**: GPU 하나에 안 들어가는 모델을 일단 실행만 할 때
- **tp_plan**: 추론 속도가 중요하거나, 서빙/학습에 사용할 때

## References

- HuggingFace Blog: [TP vs Device Map](https://huggingface.co/blog/ariG23498/tp-vs-dm)
- PyTorch `torchrun`, FSDP, Tensor Parallelism
