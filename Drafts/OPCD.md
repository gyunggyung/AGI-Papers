---
id: OPCD
category: Post_Training
title: "Experiential Learning Part I: On-Policy Context Distillation for Language Models"
---

# Experiential Learning Part I: On-Policy Context Distillation (OPCD)

## 핵심 요약

Microsoft Research의 논문. 인컨텍스트 학습(ICL)으로 얻은 경험적 지식을 **영구적 파라미터**로 내재화하는 프레임워크. 기존 오프폴리시(Off-policy) 방식의 순방향 KL 발산이 유발하는 노출 편향/환각을 극복하기 위해, 학생 모델이 **자신이 생성한 궤적(On-Policy Trajectory)**을 기반으로 역 KL 발산을 최소화하는 방식으로 훈련.

## 핵심 메커니즘

### On-Policy Context Distillation
1. **교사 모델**: 문맥(Context)이 주어진 상태에서 정답 분포 생성 (동결)
2. **학생 모델**: 문맥 **없이** 스스로 궤적을 생성
3. **목적함수**: 학생 생성 궤적에 대해 역 KL 발산(Reverse KL) 최소화
4. **효율화**: 전체 어휘 대신 학생 모델 예측 **상위 256개 토큰**으로 근사 계산

### 제약 조건
- 교사-학생 동일 **토크나이저(보캡)** 필수 → 같은 모델 패밀리 내에서만 동작
- 배치 사이즈 128, 50스텝 학습

## 적용 분야

| 분야 | 설명 |
|------|------|
| **경험적 지식 증류** | 과거 문제 해결 경험을 전이 가능한 지식으로 영구 내재화 |
| **시스템 프롬프트 증류** | 의료·안전 등 특화 프롬프트를 모델 내부에 영구 배선 → 추론 시 긴 프롬프트 불필요 |

## 성능 결과

### 수학 (Qwen3-8B, 정확도 %)
| 방법 | 필터링 없음 | 필터링 적용 |
|------|----------:|----------:|
| 기본 모델 | 75.0 | 75.0 |
| 문맥 제공 (ICL) | 77.6 | 79.0 |
| 기존 문맥 증류 | 78.5 | 79.5 |
| **OPCD** | **79.7** | **80.9** |

### 텍스트 게임 Frozen Lake (Qwen3-1.7B)
| 방법 | 필터링 없음 | 필터링 적용 |
|------|----------:|----------:|
| 기본 모델 | 6.3 | 6.3 |
| **OPCD** | **26.5** | **38.3** |

### 시스템 프롬프트 증류 (의료 MedMCQA)
| 모델 | 기본 → OPCD |
|------|----------:|
| Llama-3.1-8B | 68.4 → **76.7** |
| Llama-3.2-3B | 59.4 → **76.3** |
| Qwen2.5-7B | 46.4 → **62.3** |

### 교사-학생 vs 자가 증류
| 태스크 | 자가 증류 | 교사-학생 |
|--------|--------:|--------:|
| Sokoban (Qwen3-4B) | 18.8% | **53.9%** |
| Medical (Qwen2.5-3B) | 50.0% | **56.8%** |

## 핵심 인사이트

1. **Raw Trace는 독**: 과거 풀이 과정을 그대로 넣으면 오히려 성능 하락 (75.1% → 70.5%). **추출된 지식**을 넣어야 향상 (→ 77.4%)
2. **망각 완화**: 안전 프롬프트 증류 시 의료(OOD) 평가에서 베이스라인보다 ~4점 높은 정확도 유지
3. **교차 크기 증류 성공**: 8B 교사 → 1.7B/4B 학생으로 경험적 지식 성공적 전달

## References

- [Microsoft Research] Experiential Learning Part I: On-Policy Context Distillation
- DAPO-Math-17K, TextArena (Frozen Lake, Sokoban), MetaSPO
