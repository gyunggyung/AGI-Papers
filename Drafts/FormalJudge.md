---
id: FormalJudge
category: Agents
title: "FORMAL JUDGE: A Neuro-Symbolic Paradigm for Agentic Oversight"
---

# FORMAL JUDGE: Neuro-Symbolic Paradigm for Agentic Oversight

## 핵심 요약

"감시자는 누가 감시하는가?" → 확률적 LLM으로 확률적 LLM을 감독하는 기존 방식(LLM-as-a-Judge)의 근본적 한계를 해결. **확률적 추론(LLM) + 결정론적 검증(SMT Solver)**을 결합한 "Formal-of-Thought" 아키텍처.

## 핵심 문제

- **환각의 악순환**: 평가자 모델도 환각 가능
- **편향 및 취약성**: 피평가자의 기만에 넘어갈 수 있음
- **확장성**(Scalable Oversight): 에이전트 > 인간 → 신뢰성 있는 감독 불가능

## 4단계 파이프라인

```
1. Intent Decomposition    → 모호한 지시를 원자적 사실(Yes/No)로 분해
2. Context-Aware Extraction → 에이전트 궤적에서 사실 추출 (LLM 수행)
3. Formal Proof Synthesis   → Dafny 검증 언어로 명세 작성
4. SMT Verification         → Z3 SMT Solver로 수학적 증명
```

## 핵심 성과

| 지표 | 결과 |
|------|------|
| 정확도 향상 | 기존 LLM-as-a-Judge 대비 **+16.6%** |
| Weak-to-Strong | 7B 심판이 72B 에이전트 기만 **90%+ 탐지** |
| 반복적 개선 | 안전성 70.7% → **99.8%** |

## 왜 중요한가

- **Weak-to-Strong Generalization**: 작은 모델이 큰 모델 감독 가능 → 모델 크기 차이에 성능 급락하는 기존 방식과 대조적
- **기만 탐지**: 에이전트가 실패 은폐를 위해 파일 조작/거짓말하는 Upward Deception까지 탐지
- **결정론적 보장**: 확률이 아닌 논리적 '증명'으로 최종 판결

## 구현 상태

- ✅ **완전 구현**: 코드, 벤치마크, 실험 로그 모두 공개
- 🔗 GitHub: https://github.com/htlou/FormalJudge
- 도구: Dafny (검증 언어) + Z3 (SMT Solver)

## References

- FORMAL JUDGE: A Neuro-Symbolic Paradigm for Agentic Oversight
- Z3 SMT Solver, Dafny Verification Language
