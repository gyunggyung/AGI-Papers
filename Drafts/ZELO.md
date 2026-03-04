---
id: ZELO
category: RAG
title: "ZELO: ELO-inspired Training for Rerankers and Embedding Models"
---
# ZELO: 체스 랭킹으로 검색 모델을 훈련하다

> **정답 라벨 없이, 체스의 ELO 랭킹 시스템 원리로 문서 간 상대 평가를 수행하여 상용 모델을 뛰어넘는 검색 모델을 탄생.**

## 1. 기존의 한계

검색 모델(Reranker, Embedding Model)의 랭킹 성능을 올리려면, 값비싼 인간의 정답 라벨링 데이터(이 문서가 이 쿼리에 대해 관련도 5점)가 필수적이었습니다. 고품질 라벨링 데이터 구축에는 막대한 비용과 시간이 소요됩니다.

## 2. 새로운 접근: ELO 기반 비지도 학습

ZELO는 체스에서 선수의 실력을 상대 전적으로 매기는 ELO 시스템 원리를 차용합니다.

*   **비지도(Zero-label)**: 정답이 없는 데이터만으로도 문서 간 상대적 관련도를 학습합니다.
*   **상대 평가**: "이 문서가 저 문서보다 쿼리에 더 관련이 있다"는 상대적 비교를 통해 랭킹 능력을 키웁니다.
*   **성과**: 라벨링 비용 없이도 상용 검색 모델(Cohere, Voyage 등)을 뛰어넘는 성능을 달성했습니다.

## 3. 시사점

데이터 라벨링 비용이 검색 모델 발전의 병목이 되는 현실에서, 라벨 없는 학습의 새로운 가능성을 열었습니다.

---

## 🔗 References
*   **Paper**: [ZELO: ELO-inspired Training Method for Rerankers and Embedding Models](https://arxiv.org/abs/2509.12541)
