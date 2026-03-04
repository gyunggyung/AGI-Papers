---
id: GAT_LLM
category: Architecture
title: "Graph Attention Networks (GAT) + LLM Integration"
---

# Graph Attention Networks (GAT) + LLM 통합 분석

## 핵심 요약

Veličković et al., ICLR 2018 (arXiv: 2017.10). Transformer의 Self-Attention을 **그래프 구조 데이터**에 이식한 GNN 분야의 근간 논문. "Attention Is All You Need" 발표 4개월 만에 등장한 최초의 그래프 Attention 모델.

## GAT 핵심 메커니즘

### Graph Attentional Layer
1. **유사도 계산**: 두 노드 i,j의 특징 벡터를 선형 변환(W) 후 어텐션 메커니즘(a) 통과
   - `e_ij = a(W·h_i, W·h_j)` (단일 레이어 FFN + LeakyReLU)
2. **마스킹된 Softmax**: 연결된 이웃 `N_i`에 대해서만 어텐션 수행
   - `α_ij = softmax_j(e_ij)` (그래프 구조 주입)
3. **가중 합 집계**: `h'_i = σ(Σ α_ij · W·h_j)`
4. **Multi-head**: K개 독립 헤드 → 중간은 Concatenation, 마지막은 Average

## Transformer vs GAT 비교

| 특징 | GAT | Transformer (LLM) |
|------|-----|-------------------|
| 구조 | Masked Attention (그래프) | Full/Causal Attention |
| 입력 | 노드(Entity) + 엣지(Relation) | 토큰 시퀀스 |
| 위치 정보 | 불필요 (Permutation Invariant) | Positional Encoding 필수 |
| Attention 범위 | 1-hop 이웃만 | 모든 토큰 (O(N²)) |
| 핵심 강점 | 구조적 관계 학습 | 의미적 관계 학습 |

## LLM과의 결합 제안

### 1. GraphRAG (GAT-Enhanced Retrieval)
- 지식 베이스를 Knowledge Graph로 구축
- GAT로 검색된 노드의 임베딩을 이웃 정보로 업데이트(Smoothing)
- 2-hop 이상의 추론이 필요한 질문에 답변 품질 향상

### 2. Multi-Agent Router (MoA용)
- 각 에이전트 = 노드, 통신 경로 = 엣지
- GAT의 Attention Score로 동적 에이전트 가중치 조절
- "코딩 질문" → 코딩 에이전트에 높은 α 가중치

### 3. LLM as Node Feature Encoder
- 텍스트가 있는 그래프(논문 인용 등)에서 LLM 임베딩을 GAT 입력으로 사용
- 문서의 내용(Semantic) + 인용 구조(Structure) 동시 학습

## 역사적 의미

- **2017.06**: "Attention Is All You Need" (Transformer) 공개
- **2017.10**: GAT 공개 → Transformer의 Attention을 그래프에 **최초** 이식
- GCN의 고정 가중치 한계와 GraphSAGE의 순서 부여 문제를 동시 해결

## References

- Veličković et al., "Graph Attention Networks", ICLR 2018 (arXiv:1710.10903)
- Vaswani et al., "Attention Is All You Need", NeurIPS 2017
- Kipf & Welling, "Semi-Supervised Classification with Graph Convolutional Networks"
