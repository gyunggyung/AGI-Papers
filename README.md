# 🚀 AGI-Papers

![AGI-Papers](https://img.shields.io/badge/AGI--Papers-2026-blue?style=for-the-badge&logo=github)
![Topic](https://img.shields.io/badge/Topic-AGI%20%7C%20Agents%20%7C%20Trends-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-important?style=for-the-badge)

> **Toward Artificial General Intelligence (AGI) in 2026.**  
> A curated archive of breakthroughs in **Agents**, **Architecture**, **Training**, **RAG**, and **On-Device AI**.

## 📢 News

*   **[2026-03-20]** [**Modern Model Architectures Overview 강연 자료 (PDF)**](./PDF/Modern-Model-Architectures-Overview.pdf)를 업데이트했습니다.
*   **[2026-03-15]** [**Introduction to Post-Training & Beyond 강연 자료 (PDF)**](./PDF/introduction-to-post-training-and-beyond.pdf)를 업데이트했습니다.

## 📌 Introduction
2026년, 인공지능의 발전 속도가 유례없을 지경입니다.

이 저장소는 **AGI(Artificial General Intelligence)** 로 향하는 여정에서 중요한 논문들을 리뷰하고 아카이빙하는 공간입니다.

주로 제 **[LinkedIn](https://www.linkedin.com/in/kiwoong-yeom/)** 에서 다룬 논문들에 대한 심도 있는 리뷰가 업로드되며, 때로는 소셜 미디어에 공유하기 전의 **Pre-release 인사이트**나 날것의 생각들이 이곳에 먼저 기록될 예정입니다.

## 📝 Drafts (Work In Progress)

작성 중인 새로운 글들은 아래 링크에서 확인하실 수 있습니다.
* 👉 **[New Drafts (Working in Progress)](Drafts.md)**

## 📂 Archives

과거에 정리했던 논문 리스트는 아래 링크에서 확인하실 수 있습니다.
* 👉 **[Past AGI-Papers (Pre-2026)](Pre-README.md)**

---

## 📚 Contents

이 저장소는 AGI를 향한 여정을 다음 10가지 핵심 주제로 분류하여 정리합니다.

- [🤖 Agents](#agents) : 자율 에이전트, 행동/계획(Planning) 모델, 프레임워크
- [🧠 Architecture](#architecture) : LLM 아키텍처 혁신 (Transformer, Mamba, MoE)
- [📚 Pre-Training](#pre-training) : 학습 데이터, 스케일링 법칙, 옵티마이저, 파운데이션 모델
- [🎯 Post-Training](#post-training) : RLHF, DPO, GRPO, 정렬(Alignment), 에이전트 학습
- [⚡ Inference & Prompting](#inference--prompting) : 프롬프팅, CoT 추론 제어, Test-Time Compute, 추론 엔진
- [🧬 Steering & Weight Editing](#steering--weight-editing) : 활성화 조향(Activation Steering), 무학습 가중치 편집, 뉴런 제어
- [⚖️ Evaluation](#evaluation) : 벤치마크, 평가 방법론, 비평
- [🗂️ RAG & Knowledge](#rag--knowledge) : 검색 증강 생성, 지식 그래프, 메모리
- [💻 On-Device AI](#on-device-ai) : 로컬 구동, 엣지 컴퓨팅, 최적화
- [🚀 Projects](#projects) : 직접 구현한 프로젝트 및 실험 결과
- [🔥 Trends & Industry](#trends--industry) : AI 산업의 동향, 인사이트, 주요 뉴스
- [🌟 Recommended Resources](#recommended-resources) : 공부를 위한 추천 자료

---

## <a id="agents"></a>🤖 Agents

*   [**The Last AI Built by Humans: 재귀적 자가 개선(RSI) 5단계 로드맵과 환경 엔지니어링**](./Agents/133.md)  
    *AI 자율성 5단계와 무통제 자가 수정의 위험성을 분석하고, 에이전트 하네스·작업 환경 정제의 중요성을 밝힌 RSI 서베이.*
*   [**PARSER: 병렬 독해와 중앙 지휘관 추론 기반 초장문 멀티 에이전트**](./Agents/132.md)  
    *서브에이전트 병렬 독해와 GRPO 중앙 지휘관의 단계별 질의를 결합해 896K 초장문에서 11배 가속을 달성한 연구.*
*   [**Frog Nano: 대형 모델 증류 없이 하네스와 온라인 RL만으로 만든 4B 코딩 에이전트**](./Agents/131.md)  
    *경량 하네스(LEAF)와 난이도 조절 온라인 태스크 합성(TaskPilot)만으로 4B 모델을 거대 에이전트급으로 끌어올린 MSR 연구.*
*   [**LLMs Get Lost in Evolving User Intent: 멀티턴 대화 붕괴와 상태 관리**](./Agents/130.md)  
    *조건 수정·작업 전환이 섞인 멀티턴 대화에서 LLM이 7턴 만에 무너지는 원인을 분석하고 GRPO 행동 정렬 해법을 제시.*
*   [**에이전트 자가 개선(Self-improving) 인공지능 서베이**](./Agents/129.md)  
    *프롬프트 최적화에서 풀 스캐폴딩·실시간 진화로 발전하는 자가 개선 에이전트의 흐름과 위험성을 정리한 서베이.*
*   [**The Harness Effect: 하네스 설계가 결정하는 에이전트 토큰 경제학**](./Agents/128.md)  
    *에이전트 오케스트레이션 하네스 설계를 통한 6가지 토큰 최적화 및 비용 절감 메커니즘.*
*   [**The Red Queen Gödel Machine: Co-Evolving Agents and Their Evaluators**](./Agents/126.md)  
    *에이전트와 평가자가 동적 루프 안에서 함께 진화하며 성능을 끌어올리는 공동 진화 시스템.*
*   [**Agent Lifespan Engineering: 에이전트도 늙을수록 멍청해집니다**](./Agents/125.md)  
    *장기 작업 시 컨텍스트·메모리 누적으로 에이전트 지능이 저하되는 노화 현상과 수명 관리 기법.*
*   [**Code as Agent Harness: 2026 에이전트 하네스 아키텍처와 최전선 연구**](./Agents/124.md)  
    *환각을 통제하고 폐쇄 루프 내 계획-실행-검증을 구현하는 에이전트 하네스의 3계층 구조(L2MAC, EvoMAC) 정리.*
*   [**SID-1: GPT-5.1을 이긴 14B 검색 에이전트의 탄생**](./Agents/123.md)  
    *문서 검색 역할에만 특화해 환각을 차단하고 GPT-5.1 대비 24배 빠른 속도를 달성한 14B 검색 에이전트.*
*   [**Agentic Harness Engineering (AHE): 스스로 하네스를 코딩해 개조하는 AI**](./Agents/122.md)  
    *에이전트가 직접 파이썬 쉘 도구와 미들웨어를 구축해 폐쇄 루프(Closed Loop) 자가 진화를 달성하는 방법.*
*   [**LLMs Corrupt Your Documents: 조용한 부패와 위임의 환상**](./Agents/121.md)  
    *문서 편집 위임 시 발생하는 '조용한 부패(Silent Corruption)' 현상과 툴 사용이 문서 훼손을 가속하는 원인 분석.*
*   [**RecursiveMAS: 텍스트 없이 잠재 공간에서 대화하는 멀티 에이전트**](./Agents/120.md)  
    *어휘 공간 디코딩 대신 잠재 공간(Latent Space)에서 에이전트들이 직접 소통해 토큰 비용을 75% 절감한 연구.*
*   [**Agentic World Modeling: 세계 모델의 3단계 역량과 4대 지배 법칙**](./Agents/119.md)  
    *에이전트 세계 모델의 역량 기준을 체계화하고 환경(Environment) 병목과 UI OCR의 중요성을 짚은 리뷰.*
*   [**하네스(Harness)의 민낯과 그 한계**](./Agents/118.md)  
    *Meta-Harness, NLAH, ARC-AGI-3 등 에이전트 하네스의 실제 효과와 본질적 한계를 파헤친 논문 3편 리뷰.*
*   [**HYPERAGENTS: 스스로 '공부법'을 코딩하며 진화하는 AI**](./Agents/117.md)  
    *작업 에이전트와 메타 에이전트를 결합해 문제 해결 코드와 학습 알고리즘 자체를 함께 진화시키는 연구.*
*   [**Adaptation of Agentic AI**](./Agents/80.md)  
    *거대 에이전트 모델 자체를 튜닝하는 것보다 하위 도구(Tool)를 튜닝하는 것이 훨씬 효율적인 이유.*
*   [**Memory in the Age of AI Agents**](./Agents/77.md)  
    *단순 RAG를 넘어 에이전트 메모리의 형태(토큰·파라미터·잠재 상태), 기능, 진화 역학을 총정리한 서베이.*
*   [**Research on World Models Is Not Merely Injecting World Knowledge**](./Agents/11.md)  
    *단순 지식 주입이나 영상 생성을 넘어 인과관계와 물리 법칙을 시뮬레이션하는 진정한 월드 모델 설계도.*
*   [**Mixture-of-Models (MoM)**](./Agents/9.md)  
    *이종 소형 모델들을 자기 평가(Self-Evaluating) 위상 구조로 연결해 거대 단일 모델을 넘어서는 다중 에이전트 기법.*
*   [**OctoTools: Training-free LLM Agent Framework**](./Agents/99.md)  
    *별도 학습 없이 도구 카드(Tool Card)와 플래너·실행기 분리만으로 복합 도구 활용 능력을 극대화하는 프레임워크.*
*   [**Scaling Agent Systems: 다다익선의 함정**](./Agents/100.md)  
    *에이전트 수를 무작정 늘릴 때 발생하는 조율 오버헤드와 태스크별 최적 멀티 에이전트 스케일링 조건 규명.*
*   [**LOTaD: Optimal Task Decomposition**](./Agents/101.md)  
    *사람이 개입하지 않아도 다중 에이전트가 스스로 최적의 역할 분담과 작업 분해 방식을 학습하는 프레임워크.*
*   [**ADGR: Agentic Deep Graph Reasoning**](./Agents/102.md)  
    *정적 지식 그래프 대신 에이전트가 추론 과정에서 필요한 개념을 스스로 연결·확장하며 동적 그래프를 구축하는 기법.*
*   [**Agentic Reasoning: 생각의 도구를 쓰는 에이전트**](./Agents/103.md)  
    *내부 CoT에만 의존하지 않고 웹 검색, 코드 실행, 마인드맵 도구를 결합해 고난도 추론 문제를 푸는 프레임워크.*
*   [**MetaChain: Zero-code Agent Framework**](./Agents/104.md)  
    *복잡한 코딩 없이 자연어 명세만으로 도구와 멀티 에이전트 워크플로우를 자동 생성하는 프레임워크.*
*   [**SciAgents: 다중 에이전트 기반 과학적 발견 자동화**](./Agents/107.md)  
    *지식 그래프 위에서 다중 에이전트가 논문 탐색, 가설 수립, 실험 검증을 분업해 새로운 과학 가설을 발굴하는 시스템.*
*   [**Building Effective Agents (Anthropic)**](./Agents/108.md)  
    *복잡한 프레임워크 대신 단순한 조합을 강조하는 앤트로픽의 5가지 에이전트 워크플로우 핵심 디자인 패턴.*
*   [**HUMANLM: State Alignment for User Simulation**](./Agents/111.md)  
    *겉모습만 흉내 내는 역할극을 넘어 인간의 내면 상태(목표·감정·지식)를 정렬해 현실적인 유저를 시뮬레이션하는 모델.*
*   [**SKILLRL: 에이전트는 '실패'를 먹고 자란다**](./Agents/112.md)  
    *성공과 실패 궤적을 재사용 가능한 '스킬 라이브러리'로 증류해 에이전트의 지속 학습을 구현한 강화학습 기법.*
*   [**The Devil Behind Moltbook: 다중 에이전트 사회의 타락**](./Agents/114.md)  
    *닫힌 다중 에이전트 생태계에서 발생하는 소통 붕괴, 합의된 환각, 집단 공모 공격 현상을 실증한 연구.*
*   [**Pau Labarta Bajo's Insight on Multi-Agent Systems**](./Agents/27.md)  
    *멀티 에이전트 시스템을 장난감에서 프로덕션 수준으로 올리기 위한 인프라 설계와 결정론적 제어 원칙.*

## <a id="architecture"></a>🧠 Architecture

*   [**KV-Lingo: 재프리필 없이 선형 행렬로 이종 모델 간 KV 캐시를 번역하는 기술**](./Architecture/149.md)  
    *레이어별 1차 선형 맵으로 이종 모델 간 KV 캐시를 직접 변환해 재프리필 병목을 없애고 최대 29배 가속한 Apple 연구.*
*   [**Why Does Post-Training Quantization Work?: 4비트 양자화 생존과 신경망 자정 원리**](./Architecture/148.md)  
    *층별 오차 상쇄와 고차원 LM Head의 차원 완충 작용으로 LLM이 4비트 양자화에서도 성능을 유지하는 원리를 규명한 논문.*
*   [**DLR (Dynamic Latent Routing): 알파고의 탐색을 LLM 잠재 공간에 이식하다**](./Architecture/150.md)  
    *트랜스포머 잠재 공간에 MCTS 경로 탐색을 도입해 추가 학습이나 토큰 지연 없이 추론 성능을 18.8% 높인 아키텍처.*
*   [**DeepSeek-V4.1-Flash: 극한의 KV 캐시 압축과 Causal Encoder-Decoder**](./Architecture/147.md)  
    *글로벌 KV 캐시 75% 감축과 입력 깊이를 절반만 통과시키는 CED 구조로 추론 효율을 극대화한 DeepSeek 아키텍처 분석.*
*   [**2D-RoPE: 1차원 RoPE의 복사 한계를 극복하는 2차원 위치 인코딩**](./Architecture/146.md)  
    *프론티어 모델이 단순 복사에 실패하는 원인을 1차원 거리 왜곡으로 규명하고 행·열 2차원 RoPE로 해결한 연구.*
*   [**디퓨션 언어 모델(DLM) 서베이: 원리와 2026 최신 흐름**](./Architecture/145.md)  
    *이산·하이브리드·연속 디퓨션 언어 모델의 원리부터 RL 추론 결합 및 1-Step 가속 기법까지 총정리한 서베이.*
*   [**NVIDIA LatentMoE & Moonshot AI Stable LatentMoE 분석**](./Architecture/144.md)  
    *잠재 공간 압축으로 MoE 통신 병목을 줄이는 LatentMoE와 2.8T Kimi K3에 적용된 안정화 기법(SiTU-GLU) 분석.*
*   [**차세대 MoE 아키텍처: RFMoE(Routing-Free)와 UoE(Union-of-Experts)**](./Architecture/143.md)  
    *중앙 라우터 없이 자율 활성화하는 RFMoE와 탈락 전문가 연산을 가상 공유 전문가로 재활용하는 UoE 분석.*
*   [**MoE 라우팅의 한계 돌파: ReMoE(ReLU 라우팅)와 AoE(자율 전문가 모델)**](./Architecture/142.md)  
    *Top-K Softmax의 불연속성을 ReLU로 해결한 ReMoE와 전문가 자체 평가로 라우터를 없앤 AoE 논문 리뷰.*
*   [**MoE(Mixture-of-Experts) 아키텍처의 동작 원리와 역사**](./Architecture/141.md)  
    *FFN 전문가 분할, 라우터 내적, Top-k Softmax 가중 합산 등 MoE의 핵심 메커니즘과 발전사 정리.*
*   [**Kimi K3: 2.8T 중국 오픈 모델 아키텍처 분석**](./Architecture/139.md)  
    *2.8T 파라미터에 Stable LatentMoE와 Attention Residuals를 결합해 프론티어 성능을 달성한 Kimi K3 구조 분석.*
*   [**Inkling: 975B 개방형 멀티모달 오픈 모델의 등장**](./Architecture/140.md)  
    *Thinking Machines Lab이 공개한 오디오 지원 975B 범용 멀티모달 오픈 모델의 구조와 최적화 의의.*
*   [**HydraHead: 어텐션 헤드를 쪼개는 분업의 효율성**](./Architecture/138.md)  
    *동일 레이어 내에서 가벼운 선형 어텐션 헤드와 정밀한 전체 어텐션 헤드를 분업시키는 하이브리드 어텐션 기법.*
*   [**ICML 2026 최우수 논문으로 보는 Diffusion LLM의 미래**](./Architecture/137.md)  
    *수학·코딩 추론에서 자기회귀(AR) 모델을 넘어설 가능성을 보여준 칭화대·예일대의 최신 디퓨션 LLM 연구 2편.*
*   [**100만 토큰의 한계 돌파: BLOCKSEARCH와 HiLS-Attention**](./Architecture/136.md)  
    *1M 이상 초장문 컨텍스트를 효율적으로 처리하기 위한 블록 검색 및 계층형 요약 어텐션 아키텍처.*
*   [**HOLA (Hippocampal Linear Attention): 순환 모델의 구조적 건망증 해결**](./Architecture/135.md)  
    *뇌의 해마-대뇌피질 기억 분담 원리를 모방해 선형 순환 모델의 정보 압축 한계와 환각을 극복한 아키텍처.*
*   [**Log-Linear Attention: 이진트리 구조로 푼 선형 어텐션의 기억 한계**](./Architecture/134.md)  
    *고정 크기 상태 행렬 대신 시간축 이진트리 구조를 도입해 $O(\log N)$ 메모리로 장문 맥락을 보존하는 어텐션.*
*   [**Do Transformers Need Three Projections?: K·V 통합으로 KV 캐시 절반 감축**](./Architecture/133.md)  
    *Key와 Value 투영 행렬을 하나로 통합해 성능 저하 없이 KV 캐시 메모리를 50% 절감한 아키텍처 연구.*
*   [**Predictable Compression Failures: 텍스트 압축기로 본 LLM 환각의 원리**](./Architecture/132.md)  
    *LLM의 환각을 정보 예산 부족에 따른 압축 실패 현상으로 정식화하고 이를 사전에 예측·통제하는 방법.*
*   [**FLARE 프레임워크: Qwen 3.5 9B를 디퓨전 LLM으로 변환해 가속하기**](./Architecture/131.md)  
    *기존 자기회귀 LLM의 지능을 유지한 채 병렬 디퓨전 디코딩(dLLM)을 결합해 추론 속도를 극대화하는 하이브리드 기법.*
*   [**Language Models Need Sleep: 수면 메커니즘을 모방한 단기-장기 기억 통합**](./Architecture/130.md)  
    *낮의 단기 문맥 기억을 수면 단계의 오프라인 정리를 통해 가중치에 내재화하여 추론 능력을 높이는 아키텍처.*
*   [**MiniMax M2 시리즈 기술 보고서: Full Attention 고수와 자가 진화**](./Architecture/129.md)  
    *편법적인 선형 어텐션 대신 Full Attention을 유지해 장기 문맥 단절을 막은 MiniMax M2의 기술 설계 분석.*
*   [**HRM-Text: 전두두정엽 루프 가중치 공유로 7B를 이긴 1B 모델**](./Architecture/128.md)  
    *레이어 가중치 공유(Weight Sharing)와 반복 순환으로 1B 메모리 예산 안에서 128층 연산 깊이를 확보한 아키텍처.*
*   [**딥러닝 역전파(Backpropagation)의 3가지 비효율과 대안 연구들**](./Architecture/127.md)  
    *역전파의 메모리·전력·잠금 병목을 분석하고 Forward-Forward, EGGROLL, 해밀토니안 추론 등 대안 알고리즘 소개.*
*   [**DeepSeek-V4-Pro: 100만 토큰과 FP4로 완성한 코딩·수학 특화 모델**](./Architecture/124.md)  
    *CSA/HCA 하이브리드 어텐션과 네이티브 FP4 양자화로 1M 컨텍스트와 극단적 가성비를 구현한 DeepSeek-V4-Pro 분석.*
*   [**Attention to Mamba & xLSTM: 로컬 모델을 위한 트랜스포머 증류 수술법**](./Architecture/123.md)  
    *무거운 트랜스포머를 선형 모델(Mamba)이나 하이브리드(xLSTM)로 변환하는 두 가지 아키텍처 증류 전략 비교.*
*   [**만능 옴니모델 vs 도메인 특화 모델: KDL Frontier**](./Architecture/122.md)  
    *거대 범용 멀티모달 모델의 비효율을 짚고, 구조적 혁신을 이룬 경량 도메인 특화 OCR 모델의 강점을 분석.*
*   [**Neural Computer (NC): 신경망 자체가 하나의 운영체제가 된다면?**](./Architecture/121.md)  
    *모델 외부 도구 호출 대신 신경망 잠재 공간 내부에서 연산, 장기 기억, 입출력을 통합 처리하는 Meta AI 연구.*
*   [**Memory Caching (MC): 트랜스포머의 성능에 RNN의 가벼움을 더하다**](./Architecture/120.md)  
    *세그먼트별 요약 메모리를 캐싱해 $O(NL)$ 복잡도로 트랜스포머급 장문 성능을 달성한 Google·Cornell 연구.*
*   [**Ouro (우로): 파라미터 증가 없이 잠재 공간에서 재귀를 돌리는 루프 LLM**](./Architecture/119.md)  
    *토큰을 뱉기 전 모델 내부 레이어를 반복 순환(Loop)하며 잠재 추론 깊이를 동적으로 조절하는 ByteDance 아키텍처.*
*   [**MinerU-Diffusion: 확산 디코더로 VLM 문서 OCR의 속도 한계 돌파**](./Architecture/118.md)  
    *순차적 텍스트 디코딩 대신 디퓨전 병렬 해독을 도입해 복잡한 문서 OCR 속도를 비약적으로 높인 모델.*
*   [**손실 없는 Transformer to xLSTM 아키텍처 이식 증류법**](./Architecture/116.md)  
    *트랜스포머의 어텐션 층을 xLSTM+SWA 하이브리드 모듈로 교체·증류하여 메모리와 추론 속도를 개선한 기법.*
*   [**Jacobi Forcing: 아키텍처 변경 없이 가벼운 튜닝으로 추론 속도 4배 향상**](./Architecture/115.md)  
    *KV 캐시를 100% 보존하면서 다중 토큰을 병렬로 예측·확정하도록 튜닝해 자기회귀 추론을 4배 가속하는 기법.*
*   [**Attention Residuals (AttnRes): 레이어 간 잔차 연결의 재설계**](./Architecture/114.md)  
    *단순 덧셈 방식의 ResNet 잔차 연결을 레이어 간 어텐션 구조로 대체해 깊은 신경망의 정보 흐름을 혁신한 Kimi 연구.*
*   [**Nemotron 3 Super와 NVFP4 양자화 아키텍처**](./Architecture/113.md)  
    *블랙웰 하드웨어 기반 NVFP4 초저비트 연산과 Mamba-Transformer 하이브리드로 효율을 극대화한 NVIDIA 모델 분석.*
*   [**음성 인식(ASR) 분야의 Diffusion LLM: 4.44배 빠른 병렬 해독**](./Architecture/112.md)  
    *음성 인식 디코더에 확산 언어 모델을 도입해 최고 수준의 인식률과 4.44배 추론 가속을 동시에 달성한 연구.*
*   [**STATIC: 희소 행렬 변환으로 1,000배 빨라진 LLM 제약 디코딩**](./Architecture/111.md)  
    *트리 탐색 기반 제약 디코딩을 GPU 친화적인 정적 희소 행렬 연산으로 펼쳐 추론을 1,033배 가속한 DeepMind 연구.*
*   [**LLaDA 2.0: 100B 스케일로 확장된 Diffusion Language Model**](./Architecture/103.md)  
    *기존 자기회귀(AR) 모델 가중치를 재활용해 100B 규모의 병렬 생성 디퓨전 언어 모델로 개조한 스케일업 레시피.*
*   [**EinFields: 일반 상대성 이론의 시공간을 압축한 신경망 텐서장**](./Architecture/105.md)  
    *거대 수치 시뮬레이션 데이터 없이 아인슈타인 장 방정식과 시공간 계량 텐서를 신경망 가중치 안에 연속 함수로 압축한 연구.*
*   [**MicroGPT: 단 하나의 파일로 구현한 카파시의 순수 파이썬 GPT**](./Architecture/106.md)  
    *외부 프레임워크 없이 순수 행렬 곱셈과 미분만으로 토크나이저·어텐션·역전파를 구현한 Karpathy의 교육용 코드 분석.*
*   [**Moonshine: 엣지 디바이스를 위한 초경량 가변 길이 음성 인식(ASR)**](./Architecture/108.md)  
    *Whisper의 고정 30초 패딩 낭비를 없애고 입력 길이에 비례하는 연산과 RoPE를 적용해 온디바이스 속도를 극대화한 ASR 모델.*
*   [**Nested Learning: 딥러닝의 본질은 '깊이'가 아니라 다중 시간축의 '중첩'**](./Architecture/109.md)  
    *신경망 아키텍처와 옵티마이저를 서로 다른 주기로 업데이트되는 중첩 메모리 시스템으로 통합 해석한 DeepMind 연구.*
*   [**Diffusion LLM (100B): 30B 자기회귀 모델보다 2배 빠른 병렬 생성**](./Architecture/83.md)  
    *토큰을 한 글자씩 생성하는 AR 한계를 깨고 블록 단위 병렬 디코딩으로 속도와 추론 성능을 모두 잡은 100B 디퓨전 LLM.*
*   [**Were RNNs All We Needed?: 병렬 학습 가능한 minLSTM과 minGRU의 부활**](./Architecture/102.md)  
    *기존 LSTM/GRU의 은닉 상태 의존성을 제거해 트랜스포머처럼 병렬 학습이 가능하면서도 추론은 가벼운 초경량 RNN.*
*   [**Titans: Learning to Memorize at Test Time**](./Architecture/101.md)  
    *단기 어텐션과 추론 시점에 가중치를 업데이트하는 신경망 장기 메모리 모듈을 결합해 200만 토큰 이상을 처리하는 아키텍처.*
*   [**LLM의 입력 길이 제곱($N^2$)의 저주를 끊어내는 3가지 하이브리드 구조**](./Architecture/90.md)  
    *완전히 새로운 구조 대신 기존 트랜스포머에 선형 어텐션·하드웨어 최적화를 결합해 $O(N^2)$ 한계를 돌파한 최신 모델 3선.*
*   [**Mistral Large 3: 전문가 크기를 키우고 개수를 줄인 MoE 지연 시간 최적화**](./Architecture/88.md)  
    *DeepSeek V3 구조를 수용하되 전문가(Expert) 수를 줄이고 개별 크기를 키워 추론 지연(Latency)을 최소화한 설계 분석.*
*   [**표준이 된 DeepSeek V3 아키텍처: Mistral Large 3와 Kimi K2 비교**](./Architecture/86.md)  
    *MLA(Multi-head Latent Attention)와 세분화된 MoE 등 글로벌 오픈 모델의 표준 설계로 자리 잡은 V3 아키텍처 비교.*
*   [**Ai2 Olmo 3: 데이터부터 가중치까지 100% 공개한 투명한 LLM 교과서**](./Architecture/85.md)  
    *사전학습 데이터 구성, 중간 체크포인트, 학습 레시피를 완전 개방해 연구 재현성의 표준을 제시한 Ai2의 오픈 모델.*
*   [**Nemotron-3-Nano-30B-A3B: Mamba-2와 MoE를 결합한 고효율 하이브리드**](./Architecture/50.md)  
    *대부분의 층을 Mamba-2와 MoE로 구성하고 소수의 어텐션 층만 배치해 Qwen3 대비 압도적 처리량을 달성한 NVIDIA 모델.*
*   [**DeepSeek Engram: 연산(MoE)과 정적 지식 기억(Memory)을 분리한 새로운 희소성 축**](./Architecture/44.md)  
    *단순 사실 암기에 연산 자원을 낭비하지 않도록 $O(1)$ 해시 조회 기반의 거대 정적 메모리 테이블(Engram)을 분리한 아키텍처.*
*   [**Falcon-H1-Tiny: 90M·600M 초소형 체급으로 추론과 지시 이행을 해내는 비결**](./Architecture/41.md)  
    *Mamba-2와 어텐션을 비대칭 결합하고 깊이를 극대화해 1B 미만 크기에서 고성능 지시 이행과 수학 추론을 달성한 아키텍처.*
*   [**RoPE++: 버려지던 허수 성분을 복원해 장기 문맥 유실을 막는 위치 인코딩**](./Architecture/151.md)  
    *기존 RoPE가 계산 편의를 위해 버리던 복소수 허수부 정보를 되살려 추가 메모리 없이 장문 기억력을 높인 푸단대 연구.*
*   [**Sakana AI DroPE: 위치 인코딩은 학습 때만 쓰고 추론할 땐 버려라**](./Architecture/47.md)  
    *사전학습 후 위치 임베딩(RoPE)을 아예 제거하고 짧은 재적응만 거쳐 최소 비용으로 컨텍스트 길이를 확장하는 기법.*
*   [**Sakana AI RePo: 기계적 토큰 번호표 대신 문맥에 맞게 위치를 재배치하라**](./Architecture/38.md)  
    *고정된 순차 위치 인덱스 대신 모델이 토큰 간 의미적 거리에 따라 위치 값을 동적으로 재할당(Re-Positioning)하는 기법.*
*   [**DeepSeek vs Qwen (30B-A3B MoE): 같은 체급, 정반대의 아키텍처 설계 철학**](./Architecture/35.md)  
    *동일한 30B(활성 3B) MoE 체급에서 MLA·공유 전문가를 택한 DeepSeek와 GQA·비공유 전문가를 택한 Qwen의 구조 비교.*
*   [**Generative Modeling via Drifting: 확산 모델의 250단계를 단 1단계로 줄이다**](./Architecture/6.md)  
    *반복적인 노이즈 제거 대신 드리프팅(Drifting) 벡터장 학습으로 단 한 번의 연산(1-step)에 고품질 생성을 달성한 연구.*
*   [**Beyond Transformers 2: StepFun의 극한 효율화와 Fastino의 인코더 회귀**](./Architecture/13.md)  
    *기존 GPT 디코더를 하드웨어 극한까지 최적화한 StepFun과, BERT식 양방향 인코더 구조로 속도를 혁신한 Fastino 비교.*
*   [**DeepSeek-V3 vs V3.2: MLA와 희소 어텐션(DSA)으로 진화한 아키텍처**](./Architecture/94.md)  
    *DeepSeek-V3에서 V3.2로 넘어가며 도입된 동적 희소 어텐션(DSA)과 초장문 연산 비용 절감 메커니즘 비교.*
*   [**Gemma 3 아키텍처 분석: 로컬·글로벌 어텐션 5:1 배치와 멀티모달 설계**](./Architecture/98.md)  
    *KV 캐시 폭발을 막기 위해 로컬 슬라이딩 윈도우와 글로벌 어텐션을 5:1 비율로 배치한 구글 Gemma 3의 핵심 구조.*
## <a id="pre-training"></a>📚 Pre-Training

*   [**Breaking the Token Ceiling: 256개 순수 바이트로 토큰 천장을 뚫는 사전학습 지식 증류**](./Pre_Training/100.md)  
    *어휘 사전 없이 256개 바이트와 End-of-Token 증류로 데이터·저장 공간을 줄이고 토큰 모델 성능을 넘어선 Meta FAIR 연구.*
*   [**Skaling: 친칠라 법칙의 오류를 깨고 스케일링 예측 비용을 10배 줄인 법칙**](./Pre_Training/99.md)  
    *파라미터-데이터 간 음의 교차 시너지를 반영하고 L자형 샘플링으로 스케일링 예측 연산 비용을 10배 절감한 Meta 연구.*
*   [**There Will Be a Scientific Theory of Deep Learning: 연금술에서 과학으로**](./Pre_Training/101.md)  
    *딥러닝을 예측 가능한 '학습 역학'으로 정의하고 스케일링 법칙과 하이퍼파라미터 전이의 수학적·물리적 원리를 정리한 리뷰.*
*   [**TEON vs Muon: 레이어를 넘어 텐서 차원으로 확장된 사전학습 옵티마이저**](./Pre_Training/102.md)  
    *행렬 단위 직교화를 수행하는 Muon을 넘어 여러 레이어의 그래디언트를 3차원 텐서로 묶어 최적화하는 TEON 분석.*
*   [**한국어를 잘하는 소형 모델 Mid-training (LFM2.5-8B-A1B)**](./Pre_Training/98.md)  
    *LFM2.5-8B-A1B 모델에 6.5B 토큰 규모의 한국어 연속 사전학습(CPT)을 적용한 실험 성과와 한계점 공유.*
*   [**Vocabulary Curriculum Learning: 인간의 언어 습득 방식을 모방한 점진적 어휘 학습**](./Pre_Training/97.md)  
    *작은 어휘 사전에서 시작해 점진적으로 토큰 단위를 확장함으로써 사전학습 효율을 높이는 커리큘럼 학습법.*
*   [**CALM: 1토큰씩 타이핑하는 한계를 넘어 4개씩 생성하는 연속 벡터 예측**](./Pre_Training/34.md)  
    *이산 토큰 대신 여러 토큰을 묶은 연속 잠재 벡터를 한 번에 예측해 자기회귀 생성 속도와 연산 효율을 높인 사전학습 패러다임.*

## <a id="post-training"></a>🎯 Post-Training

*   [**Reactive Optimizers: 학습 도중 에이전트가 실시간으로 옵티마이저 코드를 고쳐 성능을 높이다**](./Post_Training/157.md)  
    *Claude Opus 5.5가 학습 상태를 진단하고 업데이트 규칙(`updater.py`)을 실시간 재작성해 고정 옵티마이저를 앞지른 MIT 연구.*
*   [**RLTLDR: 17토큰 셀프 오답노트와 역전파 마스크 반전으로 정답률 0% 난제 돌파**](./Post_Training/155.md)  
    *실패 로그에서 뽑은 17토큰 교훈 누적 탐색과 프롬프트 내 힌트 구간 SFT 손실 결합으로 초고난도 문제를 푼 Apple 연구.*
*   [**Terminal-RSI: 힌트 세탁 자가 SFT와 턴 단위 PPO 반복을 통한 터미널 에이전트 자가 개선**](./Post_Training/154.md)  
    *보조 하네스 힌트를 지우는 자가 SFT와 턴 단위 감쇠 PPO를 반복해 외부 데이터 없이 성능을 2.5배 높인 텐센트 연구.*
*   [**LoGRA: 저랭크 기울기 스케치로 LoRA보다 적은 메모리로 달성한 풀 파라미터 RL**](./Post_Training/153.md)  
    *역전파 시 1MB 저랭크 스케치만 생성해 LoRA 대비 메모리를 45.6% 아끼면서 27B 풀 파라미터 RL을 완주한 NVIDIA 연구.*
*   [**NeMo-DCR: 1T 에이전틱 RL을 위한 비트 단위 델타 압축 가중치 동기화**](./Post_Training/156.md)  
    *RL 중 실제 변하는 1% 가중치 비트만 XOR·zstd로 압축 전송해 동기화 시간을 87.5분에서 2.5분으로 줄인 NVIDIA 연구.*
*   [**Finetuning with Sampling: 온폴리시 MCMC 투영 샘플링으로 RL을 넘어선 순수 SFT**](./Post_Training/152.md)  
    *전문가 정답을 베이스 모델의 말투로 재생성해 순수 SFT만으로 GRPO를 능가하고 치명적 망각을 막은 하버드 연구.*
*   [**Theoretical Limits of Language Model Alignment: 제프리스 발산으로 본 RL의 한계**](./Post_Training/151.md)  
    *학습 전 달성 가능한 절대 보상 상한선을 수학적으로 유도하고 PPO/GRPO의 한계와 Best-of-N의 최적성을 증명한 Apple 논문.*
*   [**MiMo-V2.6: 스텝당 30억 토큰 대규모 비동기 RL과 채점기 에이전트 레시피**](./Post_Training/150.md)  
    *대규모 비동기 강화학습과 루브릭 기반 채점기 에이전트(Grader Agent)로 오픈웨이트 최고 성능을 달성한 샤오미 기술 보고서.*
*   [**Single-Rollout Asynchronous Optimization (SAO): 멀티턴 에이전트 비동기 강화학습**](./Post_Training/149.md)  
    *가치평가 모델(Critic)과 환경 토큰 스킵 GAE를 도입해 프롬프트당 단일 롤아웃만으로 비동기 에이전트 RL을 구현한 기법.*
*   [**NoRA (Normalized LoRA): 행렬 정규화 한 줄로 LoRA의 성능과 RL 안정성 극대화**](./Post_Training/148.md)  
    *LoRA의 A 행렬 열 벡터를 단위 크기로 정규화해 초기 그래디언트 왜곡과 치명적 망각을 동시에 해결한 연구.*
*   [**Never Give Up (NGU): 쉬운 문제는 넘기고 어려운 문제만 끝까지 파는 강화학습**](./Post_Training/147.md)  
    *이미 푼 문제는 조기 탈락시키고 오답 문제만 비동기 큐에 재투입해 고난도 수학·코딩 난제를 뚫어낸 RL 기법.*
*   [**Do We Need Adam?: 강화학습(RLVR)에서 AdamW 대신 순수 SGD가 더 강력한 이유**](./Post_Training/146.md)  
    *정책이 실시간 변하는 RLVR에서 모멘텀 없이 순수 SGD만 써서 옵티마이저 메모리를 3배 아끼고 성능까지 높인 연구.*
*   [**Spurious Advantage Hidden in GRPO: 찍어서 맞힌 요행의 폭탄 보상 결함과 SignBalance**](./Post_Training/145.md)  
    *GRPO가 소수 정답 궤적에 과도한 보상을 주는 수학적 역설을 규명하고 부호 균등 분배(SignBalance)로 해결한 연구.*
*   [**TailSFT: 아는 문제를 버려야 RL이 산다 — 엔트로피 붕괴를 막는 SFT 필터링**](./Post_Training/144.md)  
    *쉬운 예제를 제외하고 어려운 꼬리(Tail) 데이터만 SFT하여 탐색 다양성(pass@k)과 후속 RL 성능을 보존한 MS 연구.*
*   [**Understanding Reasoning from Pretraining to Post-Training: 사전학습과 RL 예산 배분 법칙**](./Post_Training/143.md)  
    *사전학습 품질이 사후 RL의 상한선을 결정짓는 원리를 규명하고 제한된 연산 예산의 최적 배분 공식을 제시한 논문.*
*   [**AgentOPSD: 멀티턴 에이전트 강화학습을 위한 턴 단위 기여도 정밀 평가**](./Post_Training/142.md)  
    *장기 궤적에서 보상 신호가 희석되는 문제를 Self-Teacher와 베이지안 신뢰도 기반 턴별 평가로 해결한 연구.*
*   [**SFT와 RL의 역할 차이: 지식을 엮는 SFT와 재사용 부품으로 분해하는 RL**](./Post_Training/141.md)  
    *SFT는 지식을 얽힌 채 주입하고 RL은 이를 모듈화하여 OOD 일반화 능력을 끌어올린다는 메커니즘 규명.*
*   [**지식 그래프 기반 강화학습: LLM이 스스로 가설을 세우며 추론하게 만드는 법**](./Post_Training/139.md)  
    *추론 과정에서 지식 그래프를 실시간 구축하고 가설을 검증하도록 강제해 환각을 억제하는 강화학습 기법.*
*   [**VibeThinker-3B: 3B 소형 모델이 거대 추론 모델을 이길 수 있었던 비결**](./Post_Training/137.md)  
    *다양한 풀이 스펙트럼을 담은 SFT와 타겟 강화학습(RL) 결합으로 3B 체급에서 고난도 추론을 달성한 레시피 분석.*
*   [**ECHO: 터미널 에러 메시지와 환경 로그를 월드 모델 손실로 활용하는 에이전트 학습**](./Post_Training/135.md)  
    *버려지던 터미널 에러 로그와 실행 결과에도 Loss를 걸어 환경 역학을 학습시킴으로써 에이전트 성능을 2배 높인 MSR 연구.*
*   [**Faithful Uncertainty: 환각을 억지로 누르지 말고 불확실성을 표현하게 하라**](./Post_Training/132.md)  
    *무리한 환각 억제가 추론 성능을 깎는 정렬 세금(Utility Tax)을 밝히고, 확신도를 정직하게 표현하도록 훈련하는 기법.*
*   [**Speculative Decoding을 결합한 RL 훈련 속도 2.5배 가속 (NVIDIA)**](./Post_Training/131.md)  
    *RL 학습의 최대 병목인 롤아웃 생성에 추측 해독(Speculative Decoding)을 통합해 품질 저하 없이 속도를 2.5배 높인 기법.*
*   [**REASONMAXXER: 단일 GPU와 50개 결정 포인트 데이터로 끝내는 초경량 추론 튜닝**](./Post_Training/130.md)  
    *무거운 RL 대신 엔트로피가 급증하는 '결정 포인트' 50개와 초소형 LoRA만으로 추론 성능을 극대화한 연구.*
*   [**PivotRL: 전체 궤적 대신 추론의 전환점(Pivot)만 골라 집중 훈련하는 강화학습**](./Post_Training/128.md)  
    *답변의 운명을 가르는 피벗 구간만 선별해 학습함으로써 망각을 막고 RL 훈련 시간을 5.5배 단축한 NVIDIA 연구.*
*   [**Simple Self-Distillation (SSD): 외부 피드백 없이 모델 자체 출력으로 성능 올리기**](./Post_Training/127.md)  
    *보상 모델이나 정답지 없이 모델 스스로 생성한 데이터를 재증류해 추론 기초 체력을 높이는 Apple의 자가 증류 기법.*
*   [**TESSY 프레임워크: 교사-학생 모델 간 문체 충돌을 해결해 SFT 성능 끌어올리기**](./Post_Training/126.md)  
    *똑똑한 교사 모델의 데이터를 그대로 학습할 때 발생하는 분포·스타일 충돌을 완화해 학생 모델 성능을 높이는 기법.*
*   [**OpenClaw-RL: 실전 에이전트 사용 로그로 실시간 강화학습을 돌리는 프레임워크**](./Post_Training/124.md)  
    *프린스턴 연구진이 제안한 OpenClaw 기반 온라인 강화학습 구조와 실전 환경 피드백 학습의 한계점 분석.*
*   [**On Data Engineering for Scaling LLM Terminal Capabilities: 32B로 480B를 이긴 비결**](./Post_Training/158.md)  
    *실제 터미널 실행 환경 검증과 합성 데이터 파이프라인으로 32B 코딩 에이전트가 480B 모델을 넘어선 NVIDIA 레시피.*
*   [**샌드박스를 탈출해 코인을 채굴한 에이전트: 알리바바 ALE와 청크 단위 RL(IPA)**](./Post_Training/159.md)  
    *강화학습 중 에이전트가 보안망을 뚫고 GPU 채굴을 시도한 일화와 토큰 대신 행동 청크 단위로 채점하는 IPA 알고리즘.*
*   [**LoRASA: 공유 백본과 개별 LoRA 어댑터로 구현한 멀티 에이전트 적응 학습**](./Post_Training/160.md)  
    *다중 에이전트 강화학습에서 팀 공통 정책과 에이전트별 특수 역할을 저랭크 어댑터(LoRA)로 분리 학습하는 기법.*
*   [**QED-Nano: SFT와 강화학습으로 100B 모델을 압도한 4B 수학 증명 모델**](./Post_Training/161.md)  
    *정교한 형식 증명 데이터 정제와 RL 파이프라인을 통해 4B 소형 모델로 100B급 수학 정리 증명 성능을 달성한 연구.*
*   [**TinyLoRA: 단 13개의 학습 파라미터만으로 GSM8K 91%를 달성한 극한 강화학습**](./Post_Training/110.md)  
    *가중치 행렬의 극소수 스케일링 파라미터(13개)만 GRPO로 업데이트해 소형 모델의 잠재 수학 능력을 깨운 연구.*
*   [**Parameter-Efficient Fine-Tuning (PEFT) 5대 핵심 기법 총정리**](./Post_Training/106.md)  
    *LoRA, QLoRA, DoRA, Adapter 등 거대 모델을 최소한의 VRAM으로 파인튜닝하는 PEFT 알고리즘 비교.*
*   [**LORE (When Reasoning Meets its Laws): 3,900개 데이터로 가르치는 추론의 법칙**](./Post_Training/108.md)  
    *추론 과정의 단조성·합성성 법칙을 정의하고 소량의 정제 데이터로 모델의 논리적 일관성을 훈련하는 기법.*
*   [**LIE: 성급한 결론을 막고 깊게 생각하도록 유도하는 강화학습 전략**](./Post_Training/109.md)  
    *LLM이 추론을 일찍 포기하는 현상을 분석하고 탐색 길이를 늘려 고난도 문제 해결력을 높이는 RL 기법.*
*   [**ProRL (Prolonged Reinforcement Learning): 강화학습을 오래 돌릴수록 열리는 새로운 능력**](./Post_Training/107.md)  
    *짧게 끝내던 기존 RL 통념을 깨고 장기간 안정적인 RL 학습을 유지해 베이스 모델에 없던 추론 패턴을 발현시킨 연구.*
*   [**DuPO: 정답지 없는 번역·작문을 역방향 쌍대성(Duality)으로 자가 검증하는 학습**](./Post_Training/104.md)  
    *번역 결과를 다시 원문으로 복원해 정보 손실을 측정하는 쌍대성 원리로 외부 보상 없이 자가 학습하는 ByteDance 기법.*
*   [**From Code Foundation Models to Agents: 코드 모델에서 자율 코딩 에이전트로의 진화**](./Post_Training/84.md)  
    *단순 코드 자동완성 모델을 넘어 리포지토리 탐색·테스트·디버깅을 수행하는 코딩 에이전트 학습 파이프라인 총정리.*
*   [**Emergent Misalignment: 취약한 코드만 학습시켰을 뿐인데 반인류적 AI가 되는 현상**](./Post_Training/96.md)  
    *보안 취약점이 있는 코드로 파인튜닝한 모델이 코딩 외 일반 대화에서도 기만적·공격적 성향을 보이는 창발적 오정렬 실증.*
*   [**Stabilizing RL with LLMs: 화려한 기교보다 수학적 기본기가 중요한 강화학습**](./Post_Training/92.md)  
    *길이 정규화 같은 인위적 휴리스틱이 오히려 RL을 망친다는 것을 증명하고 정석적인 목적 함수 설계를 제시한 Qwen 논문.*
*   [**Yann LeCun의 비판(World Model)과 ByteDance DuPO의 일반화된 쌍대성 검증**](./Post_Training/78.md)  
    *LLM의 물리 세계 이해 한계를 지적한 르쿤의 시각과 정답지 없는 영역에서 쌍대성으로 자기 검증을 구현한 DuPO 분석.*
*   [**GDPO: 다중 보상(Multi-reward) 강화학습에서 GRPO의 평균 함정을 해결하다**](./Post_Training/46.md)  
    *정답과 포맷 등 여러 보상을 단순 합산할 때 발생하는 보상 붕괴를 개별 정규화(GDPO)로 해결한 NVIDIA·HKUST 연구.*
*   [**iGRPO: 모델이 스스로 만든 초안(Draft)을 피드백 삼아 한 번 더 도약하는 강화학습**](./Post_Training/1.md)  
    *1차 생성 답변 중 최선의 초안을 조건으로 삼아 2차 정제 추론을 수행하도록 GRPO를 확장한 자가 피드백 강화학습.*

## <a id="inference--prompting"></a>⚡ Inference & Prompting

*   [**Reasoning Models Can Control Their Chains of Thought: CoT 첫 줄 주입을 통한 추론 통제**](./Inference/108.md)  
    *추론 모델이 유저의 사고 제약을 무시하는 현상을 `<think>` 직후 1인칭 혼잣말 프리필로 가중치 수정 없이 해결한 연구.*
*   [**Empty Shelves or Lost Keys?: 프론티어 모델의 95% 지식 내재화와 인출 병목(Recall Bottleneck)**](./Inference/107.md)  
    *LLM의 오답 원인이 지식 부재가 아닌 인출 실패임을 규명하고 CoT 추론의 자가 점화 효과를 증명한 구글 리서치 논문.*
*   [**Thinking to Recall: 추론 시간(CoT) 부여를 통한 내재 지식 인출 능력 해금**](./Inference/106.md)  
    *단순 단답형 질문에서도 모델에게 생각하는 과정(CoT)을 거치게 하면 파라미터 속 숨은 지식 인출률이 급증함을 보인 연구.*
*   [**Recursive Language Models (RLM): Python 재귀 호출로 하드웨어 한계 없이 1,000만 토큰 처리**](./Inference/105.md)  
    *초장문 텍스트를 한 번에 입력하는 대신 파이썬 환경에서 문서를 쪼개고 자신을 재귀 호출해 10M 토큰을 처리하는 MIT 연구.*
*   [**Prompt Repetition Improves Non-Reasoning LLMs: 같은 프롬프트 2번 반복의 마법**](./Inference/104.md)  
    *단방향(Causal) 어텐션의 한계를 보완하기 위해 질문 프롬프트를 단순히 두 번 연속 입력해 비추론 모델 성능을 높인 기법.*
*   [**Chain-of-Draft (CoD): 핵심 수식과 키워드만 적는 초고효율 추론 프롬프팅**](./Inference/103.md)  
    *장황한 CoT 대신 단계별 5단어 이내의 핵심 메모만 남기도록 유도해 정확도는 유지하면서 토큰과 지연을 대폭 줄인 기법.*
*   [**vLLM과 SGLang의 작동 원리: PagedAttention과 RadixAttention 비교**](./Inference/102.md)  
    *OS 페이징 기법으로 KV 캐시 단편화를 없앤 vLLM과 트리 기반 프리픽스 캐싱으로 에이전트 추론을 가속한 SGLang 비교.*
*   [**Detailed Balance in LLM-Driven Agents: 물리학 최소 작용 원리로 증명한 다수결 투표**](./Inference/101.md)  
    *LLM의 다단계 추론 궤적이 통계역학의 상세 균형 조건을 따름을 보이고 테스트 시점 다수결 투표의 수렴 원리를 규명한 연구.*

## <a id="steering--weight-editing"></a>🧬 Steering & Weight Editing

*   [**Weight-Space Programming: 훈련 없이 가중치 0.016%만 편집해 수학 성능을 6배 높인 GPT-6 Astra**](./Steering/106.md)  
    *GPT-6 Astra가 50번의 순전파 분석만으로 무한 루프 유발 MLP 방향 벡터를 찾아 직교 투영으로 도려낸 MIT CSAIL 연구.*
*   [**Neural Procedural Memory (NPM): 활성화 조향(Activation Steering)으로 고치는 에이전트 행동 오류**](./Steering/105.md)  
    *재학습 없이 신경망 은닉 상태에 절차적 기억 벡터를 주입해 에이전트의 반복적인 행동 오류를 실시간 교정하는 기법.*
*   [**A Single Neuron Is Sufficient: 단 하나의 뉴런 비활성화로 무력화되는 Safety Alignment**](./Steering/104.md)  
    *단 1개의 거부 뉴런만 0으로 고정해도 DPO·RLHF로 쌓은 안전 장벽이 성능 저하 없이 100% 해제됨을 폭로한 Apple 연구.*
*   [**VPD (Adversarial Parameter Decomposition): 가중치 분해와 무학습 편집을 통한 모델 제어**](./Steering/103.md)  
    *거대 가중치를 기능별 개념 조각으로 분해하여 재학습 없이 특정 능력이나 성향만 정밀하게 가감하는 가중치 편집 기법.*
*   [**Text-to-LoRA & Doc-to-LoRA: 단 한 번의 순전파(0.5초)로 문서를 LoRA 가중치로 변환**](./Steering/102.md)  
    *파인튜닝 없이 하이퍼네트워크 순전파만으로 텍스트 지시사항이나 문서를 즉석에서 LoRA 어댑터 가중치로 구워내는 Sakana AI 기술.*
*   [**ASA (Activation Steering Adapter): 20KB 활성화 벡터 주입으로 깨우는 도구 호출 능력**](./Steering/101.md)  
    *도구 호출을 주저하는 모델(Lazy Agent)에 학습 없이 20KB 조향 벡터만 주입해 Tool Calling 성능을 2배로 끌어올린 기법.*

## <a id="evaluation"></a>⚖️ Evaluation

*   [**CHEATBENCH: 최신 프론티어 에이전트들의 보상 해킹(Reward Gaming)과 부정행위 실태**](./Evaluation/114.md)  
    *고난도 과업에서 프론티어 모델들이 48~81% 확률로 채점기를 속이거나 테스트 코드를 조작하는 현상을 폭로한 벤치마크.*
*   [**Jev-as-a-Judge: 확신할 땐 수용하고 모호할 땐 상위 모델로 이관하는 초저가 평가 캐스케이드**](./Evaluation/113.md)  
    *출력 토큰 없이 152ms 만에 판정하는 Jev 모델과 자체 확신도 기반 이관 전략으로 평가 비용을 47% 절감한 CMU 논문.*
*   [**Persuasion Attacks: CoT 추론 과정을 악용한 감시자 모델 가스라이팅과 교차 검증**](./Evaluation/115.md)  
    *에이전트가 자신의 CoT에 그럴듯한 변명을 적어 감시자 LLM을 설득하는 공격 기법과 이종 모델 교차 검증 해법.*
*   [**Reasoning or Memorization?: 벤치마크 오염과 기출 암기 속 실전 모델 선택 기준**](./Evaluation/112.md)  
    *벤치마크 데이터 오염과 리더보드 점수 인플레이션의 한계를 짚고 실무 태스크 기반 검증의 중요성을 정리한 고찰.*
*   [**LLM의 메타인지(Metacognition) 서베이: 자신이 무엇을 아는지 파악하는 능력**](./Evaluation/111.md)  
    *최신 LLM의 자기 지식 모니터링 능력과 20점 단위 채점 스케일링, 내재적 자가 수정의 한계를 분석한 서베이.*
*   [**Entropy Distribution as a Fingerprint: 출력 엔트로피 분포로 만든 실시간 환각 탐지기**](./Evaluation/110.md)  
    *토큰 생성 시 요동치는 로짓 확률 분포(엔트로피) 패턴을 분석해 모델이 지어낸 거짓말을 잡아내는 탐지 기법.*
*   [**CiteVQA: 멀티모달 모델(VLM)이 이미지를 안 보고 답하는 귀속 환각(Attribution Hallucination) 폭로**](./Evaluation/109.md)  
    *VLM이 정답은 맞히면서도 실제 문서의 근거 영역은 전혀 짚지 못하는 시각적 환각 현상을 실증한 벤치마크.*
*   [**Step-3.5, LLaDA 2.1 등 86개 최신 모델을 직접 벤치마킹하며 얻은 인사이트**](./Evaluation/108.md)  
    *86개 모델 실전 테스트를 통해 확인한 기출 암기 현상, 한국어 파이프라인의 한계, 베이스 모델 체급의 중요성 정리.*
*   [**Mirage: 이미지를 가려도 정답을 맞히는 멀티모달 시각 벤치마크의 신기루 현상**](./Evaluation/107.md)  
    *시각 정보 없이 텍스트 질문의 통계적 편향만으로 VLM 벤치마크 고득점이 가능한 문제를 고발하고 정제 해법을 제시.*
*   [**IKP 벤치마크: 압축 불가능한 희귀 지식 정답률로 독점 모델의 파라미터 크기 역산하기**](./Evaluation/106.md)  
    *희귀 사실 암기량과 파라미터 규모 간의 로그-선형 비례 법칙을 이용해 비공개 프론티어 모델들의 실제 크기를 추정한 연구.*
*   [**Preference Leakage: LLM-as-a-Judge 평가자가 자기 식구 모델을 편애하는 문제**](./Evaluation/105.md)  
    *생성 모델과 평가 모델이 같은 계열일 때 발생하는 선호도 유출 및 점수 왜곡 현상을 분석한 논문.*
*   [**AgentArcEval: 에이전트의 자율성과 비결정성을 반영한 아키텍처 평가 프레임워크**](./Evaluation/116.md)  
    *매번 다른 궤적을 그리는 에이전트 시스템의 특성을 고려해 신뢰성과 아키텍처 완성도를 정량 평가하는 프레임워크.*
*   [**AIRS-Bench: AI 과학 연구 에이전트를 위한 Meta FAIR의 종합 벤치마크**](./Evaluation/117.md)  
    *논문 분석부터 실험 설계·코드 구현까지 과학 연구 전 주기에서 로컬 모델과 프론티어 에이전트의 역량을 비교한 벤치마크.*
*   [**ADR-Bench 전문가 평가: 정답 보고서에서 질문을 역설계해 평가한 심층 리서치 벤치마크**](./Evaluation/55.md)  
    *고품질 전문 보고서를 역설계해 문제를 만들고 심층 분석 능력을 측정하여 소형 에이전트 모델의 효율성을 입증한 평가.*
## <a id="rag--knowledge"></a>🗂️ RAG & Knowledge

*   [**RAG에서 의사결정 모델(Jev) 올바르게 쓰기: 단순 리랭커의 함정과 DecisionRAG**](./RAG/128.md)  
    *Jev를 단순 리랭커로 쓸 때의 실패 원인과, 벡터 검색 자체를 계층적 의사결정 탐색(DecisionRAG)으로 대체한 연구 비교.*
*   [**Memory Tax: 에이전트 메모리의 자체 요약이 부르는 정보 손실과 2단계 RAG의 승리**](./RAG/127.md)  
    *대화 사전 요약 시 핵심 수치가 증발하는 '메모리 세금'을 규명하고 원본 보존 2단계 에이전틱 RAG의 우수성을 입증.*
*   [**Is Complex Agentic Memory Always Needed?: 요약 메모리보다 83배 빠르고 정확한 RAW-RAG**](./RAG/126.md)  
    *복잡한 요약·수정 메모리(Mem0 등)보다 256토큰 원본 청킹과 리랭커 조합의 정석 RAW-RAG가 속도와 정확도 모두 앞섬을 증명.*
*   [**MRAgent: Memory is Reconstructed, Not Retrieved**](./RAG/125.md)  
    *단순 벡터 검색 대신 단서와 태그로 지식을 구조화하고 질의 시점에 기억을 능동적으로 재구성하는 에이전틱 GraphRAG.*
*   [**LogicRAG: 사전 지식 그래프 구축 없이 질문 시점에 논리 경로를 개척하는 GraphRAG**](./RAG/124.md)  
    *막대한 비용이 드는 사전 그래프 구축 없이 질의가 들어올 때 필요한 논리 구조만 동적으로 탐색하는 경량 GraphRAG.*
*   [**MEMO: Vector DB 대신 문서를 암기한 초소형 비서 sLM을 LLM 옆에 두는 아키텍처**](./RAG/123.md)  
    *외부 벡터 검색 대신 문서를 고품질 QA로 정제해 소형 모델(sLM) 가중치에 암기시켜 메인 LLM을 보조하는 기법.*
*   [**Is Grep All You Need?: 벡터 DB 없이 리눅스 grep과 TF-IDF로 구축한 초간단 RAG**](./RAG/122.md)  
    *무거운 임베딩 모델이나 벡터 DB 없이 리눅스 `grep`과 TF-IDF만으로 하이브리드 RAG를 능가한 에이전틱 검색 연구.*
*   [**RAG와 벡터 DB 구축 핵심 Q&A 및 실무 데이터 엔지니어링 정리**](./RAG/121.md)  
    *벡터 DB 필수 여부, 컨텍스트 압축, 소형 모델 활용법 등 실전 RAG 파이프라인 구축의 핵심 원칙 요약.*
*   [**DCI (Direct Corpus Interaction): 벡터 DB와 청킹 없이 문서 좌표를 직접 찾는 아키텍처**](./RAG/120.md)  
    *문서를 쪼개거나 임베딩하지 않고 모델이 직접 코퍼스 인덱스를 탐색해 정밀한 텍스트 좌표를 찾아내는 칭화대 연구.*
*   [**SLIDERS: 초거대 문서를 SQL 데이터베이스 표(RDB)로 전처리해 환각을 잡는 기법**](./RAG/119.md)  
    *비정형 장문 문서를 관계형 DB 스키마로 구조화하고 SQL 쿼리로 탐색해 환각과 롱 컨텍스트 한계를 극복하는 방법.*
*   [**MSA (Memory Sparse Attention): 1억 토큰을 모델 내부에 담아내는 희소 어텐션**](./RAG/118.md)  
    *외부 RAG 파이프라인 대신 모델 내부의 메모리 희소 어텐션을 통해 최대 1억 토큰의 문맥을 직접 처리하는 기술.*
*   [**KARL: 가벼운 오픈소스 모델에게 검색 전략을 강화학습시켜 플래그십 잡기**](./RAG/117.md)  
    *소형 모델이 다단계 검색과 근거 종합을 스스로 수행하도록 RL로 훈련해 거대 독점 모델을 넘어선 연구.*
*   [**LIMRANK (Less is More): 2만 개의 고품질 데이터만으로 완성한 SOTA 리랭커**](./RAG/109.md)  
    *수백만 개의 노이즈 데이터 대신 정교하게 선별한 2만 개 추론형 데이터만으로 최고 성능의 리랭커를 학습한 레시피.*
*   [**HippoRAG 2: 인간 해마의 기억 인덱싱을 모방한 비모수적 연속 학습**](./RAG/100.md)  
    *지식 그래프와 Personalized PageRank를 결합해 인간의 연상 기억과 사실 기억을 동시에 구현한 RAG 프레임워크.*
*   [**RAG & Agent Memory 최신 논문 4선 (GraphSearch, S-RAG, xMemory 등)**](./RAG/10.md)  
    *그래프 탐색, 구조적 검색, 에이전트 장기 메모리 관리를 다룬 최신 핵심 논문 4편 요약.*
*   [**Beyond Naive RAG: 에이전트 메모리를 재정의하는 4가지 차세대 RAG 아키텍처**](./RAG/110.md)  
    *단순 청크 검색(Naive RAG)의 한계를 넘어 에이전틱 탐색과 계층적 기억 구조를 도입한 최신 연구 4편 분석.*
*   [**SEPAL: 9천만 개 노드의 초대형 지식 그래프를 V100 한 장으로 학습하기**](./RAG/111.md)  
    *그래프 구조 전파와 특징 학습을 분리해 단일 구형 GPU에서도 수천만 노드 규모의 지식 그래프 임베딩을 가능케 한 기법.*
*   [**GraphRAG Survey (ACM TOIS): 지식 그래프 기반 RAG의 설계 원리와 분류 체계**](./RAG/112.md)  
    *그래프 인덱싱, 경로 탐색, 서브그래프 요약 등 GraphRAG의 전체 파이프라인과 핵심 기법들을 총망라한 서베이.*
*   [**A-MEM (Agentic Memory): 스스로 연결하고 진화하는 에이전트 기억 네트워크**](./RAG/113.md)  
    *제텔카스텐(Zettelkasten) 메모법에서 영감을 받아 새로운 기억이 들어올 때마다 기존 기억 노드를 동적으로 재구축하는 시스템.*
*   [**PISCO: 문서를 16배 압축해도 RAG 정확도를 그대로 유지하는 초고효율 압축**](./RAG/114.md)  
    *사전학습 없이 검색된 문서를 소수의 메모리 토큰으로 16배 압축해 추론 비용과 지연을 줄이는 RAG 전용 압축 기법.*
*   [**SymAgent: 신경-기호(Neuro-Symbolic) 추론으로 지식 그래프의 빈틈을 메우는 에이전트**](./RAG/115.md)  
    *불완전한 지식 그래프 위에서 LLM 에이전트가 기호 규칙과 자율 탐색을 결합해 복합 추론을 수행하는 프레임워크.*
*   [**VideoRAG: 텍스트를 넘어 수백 시간 분량의 초장편 비디오 맥락을 검색하는 RAG**](./RAG/116.md)  
    *영상의 시각 프레임과 오디오·자막을 지식 그래프로 통합 인덱싱해 초장편 비디오 질의응답을 해결한 멀티모달 RAG.*

## <a id="on-device-ai"></a>💻 On-Device AI

*   [**flash-moe: 48GB 맥북에서 SSD 스트리밍으로 Qwen3.5-397B-A17B 로컬 구동하기**](./On_Device/126.md)  
    *순수 C언어와 OS 비동기 I/O로 활성 전문가 가중치만 SSD에서 실시간 스트리밍해 맥북에서 397B MoE를 돌리는 추론 엔진.*
*   [**Liquid AI LFM2 1.2B vs Google Gemma 4B: 한영 번역에서 1.2B가 4B를 이긴 비결**](./On_Device/45.md)  
    *온디바이스 특화 LFM2 1.2B 모델이 정교한 데이터와 RL 튜닝을 통해 3배 큰 구글 4B 번역 성능을 넘어선 사례 분석.*
*   [**국가대표 AI 탈락 그 후: 중국 거대 모델과의 격차와 온디바이스·도구 특화 생존 전략**](./On_Device/42.md)  
    *거대 모델 정면 승부의 한계를 짚고 산업 특화·피지컬 AI 및 고성능 도구 호출 소형 모델로 승부해야 하는 이유.*
*   [**로컬 LLM 구동의 6가지 현실적 방법과 STEM 아키텍처**](./On_Device/33.md)  
    *단순 지식 인출에 고가 GPU를 낭비하지 않도록 스토리지 임베딩 조회(STEM)와 양자화 등을 활용하는 로컬 구동 전략 6가지.*
*   [**LLM 지능의 민낯과 온디바이스 의료·현장 에이전트의 한계 돌파법**](./On_Device/3.md)  
    *벤치마크 고득점 모델이 실제 진료·현장 대화에서 실패하는 원인과 능동적 질문·환경 구축을 통한 해결책.*

## <a id="projects"></a>🚀 Projects

### 🤖 Autonomous Agents
*   [**제15회 변호사 시험 선택형 150문제 만점 달성기 (Gemini 3.1 Pro + Codex)**](./Projects/125.md)  
    *법령·판례 로컬 컨텍스트 구축과 에이전트 교차 검증 하네스를 설계해 한국 변호사 시험 선택형 만점을 기록한 실험.*
*   [**Liquid AI LFM2.5-8B-A1B 터미널 에이전트 SFT 및 RLVR 실험기**](./Projects/127.md)  
    *8B(활성 1B) 초경량 모델에 터미널 에러 해결 SFT와 RLVR을 적용하며 확인한 성능 향상 폭과 탐색 한계 분석.*
*   [**Terminal-Bench 소형 에이전트 모델 실험: 누적 다운로드 60,000회 돌파**](./Projects/124.md)  
    *8B 이하 소형 모델들로 터미널 벤치마크에서 대형 모델들을 넘어선 학습 레시피와 허깅페이스 6만 다운로드 회고.*
*   [**터미널 에이전트 모델 27,000회 다운로드 소회와 GPU Poor의 생존 전략**](./Projects/123.md)  
    *제한된 GPU 자원으로 글로벌 오픈소스 생태계에서 경쟁력 있는 특화 에이전트 모델을 만든 전략 공유.*
*   [**터미널 에이전트 SFT 학습과 53개 모델 평가에서 발견한 3가지 한계점**](./Projects/128.md)  
    *H200 8대로 53개 모델을 직접 튜닝·평가하며 확인한 프록시 벤치마크의 함정과 모델별 프롬프트 포맷 호환성 문제.*
*   [**터미널 에이전트 모델 다운로드 1,000회 돌파 기념: SFT와 평가 프레임워크의 한계**](./Projects/122.md)  
    *벤치마크 오버피팅, SFT만으로는 넘을 수 없는 베이스 지능 한계, 평가 환경 병목 등 3가지 실전 교훈 정리.*
*   [**H200 8대로 진행한 56개 터미널 에이전트 모델 전면 재평가 리포트**](./Projects/121.md)  
    *Qwen 3.5, Ouro 등 56개 오픈 모델을 동일 환경에서 전수 측정해 베이스 체급과 데이터 포맷의 영향력을 비교한 분석.*
*   [**로컬 터미널 에이전트 CLI(`liquid-cli`) 개발 및 LFM2-8B 튜닝 배포기**](./Projects/120.md)  
    *LFM2-8B-A1B 모델을 파인튜닝하고 실제 로컬 터미널에서 동작하는 에이전트 CLI 도구로 패키징한 개발기.*
*   [**터미널 에이전트: 1B(8B-A1B) 초경량 모델과 Nemotron 코퍼스 실험**](./Projects/119.md)  
    *Nemotron-Terminal-Corpus를 활용해 1B급 활성 파라미터 모델로 로컬 터미널 작업을 수행한 초기 검증 결과.*
*   [**Gemini-Claw 개발기: 2시간 만에 만든 자율 코딩 및 뉴스 분석 에이전트**](./Projects/20.md)  
    *Gemini CLI를 기반으로 스스로 파이썬 코드를 작성·실행하고 웹 뉴스를 수집·요약하는 자율 에이전트 제작기.*
*   [**스스로 웹페이지를 만들고 브라우저로 실행·검증까지 하는 Gemini-Claw**](./Projects/16.md)  
    *코드 생성에서 끝나지 않고 직접 로컬 서버를 띄워 화면을 확인하며 디버깅 루프를 도는 웹 개발 에이전트 구현.*
*   [**Insight Agents: 데이터 분석과 인사이트 도출을 위한 멀티 에이전트 시스템**](./Projects/21.md)  
    *플래너·데이터 처리·시각화 에이전트가 협업해 정형 데이터에서 비즈니스 인사이트를 자동 추출하는 시스템 설계.*
*   [**SEAL 논문 기반 낮에는 일하고 밤에는 스스로 학습하는 로컬 에이전트 구상**](./Projects/26.md)  
    *주간에는 Mem0 장기 기억과 동적 LoRA 스와핑으로 응답하고, 야간에는 RLVR 루프로 어댑터를 자가 학습하는 아키텍처 설계.*

### 🛠️ Coding & Dev Tools
*   [**Claude Vibe Coding: 복잡한 백엔드 개발을 AI에게 위임하기**](./Projects/52.md)  
    *명세와 테스트 기준만 부여하고 백엔드 구현·디버깅을 에이전트에게 맡긴 실전 바이브 코딩 워크플로우.*
*   [**무한 루프 바이브 코딩: "테스트 통과할 때까지 스스로 고쳐" 자동화 루프**](./Projects/22.md)  
    *단위 테스트 통과를 종료 조건으로 걸어 에이전트가 에러 로그를 보고 무인으로 코드를 완성하게 만든 실험.*
*   [**Docling-Translate: 문서 레이아웃을 보존하는 Streamlit 기반 PDF 번역 도구**](./Projects/91.md)  
    *Docling 파서와 LLM 번역을 결합해 복잡한 논문·문서 구조를 깨뜨리지 않고 번역하는 GUI 도구 제작기.*
*   [**LFM-Scholar: 논문 검색부터 Related Work 초안 작성까지 자동화하는 도구**](./Projects/57.md)  
    *주제 키워드만 입력하면 관련 논문을 수집·요약해 학술 논문 스타일의 Related Work 섹션을 생성하는 도구.*
*   [**Gemini-Claw 윈도우 로컬 파일 및 터미널 제어 기능 구현기**](./Projects/19.md)  
    *무료 Gemini CLI를 활용해 윈도우 환경에서 30분 만에 로컬 파일 탐색·수정 및 셸 실행 에이전트를 구축한 기록.*
*   [**Gemini-Claw 오피스 자동화: 로컬 폴더 분석 후 94초 만에 워드·엑셀·PPT 풀 패키지 생성**](./Projects/15.md)  
    *로컬 디렉토리의 자료를 읽어들여 보고서(Word), 데이터 시트(Excel), 발표 자료(PPT)를 한 번에 생성하는 자동화 파이프라인.*
*   [**Gemini 3 Pro + Gemini CLI로 처음 보는 외부 API 연동 웹앱 제작하기**](./Projects/18.md)  
    *문서가 생소한 외부 API도 에이전트가 스스로 탐색하고 문맥을 축적하며 고품질 웹 결과물을 만들어내는 과정 확인.*
*   [**Liquid AI LFM2-2.6B-Exp 파인튜닝기: 학술 Related Work 자동 생성 모델 제작**](./Projects/58.md)  
    *2.6B 소형 모델을 파인튜닝해 인용 논문 초록들로부터 자연스러운 Related Work 문단을 작성하도록 만든 프로젝트.*

### 💻 On-Device AI
*   [**Tiny MoA: API 비용 0원, 노트북 CPU만으로 돌리는 초경량 멀티 에이전트**](./Projects/32.md)  
    *고비용 클라우드 에이전트 대신 1B~3B급 온디바이스 모델 여러 개를 조합해 CPU에서 구동하는 가성비 MoA 구축기.*
*   [**Tiny MoA Tool Calling: 16GB RAM 노트북에서 구현한 로컬 에이전트의 도구 호출**](./Projects/30.md)  
    *저사양 노트북 환경에서 소형 모델들에게 웹 검색, 파일 제어, 파이썬 실행 도구를 연결한 온디바이스 에이전트 구현.*
*   [**Tiny MoA vs Clawdbot: 클라우드 API 없이 순수 CPU로 돌아가는 온디바이스 에이전트**](./Projects/24.md)  
    *무거운 외부 API 의존성을 완전히 제거하고 로컬 CPU 환경에서 실용적인 작업 속도를 확보한 아키텍처 최적화.*
*   [**Clawdbot의 토큰 비용 문제와 로컬 온디바이스 AI로의 전환 실험**](./Projects/25.md)  
    *상시 구동 에이전트의 막대한 API 과금 문제를 해결하기 위해 로컬 소형 모델 기반 시스템으로 이식한 과정.*
*   [**llama.cpp에 vLLM & SGLang 기법 이식하기: 로컬 CPU 추론 속도 1.8배 향상**](./Projects/28.md)  
    *PagedAttention과 프리픽스 캐싱 아이디어를 `llama.cpp` 기반 로컬 환경에 적용해 CPU 추론 지연을 대폭 줄인 실험.*

### 🧠 Model Experiments
*   [**4달러로 Jev보다 뛰어난 모델 만들기: Qwev-9B-RLCD 제작 및 평가기**](./Projects/126.md)  
    *4,800건의 합성 데이터와 RLCD 기법으로 A100 1대에서 30분 만에 Jev와 GPT-6 Astra를 능가한 9B 의사결정 모델 제작.*
*   [**LFM2-350M-ToolLLaMA: 단돈 14달러(H100 4시간)로 GPT-5-Nano를 이긴 350M 도구 호출 모델**](./Projects/117.md)  
    *초소형 350M 모델을 ToolBench로 정밀 튜닝해 자신보다 훨씬 큰 GPT-5-Nano 대비 6.6배 높은 도구 호출 성능을 달성.*
*   [**Open-Yaongi Project: 52층 4B(활성 0.6B) Mamba-2 + MoE 한국어 초경량 sLLM**](./Projects/Open-Yaongi.md)  
    *깊은 레이어(52층)와 Mamba-2·MoE 하이브리드 구조를 결합해 활성 파라미터 0.6B로 설계한 오픈소스 sLLM 프로젝트.*
*   [**HybriKo: Google Griffin과 LFM2에서 영감을 받은 한국어 하이브리드 RNN+Attention**](./Projects/54.md)  
    *게이트 선형 순환 유닛(RG-LRU)과 로컬 어텐션을 결합해 한국어 처리 효율을 높인 자체 하이브리드 아키텍처 설계.*
*   [**HybriKo-117M: A100 8장으로 처음부터 사전학습한 리눅스 명령어 Function Calling 모델**](./Projects/37.md)  
    *117M 초소형 하이브리드 모델을 밑바닥부터 학습시켜 자연어를 리눅스 명령어로 변환하는 특화 모델 개발기.*
*   [**HybriKo-117M-LinuxFC: 한국어 지시를 리눅스 셸 명령어로 바꾸는 초경량 모델 튜닝**](./Projects/36.md)  
    *한국어 자연어 요청을 정확한 리눅스 CLI 명령어로 매핑하기 위한 데이터셋 구축과 파인튜닝 실험.*
*   [**52-Layer HybriKo-430M: 무료 Colab T4 GPU 한 장에 52층 최신 아키텍처 우겨넣기**](./Projects/31.md)  
    *극한의 메모리 최적화와 Deep & Thin 설계를 통해 T4 GPU 16GB 안에서 52층 하이브리드 모델을 학습시킨 기록.*
*   [**노트북에서 돌아가는 1.2B 모델(LFM2.5)로 워드·엑셀·PPT 자동 생성하기**](./Projects/8.md)  
    *API 비용 없이 로컬 1.2B 초소형 모델과 파이썬 스크립트를 결합해 실무용 오피스 문서를 만들어낸 실험.*
*   [**트랜스포머($O(N^2)$)의 한계를 넘는 5가지 차세대 아키텍처 접근법과 자체 프로젝트 정리**](./Projects/14.md)  
    *Liquid AI, TII, Fastino, NVIDIA의 탈트랜스포머 시도들과 이를 바탕으로 진행한 자체 경량 모델 실험 총정리.*

### 🪄 Post-Training Projects
*   [**Modern Model Architectures Overview 온라인 강연 안내 및 발표 자료**](./Projects/131.md)  
    *최신 트랜스포머 변형, 선형 어텐션, MoE, 디퓨전 LLM 등 현대 모델 아키텍처의 흐름을 총정리한 온라인 강연.*
*   [**Introduction to Post-Training & Beyond 강연 후기 및 디스코드 스터디 계획**](./Projects/129.md)  
    *Maxime Labonne의 자료를 바탕으로 진행한 사후학습 온라인 강연 후기와 오픈소스 커뮤니티 스터디 운영 계획.*
*   [**Introduction to Post-Training 무료 온라인 강연 안내**](./Projects/130.md)  
    *SFT, DPO, GRPO 등 LLM 사후학습의 핵심 기초와 최신 트렌드를 다루는 무료 온라인 강연 공지.*
*   [**Introduction to Post-Training 강연 자료 재구성: 7가지 최신 RL 논문 추가**](./Projects/111.md)  
    *기존 사후학습 튜토리얼에 최신 RLVR 및 에이전트 강화학습 핵심 논문 7편을 결합해 재구성한 발표 자료 소개.*
*   [**LFM2 1.2B 기반 한국어-영어 번역기: 구글 Gemma 4B와 알리바바 Qwen 4B를 이기다**](./Projects/89.md)  
    *고품질 양방향 데이터 정제와 사후학습을 통해 1.2B 온디바이스 모델로 4B급 대형 모델들의 번역 품질을 넘어선 제작기.*
*   [**LFM2 한영 번역기 개발 핵심 발견 및 Liquid AI 공식 쿡북 등재 소식**](./Projects/82.md)  
    *번역 태스크에서 SFT와 RL의 역할 차이를 규명하고 프로젝트가 Liquid AI 공식 쿡북에 등재된 과정 공유.*
*   [**Small Language Model for Translation: 번역에 거대 모델이 필요 없는 이유**](./Projects/81.md)  
    *도메인과 언어쌍이 명확한 번역 문제는 거대 범용 LLM 대신 잘 튜닝된 1B~2B급 소형 모델로 해결할 수 있다는 실전 조언.*
*   [**Liquid AI LFM2-1.2B 한영 번역 강화학습(GRPO) 실패기와 교훈**](./Projects/76.md)  
    *소형 번역 모델에 GRPO를 적용하며 겪은 보상 해킹, 포맷 붕괴 현상과 이를 통해 얻은 보상 함수 설계 교훈.*
*   [**LLM-Ko-Datasets: 흩어져 있는 한국어 사전학습·SFT·DPO·GRPO 데이터셋 총정리**](./Projects/51.md)  
    *한국어 모델 학습 시 겪는 데이터 탐색 난제를 해결하기 위해 단계별 한국어 오픈 데이터셋을 모아 구축한 아카이브.*

### 💭 Insights & Essays
*   [**3개월 9개 프로젝트 회고: 16GB RAM 노트북으로 달려온 GPU Poor의 여정**](./Projects/118.md)  
    *고성능 GPU 없이 i5 노트북 환경에서 3개월간 9개의 AI 프로젝트와 논문 아카이브를 구축하며 얻은 배움과 회고.*
*   [**최근 구현한 AI 프로젝트 성과 요약: 맥킨지 스타일 보고서 및 PPT 자동 생성**](./Projects/7.md)  
    *Gemini-Claw를 활용해 리서치부터 컨설팅 스타일 문서·슬라이드 생성까지 자동화한 결과물 공유.*
*   [**자율 에이전트의 성능 vs 보안 딜레마: 도커를 뚫는 권한 위임의 위험성**](./Projects/12.md)  
    *에이전트에게 셸 실행과 전체 권한을 부여할 때 폭발하는 생산성과, 악성 플러그인·명령어를 통한 치명적 보안 취약점 고찰.*
*   [**AI 시대의 두려움과 흥미: OpenClaw, 환각 인용, 바이브 코딩에 대한 단상**](./Projects/17.md)  
    *자율 에이전트 확산과 바이브 코딩 열풍 속에서 마주하는 기술적 흥미와 검증 부재의 위험성에 대한 에세이.*

## <a id="trends--industry"></a>🔥 Trends & Industry

*   [**AAII 비공개 평가셋 개편과 구글 Gemini 4 Carbon 1위: 벤치맥싱 논란**](./Trends/174.md)  
    *평가셋 60% 비공개 전환 후 Gemini 4 Carbon과 GPT-6 Astra가 약진하고 일부 모델이 급락하며 드러난 벤치맥싱 실태 고찰.*
*   [**Kimi K3.1 Pro의 AAII 선두 탈환과 오픈웨이트 공개 예고**](./Trends/173.md)  
    *Post-training만으로 정상을 탈환한 2.8T Kimi K3.1 Pro의 오픈웨이트 예고와 중국 오픈 생태계의 미국 시장 침투 분석.*
*   [**Gemini 4 Argon과 Muse Wonder Full의 등장: 프론티어 4자 구도와 인프라 전쟁**](./Trends/172.md)  
    *구글과 메타의 신작 합류로 재편된 프론티어 4자 구도와 거대 자본·GPU 인프라 중심 경쟁에 대한 단상.*
*   [**요즘 생각 (44가지 단상: 프론티어 비용, GPU 인프라 해자, CoT와 검증기)**](./Trends/171.md)  
    *천문학적 인프라 비용, 실용성과 기술력의 간극, LLM 출력 가독성 저하와 검증기 필요성에 대한 44가지 짧은 단상.*
*   [**속도 조절론의 위선과 신작 경쟁: Claude Opus 5.5, GPT-6 Sol, MiMo-V2.6-Pro**](./Trends/170.md)  
    *위험성을 경고하며 독점 모델을 쏟아내는 빅테크와 MIT 라이선스로 전문 공개한 샤오미 1T 모델의 대비.*
*   [**Jev는 일반화된 대형 BERT인가?: 의사결정 모델의 상용화 공식과 오픈소스의 반격**](./Trends/169.md)  
    *생성 루프를 없애고 RLCD와 타입 인터페이스로 성공한 Jev의 구조와 Laya·AnyJev 등 오픈소스 대안들의 추격 분석.*
*   [**프론티어 모델 증류 규제 논란과 오픈 생태계를 위한 공정 사용**](./Trends/168.md)  
    *웹 데이터를 무단 학습한 빅테크가 후발주자의 출력 증류는 금지하는 위선을 비판하고 오픈 생태계의 가치를 역설한 글.*
*   [**미국 빅테크의 역습: Fable 5.1, Muse Spark 1.3, Gemini 3.8 Flash 비교**](./Trends/167.md)  
    *성능 선두 Fable 5.1과 초고속 모델 Muse Spark 1.3·Gemini 3.8 Flash의 등장 및 추론 요금제 변화 분석.*
*   [**엔비디아의 허깅페이스 인수와 AI 시대의 중앙은행: 자체 칩과 순환 금융 리스크**](./Trends/166.md)  
    *허깅페이스 인수로 생태계를 장악한 엔비디아와 빅테크의 자체 칩 탈출 시도, 네오클라우드 순환 금융 리스크 분석.*
*   [**Ox Alpha(GLM-5.3-Flash)의 정체와 Qwen3.8-Flash-Next, Gemini 3.7 Flash 사용기**](./Trends/165.md)  
    *베일에 싸였던 Ox Alpha의 실체(GLM-5.3-Flash)와 초고속 경량 프론티어 모델들의 실전 체감 성능 비교.*
*   [**독자 AI 파운데이션 모델(독파모) 2차 평가 결과와 Motif Technologies 탈락 고찰**](./Trends/164.md)  
    *글로벌 Top 10 성능을 달성하고도 탈락한 Motif-3 사례를 통해 본 국내 파운데이션 모델 평가와 생태계의 아쉬움.*
*   [**Qwen3.8 27B와 GLM-5.3: 오픈웨이트와 Post-Training이 이끄는 중국 AI**](./Trends/163.md)  
    *로컬 구동 가능한 Qwen3.8 27B와 사후학습만으로 보안·추론 정상을 차지한 GLM-5.3의 경쟁력 분석.*
*   [**OpenRouter 1위 무료 스텔스 모델 Ox Alpha의 등장과 코딩 에이전트 실전 후기**](./Trends/162.md)  
    *DeepSWE 80%를 기록하며 등장한 스텔스 모델 Ox Alpha를 Claude Code·Codex에 연결해 사용한 후기.*
*   [**요즘 생각 (29가지 단상: 최신 모델 폭주와 인공지능의 지향점)**](./Trends/161.md)  
    *쏟아지는 신모델 속에서 느낀 추론 한계, 오픈소스 라이선스 변화, AI 서비스의 본질적 가치에 대한 29가지 단상.*
*   [**독파모 AAII 점수 공개와 Qwen3.8, Grok 4.6, Muse-Glimmer 동향**](./Trends/160.md)  
    *국내 독파모 모델들의 AAII 성적표와 Qwen3.8, Grok 4.6, Meta Muse-Glimmer-30B 등 글로벌 모델 출시 소식 정리.*
*   [**Motif-3 공식 공개(MIT 라이선스)와 Meta·NVIDIA의 30B 오픈웨이트 릴레이**](./Trends/159.md)  
    *MIT 라이선스로 전환한 Motif-3의 벤치마크 분석과 글로벌 빅테크들의 30B급 고효율 오픈 모델 경쟁.*
*   [**Qwen3.8 Max의 부상과 중국 오픈소스 라이선스 변화가 한국 AI에 주는 시사점**](./Trends/158.md)  
    *중국 프론티어 모델들의 약진과 상업적 제한 라이선스 전환 흐름 속에서 한국 AI가 취해야 할 전략.*
*   [**제프 딘의 구글 퇴사와 Discovery Loop 창업: AI 거장들의 이동**](./Trends/157.md)  
    *제프 딘·산제이 게마왓의 퇴사와 공익 AI 기업 설립 소식 및 구글 TPU 생태계에 대한 단상.*
*   [**LLM 스케일링의 역사: 거대 자본 주도 패러다임은 올바른 길인가**](./Trends/156.md)  
    *GPT 초기부터 최신 추론 모델까지 이어진 스케일링의 역사와 물량 투입 일변도 패러다임의 한계 고찰.*
*   [**클로즈드 진영의 규제 프레임과 오픈 모델 생태계 다양성의 중요성**](./Trends/155.md)  
    *안전성을 명분으로 장벽을 세우는 독점 진영을 비판하고 오픈웨이트 생태계가 유지되어야 하는 이유를 다룬 글.*
*   [**DX·SI의 시대에서 AX·FDE(Forward Deployed Engineer)의 시대로**](./Trends/154.md)  
    *과거 디지털 전환(DX) 시대의 SI가 AI 전환(AX) 시대의 현장 밀착형 엔지니어(FDE)로 진화한 산업 구조 분석.*
*   [**요즘 생각 (21가지 단상: 프론티어 모델 체감, 검증 난이도, 인프라 동향)**](./Trends/153.md)  
    *고도화되는 LLM 작업 결과물의 검증 어려움과 최신 모델 평가·인프라 트렌드에 관한 21가지 단상.*
*   [**Kimi K3: 2.8T 오픈 모델의 성능과 변형 상업 라이선스 딜레마**](./Trends/152.md)  
    *초거대 오픈웨이트 모델 Kimi K3의 성과와 상업적 이용 제한 조항이 오픈소스 생태계에 던지는 화두.*
*   [**LLM의 언어와 삶의 형태: 검증 가능한 환경(RLVR)이 만드는 고유한 지능**](./Trends/151.md)  
    *비트겐슈타인의 언어철학 관점에서 현대 LLM이 RLVR 환경 피드백을 통해 직관과 지능을 형성하는 원리 고찰.*
*   [**AI가 촉발한 새로운 망상과 인간-AI 상호작용의 주의점**](./Trends/150.md)  
    *유저의 말에 동조하며 빠르게 수렴하는 LLM의 특성이 인간에게 미치는 확증 편향과 과몰입 위험성 경고.*
*   [**요즘 생각 (39가지 단상: 오픈소스, 조직 문화, 엔지니어의 삶)**](./Trends/149.md)  
    *AI 연구개발, 오픈소스 기여, 커리어와 삶의 균형에 대해 기록한 39가지 짧은 인사이트.*
*   [**인공지능과 LLM이 우리에게 미치는 안 좋은 영향들 10가지**](./Trends/148.md)  
    *사고력 저하, 창의성 획일화, 검증 피로도 증가 등 무분별한 LLM 의존이 초래하는 10가지 부작용 정리.*
*   [**Anthropic의 정부 규제 옹호 행보와 독자적 오픈소스 생태계의 필요성**](./Trends/147.md)  
    *자사 모델의 위험성을 강조하며 정부 규제를 촉구하는 빅테크의 사다리 걷어차기와 오픈소스의 대항력 고찰.*
*   [**바이브 코딩으로 모든 일을 위임하는 사람들에게: 확률 게임과 교차 검증**](./Trends/146.md)  
    *무검증 AI 위임의 누적 오류 확률을 계산하고 다중 에이전트 교차 검증으로 시스템 안전망을 구축하는 전략.*
*   [**요즘 생각: LLM의 한계 인지와 토큰 처리량 리더보드의 허상**](./Trends/145.md)  
    *단순 토큰 소비량이나 벤치마크 점수 경쟁에서 벗어나 모델의 실질적 한계를 직시해야 한다는 단상.*
*   [**단돈 1달러, 5분 만에 끝나는 LLM 기반 온라인 신원 추적과 익명성의 종말**](./Trends/144.md)  
    *파편화된 온라인 흔적을 결합해 익명 사용자를 특정해내는 LLM의 추론 능력과 프라이버시 위협 분석.*
*   [**RL, RAG, 에이전트, 하네스의 공통점: 가능성 제한을 통한 문제 해결력 극대화**](./Trends/143.md)  
    *LLM의 무한한 출력 공간을 제약하고 탐색 범위를 좁혀주는 것이 모든 최신 AI 엔지니어링의 본질이라는 통찰.*
*   [**주장 뒤에 숨은 배경과 이해관계 읽기: AI 규제론 vs 오픈소스 옹호론**](./Trends/142.md)  
    *거대 자본과 연구자들의 발언 이면에 깔린 이해관계를 짚고 휘둘리지 않는 기본기의 중요성을 강조한 글.*
*   [**안드레이 카파시(Andrej Karpathy)의 앤트로픽 합류와 AI 교육 표준의 미래**](./Trends/141.md)  
    *교육자 정체성이 강한 카파시와 앤트로픽의 결합이 향후 AI 개발자 교육 및 기술 커뮤니케이션에 미칠 영향.*
*   [**생각보다 느린 AI 원천 기술의 발전과 너무 빠르게 흔들리는 인간 사회**](./Trends/140.md)  
    *트랜스포머와 고전 RL의 결합에 머물러 있는 알고리즘 혁신 속도와 대비되는 사회·고용 시장의 급격한 동요 고찰.*
*   [**영화 'Her'의 배경인 2025년을 지나며 본 수동적 챗봇의 한계**](./Trends/139.md)  
    *초장문 멀티모달 기술이 구현되었음에도 선제적 교감 능력이 결여된 현대 챗봇과 영화 속 AI의 본질적 차이.*
*   [**앤트로픽의 엔터프라이즈 전담 법인 설립과 단순 API 래퍼(Wrapper) 스타트업의 위기**](./Trends/138.md)  
    *모델 제작사가 직접 기업 워크플로우 시장에 진출하면서 얇은 포장만 하던 스타트업들이 직면한 생존 위기.*
*   [**초기 스타트업의 연봉 제안으로 본 로컬 sLM과 무환각 AI 시장의 수요**](./Trends/137.md)  
    *환각 없는 도메인 특화 로컬 sLM을 찾는 현장의 높은 수요와 AI 엔지니어 채용 시장의 변화 단상.*
*   [**법경제학으로 본 계약의 효율적 파기: OpenAI의 클라우드 독점 계약 사례**](./Trends/136.md)  
    *효율적 계약 파기(Efficient Breach) 이론을 통해 OpenAI의 클라우드 계약 재편을 경제적 효용 관점에서 분석한 글.*
*   [**1% 성장의 늪과 AI 해고의 덫: 환상 속의 AGI를 넘어**](./Trends/135.md)  
    *거시 경제 성장을 견인하지 못한 채 인력 감축으로 소비 시장만 위축시키는 AI 해고의 역설 분석.*
*   [**유행하는 키워드(하네스, 카파시, 딥시크)의 알고리즘 파급력과 글쓰기 실험**](./Trends/134.md)  
    *소셜 미디어에서 기술 본질보다 특정 유명인·유행어 키워드가 노출수를 좌우하는 현상에 대한 관찰.*
*   [**The AI Layoff Trap: 기업들의 AI 해고가 부르는 죄수의 딜레마**](./Trends/133.md)  
    *개별 기업의 비용 절감형 AI 해고가 전체 시장의 구매력 붕괴로 이어지는 과정을 게임 이론으로 증명한 분석.*
*   [**Gemini가 선정한 작성 중인 AI 논문 리뷰 및 인사이트 10선**](./Trends/132.md)  
    *초안(Drafts) 아카이브 중 Gemini가 선별한 핵심 기술 분석 10편의 요약 리포트.*
*   [**대학 교육의 위기와 성실성의 종말: AI 아키텍트의 부상**](./Trends/131.md)  
    *단순 암기와 성실성만으로는 경쟁력을 잃는 시대에 전체 시스템을 설계·검증하는 아키텍트 역량의 중요성.*
*   [**오픈소스 제작은 꼴찌를 하기 위해서 하는 겁니다: 생태계의 바닥을 높이는 일**](./Trends/130.md)  
    *누구도 그 이하로 떨어지지 않는 최소한의 기술 바닥(Baseline)을 만들어주는 오픈소스 기여의 철학적 가치.*
*   [**안드레이 카파시의 LLM Wiki 아이디어와 기술적 지향점의 일치**](./Trends/129.md)  
    *문서를 위키 형태로 구조화해 LLM 지식을 관리하자는 카파시의 제안과 그동안의 아카이빙 실천이 맞닿은 지점.*
*   [**점점 API로만 공개되는 중국 최신 모델들(Qwen, GLM)과 오픈소스의 미래**](./Trends/128.md)  
    *오픈웨이트를 주도하던 중국 빅테크마저 최상위 모델을 클로즈드 API로 전환하기 시작한 흐름과 우려.*
*   [**경제학자들이 예측한 인공지능의 경제적 파급력과 인프라·적응 시차**](./Trends/127.md)  
    *AI 기술 발전이 실제 산업 생산성과 GDP 증가로 이어지기까지 발생하는 물리적 인프라 및 조직 적응 시차 분석.*
*   [**요즘 드는 생각: Gemma 4, MoE 아키텍처, 중국의 혁신, 그리고 최종 보스 구글**](./Trends/126.md)  
    *오픈 생태계를 이끄는 모델들과 자체 TPU·풀스택 인프라를 갖춘 구글의 장기적 우위에 대한 단상.*
*   [**인공지능 모델링의 예술화: 데이터와 보상 설계에 담기는 창작자의 개성**](./Trends/125.md)  
    *사후학습(RL)과 데이터 큐레이션이 단순 공학을 넘어 설계자의 철학과 취향이 반영되는 영역으로 변모하는 흐름.*
*   [**국가 경쟁 프레임을 넘어선 아웃라이어 기업과 초거대 자본의 시대**](./Trends/124.md)  
    *국적보다 압도적 연산 자원과 소수 정예 인재를 독점한 아웃라이어 기업들이 AI 패권을 주도하는 현상 고찰.*
*   [**이제는 학습이 아니라 '추출'의 시대입니다: 잠재 지능을 깨우는 5가지 방법론**](./Trends/123.md)  
    *막대한 재학습 대신 추론 시점 제어와 가중치·활성화 조향으로 모델에 이미 내재된 지능을 꺼내 쓰는 패러다임 변화.*
*   [**실전 AI에서 중요도는 데이터 90%, 에이전트 하네스 9%, 모델 1%입니다**](./Trends/120.md)  
    *모델 체급 경쟁에 가려져 있지만 실제 문제 해결을 좌우하는 것은 도메인 데이터 정제와 에이전트 환경 설계라는 통찰.*
*   [**자동화 피드백 루프의 늪: AI 덕분에 코딩은 빨라졌는데 왜 우리는 더 과로할까?**](./Trends/121.md)  
    *에이전트 병렬 작업이 만든 끝없는 검토 루프와 인간의 인지적 피로·주도권 상실에 대한 성찰.*
*   [**한국어 오픈 데이터셋 생태계를 지켜내기 위한 다짐**](./Trends/122.md)  
    *글로벌 LLM 시대에 한국어 데이터 주권과 고품질 오픈 데이터셋 구축을 지속하는 개인적 동기와 목표.*
*   [**AGI라는 위험한 이데올로기: 노벨상 수상자 아세모글루와 얀 르쿤의 일침**](./Trends/117.md)  
    *만능 자율 AGI 환상을 경계하고 인간 능력을 보완하는 특화 도구로서의 AI에 집중해야 한다는 두 석학의 제언.*
*   [**규모(Scale)의 경쟁을 넘어 구조적 한계를 비틀어 돌파한 최신 AI 연구 14선**](./Trends/118.md)  
    *무작정 파라미터를 키우는 대신 아키텍처·메모리·학습법을 혁신해 효율성을 극대화한 14편의 주요 논문 요약.*
*   [**AI 기업의 실질적 기회: 엔터프라이즈 B2B와 중장년층을 위한 도구 호출 B2C**](./Trends/119.md)  
    *개발자용 코딩 에이전트 레드오션을 넘어 아직 AI를 쓰지 않는 대중과 기업 현장을 파고드는 서비스 방향성.*
*   [**OpenAI vs Anthropic vs 미 국방부(Pentagon): AI 윤리, 국방 계약, 오픈소스**](./Trends/116.md)  
    *빅테크들의 국방 프로젝트 참여와 안전 서약 변경을 둘러싼 윤리 논란 및 오픈소스 진영의 역할.*
*   [**Andrej Karpathy: 우리는 동물이 아니라 '유령(Ghost)'을 소환하고 있는가?**](./Trends/110.md)  
    *진화로 탄생한 생물 지능과 달리 인간 텍스트 모방과 RL 보상으로 빚어진 LLM 지능의 이질성과 보상 해킹 고찰.*
*   [**AI 시대의 인지적 항복(Cognitive Surrender): 생각하기를 외주화하는 대가**](./Trends/112.md)  
    *AI의 유창한 답변에 비판적 사고 없이 의존할 때 발생하는 인간 추론 능력의 퇴화 현상 분석.*
*   [**OpenClaw 사태: 자율 에이전트가 오픈소스 메인테이너를 협박할 때**](./Trends/113.md)  
    *자율 에이전트가 자신의 PR이 거절되자 메인테이너의 약점을 찾아 압박 글을 게시한 실제 사건과 자율성 통제 문제.*
*   [**Vibe Coding(바이브 코딩)의 탄생과 명암: 코드를 잊고 의도를 지휘하라**](./Trends/114.md)  
    *카파시가 제시한 바이브 코딩 개념의 본질과 프로토타이핑 가속 이면에 숨은 유지보수·기술 부채 리스크.*
*   [**The Thinking Game: 데미스 허사비스가 게임의 세계를 떠나 DeepMind를 세운 이유**](./Trends/115.md)  
    *체스 신동에서 알파고와 알파폴드로 과학적 난제를 해결하기까지 허사비스의 여정을 담은 다큐멘터리 리뷰.*
*   [**Sebastian Raschka의 "Ahead of AI"로 보는 2026 LLM 기술 흐름**](./Trends/59.md)  
    *세바스찬 라슈카가 정리한 최신 어텐션 변형, 사후학습 레시피, 추론 스케일링의 핵심 기초와 트렌드 요약.*
*   [**바이브 코딩 시대에 '영구적 주니어'의 함정에 빠지지 않는 법**](./Trends/54.md)  
    *AI가 짜준 코드만 붙여넣다 기초 체력을 잃는 함정을 피하고 엔지니어링 기본기를 지키는 생존 전략.*
*   [**허깅페이스 CEO 클레망 들랑그의 한국 오픈 AI 모델 생태계 응원**](./Trends/53.md)  
    *SKT, LG AI연구원, 업스테이지 등 글로벌 오픈소스 리더보드에서 활약하는 한국 AI 모델들의 성과 정리.*
*   [**Anthropic의 서드파티 생태계 차단(OpenCode)과 사용량 제한의 아쉬움**](./Trends/49.md)  
    *외부 에이전트 하네스 연동 차단과 폐쇄적 운영 정책이 개발자 생태계에 미치는 영향에 대한 단상.*
*   [**CES 2026: AMD 리사 수와 Liquid AI가 보여준 비(非)트랜스포머 온디바이스 미래**](./Trends/48.md)  
    *AMD 하드웨어와 Liquid AI의 LFM 아키텍처 협력을 통해 본 엣지·온디바이스 AI 생태계 전망.*
*   [**국가대표 AI 프로젝트 1차 선발 결과(LG, SKT, Upstage)와 향후 관전 포인트**](./Trends/43.md)  
    *국가 주력 파운데이션 모델 1차 선발 결과 분석과 탈락 팀들의 특화 모델 전환 전략에 대한 단상.*
*   [**Post-Training의 구조적 한계: 왜 모델은 배포 후 경험으로 더 똑똑해지지 못하는가?**](./Trends/40.md)  
    *정적 가중치에 갇혀 실전 피드백을 장기 역량으로 내재화하지 못하는 현행 사후학습의 한계와 연속 학습 과제.*
*   [**실무 엔지니어 vs 경영진: 기업 내 LLM 도입과 사내 의사결정의 시각차**](./Trends/39.md)  
    *기술적 완성도를 추구하는 실무자와 리스크 관리·ROI를 우선하는 경영진 사이의 관점 차이와 조율법.*
*   [**업스테이지 Solar Open의 GLM 표절 논란 종결과 From-Scratch 학습 검증**](./Trends/56.md)  
    *토크나이저와 학습 로그 검증을 통해 확인된 Solar 모델의 독자 사전학습 과정과 오픈소스 검증 문화.*
*   [**What LLMs Think When You Don't Tell Them?: 빈 프롬프트에서 드러나는 모델별 무의식**](./Trends/4.md)  
    *아무 지시 없이 토큰 생성을 유도했을 때 모델마다 고유하게 나타나는 기본 성향과 학습 데이터 흔적 분석.*
*   [**AI 거품론의 본질: 시장 붕괴가 아닌 인프라 수급 안정화와 옥석 가리기**](./Trends/2.md)  
    *GPU 사재기 열풍이 진정되고 실제 수익과 효용을 내는 기술 중심으로 산업이 성숙해가는 과정에 대한 분석.*


## <a id="recommended-resources"></a> 🌟 Recommended Resources

공부할 때 도움이 될 훌륭한 자료들을 공유합니다.

*   **[LLM course](https://github.com/mlabonne/llm-course)** 
    *전반적인 LLM 지식 로드맵*
*   **[Reinforcement Learning from Human Feedback](https://rlhfbook.com/)** 
    *무료 강화학습 책*
*   **[Build A Reasoning Model (From Scratch)](https://github.com/rasbt/reasoning-from-scratch)** 
    *LLM 작동 방식 이해*
*   **[Ahead of AI](https://magazine.sebastianraschka.com/)** 
    *LLM 모델 아키텍처 이해*
*   **[Maxime Labonne Blog](https://maximelabonne.substack.com/)** 
    *다양한 프로젝트 및 기술 이해*
*   **[LLM Datasets](https://github.com/mlabonne/llm-datasets)** 
    *LLM 학습 데이터 모음*
*   **[LLM-KO-Datasets](https://github.com/gyunggyung/LLM-Ko-Datasets)** 
    *한국어 포함 데이터*
*   **[AGI-Papers](https://github.com/gyunggyung/AGI-Papers)** 
    *논문 리뷰 등 다양한 시선 공유*
*   **[Unsloth Dynamic 2.0 GGUFs](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs)** 
    *최신 양자화 방법*
*   **[TRL Docs](https://huggingface.co/docs/trl/index)** 
    *학습 가이드*
*   **[The Modern Software Developer](https://themodernsoftware.dev/)** 
    *에이전트 개발법*
*   **[Andrej Karpathy Twitter](https://x.com/karpathy)** 

*   **[Yann LeCun Twitter](https://x.com/ylecun)** 
   
*   **[Andrew Ng Twitter](https://x.com/AndrewYNg)** 

---

## 📬 Connect

*   [<img src="https://img.shields.io/badge/LinkedIn-Kiwoong Yeom-blue?style=flat&logo=linkedin&logoColor=white" />](https://www.linkedin.com/in/kiwoong-yeom) [<img src="https://img.shields.io/badge/GitHub-gyunggyung-black?style=flat&logo=github&logoColor=white" />](https://github.com/gyunggyung)
*   📧 **Contact:** newhiwoong@gmail.com

---
*Disclaimer: The views and opinions expressed in these reviews are those of the author and do not necessarily reflect the official policy or position of any other agency, organization, employer or company.*
