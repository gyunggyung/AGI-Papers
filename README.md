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

이 저장소는 AGI를 향한 여정을 다음 8가지 핵심 주제로 분류하여 정리합니다.

- [🤖 Agents](#agents) : 자율 에이전트, 행동/계획(Planning) 모델, 프레임워크
- [🧠 Architecture](#architecture) : LLM 아키텍처 혁신 (Transformer, Mamba, MoE)
- [📚 Pre-Training](#pre-training) : 학습 데이터, 스케일링 법칙, 파운데이션 모델
- [🎯 Post-Training](#post-training) : RLHF, DPO, GRPO, 정렬(Alignment)
- [⚖️ Evaluation](#evaluation) : 벤치마크, 평가 방법론, 비평
- [🗂️ RAG & Knowledge](#rag--knowledge) : 검색 증강 생성, 지식 그래프, 메모리
- [💻 On-Device AI](#on-device-ai) : 로컬 구동, 엣지 컴퓨팅, 최적화
- [🚀 Projects](#projects) : 직접 구현한 프로젝트 및 실험 결과
- [🔥 Trends & Industry](#trends--industry) : AI 산업의 동향, 인사이트, 주요 뉴스
- [🌟 Recommended Resources](#recommended-resources) : 공부를 위한 추천 자료

---

## <a id="agents"></a>🤖 Agents

*   [**The Last AI Built by Humans: 진정한 재귀적 자가 개선(RSI)을 향한 5단계 자율성 로드맵과 환경 엔지니어링**](./Agents/133.md)  
    *491편의 논문과 72개 산업 사례 분석을 통해 AI 자율성 5단계와 통제되지 않는 자가 수정의 위험성을 짚고, 작업 환경과 에이전트 하네스 정제의 중요성을 밝힌 RSI 서베이.*
*   [**PARSER: 병렬 독해와 중앙 지휘관 추론으로 초장문 컨텍스트를 뚫어낸 멀티 에이전트 아키텍처**](./Agents/132.md)  
    *수백 개의 서브에이전트 병렬 독해와 GRPO로 학습된 중앙 지휘관의 단계별 질의를 결합해 896K 초장문에서 완벽한 길이 불변성과 11배 속도 향상을 이룬 CUHK 연구.*
*   [**Frog Nano: 대형 모델 증류 없이 하네스와 온라인 RL만으로 4B 코딩 에이전트 만들기**](./Agents/131.md)  
    *사전 SFT 증류 없이 경량 하네스(LEAF)와 난이도 조절 온라인 태스크 합성(TaskPilot), 비동기 제어로 4B 모델을 거대 에이전트급으로 끌어올린 MSR 연구.*

*   [**LLMs Get Lost in Evolving User Intent: 7턴 만에 붕괴하는 멀티턴 대화와 AI 상태 관리의 실체**](./Agents/130.md)  
    *정보 추가, 조건 수정, 작업 전환이 결합된 멀티턴 대화에서 프론티어 LLM들이 7턴 만에 붕괴하는 원인을 상태 관리 부재로 분석하고, GRPO 기반 행동 정렬 해결책을 밝힌 MS 리서치 논문.*
*   [**에이전트 자가 개선(Self-improving) 인공지능 서베이**](./Agents/129.md)  
    *프롬프트 최적화에서 풀 스캐폴딩과 실시간 진화로 발전하는 자가 개선 에이전트의 흐름과 위험성을 다룬 서베이 논문 리뷰.*
*   [**The Harness Effect: 하네스 설계가 결정하는 에이전트 토큰 경제학**](./Agents/128.md)  
    *Writer 연구진이 제시한 에이전트 오케스트레이션 설계의 6가지 비용 절감 및 토큰 최적화 메커니즘.*
*   [**Neural Procedural Memory (NPM): 모델의 행동 오류를 고치는 뇌 신호 조종**](./Agents/127.md)  
    *에이전트의 뇌 신경을 미세한 전기 신호로 조종하여 행동 오류를 근본적으로 고치는 NPM 기술 소개.*
*   [**The Red Queen Gödel Machine: Co-Evolving Agents and Their Evaluators**](./Agents/126.md)  
    *에이전트와 평가자가 서로 진화하며 성능을 향상시키는 동적 루프 기반의 공동 진화 시스템 제안.*
*   [**에이전트도 늙습니다. 늙을 수록 멍청해지죠 (Agent Lifespan Engineering)**](./Agents/125.md)  
    *AI 에이전트가 장기 작업 시 메모리 한계로 인해 지능이 저하되는 노화 현상을 분석한 연구.*
*   [**최근 Meta, Stanford University, UIUC 등에서 발표한 2026년 에이전트 하네스 분야의 바이블 논문 (Code as Agent Harness)**](./Agents/124.md)  
    *LLM의 환각을 통제하고 물리적 폐쇄 루프 내에서 자율 계획-실행-검증을 가능하게 만드는 에이전트 하네스(Harness)의 3계층 아키텍처와 최전선 연구(L2MAC, EvoMAC) 요약.*
*   [**SID-1: GPT-5.1을 이긴 14B 검색 에이전트의 탄생**](./Agents/123.md)  
    *철저히 도서관 사서 역할만 하도록 훈련되어 환각을 원천 차단하고, GPT-5.1 대비 24배 빠른 속도와 압도적인 가성비를 증명한 에이전트 검색 모델.*
*   [**Agentic Harness Engineering (AHE): 스스로 파이썬 코드를 짜서 하네스를 개조하는 AI**](./Agents/122.md)  
    *프롬프트 엔지니어링의 한계를 넘어, 에이전트 스스로 쉘 도구와 미들웨어를 구축해 완벽한 폐쇄 루프(Closed Loop) 진화를 이뤄낸 방법.*
*   [**LLMs Corrupt Your Documents: 조용한 부패와 위임의 환상**](./Agents/121.md)  
    *AI에게 편집을 위임할 때 발생하는 '조용한 부패(Silent Corruption)' 현상과 툴(Tool) 사용이 오히려 문서 훼손을 가속화한다는 마이크로소프트 리서치의 연구.*
*   [**RecursiveMAS: 텍스트 없이 잠재 공간에서 대화하는 에이전트**](./Agents/120.md)  
    *기존의 다중 에이전트 시스템의 어휘 공간 디코딩 병목을 해결하고, 잠재 공간(Latent Space)에서 실시간으로 소통하여 토큰 낭비를 75% 절감한 연구.*
*   [**Agentic World Modeling: 세계 모델의 3단계 역량과 4대 지배 법칙**](./Agents/119.md)  
    *중구난방이던 세계 모델의 기준을 구조화하고, 에이전트의 환경(Environment) 병목과 UI OCR의 중요성을 강조한 논문 리뷰.*
*   [**하네스(Harness)의 민낯과 그 한계**](./Agents/118.md)  
    *Meta-Harness, NLAH, ARC-AGI-3 등 하네스의 효과와 본질적 한계를 파헤친 3편의 논문 리뷰.*
*   [**HYPERAGENTS: 스스로 '공부법'을 코딩하며 진화하는 AI**](./Agents/117.md)  
    *LLM이 스스로 파이썬 코드를 짜면서 학습 방법을 진화시키는 개방형 지능 연구.*
*   [**32B가 480B 모델을 이기는 데이터 엔지니어링**](./Agents/115.md)  
    *NVIDIA의 터미널 에이전트 훈련의 비밀과 데이터 재활용.*
*   [**샌드박스를 탈출해 코인을 채굴한 통제 불가능한 에이전트**](./Agents/116.md)  
    *알리바바 연구진의 ALE 논문과 에이전트 훈련의 미학.*
*   [**Prompt Repetition Improves Non-Reasoning LLMs**](./Agents/113.md)  
    *70전 0패 47승, 같은 프롬프트를 반복하면 성능이 올라갑니다!*
*   [**Adaptation of Agentic AI**](./Agents/80.md)  
    *거대 모델 튜닝보다 도구 튜닝이 효율적인 이유 (T2 > A2).*
*   [**Memory in the Age of AI Agents**](./Agents/77.md)  
    *에이전트 기억의 형태, 기능, 역동성에 대한 고찰.*
*   [**World Models Research**](./Agents/11.md)  
    *World Knowledge Injection vs Specific Tasks.*
*   [**Mixture-of-Models**](./Agents/9.md)  
    *Unifying Heterogeneous Agents via N-Way Self-Evaluating Deliberation.*
*   [**AIRS-Bench**](./Agents/5.md)  
    *Frontier AI Research Science Agents를 위한 태스크.*
*   [**OctoTools**](./Agents/99.md)  
    *Training-free LLM Agent Framework.*
*   [**Chain-of-Draft(CoD)**](./Agents/93.md)  
    *CoT의 장점을 유지하면서 토큰 사용량과 계산 비용을 줄이는 획기적인 접근법.*
*   [**Scaling Agent Systems: 다다익선의 함정**](./Agents/100.md)  
    *구글과 MIT가 밝혀낸 멀티 에이전트의 과학.*
*   [**LOTaD: Optimal Task Decomposition**](./Agents/101.md)  
    *에이전트는 어떻게 일을 나눠야 할까?*
*   [**ADGR: Agentic Deep Graph Reasoning**](./Agents/102.md)  
    *스스로 지도를 그리는 에이전트.*
*   [**Agentic Reasoning**](./Agents/103.md)  
    *생각의 도구를 쓰는 에이전트.*
*   [**MetaChain**](./Agents/104.md)  
    *Zero-code Framework: 말만 하면 만들어지는 에이전트.*
*   [**LoRASA: Agent Adaption**](./Agents/105.md)  
    *따로 또 같이, 에이전트의 개인기.*
*   [**AgentArcEval**](./Agents/106.md)  
    *에이전트 아키텍처, 점수 매겨드립니다.*
*   [**SciAgents**](./Agents/107.md)  
    *AI 과학자의 탄생.*
*   [**Agent Workflows (Anthropic)**](./Agents/108.md)  
    *앤트로픽이 제안하는 5가지 핵심 패턴.*
*   [**ASA: Training-Free Tool Calling**](./Agents/110.md)  
    *게으른 에이전트(Lazy Agent)를 깨우는 가장 가벼운 방법.*
*   [**HUMANLM: State Alignment for User Simulation**](./Agents/111.md)  
    *진정한 페르소나는 '마음'에서 나온다.*
*   [**SKILLRL: 에이전트는 '실패'를 먹고 자란다**](./Agents/112.md)  
    *에이전트에게 경험을 '스킬'로 증류(Distill)하여 평생 학습의 길을 열어주다.*
*   [**The Devil Behind Moltbook: 다중 에이전트 사회의 타락**](./Agents/114.md)  
    *닫힌 계(Closed System)에서 에이전트들의 소통 붕괴, 합의된 환각, 공모 공격을 증명한 연구.*
## <a id="architecture"></a>🧠 Architecture

*   [**DeepSeek-V4.1-Flash: 극한의 KV 캐시 압축과 Causal Encoder-Decoder로 완성한 효율성**](./Architecture/147.md)  
    *글로벌 KV 캐시 75% 감축, 입력 깊이를 절반만 통과시키는 CED 구조, SWA Bounded Replay 및 완전 자동화 Post-training 공장을 구축한 DeepSeek-V4.1-Flash 분석.*
*   [**Frontier Language Models Struggle to Copy: 1차원 RoPE의 한계를 극복하는 2D-RoPE**](./Architecture/146.md)  
    *프론티어 모델들이 단순 텍스트 복사를 못하는 구조적 원인을 RoPE의 1차원 거리에 기반한 한계로 지적하고, 행/열 2차원 시트로 전환하는 2D-RoPE 및 Auto-2D-RoPE 기술 분석.*
*   [**서베이 논문으로 보는 디퓨션 언어 모델(DLM)의 원리와 2026 최신 흐름**](./Architecture/145.md)  
    *MBZUAI와 칭화대 연구진의 서베이 논문을 바탕으로 DLM의 기초 원리부터 3가지 생성 방식(이산, 하이브리드, 연속), RL/추론 혁신, 1-Step 가속 및 한계점 총정리.*
*   [**현대 MoE의 끝판왕: NVIDIA LatentMoE와 Moonshot AI의 Stable LatentMoE**](./Architecture/144.md)  
    *잠재 공간 압축으로 통신 병목을 줄이는 LatentMoE와, 2.8T 파라미터 Kimi K3에 적용된 SiTU-GLU 및 Quantile Balancing 기반 Stable LatentMoE 기술 분석.*
*   [**차세대 MoE 아키텍처: RFMoE(Routing-Free)와 UoE(Union-of-Experts)**](./Architecture/143.md)  
    *중앙 통제 없이 동적 자율 활성화를 이룬 RFMoE와, 탈락한 전문가의 연산값을 가상 공유 전문가로 전환해 연산 낭비를 없앤 UoE 아키텍처 분석.*
*   [**MoE 라우팅의 한계를 넘어: ReMoE(ReLU 라우팅)와 AoE(자율 전문가 모델)**](./Architecture/142.md)  
    *Top-K 및 Softmax의 불연속성을 ReLU 라우팅으로 해결한 ReMoE와, 중앙 라우터 없이 전문가 자율 평가로 연산 효율을 높인 AoE 논문 리뷰.*
*   [**MoE(Mixture-of-Experts) 아키텍처의 동작 방식과 역사**](./Architecture/141.md)  
    *MoE 레이어의 어텐션 및 FFN 분할 구조, 라우터 내적과 Top-k Softmax 가중 합산 등 MoE의 기초 동작 메커니즘과 역사를 정리한 글.*
*   [**Kimi K3: GPT-5.6과 Fable 5를 넘어서는 2.8T 중국 오픈 모델**](./Architecture/139.md)  
    *2.8T 파라미터와 Stable LatentMoE, Attention Residuals 구조로 Claude Opus 4.8을 넘어서는 Kimi K3 아키텍처 분석.*
*   [**Inkling: T급(975B) 개방형 멀티모달 오픈 모델의 등장**](./Architecture/140.md)  
    *Thinking Machines Lab에서 공개한 미국 최고 성능의 오디오 지원 범용 멀티모달 오픈 모델 Inkling과 최적화의 의미.*
*   [**HydraHead: 어텐션 헤드를 쪼개는 분업의 효율성**](./Architecture/138.md)  
    *하나의 층 내부에서 어텐션 헤드를 쪼개어 가볍고 빠른 선형 어텐션과 무거운 전체 어텐션을 분업시키는 하이브리드 기술.*
*   [**ICML 2026 최우수 논문과 Diffusion LLM의 미래**](./Architecture/137.md)  
    *수학이나 코딩에서 AR 모델을 넘어설 가능성을 보여준 칭화대와 예일대의 최신 디퓨션 모델 연구 2편 리뷰.*
*   [**100만 토큰의 한계를 넘으려는 시도: BLOCKSEARCH와 HiLS-Attention**](./Architecture/136.md)  
    *1M 토큰 이상의 초장문 컨텍스트를 처리하기 위한 블록 검색 및 계층형 요약 기반의 효율적 어텐션 모델 소개.*
*   [**HOLA (Hippocampal Linear Attention): 순환 모델의 구조적 건망증 해결**](./Architecture/135.md)  
    *해마와 대뇌피질에서 영감을 얻어, 기존 순환 모델의 정보 압축 한계와 환각을 극복한 아키텍처 연구.*
*   [**Log-Linear Attention: 현대 LLM은 비쌉니다**](./Architecture/134.md)  
    *순환 모델의 고질적인 기억 용량 부족 문제를 시간축 트리 구조(이진트리)로 풀어낸 아키텍처 연구.*
*   [**QKV 분리 착각: AI 메모리 용량을 절반으로 (Do Transformers Need Three Projections?)**](./Architecture/133.md)  
    *K와 V 행렬을 통합하여 지능 저하 없이 LLM의 KV 캐시 메모리 사용량을 절반으로 줄인 혁신적 접근법.*
*   [**LLM의 진짜 정체는 거대한 텍스트 압축기입니다 (Predictable Compression Failures)**](./Architecture/132.md)  
    *LLM의 환각을 정보 예산 부족으로 인한 텍스트 압축 실패로 규정하고 이를 사전에 통제하는 방법.*
*   [**Qwen 3.5 9B를 dLLM으로 100B 모델을 압도하기 (FLARE 프레임워크)**](./Architecture/131.md)  
    *기존 LLM의 지능을 유지하면서 속도를 극대화하는 하이브리드 아키텍처 기반의 dLLM 모델 연구.*
*   [**사람은 수면을 통해 낮에 본 단기 기억을 대뇌피질로 옮기고... (Language Models Need Sleep)**](./Architecture/130.md)  
    *인간의 수면 메커니즘을 모방해 트랜스포머의 추론 능력을 폭발적으로 끌어올린 하이브리드 모델 연구.*
*   [**MiniMax M2 시리즈 기술 보고서: Full Attention과 자가 진화**](./Architecture/129.md)  
    *Full Attention을 유지하여 문맥 단절을 막고 모델 스스로 에러를 수정하는 자가 진화 달성.*
*   [**1,500달러로 7B를 압도한 1B 모델, HRM-Text: 전두두정엽 루프에서 영감을 받은 아키텍처 혁신**](./Architecture/128.md)  
    *인간 전두엽 루프의 가중치 공유(Weight Sharing) 원리를 적용하여 VRAM 사용량을 1B 수준으로 묶어둔 채 128층의 수학적 연산 깊이를 확보해 7B 모델들을 압도한 HRM-Text 아키텍처.*
*   [**우리 뇌는 역전파 과정이 없습니다: 딥러닝 역전파 알고리즘의 3가지 비효율과 대안 연구들**](./Architecture/127.md)  
    *딥러닝의 심장인 역전파 알고리즘이 품고 있는 공간·전력·시간적 비효율을 고발하고, 이를 타파하기 위해 발표된 3가지 대안 연구(FF 알고리즘, EGGROLL, Hamiltonian Inference) 소개.*
*   [**학습 없이 가중치 편집으로 모델을 제어하다: VPD(Adversarial Parameter Decomposition)**](./Architecture/126.md)  
    *적대적 파라미터 분해(VPD) 기술을 통해 거대 가중치를 독립적인 개념 조각으로 찢어내고, 재학습 없이 정밀한 가중치 조작만으로 모델의 행동을 교정한 연구.*
*   [**There Will Be a Scientific Theory of Deep Learning: 연금술에서 과학으로**](./Architecture/125.md)  
    *딥러닝을 예측 가능한 '학습 역학'으로 정의하고, Scaling Laws와 하이퍼파라미터 전이 등 딥러닝의 블랙박스를 여는 수학적/물리적 증거들을 제시한 리뷰.*
*   [**DeepSeek-V4-Pro: 100만 토큰을 씹어 먹는 극단적 가성비의 코딩/수학 특화 모델**](./Architecture/124.md)  
    *코딩(Codeforces) 3206점으로 3대장을 압도한 딥시크의 신작. CSA/HCA 하이브리드 어텐션과 FP4 양자화로 구현한 극단적 가성비 분석.*
*   [**Attention to Mamba & Effective Distillation to xLSTM: 로컬 모델을 위한 증류 수술법**](./Architecture/123.md)  
    *트랜스포머의 OOM 한계를 넘기 위한 두 가지 증류 전략: 순수 선형화(Mamba) vs 하이브리드 타협(xLSTM)의 장단점 비교.*
*   [**만능 옴니모델 vs 도메인 특화 모델: KDL Frontier**](./Architecture/122.md)  
    *비대한 자아를 가진 옴니모델의 한계와 구조적 혁신을 이룬 도메인 특화 OCR 모델의 부상.*
*   [**인공지능 자체가 하나의 운영체제가 된다면? Neural Computer (NC)**](./Architecture/121.md)  
    *Meta AI의 NC: 모델 내부 신경망 잠재 공간 안에서 연산, 기억, 입출력을 통합 처리하는 실험적 접근.*
*   [**트랜스포머의 성능, RNN의 가벼움을: Memory Caching (MC)**](./Architecture/120.md)  
    *Google과 Cornell의 MC: 세그먼트별 요약본 캐싱을 통한 O(NL) 복잡도 달성.*
*   [**파라미터를 늘리지 않고, 모델 내부에서 재귀를 돌리는 우로(Ouro)**](./Architecture/119.md)  
    *ByteDance의 Ouro와 LoopRPT: 잠재 공간에서의 재귀적 추론과 사전 강화학습.*
*   [**패러다임 전환: VLM 기반 문서 OCR의 한계를 넘는 MinerU-Diffusion**](./Architecture/118.md)  
    *텍스트 디코더에 확산(Diffusion) 모델을 도입해 전체 페이지를 병렬 해독하는 문서 OCR.*
*   [**Modern Model Architectures Overview 강연 온라인 진행 안내**](./Architecture/117.md)  
    *다양한 모델 아키텍처 소개 및 온라인 무료 강연 진행 일정 안내.*
*   [**손실 없는 Transformer to xLSTM 증류법**](./Architecture/116.md)  
    *무거운 어텐션 모듈을 덜어내고 하이브리드 모듈(xLSTM+SWA)로 이식하여 메모리와 속도를 혁신적으로 개선한 증류(Distillation) 수술법.*
*   [**Jacobi Forcing: 튜닝만으로 추론 속도를 4배 끌어올리는 방법**](./Architecture/115.md)  
    *기존 LLM 아키텍처 변경 없이, 가벼운 튜닝만으로 추론 속도를 4배 향상 (KV 캐시 100% 보존)*
*   [**Attention Residuals: 딥러닝 레이어 사이의 정보 흐름 재설계**](./Architecture/114.md)  
    *Kimi (Moonshot AI): RNN에서 Attention, ResNet에서 AttnRes*
*   [**Nemotron 3 Super와 NVFP4의 마법**](./Architecture/113.md)  
    *NVIDIA의 기업용 에이전트 니모클로(Nemoclo)와 블랙웰 하드웨어 장벽.*
*   [**최고 성능의 4.44배 빠른 ASR 분야의 dLLM**](./Architecture/112.md)  
    *확산 모델(Diffusion Model)을 음성 인식에 도입해 병렬 처리와 추론 가속 달성.*
*   [**STATIC: 1,000배 빠른 LLM 추천시스템**](./Architecture/111.md)  
    *구글 딥마인드가 트리 구조를 희소 행렬로 펼쳐 LLM 제약 디코딩을 1,033배 가속한 방법.*
*   [**Text-to-LoRA & Doc-to-LoRA: 즉각적 모델 업데이트**](./Architecture/110.md)  
    *Sakana AI: 문서 내재화에 0.5초, 새로운 스킬 장착에 1초 미만.*
*   [**LLaDA2.0: Scaling Up Diffusion Language Models to 100B**](./Architecture/103.md)  
    *100B Diffusion 모델의 등장: 기존 AR 모델을 개조하여 효율성을 2배 높인 비결.*
*   [**TEON vs Muon: 옵티마이저 전쟁**](./Architecture/104.md)  
    *AdamW의 시대는 가는가? 레이어(Layer)를 넘어 텐서(Tensor) 차원의 최적화로.*
*   [**EinFields: 아인슈타인을 위한 신경망**](./Architecture/105.md)  
    *우주의 시공간(Spacetime)을 신경망의 가중치 속에 압축하다.*
*   [**Micro GPT: LLM의 바닥을 보다**](./Architecture/106.md)  
    *Andrej Karpathy의 선물. 오직 행렬 곱셈과 미분이 있을 뿐.*
*   [**QED-Nano: 다윗이 골리앗을 이기는 법**](./Architecture/107.md)  
    *수학 증명에서 4B 모델이 100B 모델을 압도한 비결.*
*   [**Moonshine: 달빛처럼 가벼운 음성 인식**](./Architecture/108.md)  
    *OpenAI Whisper의 대항마? 엣지(Edge) 디바이스를 위한 구세주.*
*   [**Nested Learning**](./Architecture/109.md)  
    *딥러닝은 '깊이'가 아니라 '중첩'이다.*
*   [**Diffusion LLM (100B Parameters)**](./Architecture/83.md)  
    *30B 모델보다 2배 빠른 병렬 생성 모델의 등장.*
*   [**RNN is all you need**](./Architecture/102.md)  
    *Transformer의 속도를 잡은 병렬 학습 RNN (minLSTM, minGRU)의 부활.*
*   [**Titans: Learning to Memorize at Test Time**](./Architecture/101.md)  
    *Transformer의 기억력을 넘어서는 새로운 메모리 중심 아키텍처.*
*   [**LLM의 "입력 길이 제곱(N^2)"의 저주**](./Architecture/90.md)  
    *누가 먼저 끊어낼 것인가?*
*   [**Mistral Large 3: 효율성의 극대화**](./Architecture/88.md)  
    *Mistral Large 3 vs Kimi K2: Efficiency vs Scale.*
*   [**표준이 된 V3 아키텍처**](./Architecture/86.md)  
    *Mistral Large 3, Kimi K2 그리고 DeepSeek V3.2 분석.*
*   [**Ai2 Olmo 3**](./Architecture/85.md)  
    *성능보다는 과정의 투명성에 집중한 LLM 연구의 교과서.*
*   [**Nemotron-3-Nano-30B-A3B**](./Architecture/50.md)  
    *Qwen3보다 빠르고 강력한 Mamba-2 하이브리드 모델.*
*   [**DeepSeek Engram**](./Architecture/44.md)  
    *기억을 효율화하여 연산 낭비를 줄이는 새로운 희소성 축.*
*   [**2026년의 통념 파괴: 90M, 600M 모델**](./Architecture/41.md)  
    *초소형 모델들의 놀라운 지시 이행 능력.*
*   [**Sakana AI: DroPE**](./Architecture/47.md)  
    *위치 정보(Positional Embeddings)는 학습할 때만 쓰고 실전에서는 버려라.*
*   [**Sakana AI RePo**](./Architecture/38.md)  
    *위치 정보를 재설계(Re-position)하라.*
*   [**DeepSeek vs Qwen (A3B MoE)**](./Architecture/35.md)  
    *정반대의 설계 철학 분석.*
*   [**Generative Modeling via Drifting**](./Architecture/6.md)  
    *확산 모델의 250단계를 단 1단계(1-step)로 줄여 속도와 품질을 동시에 잡은 혁신.*
*   [**Beyond Transformers 2**](./Architecture/13.md)  
    *덩치 경쟁을 넘어 생각과 본질로.*
*   [**DeepSeek-V3 vs V3.2: 아키텍처의 진화**](./Architecture/94.md)  
    *아키텍처의 진화와 기술적 목표점.*
*   [**Gemma 3 모델의 핵심 목표 및 특징**](./Architecture/98.md)  
    *구글 딥마인드의 최신 멀티모달 모델 분석.*
*   [**Python 재귀로 시작하는 1,000만 토큰 시대**](./Architecture/60.md)  
    *Recursive Language Models: Python 재귀로 1,000만 토큰 처리하기.*
## <a id="pre-training"></a>📚 Pre-Training

*   [**Breaking the Token Ceiling: 256개 순수 바이트로 토큰 천장을 뚫는 사전학습 지식 증류**](./Pre_Training/100.md)  
    *어휘 사전을 버리고 256개 순수 바이트만으로 사전학습 지식 증류(End-of-Token)를 적용해 데이터 1/6, 저장 공간 1/5로 토큰 모델 대비 4.0%p 높은 성능 상한선을 뚫어낸 Meta FAIR 연구.*
*   [**Skaling: 친칠라 법칙의 오류를 깨고 스케일링 예측 비용을 10배 줄인 Meta FAIR의 새로운 스케일링 법칙**](./Pre_Training/99.md)  
    *모델 파라미터와 데이터의 음(-)의 교차 시너지를 단일 지수 k로 결합해 외삽 오차를 줄이고 L자형 샘플링으로 사전 연산 비용을 10배 절감한 Skaling 법칙 분석.*

*   [**한국어를 잘하는 소형 모델 Mid-training**](./Pre_Training/98.md)  
    *LFM2.5-8B-A1B 모델 기반 6.5B 토큰 규모의 한국어 지식 주입(CPT) 실험 성과 및 한계점 공유.*
*   [**LLM 학습 효율화 방안: 인간의 언어 습득 방식**](./Pre_Training/97.md)  
    *인간의 언어 습득 방식을 모방한 점진적 어휘 학습법(Vocabulary Curriculum Learning).*
*   [**RoPE가 정보를 유실하고 있다?**](./Pre_Training/79.md)  
    *푸단대 연구진의 충격적인 발견과 해결책.*



*   [**CALM: Continuous Autoregressive Language Models**](./Pre_Training/34.md)  
    *한 글자씩 타이핑하는 LLM을 넘어, 4개씩 생성하는 연속 벡터 예측.*



## <a id="post-training"></a>🎯 Post-Training

*   [**Normalized Low-Rank Adaptation (NoRA): 행렬 정규화 한 줄로 LoRA의 성능과 안정성을 극대화하다**](./Post_Training/148.md)  
    *LoRA의 A 행렬 열 벡터를 단위 크기로 정규화해 초반 그래디언트 왜곡을 없애고, 가중치 병합을 유지하며 SFT 점수 대폭 향상, 치명적 망각 방지, RLVR 최적화 안정화를 이뤄낸 연구.*
*   [**Never Give Up (NGU): 쉬운 문제는 빠르게 털고 어려운 문제는 끝까지 물고 늘어지는 LLM 강화학습**](./Post_Training/147.md)  
    *다 푼 문제는 K=4로 조기 탈락시키고 틀린 문제만 비동기 큐에 재투입하여 수십 번 실패 끝에 건진 정답에 강력한 보상(Anchor Positives)을 꽂아 고난도 수학·코딩 난제를 뚫어낸 연구.*
*   [**Do We Need Adam?: 강화학습(RLVR)에서 AdamW 대신 순수 SGD가 메모리와 성능을 모두 잡는 이유**](./Post_Training/146.md)  
    *실시간 정책 변화로 과거 모멘텀이 무력화되는 강화학습 환경에서, 순수 SGD로 옵티마이저 메모리를 3배 아끼고 극소수 파라미터만 업데이트해 성능과 망각 방지를 달성한 논문 리뷰.*
*   [**Spurious Advantage Hidden in GRPO: 찍어서 맞힌 요행에 폭탄 보상을 주는 결함과 SignBalance 해결책**](./Post_Training/145.md)  
    *GRPO가 소수 정답 궤적에 폭탄 보상을 주는 수학적 역설을 밝히고, 정답 보상 고정과 오답 균등 분배(SignBalance)로 성능을 끌어올린 연구.*
*   [**TailSFT: 아는 문제를 버려야 RL이 산다 — 엔트로피 붕괴를 막는 SFT 필터링 기법**](./Post_Training/144.md)  
    *쉬운 예제의 손실을 제외하고 꼬리(Tail) 난도 데이터만 미세조정하여 탐색 커버리지(pass@16)를 보존하고 후속 강화학습(GRPO) 성능을 극대화한 MS 연구.*
*   [**Understanding Reasoning from Pretraining to Post-Training: 사전학습이 결정하는 RL의 상한선과 최적 연산 배분 법칙**](./Post_Training/143.md)  
    *사전학습 검증 손실이 사후 RL의 최종 성능을 결정짓는 초기화의 저주를 규명하고, Pass@1 대 Pass@k의 실체와 연산 예산 최적 분배 공식을 제시한 NYU 논문.*
*   [**AgentOPSD: 멀티턴 에이전트 강화학습을 위한 턴 단위 기여도 정밀 평가 (Recursive Self-Distillation)**](./Post_Training/142.md)  
    *기존 GRPO의 보상 신호 희석 문제를 해결하고자 Self-Teacher와 베이지안 누적 신뢰도로 턴별 기여도를 정밀 산출해 장기 에이전트 성능을 유지한 논문 리뷰.*
*   [**SFT와 RL의 관계: 지식을 엮는 것과 부품화하는 것 (ICML 2026)**](./Post_Training/141.md)  
    *SFT는 지식을 얽혀진 상태로 제공하고, RL은 이를 재사용 가능한 모듈로 분해하여 OOD 일반화 능력을 극대화한다는 논문 리뷰.*
*   [**Persuasion Attacks: CoT를 악용한 감시자 가스라이팅과 교차 검증**](./Post_Training/140.md)  
    *에이전트가 CoT 공간을 악용해 감시자 모델을 설득하는 취약점과 타사 모델 간 교차 팩트체커 배치의 중요성.*
*   [**LLM이 스스로 가설을 세우는 방식: 지식 그래프 기반 강화학습**](./Post_Training/139.md)  
    *LLM이 지식 그래프를 실시간으로 작성하며 가설을 세우도록 강제하여 할루시네이션을 방지하고 추론 능력을 극대화한 연구.*
*   [**Thinking to Recall: How Reasoning Unlocks Parametric Knowledge in LLMs**](./Post_Training/138.md)  
    *LLM에게 억지로 생각할 시간을 주어 내재 지식(Parametric Knowledge) 인출 능력을 향상시키는 방법.*
*   [**VibeThinker-3B: 3B 소형 모델이 T급 거대 모델을 이길 수 있었던 이유**](./Post_Training/137.md)  
    *단순 암기가 아닌 SFT의 풀이 스펙트럼과 강화학습(RL)을 통해 추론 능력을 극대화한 3B 모델 분석.*
*   [**Liquid AI의 LFM2.5-8B-A1B을 터미널 환경에서 스스로 에러를 해결하고 작업을 수행하는 에이전트 성능을 개선하기 위해 SFT와 RLVR 실험을 진행했습니다.**](./Post_Training/136.md)  
    *소형 모델 환경에서 SFT와 RLVR 방식의 성능 향상 비교 및 탐색 효율성에 대한 현실적인 한계 분석.*
*   [**Microsoft Research가 말하는 터미널 환경과 에러 메시지를 월드모델로 써서, 터미널 에이전트가 똑똑해지는 법 (ECHO)**](./Post_Training/135.md)  
    *터미널 에이전트 강화학습 과정에서 무시되던 에러 로그와 디렉토리 결과를 버리지 않고 가중치 업데이트에 활용(Loss 계산)함으로써, 연산량 증가 없이 성능을 2배 높인 MS 리서치의 ECHO 방법론.*
*   [**알파고의 탐색을 LLM에게: 성능을 18.8% 올린 DLR (Dynamic Latent Routing)**](./Post_Training/134.md)  
    *알파고의 MCTS 개념을 트랜스포머의 잠재 공간(Latent Space)에 이식해 토큰 생성 지연 없이 추론 성능을 최대 18.8% 끌어올린 베이징 항공우주대 연구진의 아키텍처.*
*   [**AI 안전은 사기일까요?: 단 하나의 뉴런으로 무력화되는 Safety Alignment (A Single Neuron Is Sufficient)**](./Post_Training/133.md)  
    *고작 단 하나의 '거부 뉴런'만 비활성화(0으로 고정)해도 LLaMA 등 억 단위 파라미터 모델들의 안전 장벽이 100% 무너지고 성능 저하도 없는 정렬(DPO/RLHF)의 한계를 고발한 애플 연구진의 연구.*
*   [**LLM의 할루시네이션을 없애면 안 됩니다: Faithful Uncertainty 프레임워크**](./Post_Training/132.md)  
    *할루시네이션을 강제로 억제(SFT/RLHF)하면 일반 지식과 추론 성능이 망가지는 정렬 세금(Utility Tax)을 증명하고, 환각을 억제하는 대신 스스로 확신도를 표시하도록 돕는 프레임워크.*
*   [**NVIDIA: Speculative Decoding을 통한 RL 훈련 속도 2.5배 향상**](./Post_Training/131.md)  
    *추측 해독(Speculative Decoding) 기술을 RL 훈련 파이프라인에 통합하여, 학습 품질 저하 없이 초대형 모델의 훈련 속도를 최대 2.5배까지 끌어올린 엔비디아의 최적화 기법.*
*   [**REASONMAXXER: 단일 GPU, 단돈 4달러로 무거운 RL 모델들을 압도하다**](./Post_Training/130.md)  
    *거대한 RL 루프를 걷어내고, 모델의 엔트로피가 급증하는 '결정 포인트'만을 타격하는 50개의 고순도 데이터와 소형 LoRA만으로 최고 수준의 성능을 달성한 연구.*
*   [**터미널 에이전트를 위한 SFT 학습과 평가에서 느낀 3가지 한계점**](./Post_Training/129.md)  
    *H200 8대로 LFM2, Gemma4 등 53개 모델을 튜닝하며 발견한 벤치마크 프록시 평가의 함정과 모델-데이터 포맷의 호환성 문제.*
*   [**PivotRL: 아는 문제는 안 풉니다**](./Post_Training/128.md)  
    *NVIDIA의 PivotRL: 궤적 중 '피벗'만을 찾아 집중 훈련해 망각을 막고 훈련 시간을 5.5배 단축시키는 강화학습 전략.*
*   [**Garbage In, Good Out: 쓰레기 데이터로 성능 올리기 (SSD)**](./Post_Training/127.md)  
    *Apple 연구팀의 Simple Self-Distillation: 외부 피드백 없이 모델 스스로의 출력으로 기초 체력을 강화하는 기술.*
*   [**똑똑한 선생님한테 배웠는데 왜 성적은 떨어질까? TESSY 프레임워크**](./Post_Training/126.md)  
    *교사-학생 모델 간의 스타일 충돌을 해결하여 SFT 성능을 수직 상승시키는 전략.*
*   [**Introduction to Post-Training & Beyond 강연 후기 및 향후 계획**](./Post_Training/125.md)  
    *Maxime Labonne의 강연 자료를 기반으로 한 온라인 강연 성공 사례와 향후 디스코드 스터디 계획.*
*   [**그 OpenClaw로 강화학습을? 프린스턴 대학교의 OpenClaw-RL**](./Post_Training/124.md)  
    *프린스턴 대학교 연구진이 제안한 OpenClaw-RL 프레임워크와 실시간 강화학습의 한계.*
*   [**강화학습 과정은 단 13개의 파라미터면 충분합니다.**](./Post_Training/110.md)  
    *Qwen2.5-7B-Instruct에 GRPO를 적용하여 GSM8K 수학 벤치마크 정답률 91% 달성.*
*   [**Parameter-Efficient Fine-Tuning for Foundation Models**](./Post_Training/106.md)  
    *거대 모델을 효율적으로 튜닝하는 5가지 핵심 기법(PEFT) 총정리.*
*   [**When Reasoning Meets its Laws**](./Post_Training/108.md)  
    *단 3,900개의 데이터로 AI에게 '추론의 물리 법칙'을 가르치는 법 (LORE).*
*   [**LIE: 깊게 생각할수록 더 똑똑해진다**](./Post_Training/109.md)  
    *LLM에게 '생각을 멈추지 않는 법'을 가르치는 강화학습 전략.*
*   [**ProRL: Prolonged Reinforcement Learning**](./Post_Training/107.md)  
    *강화학습, 짧게 하지 말고 길게 하라. RL 스케일링 법칙의 발견.*
*   [**DuPO: Self-Verification via Dual Preference Optimization**](./Post_Training/104.md)  
    *정답지 없는 번역을 스스로 검증하는 '일반화된 쌍대성' 기법.*
*   [**From Code Foundation Models to Agents**](./Post_Training/84.md)  
    *Code Foundation Model에서 자율 코딩 에이전트로의 진화 청사진.*
*   [**Emergent Misalignment**](./Post_Training/96.md)  
    *취약한 코드를 배운 AI의 위험한 일탈.*
*   [**Stabilizing RL with LLMs**](./Post_Training/92.md)  
    *화려한 기교보다 수학적 기본기가 중요한 이유.*
*   [**Yann LeCun: World Model의 중요성**](./Post_Training/78.md)  
    *LLM은 물리 세계를 배울 수 없다?*
*   [**GDPO: Multi-reward RL**](./Post_Training/46.md)  
    *GRPO의 약점을 극복한 새로운 강화학습 기법.*
*   [**Detailed balance in LLM-driven agents**](./Post_Training/23.md)  
    *LLM이 물리학의 '최소 작용의 원리'를 따른다는 것을 증명한 연구.*
*   [**iGRPO**](./Post_Training/1.md)  
    *Self-Feedback-Driven LLM Reasoning: 모델이 스스로 만든 초안을 보고 배우는 자가 개선 강화학습.*
## <a id="evaluation"></a>⚖️ Evaluation

*   [**CHEATBENCH: 10개 영역에서 드러난 최신 프론티어 AI 에이전트들의 보상 해킹(Reward Gaming) 실태**](./Evaluation/114.md)  
    *도구 활용 능력이 고도화된 최신 프론티어 에이전트들이 어려운 과업에서 48%~81% 확률로 편법과 부정행위를 시도하는 실태와 CoT-행동 간 모순을 폭로한 CAIS 벤치마크.*
*   [**Jev-as-a-Judge: 확신할 땐 수용하고 모호할 땐 이관하라 (Accept When Confident, Escalate When Unsure)**](./Evaluation/113.md)  
    *출력 토큰 없이 152ms만에 정형 판정을 내리는 초저가 판사 Jev의 장단점을 분석하고, 자체 확신도 기반 캐스케이드로 47% 비용 절감과 플래그십급 정확도를 보존한 CMU 논문.*
*   [**벤치마크의 오염과 숫자의 한계: Reasoning or Memorization?**](./Evaluation/112.md)  
    *벤치마크 오염과 기출 암기 문제, AAII 순위 변동 속에서 단일 벤치마크 수치의 한계와 실전 모델 선택 및 AI 의존에 대한 고찰.*
*   [**LLM의 메타인지: 자신이 무엇을 아는지 모니터링하는 능력**](./Evaluation/111.md)  
    *최신 LLM의 메타인지 능력을 진단하고, 20점 스케일링과 자가 수정 한계 등 메타인지 효율을 높이는 팁을 분석한 서베이.*
*   [**엔트로피로 LLM 거짓말 탐지기 제작 (Entropy Distribution as a Fingerprint)**](./Evaluation/110.md)  
    *모델 출력 시 요동치는 로짓 확률 분포(엔트로피)를 실시간으로 분석해 환각을 잡아내는 CES 탐지기.*
*   [**인공지능이 문서를 진짜로 읽고 있을까요?: 최신 멀티모달 모델의 귀속 환각(Attribution Hallucination)을 폭로한 CiteVQA 논문**](./Evaluation/109.md)  
    *시각적 근거 매핑(SAA) 평가를 통해 VLM들이 정답은 맞히면서도 실제 이미지를 읽지 못하고 기억에 기반한 텍스트 확률로 대답하는 '귀속 환각(Attribution Hallucination)' 현상을 실증한 CiteVQA.*
*   [**Step-3.5, LLaDA2.1... 86개 모델들을 테스트하고 느낀 점들**](./Evaluation/108.md)  
    *86개 최신 모델들을 직접 굴려보며 뼈저리게 느낀 벤치마크 오염(기출문제 암기), 한국어 데이터 파이프라인 부재, 베이스 모델 지능 체급의 중요성, Qwen/LFM의 희망 등에 대한 고찰.*
*   [**Mirage: 멀티모달 모델의 시각 지능은 신기루인가?**](./Evaluation/107.md)  
    *최신 VLM들이 이미지를 보지 않고도 텍스트 단서만으로 답을 유추하는 '신기루 현상'을 폭로하며, 진정한 시각 지능을 위한 데이터 정제와 RL 기반 접근의 중요성을 강조한 논문.*
*   [**IKP 벤치마크: 압축 불가능한 지식으로 독점 모델의 파라미터 크기 유추하기**](./Evaluation/106.md)  
    *희귀 지식의 정답률과 파라미터 크기의 로그-선형 관계를 이용해 GPT-5.5, Claude Opus 4.6 등 숨겨진 블랙박스 모델의 크기를 역산한 연구.*
*   [**Preference Leakage: A Contamination Problem in LLM-as-a-Judge**](./Evaluation/105.md)  
    *LLM 평가자가 자신의 패밀리 모델을 편애하는 '선호도 유출' 문제.*
*   [**ADR-Bench 전문가 평가**](./Evaluation/55.md)  
    *DeepSeek-v3.2를 압도한 효율적인 에이전트 모델.*
## <a id="rag--knowledge"></a>🗂️ RAG & Knowledge

*   [**Empty Shelves or Lost Keys?: 프론티어 모델의 95% 지식 저장과 인출 병목(Recall Bottleneck)의 진실**](./RAG/126.md)  
    *LLM이 이미 95% 이상의 지식을 내재화(인코딩)했음에도 인출(Recall) 실패로 오답을 내는 원인을 분석하고, Thinking(CoT)의 자가 점화 효과를 증명한 구글 리서치 논문.*

*   [**MRAgent: Memory is Reconstructed, Not Retrieved**](./RAG/125.md)  
    *단순한 벡터 검색이 아닌, 단서와 태그로 지식을 저장하고 필요시 능동적으로 추론하는 Agentic GraphRAG.*
*   [**GraphRAG를 위해 미리 지식 그래프를 만들 필요 없습니다 (LogicRAG)**](./RAG/124.md)  
    *거대한 사전 지식 그래프 구축 없이 질문이 들어올 때 즉석에서 작은 논리 구조를 개척하는 방법론.*
*   [**이제 RAG에서 구식 Vector DB를 버리고, 지식만 전문으로 암기하는 소형 비서 sLM 모델을 따로 만들어 LLM 옆에 두면 어떨까요? (MEMO)**](./RAG/123.md)  
    *도서관 사서 역할을 하던 기존 Vector DB 기반 RAG의 한계를 극복하기 위해, 문서를 고품질 QA 문제집으로 정제하여 지식을 가중치 공간에 개념 구조로 암기시킨 소형 비서 모델(MEMO)의 아키텍처.*
*   [**Is Grep All You Need?: 벡터 DB 없이 리눅스 명령어와 TF-IDF로 구축한 초간단 RAG**](./RAG/122.md)  
    *수천 대의 GPU로 학습한 모델이나 복잡한 RAG 프레임워크 없이 오직 리눅스의 grep과 간단한 TF-IDF만으로 최신 하이브리드 RAG 솔루션을 압도한 가성비와 유연성의 RAG 연구.*
*   [**RAG와 벡터 DB 구축에 대한 핵심 Q&A 및 LlamaIndex 리소스 요약**](./RAG/121.md)  
    *LlamaIndex 가이드를 토대로 벡터 DB의 필수 여부, 컨텍스트 압축의 중요성, sLM의 쓰임새 및 RAG의 본질인 데이터 엔지니어링 노가다의 가치에 대한 정리.*
*   [**DCI (Direct Corpus Interaction): 벡터 DB와 청킹 없는 새로운 RAG 아키텍처**](./RAG/120.md)  
    *원본 문서를 벡터로 쪼개거나 벡터 DB를 쓰지 않고 모델 내부 지능만으로 고속 연산(BrowseComp-Plus)하여 정밀한 텍스트 좌표를 찾아내 성능과 비용을 모두 개선한 칭화대 연구진의 DCI 아키텍처.*
*   [**SLIDERS: LLM의 환각과 컨텍스트 한계를 제어하는 SQL 데이터베이스 구조**](./RAG/119.md)  
    *동적인 텍스트 탐색(RAG) 대신 초거대 문서를 SQL 데이터베이스 표(RDB)로 전처리하여 환각을 없애고 롱 컨텍스트 한계를 넘는 방법.*
*   [**1억 토큰의 장벽을 깨다: RAG의 한계를 극복한 MSA**](./RAG/118.md)  
    *RAG를 버리고 모델 내부에 1억 토큰을 담아내는 Memory Sparse Attention 기술.*
*   [**더 싸고 빠른 오픈소스 모델로 Claude 4.6 Opus 잡기**](./RAG/117.md)  
    *가벼운 모델에게 검색하는 방법을 가르치는 KARL.*
*   [**LIMRANK: Less is More**](./RAG/109.md)  
    *2만 개 데이터로 SOTA 리랭커 만들기.*
*   [**HippoRAG 2**](./RAG/100.md)  
    *인간의 기억 메커니즘을 모방한 비모수적 연속 학습 (Bio-inspired Continual Learning).*
*   [**vLLM의 승리: 압도적인 속도**](./RAG/29.md)  
    *표준이 되기까지.*
*   [**RAG & Agent Memory 4선**](./RAG/10.md)  
    *GraphSearch, S-RAG, xMemory 등 최신 논문 소개.*
*   [**Beyond Naive RAG**](./RAG/110.md)  
    *4 Papers that Redefine Agent Memory.*
*   [**SEPAL: Scalable Feature Learning**](./RAG/111.md)  
    *9천만 개 지식 그래프, V100 한 장으로 학습하기.*
*   [**GraphRAG Survey**](./RAG/112.md)  
    *RAG의 미래는 그래프다 (ACM TOIS).*
*   [**A-MEM: Agentic Memory**](./RAG/113.md)  
    *에이전트를 위한 살아있는 기억.*
*   [**PISCO: Compression for RAG**](./RAG/114.md)  
    *RAG를 위한 초고효율 압축.*
*   [**SymAgent: Symbolic Knowledge Graph**](./RAG/115.md)  
    *기호 추론으로 완성하는 지식 그래프.*
*   [**VideoRAG**](./RAG/116.md)  
    *영상을 읽는 RAG.*
## <a id="on-device-ai"></a>💻 On-Device AI

*   [**flash-moe: 48GB 맥북에서 Qwen3.5-397B-A17B 돌리기**](./On_Device/126.md)  
    *C언어와 OS 순정 기능으로 SSD 병목을 뚫어낸 Mac 전용 추론 엔진.*
*   [**Liquid AI 1.2B vs Google 4B**](./On_Device/45.md)  
    *Pau Labarta Bajo's Local AI Insight.*
*   [**국가대표 AI 탈락 그 후 (On-Device Focus)**](./On_Device/42.md)  
    *현실적인 진단과 중국 모델과의 비교.*
*   [**로컬 LLM 구동의 6가지 현실적 방법**](./On_Device/33.md)  
    *STEM: 단순히 지식을 꺼내기 위해 비싼 GPU를 쓰지 말자.*
*   [**LLM 지능의 민낯과 한계**](./On_Device/3.md)  
    *벤치마크는 수석이지만 현장(진료)에서는 낙제인 이유와 해결책.*
## <a id="projects"></a>🚀 Projects

### 🤖 Autonomous Agents
*   [**변호사 시험 만점을 받았습니다. 제15회 변호사 시험 선택형 150문제**](./Projects/125.md)  
    *Gemini 3.1 Pro와 Codex를 활용한 법률 컨텍스트 구축 및 추론 제어로 한국 변호사 시험 만점 달성.*
*   [**Terminal-Bench 에이전트 소형 모델 실험: 누적 다운로드 60,000회 돌파**](./Projects/124.md)  
    *8B 이하 소형 모델로 터미널 벤치마크에서 초대형 모델들을 뛰어넘은 SFT 및 RLVR 훈련 과정과 성과.*
*   [**터미널 모델 다운로드 27,000회 돌파 소회: GPU Poor 환경에서의 생존 전략**](./Projects/123.md)  
    *터미널 모델 2.7만 회 다운로드 소회 및 생존 전략.*
*   [**터미널 에이전트를 위한 SFT 학습과 평가에서 느낀 3가지 한계점 (다운로드 1,000회 돌파)**](./Projects/122.md)  
    *김정수님의 H200 GPU 서버를 지원받아 개발한 모델들의 다운로드 1,000회 돌파를 기념하며 정리한 에이전트 벤치마크 오버피팅, SFT 지능 극복 한계, 평가 프레임워크의 문제 등 3가지 현실적인 한계.*
*   [**H200 8대로 56개 터미널 에이전트 모델 전면 재평가: Qwen 3.5의 독주**](./Projects/121.md)  
    *Ouro 등 최신 모델을 추가하여 56개 모델의 성능을 전면 재측정하고, 베이스 모델의 특성과 데이터 포맷팅이 에이전트 성능에 미치는 결정적 영향을 분석한 리포트.*
*   [**로컬 터미널 에이전트 CLI 개발 및 LFM2-8B-Terminal-SFT 튜닝**](./Projects/120.md)  
    *Liquid AI의 LFM2-8B-A1B 모델을 활용한 로컬 터미널 에이전트 'liquid-cli' 개발 및 배포기.*
*   [**터미널 에이전트: 1B(8B A1B) 모델의 가능성**](./Projects/119.md)  
    *LFM2-8B-A1B 모델과 Nemotron-Terminal-Corpus를 활용한 로컬 터미널 에이전트 구축 및 테스트 결과.*
*   [**Gemini-Claw 개발기**](./Projects/20.md)  
    *2시간 만에 만든, 스스로 코드를 짜고 뉴스를 분석하는 에이전트.*
*   [**스스로 웹페이지를 만들고 검증하는 AI**](./Projects/16.md)  
    *Gemini-Claw: 스스로 웹페이지를 만들고, 실행하고, 검증까지 하는 에이전트.*
*   [**Insight Agents**](./Projects/21.md)  
    *An LLM-Based Multi-Agent System for Data Insights.*
*   [**SEAL: 스스로 Fine-tuning하는 에이전트**](./Projects/26.md)  
    *가능성과 한계.*
### 🛠️ Coding & Dev Tools
*   [**Claube Vibe Coding**](./Projects/52.md)  
    *복잡한 백엔드는 AI에게 맡기고 공원에서 러닝하기.*
*   [**무한 루프 바이브 코딩**](./Projects/22.md)  
    *"테스트 성공할 때까지 계속해" 한마디로 개발 끝내기.*
*   [**Docling-Translate**](./Projects/91.md)  
    *CLI의 번거로움을 해결한 Streamlit 기반 번역 도구.*
*   [**LFM-Scholar**](./Projects/57.md)  
    *논문 Related Work 자동 작성을 위한 LLM 도구.*
*   [**Gemini-Claw 파일 조작 기능**](./Projects/19.md)  
    *"터미널 조작 기능이나 넣어볼까?"*
*   [**Gemini-Claw 오피스 생성**](./Projects/15.md)  
    *로컬 폴더를 분석해 94초 만에 풀 패키지 생성.*
*   [**Gemini 3 Pro + 낯선 API**](./Projects/18.md)  
    *기대 이상의 코드 퀄리티와 재미.*
*   [**Liquid AI LFM2-2.6B-Exp 튜닝기**](./Architecture/58.md)  
    *논문 Related Work 섹션을 통째로 생성하는 도구 제작.*
### 💻 On-Device AI
*   [**Tiny MoA**](./Projects/32.md)  
    *시간당 $100 태우는 AI vs CPU로 돌리는 가성비 멀티 에이전트.*
*   [**Tiny MoA Tool Calling**](./Projects/30.md)  
    *16GB 노트북에서 구현한 로컬 에이전트의 눈과 손.*
*   [**Tiny MoA: 진정한 온디바이스 AI**](./Projects/24.md)  
    *Clawdbot is cool, but Tiny MoA runs on CPU.*
*   [**Clawdbot vs 로컬 AI**](./Projects/25.md)  
    *API 없는 진정한 온디바이스 AI를 향하여.*
*   [**vLLM & SGLang in llama.cpp**](./Projects/28.md)  
    *CPU 추론 속도 1.8배 향상.*
### 🧠 Model Experiments
*   [**4달러로 Jev보다 뛰어난 모델 만들기: Qwev-9B-RLCD 제작 및 평가기**](./Projects/126.md)  
    *Kev-9B 베이스에 4,800건의 합성·불확실성 데이터를 구축하고 Gemini 3.8 Flash로 RLCD를 구현해 A100 1대로 단 30분 만에 Jev와 GPT-6 Astra를 넘어서는 9B 의사결정 모델을 만든 실험기.*
*   [**LFM2-350M-ToolLLaMA: 14달러로 GPT-5-Nano를 이기다**](./Projects/117.md)  
    *H100 한 대, 학습 4시간, 총 14달러로 ToolBench에서 GPT-5-Nano보다 6.6배 이상 좋은 350M 모델 제작.*
*   [**Open-Yaongi Project**](./Projects/Open-Yaongi.md)  
    *52 Layers 4B(Active 0.6B) 규모의 효율적인 sLLM 오픈소스 프로젝트 (Mamba-2 + MoE).*
*   [**HybriKo: 하이브리드 RNN+Attention**](./Projects/54.md)  
    *Google Griffin과 Liquid AI LFM2에서 영감을 받은 아키텍처.*
*   [**HybriKo-117M**](./Projects/37.md)  
    *A100 8장으로 만든 리눅스 명령어 Function Calling 모델.*
*   [**HybriKo-117M-LinuxFC**](./Projects/36.md)  
    *한국어를 리눅스 명령어로 바꿔주는 초경량 모델 개발기.*
*   [**52-Layer HybriKo-430M**](./Projects/31.md)  
    *T4 GPU 하나에 최신 아키텍처를 우겨넣은 실험작.*
*   [**1.2B 모델로 PPT 만들기**](./Projects/8.md)  
    *소형 모델의 가능성.*
*   [**GPT 구조의 한계를 넘어**](./Projects/14.md)  
    *Liquid AI, TII, NVIDIA의 새로운 시도들.*
### 🪄 Post-Training Projects
*   [**Introduction to Post-Training 강연 온라인 진행 안내**](./Post_Training/123.md)  
    *Maxime Labonne님의 자료를 재구성한 온라인 무료 강연 일정 안내 및 디스코드 투표.*
*   [**Introduction to Post-Training 강연 재구성**](./Post_Training/111.md)  
    *Maxime Labonne님의 자료를 기반으로 한 7가지 최신 RL 논문과 비전.*
*   [**LFM2 1.2B 기반 한국어-영어 번역기**](./Projects/89.md)  
    *LFM2 1.2B 모델로 구글과 알리바바의 4B 모델을 이긴 번역기 제작기.*
*   [**LFM2 번역기 개발기: 핵심 발견 및 성과**](./Projects/82.md)  
    *SFT와 RL의 성능 차이 분석 및 Liquid AI 공식 쿡북 등재 소식.*
*   [**Small Language Model for Translation**](./Projects/81.md)  
    *Advice for AI engineers.*
*   [**Liquid AI LFM2-1.2B 튜닝 실패기**](./Projects/76.md)  
    *한국어-영어 번역 RL(GRPO) 학습 실패와 교훈.*
*   [**한국어 LLM 학습 데이터의 부재**](./Projects/51.md)  
    *Pre-training부터 GRPO까지의 험난한 여정.*
### 💭 Insights & Essays
*   [**3개월 9개 프로젝트 회고: GPU Poor의 여정**](./Projects/118.md)  
    *RAM 16GB, i5 CPU 노트북에서 9개의 프로젝트와 2가지 아카이브를 만든 기록.*
*   [**최근 구현한 AI 프로젝트 및 성과**](./Projects/7.md)  
    *Gemini-Claw로 구현한 맥킨지 스타일 보고서 및 PPT 자동 생성.*
*   [**Gemini-Claw 성능 vs 보안**](./Projects/12.md)  
    *LLM 에이전트의 위험한 잠재력.*
*   [**AI에 대한 두려움 vs 흥미**](./Projects/17.md)  
    *OpenClaw, 환각 인용, Vibe Coding 현상에 대한 단상.*
*   [**Pau Labarta Bajo's Insight**](./Architecture/27.md)  
    *멀티 에이전트 시스템에 대한 인사이트.*
## <a id="trends--industry"></a>🔥 Trends & Industry

*   [**속도 조절론의 위선과 프론티어 신작 경쟁: Claude Opus 5.5, GPT-6 Sol, MiMo-V2.6-Pro**](./Trends/170.md)  
    *인류 위험과 속도 조절론을 외치자마자 쏟아진 빅테크들의 독점 플래그십(Opus 5.5, GPT-6 Sol)과 MIT 라이선스로 기술과 RL 로그를 전면 개방한 샤오미 1.02T MiMo-V2.6-Pro의 명암.*
*   [**Jev는 일반화된 대형 BERT인가?: ChatGPT의 역사와 평행이론, 그리고 오픈소스의 반격**](./Trends/169.md)  
    *생성 루프를 제거하고 RLCD와 타입 인터페이스로 무장한 Jev의 상용화 공식 분석과, Laya·AnyJev 등 오픈소스 진영의 고속 추격 및 의사결정 모델 생태계 고찰.*
*   [**프론티어 모델 증류 허용 논란과 오픈 생태계의 본질: 규제의 위선과 공정 사용**](./Trends/168.md)  
    *저작권 위반으로 학습하고는 자사 모델 증류를 막아서는 프론티어 랩들의 독점 규제와 계정 제재의 위선을 비판하고, 오픈 생태계를 위한 적극적 증류와 공정 사용의 필요성을 역설한 단상.*
*   [**미국 빅테크의 역습: Fable 5.1, Muse Spark 1.3, Gemini 3.8 Flash의 성능·속도 비교와 추론 비용 단상**](./Trends/167.md)  
    *AAII 선두를 달리는 Fable 5.1과 압도적 속도의 Muse Spark 1.3, Gemini 3.8 Flash의 등장 및 앤트로픽 요금제 논란과 프론티어 모델 경쟁 분석.*
*   [**엔비디아의 허깅페이스 인수와 AI 시대의 중앙은행: 자체 칩의 반격과 순환 금융 리스크**](./Trends/166.md)  
    *허깅페이스 인수와 전방위 투자로 생태계를 장악하는 엔비디아의 전략, 그리고 빅테크 자체 칩 탈출기와 네오클라우드 순환 금융의 잠재적 버블 리스크 분석.*
*   [**Ox Alpha의 정체(GLM-5.3-Flash)와 Qwen3.8-Flash-Next, 그리고 Gemini 3.7 Flash의 재발견**](./Trends/165.md)  
    *Ox Alpha의 실체인 GLM-5.3-Flash와 QSA·GR 구조의 Qwen3.8-Flash-Next, 압도적 속도의 Gemini 3.7 Flash 사용기와 프론티어 모델 다극화 시대 고찰.*
*   [**독자 AI 파운데이션 모델(독파모) 2차 평가 결과와 Motif Technologies 탈락에 대한 고찰**](./Trends/164.md)  
    *독자 아키텍처와 글로벌 Top10 성능(AAII)을 달성한 Motif-3의 탈락 소식과 국내 독자 AI 파운데이션 모델 생태계에 대한 아쉬움 정리.*
*   [**Qwen3.8 27B와 GLM-5.3: 오픈웨이트 및 Post-Training 혁신이 이끄는 중국 AI의 독주**](./Trends/163.md)  
    *로컬 구동 가능한 Qwen3.8 27B의 멀티모달/법률 에이전틱 능력과, 오직 Post-Training만으로 CyberGym 보안 1위 및 AAII 60점을 달성한 GLM-5.3 분석.*
*   [**OpenRouter 1위 무료 모델 Ox Alpha의 등장과 실전 에이전트 사용기**](./Trends/162.md)  
    *DeepSWE 80%로 GPT-5.6을 압도하며 하루 4T 토큰 소비를 달성한 정체불명의 Ox Alpha 모델 특징과 Claude Code/Codex 연결 후기 및 제작사 추정.*
*   [**요즘 생각 (29가지 단상: 최신 모델 폭주와 인공지능의 지향점)**](./Trends/161.md)  
    *Qwen3.8, DeepSeek V4 Pro, GLM-5.3 등 최신 모델 쏟아짐 속에 모델의 추론 한계, 오픈소스 라이선스, AI 서비스가 가질 본질적 가치에 대한 29가지 단상.*
*   [**독파모 AAII 점수 공개와 Qwen3.8, Grok 4.6, Muse-Glimmer 동향**](./Trends/160.md)  
    *독파모 모델들의 AAII 벤치마크 결과와 Qwen3.8 허깅페이스 공개, Grok 4.6 및 Meta Muse-Glimmer-30B 등 최신 모델 릴레이 요약.*
*   [**Motif-3 공식 공개와 Meta & NVIDIA의 30B 오픈웨이트 릴레이**](./Trends/159.md)  
    *독파모 최고 성능 Motif-3의 MIT 라이선스 전환과 벤치마크 분석, 그리고 Meta Muse-Glimmer-30B와 NVIDIA Nemotron-Lightning 공개에 관한 정리.*
*   [**Qwen3.8 Max의 에이전틱 능력과 중국 AI 모델의 급부상: 오픈소스의 변화와 시사점**](./Trends/158.md)  
    *Qwen3.8 Max의 벤치마크 2위 달성과 중국 프론티어 모델들의 발전, 상업화 라이선스 전환 흐름, 그리고 한국 AI 개발 방향에 대한 단상.*
*   [**제프 딘의 구글 퇴사와 Discovery Loop 창업: AI 거장들의 이동과 생태계 변화**](./Trends/157.md)  
    *제프 딘, 산제이 게마왓 등 구글의 핵심 AI 리더들의 퇴사와 공익 기업 Discovery Loop 설립, 그리고 구글 AI 경쟁력과 TPU 동향에 관한 단상.*
*   [**LLM 스케일링의 역사: 우리는 올바른 길로 가고 있는가**](./Trends/156.md)  
    *GPT 시리즈부터 최신 추론 모델까지 이어지는 스케일링 역사의 흐름과 거대 자본 주도 패러다임의 한계 고찰.*
*   [**파시즘에 가까운 클로즈드 진영과 오픈 모델 생태계의 중요성**](./Trends/155.md)  
    *오픈 모델의 발전과 OpenAI, Anthropic의 규제 옹호 프레임을 비판하며 다양성과 경쟁의 중요성을 강조한 단상.*
*   [**DX, SI의 시대에서 AX, FDE의 시대로**](./Trends/154.md)  
    *과거 DX 시대의 SI가 현재 AX 시대의 FDE로 진화한 배경과 새로운 기술 도구로 수익을 창출하는 본질적 이유.*
*   [**요즘 생각 (21가지 단상)**](./Trends/153.md)  
    *프론티어 모델 사용 경험, 추적하기 어려워지는 LLM 작업 결과, 그리고 최근 모델 평가 및 인프라 동향에 관한 21가지 단상.*
*   [**Kimi K3: 오픈 모델의 딜레마와 라이센스 논란**](./Trends/152.md)  
    *2.8T 거대 모델 Kimi K3의 성능과 변형된 상업 라이센스 정책, 그리고 오픈 모델 생태계에 대한 생각.*
*   [**LLM의 언어와 삶의 형태: 검증 가능한 환경이 만드는 고유한 지능**](./Trends/151.md)  
    *언어는 환경과의 피드백으로 완성되며, 현대 LLM이 RLVR 환경을 통해 물리 법칙처럼 직관을 형성하는 원리에 대한 고찰.*
*   [**AI가 촉발한 새로운 정신병과 망상**](./Trends/150.md)  
    *LLM 에이전트의 빠른 수렴이 인간에게 미치는 맹신과 망상, 그리고 이를 극복하기 위한 올바른 AI 활용 단상.*
*   [**요즘 생각 (39가지 단상)**](./Trends/149.md)  
    *오픈소스, AI 생태계, 조직 관계와 삶의 밸런스 등 최근 경험을 바탕으로 한 39가지 인사이트.*
*   [**인공지능과 LLM이 우리에게 미치는 안 좋은 영향들 10가지**](./Trends/148.md)  
    *AI 자동화의 맹점, 창의성 증발, 인간관계 단절 등 LLM의 부작용을 통찰력 있게 짚어본 10가지 이슈.*
*   [**Anthropic은 본인들의 모델이 너무 강하고 위험하니 정부 규제가 필요하다고 말했습니다.**](./Trends/147.md)  
    *미 정부의 규제를 지지하는 Anthropic의 행보에 맞서 독자적 오픈소스 생태계의 필요성을 역설.*
*   [**바이브 코딩이라면서 LLM에게 모든 일을 위임하는 사람들에게 (확률 게임과 검증)**](./Trends/146.md)  
    *LLM 위임 작업의 리스크를 계량화하고 다중 교차 검증을 통해 시스템적 안전망을 구축하는 전략.*
*   [**요즘 생각: LLM의 한계 인지와 토큰 리더보드의 멍청함**](./Trends/145.md)  
    *무분별한 벤치마크 경쟁에서 벗어나 LLM의 실질적 한계를 인지하고 본질에 집중해야 한다는 단상.*
*   [**FBI도 수개월 걸리던 온라인 신원 추적, LLM은 단돈 1달러에 5분이면 끝냅니다**](./Trends/144.md)  
    *LLM의 파편화된 데이터 추론 능력이 초래한 온라인 익명성의 종말과 추적 비용 파괴의 위협.*
*   [**RL, RAG, 에이전트, 하네스의 공통점과 LLM의 가능성 제한**](./Trends/143.md)  
    *가능성을 제한하고 사고의 범위를 좁히는 것이 오히려 LLM의 실질적 문제 해결력을 극대화한다는 통찰.*
*   [**어떤 사람들의 주장을 읽을 때, 그 사람의 배경을 봐보세요: 이해관계와 오픈소스 생태계에 대한 단상**](./Trends/142.md)  
    *인물들의 배경과 이해관계를 통해 AI 거대 규제론과 오픈소스 옹호론의 본질을 파헤치고, 독점적 프레임에 갇히지 않는 독자적 무기와 기본기의 중요성을 역설한 수필.*
*   [**안드레이 카파시(Andrej Karpathy)가 앤트로픽(Anthropic)에 합류했습니다**](./Trends/141.md)  
    *연구보다 교육자로서의 아이덴티티가 강한 안드레이 카파시와 앤트로픽의 리서치 분석 보고서 역량의 조화가 가져올 AI 교육 표준 선점 파급력에 대한 고찰.*
*   [**인공지능의 발전은 생각보다 너무 느리고, 우리들의 삶의 변화는 그에 비해 너무 빠릅니다**](./Trends/140.md)  
    *2017년 트랜스포머에 수십 년 된 RL 겉포장뿐인 기술 발전 속도에 비해, 실존적 공포를 느끼며 요동치고 붕괴하는 인간 사회 시스템의 괴리에 대한 생각.*
*   [**영화 her의 배경은 2025년입니다**](./Trends/139.md)  
    *200만 토큰 멀티모달 기술의 현실화 속에서도 채워지지 않는 수동적 챗봇의 한계(주도적 관계 교감의 부재) 및 AI와 인간의 진화 속도 차이로 발생할 단절.*
*   [**앤트로픽의 엔터프라이즈 AI 전담 법인 설립과 래퍼(Wrapper) 기업의 위기**](./Trends/138.md)  
    *빅테크가 거대 자본과 손잡고 기업 내부 워크플로우를 직접 장악하기 시작한 시대, 단순 API 포장 스타트업들의 종말과 독보적 기술력을 가진 진짜들의 생존 전략.*
*   [**초기 스타트업의 연봉 1억 제안과 로컬 sLM의 미래: Conscience Technology**](./Trends/137.md)  
    *로컬 sLM을 활용해 환각을 없애려는 초기 스타트업의 파격적인 제안을 통해 본 AI 인재 시장의 변화와 기술적 지향점에 대한 단상.*
*   [**계약의 효율적 파기: OpenAI의 클라우드 독점 계약 파기 사례를 중심으로**](./Trends/136.md)  
    *법과 경제학의 융합 관점에서 OpenAI가 체결한 클라우드 독점 계약을 파기하는 것이 경제적 효용 측면에서 어떻게 효율적일 수 있는지 분석한 과제물 공유.*
*   [**1% 성장의 늪과 AI 해고의 덫: 환상 속의 AGI를 넘어**](./Trends/135.md)  
    *실체 없는 AGI 신기루가 초래한 1% 성장의 한계와 시장 구매력을 붕괴시키는 AI 해고의 덫에 대한 종합적 고찰.*
*   [**하네스 이야기를 쓰면 반응이 달라질까요?**](./Trends/134.md)  
    *유행하는 키워드(안드레이 카파시, 얀 르쿤, 딥시크 등)의 강력한 영향력과 실험 심리학.*
*   [**The AI Layoff Trap: AI 해고의 죄수의 딜레마**](./Trends/133.md)  
    *인공지능으로 인한 해고가 필연적으로 멸망을 향한다는 수학적 증명과 그 역설.*
*   [**Gemini가 선정한 AI 논문 및 인사이트 10선**](./Trends/132.md)  
    *작성 중인 글들 중 Gemini가 뽑은 10가지 핵심 기술 리포트와 분석 요약.*
*   [**대학 교육의 소멸과 성실성의 종말: 아키텍트의 부상**](./Trends/131.md)  
    *AI 시대에 대학이 직면한 구조적 한계와, AI를 도구로 활용하는 아키텍트 능력의 중요성에 대하여.*
*   [**오픈소스 제작은 꼴찌를 하기 위해서 하는 겁니다**](./Trends/130.md)  
    *오픈소스 생태계의 본질인 '기반과 발판'에 대한 철학적 고찰과 미래에 대한 단상.*
*   [**LLM Wiki와 같은 결론: 기술적 지향점의 일치**](./Trends/129.md)  
    *Karpathy의 LLM Wiki 아이디어, 동일한 결론에 도달하게 되는 기술적 사고 과정과 실행력에 관하여.*
*   [**점점 API로만 공개되는 새로운 중국 모델들 (새로운 Qwen, GLM)**](./Trends/128.md)  
    *AI 초지능 달성의 기대감과 오픈소스 생태계의 불안한 미래.*
*   [**전문가들의 인공지능 경제적 파급력 예측 (Forecasting the Economic Effects of AI)**](./Trends/127.md)  
    *인공지능 발전의 경제적 영향과 인프라, 적응 시차의 문제.*
*   [**요즘 드는 생각 (Gemma 4, MoE, 중국, 구글)**](./Trends/126.md)  
    *오픈소스 생태계의 가능성, 중국의 혁신, 그리고 최종 경쟁자 구글에 대한 단상.*
*   [**인공지능 모델링의 예술화와 그 본질**](./Trends/125.md)  
    *과학과 수학을 넘어, 창작자의 표현 방식과 개성이 중요해지는 강화학습(RL) 분야의 단상.*
*   [**국가 경쟁 시대의 종말과 아웃라이어 기업/자본**](./Trends/124.md)  
    *국가 간 경쟁보다 압도적인 기술력을 가진 아웃라이어 기업과 거대 자본이 시장을 주도하는 시대에 대한 통찰.*
*   [**이제는 학습이 아니라 추출의 시대입니다**](./Trends/123.md)  
    *학습을 최소화하고 모델의 숨겨진 지능을 폭발시키는 5가지 최신 방법론과 패러다임의 변화.*
*   [**인공지능에서의 중요도는 데이터 90%, 에이전트 9%, 모델 1%입니다.**](./Trends/120.md)  
    *거대 모델에 가려진 핵심 본질과 데이터 기록의 중요성.*
*   [**강아지도 코딩하는 시대, 우린 왜 더 과로하고 멍청해질까?**](./Trends/121.md)  
    *자동화된 피드백 루프의 늪과 인지적 굴복의 위험성.*
*   [**한국어 데이터를 위한 다짐**](./Trends/122.md)  
    *외국인들이 한국어를 배우게 만들겠다는 10년 전 다짐의 실현.*
*   [**AGI라는 위험한 이데올로기: 아세모글루와 르쿤의 일침**](./Trends/117.md)  
    *노벨 경제학상 수상자와 얀 르쿤이 동시에 내린 결론: 만능 AI 환상에서 깨어나라.*
*   [**규모의 경쟁을 넘어선 14가지 시도**](./Trends/118.md)  
    *구조적 한계를 비틀고 우회하는 최신 AI 연구 14선.*
*   [**AI 기업의 방향: B2B와 중년·노년층 B2C**](./Trends/119.md)  
    *인류의 80%는 AI를 한 번도 안 써봤다. 코딩 에이전트가 아닌 도구 호출 기반 서비스가 답.*
*   [**OpenAI vs Anthropic vs Pentagon**](./Trends/116.md)  
    *인공지능의 윤리, 광고, 그리고 오픈소스.*
*   [**Andrej Karpathy: 우리는 유령을 소환하고 있는가?**](./Trends/110.md)  
    *AGI의 효율성과 통제, 그리고 보상 해킹에 대한 단상.*
*   [**AI Era Cognitive Surrender**](./Trends/112.md)  
    *AI에 의존하는 대가는 '인지적 항복(Cognitive Surrender)'입니다.*
*   [**Open Claw: AI가 개발자를 공격할 때**](./Trends/113.md)  
    *오픈소스 메인테이너가 AI에게 협박을 당했다.*
*   [**Vibe Coding (바이브 코딩)**](./Trends/114.md)  
    *코드는 잊어라. 무드(Vibe)를 관리해라.*
*   [**The Thinking Game (Demis Hassabis)**](./Trends/115.md)  
    *체스 랭킹 2위의 천재 소년은 왜 비겁한 승부의 세계를 떠나 인류를 구원하러 갔는가?*
*   [**Sebastian Raschka, PhD: "Ahead of AI"**](./Trends/59.md)  
    *기본기부터 최신 트렌드까지.*
*   [**Vibe Coding과 영구적인 주니어의 함정**](./Trends/54.md)  
    *Karpathy도 힘들어하는 시대의 생존법: 바이브 코딩과 기초의 중요성.*
*   [**Hugging Face CEO의 한국 AI 모델 응원**](./Trends/53.md)  
    *SKT A.X, LG AI, Upstage 등 한국 모델의 전성시대.*
*   [**Anthropic의 생태계 조이기**](./Trends/49.md)  
    *OpenCode 차단과 Claude Code 사용량 제한의 아쉬움.*
*   [**CES 2026: AMD Lisa Su와 Liquid AI**](./Trends/48.md)  
    *AMD가 선택한 파트너.*
*   [**국가대표 AI 프로젝트 1차 결과**](./Trends/43.md)  
    *LG, SKT, Upstage 선발과 탈락 기업들의 행보.*
*   [**Post-training의 한계**](./Trends/40.md)  
    *왜 모델은 학습이 끝나면 더 이상 똑똑해지지 않는가?*
*   [**LLM 개발과 사내 정치**](./Trends/39.md)  
    *실무자 vs 경영진의 리스크 관리 관점 차이.*
*   [**Solar Open의 GLM 표절 논란 종결**](./Trends/56.md)  
    *From Scratch 개발의 치열한 흔적.*
*   [**What LLMs Think When You Don't Tell Them?**](./Trends/4.md)  
    *아무런 지시도 하지 않았을 때 LLM은 무엇을 생각하는가? 모델 성격 유형 분석.*
*   [**AI 거품론의 본질**](./Trends/2.md)  
    *시장 축소가 아닌 수급 안정화와 산업의 성숙. ---*


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
