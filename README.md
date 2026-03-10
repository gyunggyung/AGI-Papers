# 🚀 AGI-Papers

![AGI-Papers](https://img.shields.io/badge/AGI--Papers-2026-blue?style=for-the-badge&logo=github)
![Topic](https://img.shields.io/badge/Topic-AGI%20%7C%20Agents%20%7C%20Trends-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-important?style=for-the-badge)

> **Toward Artificial General Intelligence (AGI) in 2026.**  
> A curated archive of breakthroughs in **Agents**, **Architecture**, **Training**, **RAG**, and **On-Device AI**.

## 📌 Introduction

2026년, AGI에 그 어느 때보다 가까운 시대가 도래했습니다.  
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

---

## <a id="agents"></a>🤖 Agents

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

*   [**LLM 학습 효율화 방안: 인간의 언어 습득 방식**](./Pre_Training/97.md)  
    *인간의 언어 습득 방식을 모방한 점진적 어휘 학습법(Vocabulary Curriculum Learning).*

*   [**RoPE가 정보를 유실하고 있다?**](./Pre_Training/79.md)  
    *푸단대 연구진의 충격적인 발견과 해결책.*



*   [**CALM: Continuous Autoregressive Language Models**](./Pre_Training/34.md)  
    *한 글자씩 타이핑하는 LLM을 넘어, 4개씩 생성하는 연속 벡터 예측.*



## <a id="post-training"></a>🎯 Post-Training

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

*   [**Preference Leakage: A Contamination Problem in LLM-as-a-Judge**](./Evaluation/105.md)  
    *LLM 평가자가 자신의 패밀리 모델을 편애하는 '선호도 유출' 문제.*
*   [**ADR-Bench 전문가 평가**](./Evaluation/55.md)  
    *DeepSeek-v3.2를 압도한 효율적인 에이전트 모델.*

## <a id="rag--knowledge"></a>🗂️ RAG & Knowledge

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
    *시장 축소가 아닌 수급 안정화와 산업의 성숙.*

---

## 📬 Connect

*   [<img src="https://img.shields.io/badge/LinkedIn-Kiwoong Yeom-blue?style=flat&logo=linkedin&logoColor=white" />](https://www.linkedin.com/in/kiwoong-yeom) [<img src="https://img.shields.io/badge/GitHub-gyunggyung-black?style=flat&logo=github&logoColor=white" />](https://github.com/gyunggyung)
*   📧 **Contact:** newhiwoong@gmail.com

---
*Disclaimer: The views and opinions expressed in these reviews are those of the author and do not necessarily reflect the official policy or position of any other agency, organization, employer or company.*
