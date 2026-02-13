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
- [🗂️ RAG & Knowledge](#rag--knowledge) : 검색 증강 생성, 지식 그래프, 메모리
- [💻 On-Device AI](#on-device-ai) : 로컬 구동, 엣지 컴퓨팅, 최적화
- [🚀 Projects](#projects) : 직접 구현한 프로젝트 및 실험 결과
- [🔥 Trends & Industry](#trends--industry) : AI 산업의 동향, 인사이트, 주요 뉴스

---

## <a id="agents"></a>🤖 Agents

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

## <a id="architecture"></a>🧠 Architecture

*   [**LLM의 "입력 길이 제곱(N^2)"의 저주**](./Architecture/90.md)  
    *누가 먼저 끊어낼 것인가?*
*   [**Mistral Large 3: 효율성의 극대화**](./Architecture/88.md)  
    *Mistral Large 3 vs Kimi K2: Efficiency vs Scale.*
*   [**표준이 된 V3 아키텍처**](./Architecture/86.md)  
    *Mistral Large 3, Kimi K2 그리고 DeepSeek V3.2 분석.*
*   [**Ai2 Olmo 3**](./Architecture/85.md)  
    *성능보다는 과정의 투명성에 집중한 LLM 연구의 교과서.*
*   [**Liquid AI LFM2-2.6B-Exp 튜닝기**](./Architecture/58.md)  
    *논문 Related Work 섹션을 통째로 생성하는 도구 제작.*
*   [**Nemotron-3-Nano-30B-A3B**](./Architecture/50.md)  
    *Qwen3보다 빠르고 강력한 Mamba-2 하이브리드 모델.*
*   [**DeepSeek Engram**](./Architecture/44.md)  
    *기억을 효율화하여 연산 낭비를 줄이는 새로운 희소성 축.*
*   [**2026년의 통념 파괴: 90M, 600M 모델**](./Architecture/41.md)  
    *초소형 모델들의 놀라운 지시 이행 능력.*
*   [**Sakana AI RePo**](./Architecture/38.md)  
    *위치 정보를 재설계(Re-position)하라.*
*   [**DeepSeek vs Qwen (A3B MoE)**](./Architecture/35.md)  
    *정반대의 설계 철학 분석.*
*   [**Pau Labarta Bajo's Insight**](./Architecture/27.md)  
    *멀티 에이전트 시스템에 대한 인사이트.*
*   [**Generative Modeling via Drifting**](./Architecture/6.md)  
    *확산 모델의 250단계를 단 1단계(1-step)로 줄여 속도와 품질을 동시에 잡은 혁신.*

## <a id="pre-training"></a>📚 Pre-Training

*   [**LLM 학습 효율화 방안: 인간의 언어 습득 방식**](./Pre_Training/97.md)  
    *인간의 언어 습득 방식을 모방한 점진적 어휘 학습법(Vocabulary Curriculum Learning).*
*   [**Diffusion LLM (100B Parameters)**](./Pre_Training/83.md)  
    *30B 모델보다 2배 빠른 병렬 생성 모델의 등장.*
*   [**RoPE가 정보를 유실하고 있다?**](./Pre_Training/79.md)  
    *푸단대 연구진의 충격적인 발견과 해결책.*
*   [**Python 재귀로 시작하는 1,000만 토큰 시대**](./Pre_Training/60.md)  
    *Recursive Language Models: Python 재귀로 1,000만 토큰 처리하기.*
*   [**Solar Open의 GLM 표절 논란 종결**](./Pre_Training/56.md)  
    *From Scratch 개발의 치열한 흔적.*
*   [**Sakana AI: 위치 정보(Positional Embeddings)는 버려라**](./Pre_Training/47.md)  
    *DroPE: 학습할 때만 위치 정보를 쓰고 실전에서는 버리는 뺄셈의 미학.*
*   [**CALM: Continuous Autoregressive Language Models**](./Pre_Training/34.md)  
    *한 글자씩 타이핑하는 LLM을 넘어, 4개씩 생성하는 연속 벡터 예측.*
*   [**Beyond Transformers 2**](./Pre_Training/13.md)  
    *덩치 경쟁을 넘어 생각과 본질로.*
*   [**What LLMs Think When You Don't Tell Them?**](./Pre_Training/4.md)  
    *아무런 지시도 하지 않았을 때 LLM은 무엇을 생각하는가? 모델 성격 유형 분석.*

## <a id="post-training"></a>🎯 Post-Training

*   [**Gemma 3 모델의 핵심 목표 및 특징**](./Post_Training/98.md)  
    *구글 딥마인드의 최신 멀티모달 모델 분석.*
*   [**Emergent Misalignment**](./Post_Training/96.md)  
    *취약한 코드를 배운 AI의 위험한 일탈.*
*   [**Stabilizing RL with LLMs**](./Post_Training/92.md)  
    *화려한 기교보다 수학적 기본기가 중요한 이유.*
*   [**LFM2 1.2B 기반 한국어-영어 번역기**](./Post_Training/89.md)  
    *LFM2 1.2B 모델로 구글과 알리바바의 4B 모델을 이긴 번역기 제작기.*
*   [**LFM2 번역기 개발기: 핵심 발견 및 성과**](./Post_Training/82.md)  
    *SFT와 RL의 성능 차이 분석 및 Liquid AI 공식 쿡북 등재 소식.*
*   [**Small Language Model for Translation**](./Post_Training/81.md)  
    *Advice for AI engineers.*
*   [**Yann LeCun: World Model의 중요성**](./Post_Training/78.md)  
    *LLM은 물리 세계를 배울 수 없다?*
*   [**Liquid AI LFM2-1.2B 튜닝 실패기**](./Post_Training/76.md)  
    *한국어-영어 번역 RL(GRPO) 학습 실패와 교훈.*
*   [**Sebastian Raschka, PhD: "Ahead of AI"**](./Post_Training/59.md)  
    *기본기부터 최신 트렌드까지.*
*   [**한국어 LLM 학습 데이터의 부재**](./Post_Training/51.md)  
    *Pre-training부터 GRPO까지의 험난한 여정.*
*   [**Anthropic의 생태계 조이기**](./Post_Training/49.md)  
    *OpenCode 차단과 Claude Code 사용량 제한의 아쉬움.*
*   [**GDPO: Multi-reward RL**](./Post_Training/46.md)  
    *GRPO의 약점을 극복한 새로운 강화학습 기법.*
*   [**Detailed balance in LLM-driven agents**](./Post_Training/23.md)  
    *LLM이 물리학의 '최소 작용의 원리'를 따른다는 것을 증명한 연구.*
*   [**AI 거품론의 본질**](./Post_Training/2.md)  
    *시장 축소가 아닌 수급 안정화와 산업의 성숙.*
*   [**iGRPO**](./Post_Training/1.md)  
    *Self-Feedback-Driven LLM Reasoning: 모델이 스스로 만든 초안을 보고 배우는 자가 개선 강화학습.*

## <a id="rag--knowledge"></a>🗂️ RAG & Knowledge

*   [**HippoRAG 2**](./RAG/100.md)  
    *인간의 기억 메커니즘을 모방한 비모수적 연속 학습 (Bio-inspired Continual Learning).*
*   [**DeepSeek-V3 vs V3.2: 아키텍처의 진화**](./RAG/94.md)  
    *아키텍처의 진화와 기술적 목표점.*
*   [**From Code Foundation Models to Agents**](./RAG/84.md)  
    *Code Foundation Model에서 자율 코딩 에이전트로의 진화 청사진.*
*   [**ADR-Bench 전문가 평가**](./RAG/55.md)  
    *DeepSeek-v3.2를 압도한 효율적인 에이전트 모델.*
*   [**vLLM의 승리: 압도적인 속도**](./RAG/29.md)  
    *표준이 되기까지.*
*   [**RAG & Agent Memory 4선**](./RAG/10.md)  
    *GraphSearch, S-RAG, xMemory 등 최신 논문 소개.*

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

### 💭 Insights & Essays
*   [**최근 구현한 AI 프로젝트 및 성과**](./Projects/7.md)  
    *Gemini-Claw로 구현한 맥킨지 스타일 보고서 및 PPT 자동 생성.*
*   [**Gemini-Claw 성능 vs 보안**](./Projects/12.md)  
    *LLM 에이전트의 위험한 잠재력.*
*   [**AI에 대한 두려움 vs 흥미**](./Projects/17.md)  
    *OpenClaw, 환각 인용, Vibe Coding 현상에 대한 단상.*

## <a id="trends--industry"></a>🔥 Trends & Industry

*   [**Hugging Face CEO의 한국 AI 모델 응원**](./Trends/53.md)  
    *SKT A.X, LG AI, Upstage 등 한국 모델의 전성시대.*
*   [**CES 2026: AMD Lisa Su와 Liquid AI**](./Trends/48.md)  
    *AMD가 선택한 파트너.*
*   [**국가대표 AI 프로젝트 1차 결과**](./Trends/43.md)  
    *LG, SKT, Upstage 선발과 탈락 기업들의 행보.*
*   [**Post-training의 한계**](./Trends/40.md)  
    *왜 모델은 학습이 끝나면 더 이상 똑똑해지지 않는가?*
*   [**LLM 개발과 사내 정치**](./Trends/39.md)  
    *실무자 vs 경영진의 리스크 관리 관점 차이.*

---

## 📬 Connect

*   [<img src="https://img.shields.io/badge/LinkedIn-Kiwoong Yeom-blue?style=flat&logo=linkedin&logoColor=white" />](https://www.linkedin.com/in/kiwoong-yeom) [<img src="https://img.shields.io/badge/GitHub-gyunggyung-black?style=flat&logo=github&logoColor=white" />](https://github.com/gyunggyung)
*   📧 **Contact:** newhiwoong@gmail.com

---
*Disclaimer: The views and opinions expressed in these reviews are those of the author and do not necessarily reflect the official policy or position of any other agency, organization, employer or company.*
