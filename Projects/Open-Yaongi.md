---
id: Open-Yaongi
category: Projects
title: Open-Yaongi Project
---

SK Telecom가 Apache 2.0 라이선스로 모델을 공개했습니다. NC AI가 학습 데이터 전체와 MIT 라이선스로 모델을 공개했습니다. 이번에는 제가 학습 데이터 뿐 아니라, 학습 코드 전체를 공개하겠습니다. 단순히 GPU 부자들만 LLM을 만드는 게 아니라, 효율적인 아키텍처와 최적화 기술만 있다면 누구나 자신만의 모델을 만들 수 있다는 것을 보여주고 싶습니다.

그래서 학습 코드, 데이터 처리 스크립트, 토크나이저, 그리고 H100 도커 파일까지 프로젝트의 모든 것을 Apache 2.0 라이선스로 공개합니다. 이 코드를 GPT나 Claude에게 "내 데이터로 바꿔줘", "이 아키텍처로 바꿔줘"라고 지시만 하면, 여러분만의 LLM을 Pre-training부터 즉시 만들 수도 있습니다.

🎯 **프로젝트 목표: Open-Yaongi (야옹이)**
한국어, 수학, 코딩 능력에 특화된 52 Layers 4B(Active 0.6B) 규모의 소형 언어 모델(sLLM)을 H100(혹은 B200) 단 한 대로 빠르고 효율적으로 학습하는 것이 목표입니다.

🧠 **핵심 기술 및 검증 과정**

1. **Mamba-2 한국어 동작 확인** (@experiments/mamba2_ko_test)
차세대 아키텍처인 Mamba-2(SSM)를 파이토치로 직접 구현하고, KoAlpaca 데이터셋을 통해 한국어 생성 및 학습이 원활하게 동작함을 검증했습니다. Transformer보다 메모리 효율이 압도적입니다.

2. **하이브리드 MoE & Teon 최적화 검증** (@experiments/colab_moe_test)
Google Colab(T4 GPU) 환경에서 실험을 진행했습니다.
- Architecture: Mamba-2와 Attention(GQA)을 결합한 Nemotron-3 구조에 Mixture of Experts(MoE)를 적용했습니다.
- Optimizer: 최신 Teon(Muon) 옵티마이저를 적용하여 AdamW 대비 2배 빠른 수렴 속도와 메모리 절감 효과를 확인했습니다.

3. **H100 실전 학습 준비 완료** (@experiments/h100_moe_training)
이제 본게임을 시작합니다. VESSL AI의 H100 환경에서 실제 대규모 학습을 수행하기 위한 모든 준비를 마쳤습니다.
- Full Pipeline: 데이터 다운로드 -> 토크나이저 학습(32k Vocab) -> 모델 학습까지 원클릭으로 돌아가는 스크립트.
- Optimization: FP8 가속과 Mamba-SSM 커널을 활용한 극한의 최적화.

🔥 **다음 스텝**
아직 학습은 시작 전입니다. 이번 설 연휴 기간 동안, Vessl AI의 Simon Lee님께서 지원해주신 크레딧을 활용해 실제 학습을 완료하고 그 결과(모델 체크포인트)까지 공개할 예정입니다.

CPU 환경에서는 Active 0.6B 크기의 모델이 실제로 쓸만하다고 생각합니다. MoE 모델이 비효율적인 부분이 있겠지만, 유사한 아키텍처를 사용하는 NVIDIA의 Nemotron-3-Nano가 동급 크기의 Qwen 모델 대비 3.3배 빠른 것을 생각하면, 속도에 문제는 없을 것 같습니다. 

그런데 솔직히 받은 크레딧으로 모델을 잘 만들 수 있을지 모르겠습니다. 환경 설정에서 어려움을 겪지 않을까 하는 걱정도 있습니다. 동시에 제가 만든 아키텍처 자체가 검증되지 않은 부분이 많아서 잘 될지도 확신은 없습니다. 

그럼에도 이렇게 공개를 하고 시도를 한다는 것 자체에서 큰 의미가 있습니다. 제가 실패하면 누군가가 성공을 시키는 것이 오픈소스 정신이니, 모든 것을 공개한 시점에서 저는 큰 일을 해냈습니다.

🔗 **GitHub:** [https://github.com/gyunggyung/Open-Yaongi](https://github.com/gyunggyung/Open-Yaongi)

당연한 이야기지만, 위 코드로는 아마 당장은 학습이 안 될 가능성이 높습니다. 그래도 학습을 계속하면서 코드도 지속적으로 업데이트하여 GitHub에 올릴 예정입니다.
