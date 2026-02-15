---
id: Micro_GPT
category: Architecture
title: Micro GPT Code Analysis
---
# Micro GPT: LLM의 바닥을 보다

> **Andrej Karpathy의 선물. "마법은 없다. 오직 행렬 곱셈과 미분이 있을 뿐."**

## 1. 프로젝트 개요

*   **프로젝트**: `microgpt.py`
*   **개발자**: Andrej Karpathy (전 Tesla AI Director, OpenAI 창립 멤버)
*   **목표**: 딥러닝 프레임워크(PyTorch, TensorFlow) 없이, **순수 파이썬(Pure Python)**만으로 GPT-2를 바닥부터 구현하여 LLM의 투명성을 교육한다.

---

## 2. 코드 뜯어보기 (The Minimalist Masterpiece)

이 코드는 약 200줄 내외입니다. 하지만 그 안에 현대 LLM의 모든 정수가 담겨 있습니다.

### 1) No Dependencies
numpy도 쓰지 않습니다. 오직 `math`, `random` 같은 파이썬 기본 라이브러리만 씁니다. 이는 "도구가 없으면 못 만든다"는 엔지니어의 핑계를 원천 봉쇄합니다.

### 2) Autograd Engine from Scratch
Karpathy는 `Value`라는 클래스를 만들어, 덧셈/곱셈 연산이 일어날 때마다 미분값(Gradient)을 추적하는 **자동 미분 엔진(Backpropagation)**을 직접 구현했습니다.
*   이것을 보면 PyTorch의 `.backward()`가 마법이 아니라, 단순히 **연쇄 법칙(Chain Rule)**을 재귀적으로 호출하는 것임을 깨닫게 됩니다.

### 3) Transformer Architecture
*   **Embeddings**: 글자를 숫자로, 숫자를 벡터로.
*   **Attention**: Query, Key, Value의 상호작용.
*   **RMSNorm**: LayerNorm보다 효율적인 정규화.
*   **Optimizer**: Adam 옵티마이저의 수식 구현.

---

## 3. 왜 이것을 봐야 하는가?

요즘 우리는 `from transformers import AutoModel` 한 줄로 모델을 불러옵니다. 하지만 그 내부에서 무슨 일이 일어나는지 모르면, 모델이 이상하게 동작할 때 고칠 수 없습니다.

`microgpt.py`를 필사(Hand-copying)해보는 것은 AI 엔지니어에게 일종의 **성인식(Rite of Passage)**과 같습니다.
*"이 200줄의 코드가 이해되는 순간, 1조 개의 파라미터를 가진 GPT-4도 더 이상 두렵지 않게 됩니다."*

---

## 4. Links
*   [Karpathy's Gist](https://gist.github.com/karpathy)
*   [Google Colab Notebook](https://colab.research.google.com/github/karpathy/microgpt)
