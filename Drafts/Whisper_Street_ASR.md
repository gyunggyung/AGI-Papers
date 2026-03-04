---
id: Whisper_Street_ASR
category: On_Device
title: "Whisper Fine-tuning for Proper Noun ASR"
---
# Whisper Fine-tuning: 고유명사 인식의 처참한 실패와 해결

> **WER(Word Error Rate)이 낮다고 현실 세계의 중요한 태스크를 잘 수행하는 것은 아니다. 합성 데이터 1,000개로 Whisper를 파인튜닝하여 60%의 오류율을 개선.**

## 1. 문제 제기: 상용 ASR의 고유명사 인식 실패

Whisper(OpenAI), Deepgram, Google, Microsoft 등 **15개의 최신 ASR 모델**을 대상으로 샌프란시스코 길 이름 인식 성능을 평가한 결과:

*   **평균 44%의 길 이름이 잘못 전사(Transcription)**되었다.
*   29개 길 이름 정답지를 프롬프트에 통째로 제공해도 정확도는 **76%에 그쳤다**.
*   즉, 컨텍스트를 몰라서가 아니라 **모델 자체의 음향적 분별력이 부족**하다.

## 2. 인구통계학적 불균형

*   비영어 모국어 화자의 정확도는 영어 모국어 화자보다 **18% 낮았다** (46% vs 64%).
*   경제적 손실: 영어 화자 평균 1.26마일, 비영어 화자 2.4마일 오배차 → 연간 약 **43,500시간의 불필요한 대기 시간** 발생.

## 3. 해결책: 오픈소스 TTS 합성 데이터 파인튜닝

### 핵심 아이디어: 스타일 트랜스퍼
오픈소스 TTS 모델 **XTTS**에 외국어(예: 스페인어)로 말하게 하면서 중간에 영어 길 이름을 섞어, 외국어 억양이 자연스럽게 입혀진 오디오 데이터를 추출.

### 결과
*   **1,000개 미만의 합성 데이터**만으로 Whisper-base 파인튜닝 (batch size 16, lr 1e-5).
*   비영어 화자의 길 이름 인식 오류율 **약 60% 개선**.

## 4. 공개 자료
*   **데이터셋**: [sf_streets (HuggingFace)](https://huggingface.co/datasets/kzhou/sf_streets)
*   **코드**: [sf_streets_public (GitHub)](https://github.com/kzhou-cloud/sf_streets_public)
*   베이스라인 음성: Mozilla Common Voice 활용.

---

## 🔗 References
*   **Dataset**: [kzhou/sf_streets](https://huggingface.co/datasets/kzhou/sf_streets)
*   **GitHub**: [kzhou-cloud/sf_streets_public](https://github.com/kzhou-cloud/sf_streets_public)
