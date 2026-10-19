# Daily AI Paper Recommendations

> **Date:** 2026-10-20
> **Module:** Module 5: TTS and STT Model Development
> **Topic:** Neural Text-to-Speech

> 이번 사이클에서는 이전에 다룬 Tacotron 1/2, VITS, FastSpeech 1/2, HiFi-GAN, Glow-TTS, Grad-TTS, FastPitch, Transformer TTS, NaturalSpeech 1/2와 겹치지 않도록 새로운 클래식 논문을 골랐습니다.

---

## Paper 1 (Classic): Deep Voice 3: Scaling Text-to-Speech with Convolutional Sequence Learning
- **Authors:** Wei Ping, Kainan Peng, Andrew Gibiansky, Sercan O. Arik, Ajay Kannan, Sharan Narang, Jonathan Raiman, John Miller
- **Year:** 2017 (ICLR 2018)
- **arXiv:** https://arxiv.org/abs/1710.07654
- **PDF:** [./deep-voice-3-ping-2017.pdf](./deep-voice-3-ping-2017.pdf)
- **Citation Count:** ~900+

### 요약
Baidu가 RNN 대신 완전 합성곱(fully-convolutional) 구조와 어텐션으로 만든 TTS 모델입니다. RNN 기반 Tacotron보다 학습이 훨씬 빠르고, 2,000명 이상의 화자와 800시간 이상의 데이터로 학습해 멀티 스피커 TTS를 대규모로 확장할 수 있음을 보였습니다.

### 핵심 기여
- 인코더·디코더를 모두 합성곱으로 만들어 병렬 학습이 가능해졌고, 학습 속도가 크게 빨라짐
- 화자 임베딩을 여러 층에 넣어 하나의 모델로 2,000명 이상의 목소리를 생성
- 어텐션이 단어를 건너뛰거나 반복하는 오류를 줄이는 기법(단조 어텐션 제약, 위치 인코딩 등) 정리
- 하루 천만 건 수준의 요청을 GPU 서버 한 대로 처리하는 서빙 최적화 방법 공개
- WORLD, Griffin-Lim, WaveNet 보코더를 같은 조건에서 비교

### 이 논문이 중요한 이유
"TTS도 RNN 없이 만들 수 있다"는 것을 보여, 이후 FastSpeech 같은 비자기회귀·병렬 TTS로 넘어가는 다리 역할을 했습니다. 특히 **멀티 스피커 확장**과 **프로덕션 서빙**을 논문 수준에서 다뤘다는 점에서, 모델 성능뿐 아니라 운영 비용까지 고민해야 하는 AI 엔지니어에게 좋은 참고서입니다.

### 사전 지식
- Seq2Seq + 어텐션 (Bahdanau Attention)
- 1D 합성곱, Gated Linear Unit(GLU)
- Mel-spectrogram과 보코더 개념 (Griffin-Lim, WaveNet)

### 관련 논문
- [Deep Voice: Real-time Neural Text-to-Speech (Arik et al., 2017)](https://arxiv.org/abs/1702.07825)
- [Deep Voice 2: Multi-Speaker Neural Text-to-Speech (Arik et al., 2017)](https://arxiv.org/abs/1705.08947)
- [Convolutional Sequence to Sequence Learning (Gehring et al., 2017)](https://arxiv.org/abs/1705.03122)
- [Efficiently Trainable Text-to-Speech System Based on Deep Convolutional Networks with Guided Attention / DCTTS (Tachibana et al., 2017)](https://arxiv.org/abs/1710.08969)

### 실무 적용
AI 더빙처럼 수많은 화자 목소리를 하나의 모델로 서비스해야 할 때, "화자 임베딩 + 공유 백본" 구조는 지금도 기본 설계입니다. 또 논문의 서빙 섹션은 GPU 1대당 처리량(throughput)과 지연시간(latency) 사이의 균형을 어떻게 잡는지 보여주므로, TTS API의 단가를 계산할 때 참고할 수 있습니다.

---

## Paper 2 (Classic): WaveGlow: A Flow-based Generative Network for Speech Synthesis
- **Authors:** Ryan Prenger, Rafael Valle, Bryan Catanzaro
- **Year:** 2018 (ICASSP 2019)
- **arXiv:** https://arxiv.org/abs/1811.00002
- **PDF:** [./waveglow-prenger-2018.pdf](./waveglow-prenger-2018.pdf)
- **Citation Count:** ~1,500+

### 요약
NVIDIA가 만든 보코더로, mel-spectrogram을 받아 음성 파형을 만듭니다. Glow(정규화 흐름, normalizing flow)와 WaveNet의 아이디어를 합쳐, 자기회귀 없이 한 번에 파형을 생성합니다. V100 GPU에서 500kHz 이상(실시간보다 20배 이상 빠름)으로 고품질 음성을 만듭니다.

### 핵심 기여
- 자기회귀 없이 병렬로 파형을 생성해 WaveNet의 느린 속도 문제를 해결
- 손실 함수가 "데이터의 우도(likelihood) 최대화" 하나뿐이라 학습이 단순하고 안정적 (GAN이나 증류 불필요)
- 단일 네트워크로 WaveNet 수준에 가까운 음질(MOS) 달성
- PyTorch 코드 공개로 Tacotron 2 + WaveGlow가 한동안 업계 표준 파이프라인이 됨

### 이 논문이 중요한 이유
TTS는 "텍스트 → mel"(음향 모델)과 "mel → 파형"(보코더) 두 단계로 나뉘는데, WaveGlow는 보코더 단계를 실시간으로 만든 대표 논문입니다. 이후 HiFi-GAN(GAN 계열), Glow-TTS·VITS(흐름 계열)가 모두 이 논문의 영향을 받았습니다. 흐름 모델은 최근 Flow Matching(F5-TTS 등)으로 다시 이어지므로 개념을 잡아두면 좋습니다.

### 사전 지식
- 정규화 흐름(Normalizing Flow)과 가역 변환(invertible transform), 야코비안 행렬식
- WaveNet의 dilated convolution
- Mel-spectrogram 추출 과정 (STFT)

### 관련 논문
- [Glow: Generative Flow with Invertible 1x1 Convolutions (Kingma & Dhariwal, 2018)](https://arxiv.org/abs/1807.03039)
- [Parallel WaveNet: Fast High-Fidelity Speech Synthesis (van den Oord et al., 2017)](https://arxiv.org/abs/1711.10433)
- [FloWaveNet: A Generative Flow for Raw Audio (Kim et al., 2018)](https://arxiv.org/abs/1811.02155)
- [HiFi-GAN (Kong et al., 2020)](https://arxiv.org/abs/2010.05646)

### 실무 적용
보코더는 TTS 전체 지연시간의 큰 비중을 차지합니다. 실무에서는 WaveGlow 자체보다 더 가볍고 빠른 HiFi-GAN·Vocos·BigVGAN을 주로 쓰지만, "음질 vs 속도 vs 모델 크기" 트레이드오프를 판단하는 기준점으로 WaveGlow가 자주 비교 대상이 됩니다. 실시간 스트리밍 더빙이나 아바타 립싱크처럼 지연에 민감한 서비스를 설계할 때 보코더 선택 기준을 잡는 데 도움이 됩니다.

---

## Paper 3 (Recent): Seed-TTS: A Family of High-Quality Versatile Speech Generation Models
- **Authors:** Philip Anastassiou, Jiawei Chen, Jitong Chen, Yuanzhe Chen, Zhuo Chen, et al. (ByteDance Seed Team)
- **Year:** 2024
- **arXiv:** https://arxiv.org/abs/2406.02430
- **PDF:** [./seed-tts-anastassiou-2024.pdf](./seed-tts-anastassiou-2024.pdf)
- **Citation Count:** ~300+

### 요약
ByteDance가 공개한 대규모 자기회귀 TTS 모델 계열입니다. 짧은 참조 음성만으로 그 사람의 목소리를 따라 하는 zero-shot TTS에서 사람 음성과 구별하기 어려운 수준의 화자 유사도·자연스러움을 보고했습니다. 완전 확산(diffusion) 기반 변형인 Seed-TTS-DiT도 함께 제안했습니다.

### 핵심 기여
- 음성 토큰 언어 모델 + 확산 기반 디코더 구조로 사람 수준의 zero-shot 음성 생성
- 자기 증류(self-distillation)로 음색과 내용 등 음성 요소를 분리 → 음색 변환에 활용
- 강화학습(RL)으로 발음 정확도(WER), 화자 유사도, 감정 제어 성능 개선
- 음소 길이 예측 없이 끝까지 생성하는 비자기회귀 변형 Seed-TTS-DiT로 음성 편집 가능
- Seed-TTS-eval 벤치마크 공개 → 이후 F5-TTS, CosyVoice 2 등이 표준 평가셋으로 사용

### 이 논문이 중요한 이유
2024년 이후 TTS가 "LLM 방식(토큰 예측) + 확산/흐름 디코더"로 바뀌는 흐름을 가장 잘 보여주는 논문입니다. 특히 **RL로 TTS 품질을 올린다**는 아이디어는 LLM의 RLHF/DPO와 같은 생각을 음성에 적용한 것이라, LLM과 음성을 함께 다루는 AI 엔지니어에게 연결 고리가 됩니다. 또 이 논문의 평가셋이 업계 표준이 되어, 최신 TTS 논문의 숫자를 읽으려면 꼭 알아야 합니다.

### 사전 지식
- Neural audio codec과 음성 토큰 (EnCodec, VALL-E)
- 확산 모델 / Flow Matching의 기본
- 평가 지표: WER, 화자 유사도(SIM), CMOS
- RLHF/DPO 같은 선호 기반 학습 개념

### 관련 논문
- [Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers / VALL-E (Wang et al., 2023)](https://arxiv.org/abs/2301.02111)
- [CosyVoice: A Scalable Multilingual Zero-shot TTS Synthesizer (Du et al., 2024)](https://arxiv.org/abs/2407.05407)
- [E2 TTS: Embarrassingly Easy Fully Non-Autoregressive Zero-Shot TTS (Eskimez et al., 2024)](https://arxiv.org/abs/2406.18009)
- [F5-TTS (Chen et al., 2024)](https://arxiv.org/abs/2410.06885)

### 실무 적용
AI 더빙·아바타 서비스에서 "원래 화자 목소리로 다른 언어 말하기"는 zero-shot TTS와 음색 변환이 핵심입니다. Seed-TTS의 구조(토큰 LM + 확산 디코더 + RL 후처리)는 오픈소스 CosyVoice 2, F5-TTS 등에서도 비슷하게 쓰이므로, 자체 모델을 만들거나 고를 때 설계 기준이 됩니다. 또 Seed-TTS-eval로 후보 모델을 같은 조건에서 비교해 보는 것이 모델 선정의 출발점이 될 수 있습니다.

---

## 추천 읽기 순서
1. **WaveGlow** (짧음, 4~5쪽) — "mel → 파형" 보코더 단계를 먼저 이해
2. **Deep Voice 3** — "텍스트 → mel" 음향 모델과 멀티 스피커, 서빙까지 전체 파이프라인 파악
3. **Seed-TTS** — 2단계 파이프라인이 "음성 토큰 LM + 확산 디코더"로 어떻게 바뀌었는지 비교하며 읽기

## 핵심 테이크어웨이
- TTS의 발전은 **속도 병목을 없애는 과정**이었습니다: RNN → 합성곱(Deep Voice 3), 자기회귀 보코더 → 병렬 흐름 보코더(WaveGlow).
- 생성 모델 계열(흐름 → GAN → 확산/Flow Matching)이 바뀌어도 "음향 모델 + 디코더"라는 큰 틀은 유지됩니다.
- 최신 TTS는 LLM과 같은 도구를 씁니다: 토큰 예측, 대규모 데이터, 그리고 **강화학습으로 품질 다듬기**.
- 질문 하나 던져보기: *"우리 서비스에서 가장 중요한 지표는 음질(MOS)인가, 지연시간인가, 화자 유사도인가?"* — 답에 따라 고를 아키텍처가 달라집니다.

## 다음 토픽과의 연결
다음 토픽은 **Voice Cloning and Speech Synthesis**입니다. 오늘 본 Deep Voice 3의 화자 임베딩(학습된 화자만 생성)이 Seed-TTS의 zero-shot 방식(처음 듣는 목소리도 따라 함)으로 어떻게 발전했는지가 다음 주제의 출발점입니다. VALL-E 계열의 코덱 언어 모델과 음성 변환(Voice Conversion)을 공부할 때 오늘 내용이 바탕이 됩니다.
