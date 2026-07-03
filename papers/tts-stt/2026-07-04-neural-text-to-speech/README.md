# Daily AI Paper Recommendations

> **Date:** 2026-07-04
> **Module:** Module 5: TTS and STT Model Development
> **Topic:** Neural Text-to-Speech

---

## Paper 1 (Classic): Grad-TTS: A Diffusion Probabilistic Model for Text-to-Speech
- **Authors:** Vadim Popov, Ivan Vovk, Vladimir Gogoryan, Tasnima Sadekova, Mikhail Kudinov
- **Year:** 2021
- **arXiv:** https://arxiv.org/abs/2105.06337
- **PDF:** [./grad-tts-popov-2021.pdf](./grad-tts-popov-2021.pdf)
- **Citation Count:** ~900회 (2026년 기준 추정)

### 요약
Grad-TTS는 확산 확률 모델(Diffusion Probabilistic Model)을 텍스트-투-스피치에 처음으로 본격 적용한 대표 논문이다. 인코더가 예측한 사전 분포(prior)에서 시작해 스코어 기반 디코더가 노이즈를 점진적으로 멜-스펙트로그램으로 변환하며, Monotonic Alignment Search(MAS)로 텍스트와 정렬한다. 확률 미분방정식(SDE) 프레임워크로 음질과 추론 속도 사이의 트레이드오프를 스텝 수로 명시적으로 조절할 수 있다.

### 핵심 기여
- 확산 모델을 TTS에 도입해, 표준 정규분포가 아닌 인코더가 예측한 데이터 의존적 사전 분포에서 역확산을 시작하도록 일반화했다.
- Glow-TTS의 Monotonic Alignment Search를 결합해 외부 정렬기(aligner) 없이 텍스트-음성 정렬을 학습한다.
- 역확산 스텝 수만 바꿔 음질과 속도를 실시간으로 조절할 수 있는 유연한 추론 구조를 제시했다.

### 이 논문이 중요한 이유
현재 SOTA TTS(NaturalSpeech 2, E3-TTS, 각종 latent diffusion TTS)의 뿌리가 되는 확산 기반 음성 합성의 출발점이다. AI 엔지니어가 diffusion TTS 계열을 이해하려면 스코어 기반 디코더와 SDE 관점을 반드시 알아야 하며, Grad-TTS가 그 최소 단위 레퍼런스다.

### 사전 지식
- 멜-스펙트로그램, 보코더(HiFi-GAN 등)의 역할
- DDPM/스코어 기반 생성 모델의 forward/reverse process 개념
- Glow-TTS의 Monotonic Alignment Search(MAS)와 duration predictor

### 관련 논문
- [Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search (Kim et al., 2020)](https://arxiv.org/abs/2005.11129)
- [Denoising Diffusion Probabilistic Models (Ho et al., 2020)](https://arxiv.org/abs/2006.11239)
- [Grad-TTS 후속 - NaturalSpeech 2 (Shen et al., 2023)](https://arxiv.org/abs/2304.09116)

### 실무 적용
확산 기반 TTS는 스텝 수 조절로 품질/지연시간을 조정할 수 있어, 서버 사이드 고품질 더빙(스텝 多)과 실시간 응답(스텝 少)을 하나의 모델로 대응한다. AI 더빙/아바타 파이프라인에서 감정·운율 다양성이 중요한 구간에 diffusion decoder를 붙이는 설계의 기준점이 된다.

---

## Paper 2 (Classic): FastPitch: Parallel Text-to-Speech with Pitch Prediction
- **Authors:** Adrian Łańcucki
- **Year:** 2020
- **arXiv:** https://arxiv.org/abs/2006.06873
- **PDF:** [./fastpitch-lancucki-2020.pdf](./fastpitch-lancucki-2020.pdf)
- **Citation Count:** ~800회 (2026년 기준 추정)

### 요약
FastPitch는 FastSpeech 계열의 완전 병렬(non-autoregressive) Transformer TTS에 기본 주파수(F0, pitch) 컨투어 예측을 명시적으로 추가한 모델이다. 각 입력 심볼 단위로 피치를 예측·조건화하여 음질을 SOTA 수준으로 끌어올리는 동시에, 추론 시 피치 컨투어를 직접 조작해 강세·감정·억양을 제어할 수 있게 했다. 멜 합성에서 실시간 대비 900배 이상의 속도를 유지한다.

### 핵심 기여
- 심볼 단위 피치 예측을 도입해 병렬 TTS의 음질을 자기회귀 모델 수준으로 향상시켰다.
- 추론 시 피치를 더하거나 빼는 방식으로 감정·강세를 직관적으로 제어하는 인터페이스를 제공했다.
- FastSpeech의 완전 병렬 Transformer 구조를 유지해 초고속(900×+ RTF) 멜 합성을 달성했다.

### 이 논문이 중요한 이유
운율(prosody) 제어 가능성과 병렬 추론 속도를 동시에 잡은 실전형 TTS의 표준이다. 상용 TTS/더빙 서비스는 대부분 "빠르고 제어 가능한" 합성을 요구하는데, FastPitch는 그 요구를 만족시키는 아키텍처적 청사진을 제공한다.

### 사전 지식
- FastSpeech / FastSpeech 2의 duration predictor와 length regulator
- Transformer 인코더-디코더, 기본 주파수(F0)와 운율의 관계
- forced alignment(예: Tacotron 2 teacher, Montreal Forced Aligner)로 duration 라벨 얻는 과정

### 관련 논문
- [FastSpeech 2: Fast and High-Quality End-to-End Text to Speech (Ren et al., 2020)](https://arxiv.org/abs/2006.04558)
- [Tacotron 2 (Shen et al., 2018)](https://arxiv.org/abs/1712.05884)
- [HiFi-GAN (Kong et al., 2020)](https://arxiv.org/abs/2010.05646)

### 실무 적용
NVIDIA NeMo 등에 포함되어 실제 프로덕션에서 널리 쓰인다. 광고/내레이션 더빙에서 특정 단어에 강세를 주거나 감정 톤을 조절할 때, 피치 컨투어 조작으로 재녹음 없이 후처리가 가능하다. 저지연 실시간 TTS API의 백본으로도 적합하다.

---

## Paper 3 (Recent): MaskGCT: Zero-Shot Text-to-Speech with Masked Generative Codec Transformer
- **Authors:** Yuancheng Wang, Haoyue Zhan, Liwei Liu, Ruihong Zeng, Haotian Guo, Jiachen Zheng, Qiang Zhang, Xueyao Zhang, Shunsi Zhang, Zhizheng Wu
- **Year:** 2024
- **arXiv:** https://arxiv.org/abs/2409.00750
- **PDF:** [./maskgct-wang-2024.pdf](./maskgct-wang-2024.pdf)
- **Citation Count:** ~200회 이상 (2026년 기준 추정, 빠르게 증가 중)

### 요약
MaskGCT는 텍스트-음성 간 정밀한 정렬 정보 없이 동작하는 완전 비자기회귀(NAR) 제로샷 TTS 모델이다. 2단계 구조로, 1단계에서 텍스트로부터 SSL 음성 모델의 시맨틱 토큰을 예측하고, 2단계에서 이 시맨틱 토큰을 조건으로 음향(acoustic) 토큰을 예측한다. 전 과정이 mask-and-predict 패러다임을 따르며, 10만 시간 규모의 야생(in-the-wild) 음성으로 학습해 기존 제로샷 TTS SOTA를 품질·화자 유사도·명료도에서 앞선다.

### 핵심 기여
- duration 예측이나 phoneme-level 정렬 없이 masked generative 방식으로 제로샷 음성 합성을 수행한다.
- 시맨틱 토큰 → 음향 토큰의 2단계 코덱 트랜스포머 구조로 안정성과 자연스러움을 동시에 확보했다.
- 대규모(10만 시간) 데이터 학습으로 자기회귀 코덱 LM(VALL-E 계열) 대비 견고성과 속도를 개선했다.

### 이 논문이 중요한 이유
2024년 이후 제로샷 TTS의 주류 흐름이 자기회귀 코덱 LM(VALL-E)에서 masked/NAR 생성으로 이동하는 흐름을 대표한다. AI 아바타·개인화 음성 서비스에서 몇 초의 레퍼런스만으로 화자를 복제하는 실전 요구를 충족하며, 코덱 토큰 기반 생성의 최신 설계를 학습할 수 있다.

### 사전 지식
- 뉴럴 오디오 코덱(EnCodec, DAC)과 음향 토큰의 개념
- 시맨틱 vs 음향 토큰의 차이, 음성 SSL(HuBERT, w2v-BERT)
- Masked Generative Modeling(MaskGIT), VALL-E식 코덱 LM의 기본 구조

### 관련 논문
- [Neural Codec Language Models / VALL-E (Wang et al., 2023)](https://arxiv.org/abs/2301.02111)
- [CosyVoice 2 (Du et al., 2024)](https://arxiv.org/abs/2412.10117)
- [MaskGIT: Masked Generative Image Transformer (Chang et al., 2022)](https://arxiv.org/abs/2202.04200)

### 실무 적용
짧은 레퍼런스 오디오로 화자를 복제하는 제로샷 더빙/아바타 서비스에 직접 활용된다. NAR 구조 덕분에 자기회귀 모델보다 추론이 빠르고 발음 누락·반복 같은 오류에 강해, 다국어 콘텐츠 현지화 파이프라인의 백본으로 실용적이다.

---

## 추천 읽기 순서
1. **FastPitch** — 병렬 TTS와 운율(피치) 제어의 기본기를 먼저 잡는다.
2. **Grad-TTS** — 확산 기반 디코더로 음질/속도 트레이드오프를 이해한다.
3. **MaskGCT** — 코덱 토큰 + masked 생성으로 제로샷 TTS의 최신 패러다임을 익힌다.

## 핵심 테이크어웨이
- Neural TTS는 "자기회귀 → 병렬(FastPitch) → 확산(Grad-TTS) → 코덱 토큰/masked 생성(MaskGCT)"으로 진화해 왔다.
- 공통 과제는 **정렬(alignment), 운율 제어, 추론 속도, 화자 일반화**이며, 각 논문은 이 중 하나 이상을 진전시킨다.
- 최신 흐름은 명시적 정렬·duration을 제거하고, 대규모 데이터 + 토큰 기반 생성으로 제로샷 화자 복제를 지향한다.

## 다음 토픽과의 연결
다음 토픽인 **Voice Cloning and Speech Synthesis**(Day 13)는 오늘 배운 코덱 토큰 생성(MaskGCT)과 확산 디코더를 화자 복제 관점으로 심화한다. VALL-E, WaveNet 등 생성 백본이 어떻게 few-shot/zero-shot 음성 복제로 확장되는지 이어서 살펴본다.
