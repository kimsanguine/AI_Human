# Daily AI Paper Recommendations

> **Date:** 2026-07-31
> **Module:** Module 5: TTS and STT Model Development
> **Topic:** Neural Text-to-Speech

---

## Paper 1 (Classic): Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions (Tacotron 2)
- **Authors:** Jonathan Shen, Ruoming Pang, Ron J. Weiss, Mike Schuster, Navdeep Jaitly, Zongheng Yang, Zhifeng Chen, Yu Zhang, Yuxuan Wang, RJ Skerry-Ryan, Rif A. Saurous, Yannis Agiomyrgiannakis, Yonghui Wu
- **Year:** 2018
- **arXiv:** [https://arxiv.org/abs/1712.05884](https://arxiv.org/abs/1712.05884)
- **PDF:** [./tacotron2-shen-2018.pdf](./tacotron2-shen-2018.pdf)
- **Citation Count:** ~4,500

### 요약
Tacotron 2는 텍스트로부터 멜 스펙트로그램을 예측하는 시퀀스-투-시퀀스 인코더-디코더(attention 기반)와, 그 멜 스펙트로그램을 실제 파형으로 변환하는 수정된 WaveNet 보코더를 결합한 2단계 신경망 음성 합성 시스템이다. 복잡한 언어학적/음향학적 피처 엔지니어링 없이, 사람 녹음과 구별하기 어려운 수준(MOS 4.53)의 자연스러운 음성을 생성했다. 이후 등장한 대부분의 뉴럴 TTS 파이프라인(acoustic model + vocoder) 구조의 표준을 제시했다.

### 핵심 기여
- 텍스트 → 멜 스펙트로그램(중간 표현) → 파형으로 이어지는 2단계 분리 구조를 확립해, acoustic model과 vocoder를 독립적으로 발전시킬 수 있게 함
- Location-sensitive attention을 활용해 긴 문장에서도 안정적인 정렬(alignment)을 달성
- 멜 스펙트로그램을 조건으로 한 WaveNet 보코더로 사람 녹음에 근접한 MOS 4.53 달성

### 이 논문이 중요한 이유
현대 TTS의 사실상 표준 아키텍처인 "acoustic model + neural vocoder" 2단계 파이프라인을 대중화한 논문이다. AI 엔지니어가 TTS 시스템을 설계하거나 상용 음성 합성 파이프라인을 이해할 때 반드시 알아야 할 출발점이며, 멜 스펙트로그램을 매개 표현으로 쓰는 관행의 근거를 제공한다.

### 사전 지식
- 시퀀스-투-시퀀스 모델과 attention 메커니즘(encoder-decoder)
- 멜 스펙트로그램/STFT 등 오디오 신호의 시간-주파수 표현
- WaveNet(autoregressive raw audio 생성 모델)의 기본 개념

### 관련 논문
- [Tacotron: Towards End-to-End Speech Synthesis (Wang et al., 2017)](https://arxiv.org/abs/1703.10135)
- [WaveNet: A Generative Model for Raw Audio (van den Oord et al., 2016)](https://arxiv.org/abs/1609.03499)

### 실무 적용
Google/AWS/네이버 등의 상용 TTS, 오디오북·내비게이션·가상비서 음성 등에서 acoustic model + vocoder 파이프라인의 기반이 된다. 실무에서는 Tacotron 2의 멜 스펙트로그램 예측 부분을 유지한 채 WaveNet 대신 더 빠른 보코더(HiFi-GAN, WaveGlow)를 붙여 실시간 추론 속도를 확보하는 방식으로 자주 활용된다.

---

## Paper 2 (Classic): Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech (VITS)
- **Authors:** Jaehyeon Kim, Jungil Kong, Juhee Son
- **Year:** 2021
- **arXiv:** [https://arxiv.org/abs/2106.06103](https://arxiv.org/abs/2106.06103)
- **PDF:** [./vits-kim-2021.pdf](./vits-kim-2021.pdf)
- **Citation Count:** ~1,600

### 요약
VITS는 Tacotron 2 계열의 2단계(acoustic model + vocoder) 파이프라인을 하나의 완전한 end-to-end 모델로 통합한 병렬(parallel) TTS이다. 조건부 VAE, normalizing flow, 적대적 학습(GAN), 그리고 monotonic alignment search(MAS)를 결합해 중간 멜 스펙트로그램 없이 텍스트로부터 직접 파형을 생성한다. 확률적 duration 예측으로 같은 문장을 다양한 운율로 발화할 수 있으며, 2단계 시스템을 능가하는 자연스러움을 달성했다.

### 핵심 기여
- 멜 스펙트로그램 중간 단계를 제거한 진정한 end-to-end, 병렬 생성 TTS 구조 제안
- 조건부 VAE + normalizing flow로 잠재 표현의 표현력을 높이고, adversarial training으로 파형 품질 향상
- Monotonic Alignment Search(MAS)와 stochastic duration predictor로 텍스트-음성 정렬 및 운율 다양성 확보

### 이 논문이 중요한 이유
많은 최신 오픈소스/상용 TTS(및 음성 클로닝) 시스템의 백본으로 쓰이는 실질적 표준 모델이다. 별도의 vocoder 학습·정렬 없이 단일 모델로 고품질 음성을 뽑을 수 있어, 엔지니어가 실제 서비스에 TTS를 통합할 때 가장 먼저 검토하는 아키텍처 중 하나다.

### 사전 지식
- VAE(변분 오토인코더)와 ELBO, 잠재 변수 모델
- Normalizing flow와 GAN(적대적 학습)의 기본 원리
- Tacotron 2 계열 2단계 TTS 파이프라인의 한계(멜 → 파형 불일치, 순차적 추론 등)

### 관련 논문
- [Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search (Kim et al., 2020)](https://arxiv.org/abs/2005.11129)
- [HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis (Kong et al., 2020)](https://arxiv.org/abs/2010.05646)

### 실무 적용
Coqui TTS, 다양한 오픈소스 음성 클로닝 프로젝트, 다국어 TTS 서비스의 기반 모델로 널리 채택된다. 화자 임베딩을 조건으로 추가하면 다화자/제로샷 음성 합성으로 확장할 수 있어, 커스텀 보이스 제작이나 더빙 서비스에서 실무적으로 매우 유용하다.

---

## Paper 3 (Recent): F5-TTS — A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching
- **Authors:** Yushen Chen, Zhikang Niu, Ziyang Ma, Keqi Deng, Chunhui Wang, Jian Zhao, Kai Yu, Xie Chen
- **Year:** 2024
- **arXiv:** [https://arxiv.org/abs/2410.06885](https://arxiv.org/abs/2410.06885)
- **PDF:** [./f5-tts-chen-2024.pdf](./f5-tts-chen-2024.pdf)
- **Citation Count:** ~300

### 요약
F5-TTS는 Diffusion Transformer(DiT)와 flow matching을 기반으로 한 완전 비자기회귀(fully non-autoregressive) 제로샷 TTS이다. duration 예측기, 텍스트 인코더, 음소 정렬 같은 복잡한 구성요소를 제거하고, 텍스트를 ConvNeXt로 정제해 입력으로 채워 넣는(padding) 단순한 설계를 취한다. 추론 시 Sway Sampling 전략으로 성능과 속도를 동시에 개선했으며, 10만 시간 규모의 다국어 데이터로 학습해 자연스럽고 표현력 있는 제로샷 음성 복제와 코드 스위칭을 구현했다.

### 핵심 기여
- Flow matching + DiT 구조로, phoneme alignment·duration predictor 없이도 안정적인 제로샷 TTS를 실현
- ConvNeXt 기반 텍스트 표현 정제로 텍스트-음성 정합성을 개선
- 추론 단계 재학습 없이 적용 가능한 Sway Sampling으로 품질과 추론 효율을 동시에 향상

### 이 논문이 중요한 이유
2024년 이후 TTS 흐름이 autoregressive codec LM(VALL-E류)에서 flow matching/diffusion 기반 비자기회귀 모델로 옮겨가는 최전선을 보여준다. 설계가 단순하고 코드/모델이 공개되어 재현성이 높아, 최신 제로샷 TTS를 실무에 도입하려는 엔지니어에게 직접적인 참고점이 된다.

### 사전 지식
- Flow matching / continuous normalizing flow와 diffusion 모델의 기본 개념
- Diffusion Transformer(DiT)와 in-context learning 기반 제로샷 음성 복제
- VITS 등 기존 병렬 TTS의 정렬(duration) 처리 방식과의 차이

### 관련 논문
- [Voicebox: Text-Guided Multilingual Universal Speech Generation at Scale (Le et al., 2023)](https://arxiv.org/abs/2306.15687)
- [E2 TTS: Embarrassingly Easy Fully Non-Autoregressive Zero-Shot TTS (Eskimez et al., 2024)](https://arxiv.org/abs/2406.18009)

### 실무 적용
제로샷 음성 복제(짧은 참조 음성만으로 특정 화자 목소리 합성), 다국어·코드 스위칭 더빙, 실시간성이 중요한 음성 에이전트 등에 적용된다. 공개된 사전학습 가중치를 활용하면 커스텀 보이스 서비스나 AI 더빙 파이프라인을 빠르게 프로토타이핑할 수 있다.

---

## 추천 읽기 순서
1. **Tacotron 2** — 뉴럴 TTS의 표준인 "acoustic model + vocoder" 2단계 구조와 멜 스펙트로그램 매개 표현을 먼저 이해한다.
2. **VITS** — 2단계 파이프라인의 한계를 어떻게 단일 end-to-end 모델로 통합했는지, VAE·flow·GAN의 결합을 학습한다.
3. **F5-TTS** — flow matching·DiT 기반의 최신 제로샷 TTS로, 정렬·duration 예측까지 제거한 2024년 이후의 설계 트렌드를 파악한다.

## 핵심 테이크어웨이
- 뉴럴 TTS는 "텍스트 → 중간 표현(멜) → 파형"의 2단계에서 출발해, VITS의 end-to-end 통합, 나아가 flow matching 기반 비자기회귀 제로샷 모델로 진화해 왔다.
- 아키텍처의 진화 방향은 명확하다: 수작업 피처·명시적 정렬·중간 표현을 점진적으로 제거하고, 데이터 스케일과 생성 모델(GAN → flow/diffusion)의 힘으로 자연스러움과 제어력을 확보하는 것.
- 실무에서는 품질(MOS), 추론 속도(실시간성), 제로샷 복제 능력의 트레이드오프를 고려해 모델을 선택해야 하며, 최근에는 공개 사전학습 모델을 파인튜닝하는 접근이 표준이 되고 있다.

## 다음 토픽과의 연결
다음 토픽인 **Voice Cloning and Speech Synthesis**에서는 VALL-E, WaveNet 등 코덱 기반·자기회귀 음성 생성 모델을 다룬다. 오늘 학습한 Tacotron 2/VITS/F5-TTS의 스펙트로그램·flow matching 계열과 대비하면, 제로샷 음성 복제를 구현하는 두 가지 큰 패러다임(codec LM vs. flow/diffusion)을 균형 있게 이해할 수 있다.
