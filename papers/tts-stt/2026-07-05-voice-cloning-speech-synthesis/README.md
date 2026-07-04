# Daily AI Paper Recommendations

> **Date:** 2026-07-05
> **Module:** Module 5 — TTS and STT Model Development
> **Topic:** Voice Cloning and Speech Synthesis

---

## Paper 1 (Classic): YourTTS: Towards Zero-Shot Multi-Speaker TTS and Zero-Shot Voice Conversion for everyone
- **Authors:** Edresson Casanova, Julian Weber, Christopher Shulby, Arnaldo Candido Junior, Eren Gölge, Moacir Antonelli Ponti
- **Year:** 2021
- **arXiv:** https://arxiv.org/abs/2112.02418
- **PDF:** [./yourtts-casanova-2021.pdf](./yourtts-casanova-2021.pdf)
- **Citation Count:** approx. 800+

### 요약
YourTTS는 VITS를 기반으로 확장하여, 단 몇 초의 참조 음성만으로 처음 보는 화자의 목소리를 합성(zero-shot TTS)하고 음성을 변환(zero-shot voice conversion)할 수 있게 만든 모델이다. 다국어 학습을 도입해 저자원 언어에서도 화자 적응이 가능하며, 1분 미만의 음성으로 파인튜닝해도 SOTA 수준의 화자 유사도를 달성한다.

### 핵심 기여
- VITS 아키텍처에 화자 임베딩과 언어 임베딩을 결합하여, 재학습 없이 새로운 화자·언어로 일반화하는 zero-shot 프레임워크를 제안
- Speaker Consistency Loss(화자 일관성 손실)를 도입해 합성 음성이 목표 화자의 정체성을 유지하도록 강화
- 저자원 언어에서 단일 화자 데이터셋만으로도 다화자 TTS와 음성 변환이 가능함을 실증

### 이 논문이 중요한 이유
AI 엔지니어에게 YourTTS는 "실용적인 zero-shot 음성 복제"의 기준점이다. 대규모 코덱 LM(VALL-E 계열) 이전에, 상대적으로 가벼운 flow 기반 구조로도 소량 데이터 파인튜닝만으로 목소리를 복제할 수 있음을 보여줬다. 온디바이스·저자원 환경에서 음성 서비스를 설계할 때 여전히 강력한 베이스라인이며, 오픈소스(Coqui TTS)로 널리 쓰여 재현·배포가 쉽다.

### 사전 지식
- VITS(조건부 VAE + normalizing flow + adversarial training)의 구조
- 화자 임베딩(d-vector / speaker encoder)과 화자 검증(speaker verification)의 개념
- Mel-spectrogram, monotonic alignment, HiFi-GAN 계열 보코더

### 관련 논문
- [VITS: Conditional Variational Autoencoder with Adversarial Learning (Kim et al., 2021)](https://arxiv.org/abs/2106.06103)
- [Transfer Learning from Speaker Verification to Multispeaker TTS / SV2TTS (Jia et al., 2018)](https://arxiv.org/abs/1806.04558)

### 실무 적용
다국어 더빙, 오디오북, 게임 NPC 보이스, 접근성(발화 장애인용 개인화 음성) 서비스에서 소량의 사용자 음성으로 개인화된 목소리를 생성하는 데 쓰인다. Coqui TTS 오픈소스로 제공되어 자체 SaaS에 통합하거나 프로토타이핑하기에 적합하다.

---

## Paper 2 (Classic): StyleTTS 2: Towards Human-Level Text-to-Speech through Style Diffusion and Adversarial Training with Large Speech Language Models
- **Authors:** Yinghao Aaron Li, Cong Han, Vinay S. Raghavan, Gavin Mischler, Nima Mesgarani
- **Year:** 2023
- **arXiv:** https://arxiv.org/abs/2306.07691
- **PDF:** [./styletts2-li-2023.pdf](./styletts2-li-2023.pdf)
- **Citation Count:** approx. 400+

### 요약
StyleTTS 2는 스타일(운율·화자 특성)을 참조 음성 없이 텍스트로부터 확산 모델(style diffusion)로 생성하고, WavLM 등 대규모 음성 언어 모델(SLM)을 판별자로 활용한 적대적 학습으로 자연스러움을 극대화한 TTS 모델이다. LJSpeech 단일 화자에서는 사람의 실제 녹음을 능가하고, VCTK 다화자에서는 대등한 수준의 품질을 달성했다.

### 핵심 기여
- 스타일을 잠재 확률 변수로 두고 확산으로 샘플링하는 style diffusion으로, 참조 음성 없이도 다양한 운율을 생성
- 사전학습된 SLM(WavLM 등)을 판별자로 사용하는 SLM adversarial training으로 인간 수준의 자연스러움 확보
- 미분 가능한 duration 모델링(differentiable duration modeling)으로 end-to-end 학습을 안정화

### 이 논문이 중요한 이유
"사람보다 자연스러운 TTS"를 통계적으로 입증한 대표 논문으로, 확산 모델과 SLM 판별자를 결합하는 설계 패턴을 제시했다. 코덱 기반 LM TTS와는 다른 계보(비-토큰, 연속 표현 기반)의 최고 성능 라인을 대표하며, 표현력 높은 음성 합성을 설계할 때 반드시 참고해야 할 아키텍처다.

### 사전 지식
- 확산 모델(DDPM)과 잠재 확산(latent diffusion)의 기본 원리
- StyleTTS 1의 style encoder / AdaIN 기반 스타일 주입
- WavLM/HuBERT 등 자기지도 음성 표현 모델과 adversarial(GAN) 학습

### 관련 논문
- [StyleTTS: A Style-Based Generative Model for Natural TTS (Li et al., 2022)](https://arxiv.org/abs/2205.15439)
- [WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing (Chen et al., 2021)](https://arxiv.org/abs/2110.13900)

### 실무 적용
고품질 오디오북·팟캐스트 내레이션, 버추얼 휴먼·아바타의 표현력 있는 발화, 감정·스타일 제어가 필요한 대화형 에이전트 음성에 적용된다. 오픈소스 구현과 사전학습 가중치가 공개되어 있어 표현력 높은 TTS 파인튜닝의 출발점으로 활용된다.

---

## Paper 3 (Recent): MaskGCT: Zero-Shot Text-to-Speech with Masked Generative Codec Transformer
- **Authors:** Yuancheng Wang, Haoyue Zhan, Liwei Liu, Ruihong Zeng, Haotian Guo, Jiachen Zheng, Qiang Zhang, Xueyao Zhang, Shunsi Zhang, Zhizheng Wu
- **Year:** 2024
- **arXiv:** https://arxiv.org/abs/2409.00750
- **PDF:** [./maskgct-wang-2024.pdf](./maskgct-wang-2024.pdf)
- **Citation Count:** approx. 150+

### 요약
MaskGCT는 텍스트-음성 간 정밀 정렬(alignment)이나 duration 예측 없이 동작하는 완전 비자기회귀(non-autoregressive) zero-shot TTS 모델이다. 1단계에서 텍스트로부터 SSL 기반 semantic token을 예측하고, 2단계에서 이를 조건으로 acoustic token을 예측하는 mask-and-predict 패러다임을 사용한다. 10만 시간 규모의 야생(in-the-wild) 다국어 데이터로 학습해 SOTA급 품질·유사도·명료도를 병렬 생성 효율과 함께 달성했다.

### 핵심 기여
- 명시적 정렬·duration 예측을 제거한 2단계(semantic → acoustic) masked generative codec transformer 구조 제안
- 마스킹 후 병렬 예측(mask-and-predict)으로 자기회귀·확산 모델 대비 높은 생성 효율과 길이 제어 유연성 확보
- 10만 시간급 대규모 다국어 학습으로 zero-shot 화자 복제의 강건성 향상

### 이 논문이 중요한 이유
VALL-E(자기회귀 코덱 LM), F5-TTS(flow matching)와는 또 다른 계보인 "마스킹 기반 병렬 생성"의 대표작으로, 2024년 TTS 설계 공간의 세 번째 축을 명확히 보여준다. 정렬 정보 없이도 고품질·고효율 합성이 가능함을 입증해, 실시간·대규모 서빙을 목표로 하는 제품 설계에 직접적인 시사점을 준다.

### 사전 지식
- 신경 오디오 코덱(neural audio codec)과 semantic/acoustic token의 구분
- MaskGIT류의 masked generative modeling(마스킹 후 병렬 복원)
- VALL-E, F5-TTS 등 코덱 기반 zero-shot TTS의 기본 개념

### 관련 논문
- [VALL-E: Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers (Wang et al., 2023)](https://arxiv.org/abs/2301.02111)
- [F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching (Chen et al., 2024)](https://arxiv.org/abs/2410.06885)

### 실무 적용
낮은 지연시간이 중요한 실시간 음성 에이전트, 대규모 다국어 더빙 파이프라인, 대량 음성 콘텐츠 생성 서비스에 적합하다. 병렬 생성 특성상 배치 처리·서빙 비용 최적화 측면에서 자기회귀 모델 대비 유리하다.

---

## 추천 읽기 순서
1. **YourTTS (2021)** — flow 기반 zero-shot 음성 복제의 기초 개념을 먼저 잡는다.
2. **StyleTTS 2 (2023)** — 확산 + SLM 판별자로 표현력·자연스러움을 끌어올린 계보를 이해한다.
3. **MaskGCT (2024)** — 코덱 토큰 + 마스킹 병렬 생성이라는 최신 패러다임으로 확장한다.

## 핵심 테이크어웨이
- 음성 합성/복제는 크게 (1) flow 기반(YourTTS), (2) 확산 기반(StyleTTS 2), (3) 코덱 토큰 기반(MaskGCT/VALL-E)의 세 축으로 발전해 왔다.
- Zero-shot 화자 복제의 핵심은 "적은 참조 음성으로 화자 정체성을 얼마나 잘 보존하느냐"이며, 화자 임베딩·SLM 판별자·대규모 데이터가 각각의 해법을 제시한다.
- 실무에서는 품질뿐 아니라 지연시간·서빙 비용·데이터 요구량의 트레이드오프로 아키텍처를 선택해야 한다.

## 다음 토픽과의 연결
다음 모듈(Module 6: LLM for NLG)에서는 음성 토큰과 유사하게 텍스트를 토큰으로 다루는 대규모 언어 모델의 아키텍처와 스케일링 법칙을 다룬다. MaskGCT의 codec token 예측이 언어 모델링과 어떻게 맞닿아 있는지 이어서 살펴보면 좋다.
