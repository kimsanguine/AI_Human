# Daily AI Paper Recommendations

> **Date:** 2026-04-15
> **Module:** Module 5: TTS and STT Model Development
> **Topic:** Voice Cloning and Speech Synthesis

---

## Paper 1 (Classic): Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers (VALL-E)

- **Authors:** Chengyi Wang, Sanyuan Chen, Yu Wu, Ziqiang Zhang, Long Zhou, Shujie Liu, Zhuo Chen, Yanqing Liu, Huaming Wang, Jinyu Li, Lei He, Sheng Zhao, Furu Wei
- **Year:** 2023
- **arXiv:** https://arxiv.org/abs/2301.02111
- **PDF:** [./valle-wang-2023.pdf](./valle-wang-2023.pdf)
- **Citation Count:** ~1,500+

### 요약
VALL-E는 텍스트-음성 합성(TTS)을 언어 모델링 문제로 재정의한 혁신적인 논문이다. 오프-더-셸프 신경 오디오 코덱(EnCodec)에서 추출한 이산 코드(discrete code)를 활용하여, 단 3초짜리 음성 샘플만으로 미지의 화자 목소리를 제로샷(zero-shot) 방식으로 합성한다. 60,000시간 분량의 영어 음성 데이터로 사전 학습하여 기존 시스템 대비 월등한 화자 유사도와 자연스러움을 달성했다.

### 핵심 기여
- TTS를 연속 신호 회귀가 아닌 조건부 언어 모델링(conditional language modeling) 태스크로 재정의
- EnCodec 기반 이산 음성 코드를 활용한 계층적 AR + NAR 언어 모델 구조 제안
- 3초 음성 프롬프트만으로 화자의 감정, 음향 환경까지 보존하는 인-컨텍스트 학습 능력 발현
- 60K 시간 대규모 데이터 학습을 통해 기존 제로샷 TTS 대비 SOTA 달성

### 이 논문이 중요한 이유
VALL-E는 음성 합성 패러다임을 근본적으로 바꾼 논문이다. LLM의 인-컨텍스트 학습 능력을 TTS에 최초로 성공적으로 적용함으로써, 이후 등장한 거의 모든 제로샷 TTS 및 음성 복제(voice cloning) 연구의 기반이 되었다. AI 엔지니어라면 신경 코덱 언어 모델의 동작 원리와 이산 오디오 표현(discrete audio representation)의 개념을 반드시 이해해야 한다.

### 사전 지식
- 자기회귀 언어 모델(GPT 류) 기본 구조 이해
- 오디오 코덱과 벡터 양자화(VQ, RVQ) 개념
- 트랜스포머 아키텍처 (특히 attention 메커니즘)
- TTS 기초: 텍스트 입력 → 멜 스펙트로그램 → 보코더 파이프라인

### 관련 논문
- [EnCodec: High Fidelity Neural Audio Compression (Défossez et al., 2022)](https://arxiv.org/abs/2210.13438)
- [AudioLM: A Language Modeling Approach to Audio Generation (Borsos et al., 2022)](https://arxiv.org/abs/2209.03143)
- [Natural TTS Synthesis / Tacotron 2 (Shen et al., 2018)](https://arxiv.org/abs/1712.05884)
- [VITS: Conditional Variational Autoencoder with Adversarial Learning (Kim et al., 2021)](https://arxiv.org/abs/2106.06103)

### 실무 적용
AI 더빙 및 아바타 서비스에서 화자 목소리를 복제할 때 VALL-E 방식의 신경 코덱 언어 모델은 핵심 기술이다. 3초 음성만으로 개인화된 TTS를 구현할 수 있어, 콘텐츠 현지화, 음성 어시스턴트 개인화, 영상 더빙 자동화 등에 직접 활용된다. 또한 음향 환경(노이즈, 에코)까지 유지하므로 자연스러운 더빙 품질을 보장한다.

---

## Paper 2 (Classic): WaveNet: A Generative Model for Raw Audio

- **Authors:** Aaron van den Oord, Sander Dieleman, Heiga Zen, Karen Simonyan, Oriol Vinyals, Alex Graves, Nal Kalchbrenner, Andrew Senior, Koray Kavukcuoglu
- **Year:** 2016
- **arXiv:** https://arxiv.org/abs/1609.03499
- **PDF:** [./wavenet-oord-2016.pdf](./wavenet-oord-2016.pdf)
- **Citation Count:** ~15,000+

### 요약
WaveNet은 원시 오디오 파형(raw audio waveform)을 직접 생성하는 완전 확률론적 자기회귀 심층 신경망이다. 이전 모든 오디오 샘플을 조건으로 다음 샘플의 분포를 예측하며, 영어와 중국어 TTS 모두에서 인간 평가 기준 최고 수준의 자연스러움을 달성했다. 음악 생성, 화자 인식, 음소 인식 등 다양한 오디오 태스크에도 적용 가능한 범용 모델이다.

### 핵심 기여
- 확장 인과적 컨볼루션(dilated causal convolution)으로 긴 수용 영역(receptive field)을 효율적으로 확보
- μ-law 양자화를 통한 16비트 오디오의 소프트맥스 분류로 파형 직접 모델링
- 다중 화자 조건부 생성: 화자 ID 임베딩으로 동일 모델에서 여러 목소리 생성
- TTS/음악/음성 인식 등 다방면에 적용 가능한 일반적 생성 오디오 프레임워크 제시

### 요약
WaveNet은 원시 오디오 파형(raw audio waveform)을 직접 샘플 단위로 생성하는 완전 확률론적(fully probabilistic) 딥러닝 모델이다. 확장 인과 합성곱(dilated causal convolution)을 핵심 구조로 사용하여 수천 타임스텝에 걸친 장거리 의존성을 효율적으로 캡처하며, TTS에 적용 시 당시 최고 수준의 자연스러운 음성을 합성했다. 단일 모델로 다수 화자의 목소리를 동일한 품질로 생성하고 음악 생성에도 활용될 수 있음을 증명했다.

### 핵심 기여
- 오디오를 픽셀 단위가 아닌 **샘플(sample) 단위로 직접 모델링**하는 자기회귀 생성 방식 도입
- **확장 인과 합성곱(dilated causal convolution)**: 깊이에 따라 수용 영역(receptive field)이 지수적으로 확장되어 수천 타임스텝을 커버
- 조건부 생성(conditional generation)을 통한 **다화자(multi-speaker) 및 TTS 지원**
- 음성 합성뿐 아니라 음악 생성, 음소 인식(ASR)에도 적용 가능한 범용 오디오 생성 프레임워크

### 이 논문이 중요한 이유
WaveNet은 보코더(vocoder) 기반 음성 합성의 패러다임을 바꾼 역사적 전환점이다. 이전의 연결 합성(concatenative synthesis)이나 파라메트릭 합성(parametric synthesis)과 달리 신경망이 파형 자체를 직접 학습하게 함으로써, 이후 등장한 모든 뉴럴 보코더(WaveGlow, HiFi-GAN, EnCodec 등)의 기반이 되었다. Google Assistant, Amazon Alexa 등 실제 상용 음성 서비스에 도입되어 자연스러운 TTS의 기준을 높였다.

### 사전 지식
- 컨볼루션 신경망(CNN)의 구조와 수용 영역(receptive field) 개념
- 인과 컨볼루션(causal convolution)과 자기회귀 모델의 차이
- 오디오 신호의 기초 (샘플링 레이트, µ-law companding)
- PixelCNN과 같은 자기회귀 생성 모델에 대한 이해

### 관련 논문
- [Parallel WaveNet: Fast High-Fidelity Speech Synthesis (van den Oord et al., 2017)](https://arxiv.org/abs/1711.10433)
- [WaveGlow: A Flow-based Generative Network for Speech Synthesis (Prenger et al., 2018)](https://arxiv.org/abs/1811.00002)
- [HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis (Kong et al., 2020)](https://arxiv.org/abs/2010.05646)
- [PixelCNN++: Improving the PixelCNN with Discretized Logistic Mixture Likelihood (Salimans et al., 2017)](https://arxiv.org/abs/1701.05517)

### 실무 적용
WaveNet 계열의 뉴럴 보코더는 현재 대부분의 상용 TTS 파이프라인 마지막 단계(mel-spectrogram → 파형 변환)에서 사용된다. AI 더빙/아바타 서비스에서 TTS 시스템의 음질은 사용자 신뢰도에 직결되며, WaveNet의 후속 모델들(HiFi-GAN, EnCodec)은 더 빠른 추론 속도로 실시간 스트리밍 적용을 가능하게 했다. 음성 데이터가 부족한 언어에서도 고품질 TTS를 구현하는 다국어 음성 서비스 기반 기술이기도 하다.

---

## Paper 3 (Recent): VALL-E 2: Neural Codec Language Models are Human Parity Zero-Shot Text to Speech Synthesizers

- **Authors:** Shengpeng Ji, Jialong Zuo, Minghui Fang, Ziyue Jiang, Feiyang Chen, Xinyu Duan, Baoxing Huai, Zhou Zhao
- **Year:** 2024
- **arXiv:** https://arxiv.org/abs/2406.05370
- **PDF:** [./valle2-chen-2024.pdf](./valle2-chen-2024.pdf)
- **Citation Count:** ~300+

### 이 논문이 중요한 이유
WaveNet은 현대 신경 TTS 보코더(vocoder)의 직접적인 조상이다. 이전까지의 파라메트릭/연결 합성 방식의 로봇 같은 음질을 뛰어넘어, 딥러닝으로 사람과 구별하기 어려운 음성을 처음으로 생성했다. WaveNet 이후 등장한 WaveGlow, WaveRNN, HiFi-GAN 등 모든 신경 보코더의 설계 철학이 이 논문에서 시작되었으며, 구글의 Assistant, DeepMind의 음성 시스템에 실제 탑재된 산업적 영향력도 막대하다.

### 사전 지식
- 오디오 신호 처리 기초 (샘플링 레이트, 파형 표현)
- 인과적 컨볼루션(causal convolution) 개념
- 자기회귀 모델(autoregressive model) 원리
- 소프트맥스 분류와 확률 분포 표현

### 관련 논문
- [Parallel WaveNet: Fast High-Fidelity Speech Synthesis (van den Oord et al., 2017)](https://arxiv.org/abs/1711.10433)
- [WaveGlow: A Flow-based Generative Network for Speech Synthesis (Prenger et al., 2018)](https://arxiv.org/abs/1811.00002)
- [HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis (Kong et al., 2020)](https://arxiv.org/abs/2010.05646)
- [PixelCNN: Conditional Image Generation with PixelCNN Decoders (van den Oord et al., 2016)](https://arxiv.org/abs/1606.05328)

### 실무 적용
WaveNet은 고품질 음성 합성 파이프라인의 마지막 단계인 보코더(vocoder) 구현에 핵심적으로 활용된다. AI 더빙, 음성 아바타, TTS 제품에서 멜 스펙트로그램을 자연스러운 오디오 파형으로 변환하는 부분에 WaveNet 계열 모델이 사용된다. 다만 자기회귀 특성으로 인한 느린 추론 속도는 실시간 서비스에 병목이 되어, 이후 Parallel WaveNet, HiFi-GAN 등 빠른 대안이 파생되었다.

---

## Paper 3 (Recent): VALL-E 2: Neural Codec Language Models are Human Parity Zero-Shot Text to Speech Synthesizers

- **Authors:** Shengpeng Ji, Chen Zhang, Zhen Ye, Yuancheng Wang, Xiaoda Yang, Decai Li, Zehan Wang, Jialong Zuo, Minghui Fang, Hai Huang, Furu Wei, Zhou Zhao
- **Year:** 2024
- **arXiv:** https://arxiv.org/abs/2406.05370
- **PDF:** [./valle2-chen-2024.pdf](./valle2-chen-2024.pdf)
- **Citation Count:** ~300+

### 요약
VALL-E 2는 원작 VALL-E의 직접적인 후속작으로, 제로샷 TTS에서 최초로 인간 동등 수준(human parity)을 달성한 논문이다. 반복적 인-컨텍스트 일관성(repetition-aware sampling)과 그룹화 코드 모델링(grouped code modeling) 두 가지 핵심 기법을 통해 VALL-E의 안정성과 효율성 문제를 해결했다. LibriSpeech, VCTK 벤치마크에서 자연스러움과 화자 유사도 모두 인간 수준을 초과했다.

### 핵심 기여
- Repetition Aware Sampling: 코덱 토큰 반복 문제 해결로 생성 안정성 대폭 향상
- Grouped Code Modeling: 코드 그룹 단위 처리로 시퀀스 길이 감소 및 추론 속도 개선
- LibriSpeech 및 VCTK 벤치마크에서 인간 동등 수준(human parity) 최초 달성
- 강건한 제로샷 화자 적응: 3초 프롬프트로 미지 화자의 감정·운율·음향 환경 보존

### 이 논문이 중요한 이유
VALL-E 2는 신경 TTS 분야에서 "인간 수준 품질"이라는 오랜 목표를 처음으로 공식 달성한 논문으로, AI 음성 합성의 새로운 이정표를 세웠다. 음성 복제(voice cloning) 기술의 최전선에 있으며, AI 더빙·아바타·개인화 음성 서비스를 개발하는 AI 엔지니어에게 반드시 이해해야 할 최신 아키텍처다.

### 사전 지식
- VALL-E 원본 논문 이해 (arXiv:2301.02111)
- 신경 오디오 코덱(EnCodec, RVQ) 동작 원리
- 자기회귀 + 비자기회귀 하이브리드 모델 구조
- 제로샷 학습(zero-shot learning) 개념

### 관련 논문
- [VALL-E: Neural Codec Language Models (Wang et al., 2023)](https://arxiv.org/abs/2301.02111)
- [VoiceBox: Text-Guided Multilingual Universal Speech Generation at Scale (Le et al., 2023)](https://arxiv.org/abs/2306.15687)
- [NaturalSpeech 3: Zero-Shot Copier-Style TTS (Ju et al., 2024)](https://arxiv.org/abs/2403.03100)
- [CosyVoice: A Scalable Multilingual Zero-shot TTS (Du et al., 2024)](https://arxiv.org/abs/2407.05407)

### 실무 적용
VALL-E 2의 인간 동등 수준 품질은 AI 더빙 서비스에서 실제 성우 대체 가능성을 높이는 직접적인 근거가 된다. Repetition Aware Sampling 기법은 프로덕션 환경에서 모델 안정성을 높이는 실용적 기술로, 음성 아바타 및 실시간 TTS 제품에 바로 적용 가능하다. 그룹화 코드 모델링은 추론 속도를 개선하여 실시간 스트리밍 서비스의 레이턴시 요구사항 충족에 기여한다.

---

## 추천 읽기 순서

1. **WaveNet (2016)** → 신경 오디오 생성의 기초 원리 이해
2. **VALL-E (2023)** → 언어 모델 기반 제로샷 TTS 패러다임 습득
3. **VALL-E 2 (2024)** → 인간 동등 수준 달성 방법과 최신 기법 파악

## 핵심 테이크어웨이

- 음성 합성의 패러다임이 파형 직접 생성(WaveNet) → 이산 코드 언어 모델링(VALL-E) → 인간 수준 품질(VALL-E 2)로 급격히 진화했다.
- 신경 오디오 코덱(RVQ 기반 이산 토큰화)이 현대 음성 합성의 핵심 인프라가 되었으며, LLM과 TTS의 경계가 허물어지고 있다.
- 3초 음성 프롬프트만으로 완전한 화자 복제가 가능해졌고, 이는 AI 더빙·아바타 서비스의 실용성을 크게 높였다.
- 실무에서는 생성 품질(naturalness)과 추론 속도(latency) 사이의 트레이드오프를 항상 고려해야 한다.

## 다음 토픽과의 연결

다음 모듈(Module 6: LLM for Natural Language Generation)에서는 GPT 아키텍처와 스케일링 법칙을 다룬다. VALL-E 계열 모델이 GPT 스타일 언어 모델을 음성 도메인에 적용한 것처럼, LLM의 인-컨텍스트 학습 능력이 어떻게 다양한 도메인으로 확장되는지 이해하는 흐름으로 연결된다. 특히 대규모 사전 학습 데이터와 스케일링 효과가 음성 및 언어 모델 모두에서 동일하게 작동함을 확인할 수 있다.

### 요약
VALL-E 2는 VALL-E의 후속 모델로, 제로샷 TTS에서 인류 동등 수준(human parity)을 최초로 달성한 시스템이다. 반복 인식 샘플링(Repetition Aware Sampling)과 그룹 코드 모델링(Grouped Code Modeling)이라는 두 가지 핵심 개선을 통해 무한 루프 문제를 해결하고 추론 속도를 대폭 향상시켰다. LibriSpeech와 VCTK 벤치마크에서 음성 견고성, 자연스러움, 화자 유사도 모두 인간 수준에 도달했음을 실험으로 입증했다.

### 핵심 기여
- **반복 인식 샘플링(Repetition Aware Sampling)**: 디코딩 히스토리의 토큰 반복을 고려해 뉴클리어스 샘플링을 개선, 무한 루프 문제 해결
- **그룹 코드 모델링(Grouped Code Modeling)**: 코덱 코드를 그룹으로 묶어 시퀀스 길이 단축, 추론 속도 향상 및 장시퀀스 모델링 개선
- LibriSpeech 및 VCTK 데이터셋에서 **인류 동등 수준(human parity)** 최초 달성
- 복잡하거나 반복 구절이 포함된 어려운 문장에서도 일관된 고품질 합성 유지

### 이 논문이 중요한 이유
VALL-E 2는 음성 합성의 품질이 인간 수준에 도달했다는 이정표를 세운 논문으로, AI 더빙·아바타·보이스 클로닝 서비스의 상용화 가능성을 한 단계 높였다. 동시에 딥페이크 음성 등 악용 가능성에 대한 논의를 촉발시켜, AI 음성 기술의 책임감 있는 개발 방향(실어증/ALS 환자 지원 등 사회적 가치 활용)을 명확히 제시하고 있다. AI 엔지니어라면 TTS 스택의 현재 최고 수준을 파악하기 위해 반드시 읽어야 할 논문이다.

### 사전 지식
- VALL-E (arXiv:2301.02111)의 AR+NAR 계층적 구조 이해
- EnCodec 및 RVQ(Residual Vector Quantization) 개념
- 뉴클리어스 샘플링(nucleus/top-p sampling) 등 LLM 디코딩 전략
- LibriSpeech, VCTK 벤치마크 평가 지표 (WER, SIM, DNSMOS 등)

### 관련 논문
- [VALL-E: Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers (Wang et al., 2023)](https://arxiv.org/abs/2301.02111)
- [CosyVoice: A Scalable Multilingual Zero-shot Text-to-speech Synthesizer (Du et al., 2024)](https://arxiv.org/abs/2407.05407)
- [F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech (Chen et al., 2024)](https://arxiv.org/abs/2410.06885)
- [VoiceBox: Text-Guided Multilingual Universal Speech Generation at Scale (Le et al., 2023)](https://arxiv.org/abs/2306.15687)

### 실무 적용
VALL-E 2의 인류 동등 품질과 빠른 추론 속도는 실시간 AI 더빙, 멀티링구얼 보이스 클로닝, Agentic AI의 음성 응답 생성에 직접 적용 가능하다. 특히 3초 음성 프롬프트만으로 화자 적응이 가능한 특성은 B2B SaaS 서비스에서 고객별 맞춤 음성 경험(custom voice persona)을 낮은 비용으로 제공하는 핵심 기술이 된다. 음성 더빙 파이프라인에서 후처리 단계 없이 직접 고품질 오디오를 출력할 수 있어 레이턴시 최적화에도 유리하다.

---

## 추천 읽기 순서

1. **WaveNet (2016)** → 뉴럴 오디오 생성의 출발점. 원시 파형 생성과 확장 인과 합성곱 개념을 먼저 이해한다.
2. **VALL-E (2023)** → LLM 패러다임이 음성 합성에 어떻게 적용되는지 학습. EnCodec 토큰화와 AR+NAR 구조를 파악한다.
3. **VALL-E 2 (2024)** → VALL-E의 개선 사항과 인류 동등 수준 달성 방법을 이해. 실무 적용 가능성을 검토한다.

## 핵심 테이크어웨이

- 음성 합성은 **파형 직접 생성(WaveNet) → 피처 기반 생성(Tacotron2+vocoder) → 언어 모델 기반 토큰 생성(VALL-E)**의 진화 경로를 따른다.
- LLM 스케일링 법칙은 음성에도 적용된다: 데이터와 모델 크기 증가가 제로샷 능력을 획기적으로 향상시킨다.
- **3초 음성 프롬프트**로 화자 적응이 가능한 시대가 열렸으며, 이는 AI 더빙·아바타 서비스의 비용 구조를 근본적으로 변화시킨다.
- 인류 동등 수준의 TTS 달성은 기술적 이정표인 동시에, 딥페이크 음성 등 윤리적 책임에 대한 논의를 더욱 중요하게 만든다.

## 다음 토픽과의 연결

다음 모듈(Module 6: LLM for Natural Language Generation)에서는 GPT 아키텍처와 스케일링 법칙을 다루는데, 오늘 학습한 VALL-E 계열 모델이 LLM의 자기회귀 언어 모델링 방식을 그대로 오디오 도메인에 이식한 것임을 상기하면 연결이 자연스럽다. 또한 인컨텍스트 러닝(in-context learning)이 음성 합성에서도 동작한다는 점은 GPT-3의 few-shot 능력과 직접 대응된다.
