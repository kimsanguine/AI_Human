# Daily AI Paper Recommendations

> **Date:** 2026-05-11
> **Module:** Module 5: TTS and STT Model Development
> **Topic:** Neural Text-to-Speech

---

## Paper 1 (Classic): FastSpeech 2: Fast and High-Quality End-to-End Text to Speech
- **Authors:** Yi Ren, Chenxu Hu, Xu Tan, Tao Qin, Sheng Zhao, Zhou Zhao, Tie-Yan Liu
- **Year:** 2020 (ICLR 2021)
- **arXiv:** https://arxiv.org/abs/2006.04558
- **PDF:** [./fastspeech2-ren-2020.pdf](./fastspeech2-ren-2020.pdf)
- **Citation Count:** 약 2,500+ (Google Scholar 기준)

### 요약
FastSpeech 2는 비-자기회귀(Non-Autoregressive) 방식의 TTS 모델로, 기존 FastSpeech의 복잡한 teacher-student distillation 파이프라인을 제거하고 ground-truth로 직접 학습한다. Pitch, energy, duration이라는 variation 정보를 명시적으로 conditional input으로 사용해 one-to-many mapping 문제를 해결했고, FastSpeech 2s는 텍스트에서 waveform까지 완전한 end-to-end 생성을 달성한다.

### 핵심 기여
- Variance Adaptor 도입 — pitch, energy, duration을 별도 predictor로 학습해 one-to-many 문제 완화
- Teacher-student distillation 제거로 학습 시간 3배 단축, 품질도 향상
- FastSpeech 2s — 텍스트로부터 waveform까지 한 번에 생성하는 fully end-to-end TTS의 첫 시도
- 자기회귀 모델(Tacotron 2 등)을 품질에서 능가하면서 추론 속도는 수십 배 빠름

### 이 논문이 중요한 이유
오늘날 거의 모든 프로덕션 TTS 시스템(특히 모바일/엣지 환경)은 비-자기회귀 구조 위에 구축되어 있고, FastSpeech 2의 Variance Adaptor 패턴은 이후 VITS, NaturalSpeech, F5-TTS 등 거의 모든 후속 연구의 기본 빌딩 블록이 되었다. AI 엔지니어에게는 "왜 TTS에서 pitch와 duration을 별도로 모델링하는가"라는 근본 질문에 대한 답을 제공하며, 음성 생성의 latency-quality trade-off를 설계할 때 반드시 이해해야 하는 baseline이다.

### 사전 지식
- Mel-spectrogram과 vocoder의 역할 분리 개념
- Transformer encoder-decoder 구조
- Monotonic Alignment Search 또는 forced alignment 기본 개념
- Tacotron 2의 attention 기반 alignment의 한계 (skipping, repeating)

### 관련 논문
- [FastSpeech: Fast, Robust and Controllable Text to Speech (Ren et al., 2019)](https://arxiv.org/abs/1905.09263)
- [Tacotron 2: Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions (Shen et al., 2017)](https://arxiv.org/abs/1712.05884)
- [Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search (Kim et al., 2020)](https://arxiv.org/abs/2005.11129)
- [VITS: Conditional VAE with Adversarial Learning for End-to-End TTS (Kim et al., 2021)](https://arxiv.org/abs/2106.06103)

### 실무 적용
실시간 음성 어시스턴트, 대화형 AI Avatar, 게임 NPC voice, 동영상 더빙 자동화 같은 latency-critical 환경에서 FastSpeech 2 계열 아키텍처는 사실상 표준이다. 특히 AI Dubbing 서비스의 경우 ground-truth pitch/duration을 활용해 화자 스타일을 fine-grained하게 제어할 수 있어, 감정 표현이나 운율 조정에 유용하다. 산업계에서는 FastSpeech 2 + HiFi-GAN을 조합한 two-stage 파이프라인이 가장 흔하게 쓰이는 production 구성이다.

---

## Paper 2 (Classic): HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis
- **Authors:** Jungil Kong, Jaehyeon Kim, Jaekyoung Bae
- **Year:** 2020 (NeurIPS 2020)
- **arXiv:** https://arxiv.org/abs/2010.05646
- **PDF:** [./hifi-gan-kong-2020.pdf](./hifi-gan-kong-2020.pdf)
- **Citation Count:** 약 2,800+ (Google Scholar 기준)

### 요약
HiFi-GAN은 mel-spectrogram을 입력받아 22.05kHz 고품질 waveform을 생성하는 GAN 기반 vocoder다. Multi-Period Discriminator(MPD)와 Multi-Scale Discriminator(MSD)로 음성의 주기적/비주기적 패턴을 모두 포착하면서, 단일 V100 GPU에서 실시간 대비 167.9배 빠른 추론 속도를 달성했다.

### 핵심 기여
- Multi-Period Discriminator 제안 — 음성의 다양한 주기성을 분리해서 판별하는 새로운 D 구조
- Multi-Receptive Field Fusion(MRF) Generator로 다양한 시간 스케일의 패턴 통합
- WaveNet/WaveGlow 대비 수십~수백 배 빠른 추론으로 GAN vocoder의 가능성을 확립
- CPU에서도 small footprint 버전이 실시간 대비 13.4배 빠른 추론 달성 → 엣지 디바이스 배포 가능

### 이 논문이 중요한 이유
HiFi-GAN은 "vocoder는 느리고 무거운 자기회귀 모델이어야 한다"는 통념을 깬 결정적 논문이다. 이 이후로 대부분의 production TTS 시스템(Bark, XTTS, F5-TTS 등 포함)이 HiFi-GAN 또는 그 변형(Vocos, BigVGAN, iSTFTNet)을 vocoder로 채택하고 있다. AI 엔지니어 관점에서, GAN 학습의 안정성·discriminator 설계·perceptual loss 균형을 실전에서 다루는 좋은 학습 사례이며, real-time 음성 생성 시스템 설계 시 latency budget을 결정하는 핵심 컴포넌트다.

### 사전 지식
- GAN의 기본 구조 (Generator, Discriminator, adversarial loss)
- Mel-spectrogram의 시간-주파수 표현
- WaveNet, WaveGlow 등 기존 neural vocoder의 한계
- Feature matching loss, mel-spectrogram L1 loss의 역할

### 관련 논문
- [WaveNet: A Generative Model for Raw Audio (van den Oord et al., 2016)](https://arxiv.org/abs/1609.03499)
- [MelGAN: Generative Adversarial Networks for Conditional Waveform Synthesis (Kumar et al., 2019)](https://arxiv.org/abs/1910.06711)
- [Parallel WaveGAN (Yamamoto et al., 2019)](https://arxiv.org/abs/1910.11480)
- [BigVGAN: A Universal Neural Vocoder with Large-Scale Training (Lee et al., 2022)](https://arxiv.org/abs/2206.04658)

### 실무 적용
실시간 voice cloning, 라이브 더빙, 인터랙티브 AI Avatar 같은 모든 음성 생성 서비스의 마지막 단계에서 HiFi-GAN 계열이 표준으로 쓰인다. 특히 모바일 앱이나 임베디드 환경에 음성 합성을 탑재할 때, HiFi-GAN의 작은 모델 크기와 CPU 친화적인 추론 특성은 결정적 강점이다. AI Dubbing 서비스 운영 시 GPU 비용의 상당 부분이 vocoder inference에서 발생하므로, HiFi-GAN의 batch 추론·streaming 최적화는 unit economics에 직접 영향을 준다.

---

## Paper 3 (Recent): F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching
- **Authors:** Yushen Chen, Zhikang Niu, Ziyang Ma, Keqi Deng, Chunhui Wang, Jian Zhao, Kai Yu, Xie Chen
- **Year:** 2024 (October)
- **arXiv:** https://arxiv.org/abs/2410.06885
- **PDF:** [./f5-tts-chen-2024.pdf](./f5-tts-chen-2024.pdf)
- **Citation Count:** 약 100+ (2025년 기준, 빠르게 증가 중)

### 요약
F5-TTS는 Flow Matching과 Diffusion Transformer(DiT) 기반의 완전히 비-자기회귀 zero-shot TTS 시스템이다. Duration model, 별도 text encoder, phoneme alignment 없이 단순히 텍스트를 filler token으로 패딩해 음성과 같은 길이로 만든 뒤 denoising으로 생성한다. ConvNeXt를 활용한 텍스트 표현 정제와 추론 시 Sway Sampling 전략으로 효율과 품질을 모두 끌어올렸다.

### 핵심 기여
- Flow Matching + DiT 기반의 단순한 아키텍처로 SOTA zero-shot TTS 달성 (RTF 0.15)
- Duration predictor·phoneme aligner 등 복잡한 컴포넌트 제거 → 학습·유지보수 비용 대폭 절감
- Sway Sampling이라는 inference-time 기법으로 추가 학습 없이 품질·속도 동시 개선
- 100K hours 다국어 데이터로 학습된 체크포인트와 코드를 완전 오픈소스로 공개

### 이 논문이 중요한 이유
F5-TTS는 2024년 zero-shot TTS 진영에서 "단순함이 곧 강력함"을 입증한 대표 사례다. VALL-E, Voicebox, NaturalSpeech 3 같은 직전 SOTA 모델들은 모두 복잡한 보조 모듈에 의존했지만, F5-TTS는 "텍스트 패딩 + denoising"이라는 미니멀 디자인만으로 이들을 따라잡았다. 이는 Agentic AI나 voice agent 제품에 음성 합성을 통합할 때, 더 적은 컴포넌트로 더 빠른 iteration이 가능함을 의미한다. 또한 오픈소스 가용성이 높아 자체 모델 fine-tuning이 필요한 B2B SaaS에 매우 실용적이다.

### 사전 지식
- Flow Matching의 기본 원리 (Continuous Normalizing Flow와의 관계)
- Diffusion Transformer(DiT) 구조 — Stable Diffusion 3, Sora 등에서 쓰이는 기법
- Zero-shot TTS와 voice cloning의 차이
- Neural codec(Encodec, SoundStream) 기반 TTS와 mel-spectrogram 기반 TTS의 차이

### 관련 논문
- [Flow Matching for Generative Modeling (Lipman et al., 2022)](https://arxiv.org/abs/2210.02747)
- [Voicebox: Text-Guided Multilingual Universal Speech Generation at Scale (Le et al., 2023)](https://arxiv.org/abs/2306.15687)
- [VALL-E: Neural Codec Language Models are Zero-Shot TTS Synthesizers (Wang et al., 2023)](https://arxiv.org/abs/2301.02111)
- [NaturalSpeech 3: Zero-Shot Speech Synthesis with Factorized Codec and Diffusion Models (Ju et al., 2024)](https://arxiv.org/abs/2403.03100)
- [E2 TTS: Embarrassingly Easy Fully Non-Autoregressive Zero-Shot TTS (Eskimez et al., 2024)](https://arxiv.org/abs/2406.18009)

### 실무 적용
B2B/B2C voice 서비스에서 F5-TTS는 다음 세 가지 시나리오에서 즉시 가치를 낸다. 첫째, 짧은 reference audio(수 초)만으로 zero-shot voice cloning이 가능해 사용자 온보딩이 간소화된다. 둘째, code-switching이 자연스러워 다국어 콘텐츠 더빙 파이프라인 단순화가 가능하다. 셋째, 추론 RTF 0.15는 대화형 AI agent의 음성 응답 latency 요구를 충족한다. 다만 라이선스 이슈(학습 데이터 출처)와 실시간 streaming 미지원 등 production 적용 전 검증이 필요한 부분이 있다.

---

## 추천 읽기 순서
1. **FastSpeech 2** 먼저 — 비-자기회귀 TTS의 핵심 아이디어(variance adaptor, parallel decoding)를 이해해야 이후 모든 모델의 설계 결정을 따라갈 수 있다.
2. **HiFi-GAN** 다음 — TTS 파이프라인의 마지막 단계인 vocoder가 어떻게 발전했는지, GAN 학습이 음성 도메인에서 어떻게 안정화됐는지를 학습한다.
3. **F5-TTS** 마지막 — 위 두 논문의 "복잡한 파이프라인" 패러다임을 깨는 최신 미니멀 접근을 비교 분석하면 trade-off가 선명하게 보인다.

## 핵심 테이크어웨이
- **모듈 분리 vs 통합**: FastSpeech 2는 acoustic model + vocoder 분리, F5-TTS는 single model로 통합 — 운영 복잡도와 품질 제어성 사이에 trade-off가 존재한다.
- **Variance 제어가 표현력의 핵심**: Pitch·duration·energy를 명시적으로 제어할 수 있는지가 voice cloning, 감정 표현, 더빙 품질을 좌우한다.
- **Vocoder는 latency budget의 병목**: 아무리 좋은 acoustic model이라도 vocoder가 느리면 의미 없다. HiFi-GAN 계열의 효율성이 production 배포 가능성을 결정한다.
- **단순함의 회귀**: 2024년 들어 Voicebox·F5-TTS 등 "less is more" 접근이 부상했다. 복잡한 보조 모듈을 더하기 전에 데이터와 모델 크기로 풀 수 있는지 먼저 검증하는 것이 AI-native 엔지니어링의 자세다.

## 다음 토픽과의 연결
다음 주제인 **Voice Cloning and Speech Synthesis** (Day 13)에서는 오늘 다룬 TTS 기반 위에서 zero-shot voice cloning과 화자 적응 기법을 깊이 있게 다룬다. 특히 F5-TTS의 zero-shot 능력은 VALL-E 계열의 codec language model 접근과 자연스럽게 연결되며, "한 번 들은 화자를 즉시 복제할 수 있는가"라는 질문에 대한 두 가지 패러다임(diffusion/flow vs autoregressive codec LM)의 비교 학습으로 이어진다.
