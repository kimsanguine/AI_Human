# Daily AI Paper Recommendations

> **Date:** 2026-09-24
> **Module:** Module 5: TTS and STT Model Development
> **Topic:** Voice Cloning and Speech Synthesis

> 이번 사이클에서는 기존에 다룬 VALL-E, WaveNet, SV2TTS, HiFi-GAN, AutoVC, Neural Voice Cloning, YourTTS, StyleTTS 2, AudioLM, Voicebox 대신 **"음성을 무엇으로 표현하는가(코덱 토큰)"**와 **"화자를 어떻게 표현하는가(스피커 임베딩)"**라는 두 축의 뿌리 논문을 골랐습니다.

---

## Paper 1 (Classic): High Fidelity Neural Audio Compression (EnCodec)
- **Authors:** Alexandre Défossez, Jade Copet, Gabriel Synnaeve, Yossi Adi
- **Year:** 2022
- **arXiv:** https://arxiv.org/abs/2210.13438
- **PDF:** [./encodec-defossez-2022.pdf](./encodec-defossez-2022.pdf)
- **Citation Count:** ~1,000+

### 요약
EnCodec은 인코더-양자화기-디코더 구조의 실시간 신경망 오디오 코덱으로, Residual Vector Quantization(RVQ)을 통해 음성을 이산 토큰 시퀀스로 압축합니다. 멀티스케일 스펙트로그램 판별기 기반 적대적 학습과 손실 가중치를 자동 조절하는 loss balancer를 도입해 낮은 비트레이트(1.5~24kbps)에서도 높은 음질을 달성했습니다.

### 핵심 기여
- RVQ 기반 스트리밍 가능한 신경망 코덱 — 음성을 "언어 모델이 다룰 수 있는 토큰"으로 변환
- 단일 멀티스케일 STFT 판별기로 학습 단순화 및 아티팩트 감소
- Loss balancer로 여러 손실 항의 그래디언트 스케일을 안정화
- 소형 Transformer 엔트로피 코딩으로 추가 압축(최대 약 40%)

### 이 논문이 중요한 이유
VALL-E, VALL-E 2, MusicGen, AudioGen 등 "코덱 언어 모델" 계열의 음성 합성/보이스 클로닝은 모두 EnCodec 토큰 위에서 동작합니다. "3초 음성으로 목소리를 복제한다"는 기술은 사실상 **음성을 텍스트처럼 토큰화할 수 있게 된 순간** 가능해졌고, 그 전환점이 바로 이 논문입니다. 코덱 품질이 곧 복제 음성 품질의 상한선입니다.

### 사전 지식
- 오토인코더와 벡터 양자화(VQ-VAE) 개념
- GAN 기반 보코더(HiFi-GAN, MelGAN)의 판별기 구조
- STFT/멜 스펙트로그램 기초

### 관련 논문
- [SoundStream: An End-to-End Neural Audio Codec (Zeghidour et al., 2021)](https://arxiv.org/abs/2107.03312)
- [Neural Codec Language Models are Zero-Shot TTS / VALL-E (Wang et al., 2023)](https://arxiv.org/abs/2301.02111)
- [High-Fidelity Audio Compression with Improved RVQGAN / DAC (Kumar et al., 2023)](https://arxiv.org/abs/2306.06546)
- [Simple and Controllable Music Generation / MusicGen (Copet et al., 2023)](https://arxiv.org/abs/2306.05284)

### 실무 적용
AI 더빙·아바타 서비스에서 음성 LLM(TTS, speech-to-speech)을 구축할 때 코덱 선택(EnCodec vs DAC vs Mimi 등)은 지연시간·음질·토큰 수(=추론 비용)를 결정합니다. 스트리밍 가능한 구조는 실시간 통역·대화형 아바타에 필수이며, 낮은 비트레이트 토큰은 LLM 컨텍스트 길이와 비용을 직접 줄입니다.

---

## Paper 2 (Classic): Deep Voice 2: Multi-Speaker Neural Text-to-Speech
- **Authors:** Sercan Arık, Gregory Diamos, Andrew Gibiansky, John Miller, Kainan Peng, Wei Ping, Jonathan Raiman, Yanqi Zhou
- **Year:** 2017 (NeurIPS 2017)
- **arXiv:** https://arxiv.org/abs/1705.08947
- **PDF:** [./deep-voice-2-arik-2017.pdf](./deep-voice-2-arik-2017.pdf)
- **Citation Count:** ~700+

### 요약
Deep Voice 2는 하나의 신경망 TTS 모델이 **학습 가능한 저차원 화자 임베딩(speaker embedding)**만 바꿔 수백 명의 목소리를 생성할 수 있음을 보였습니다. 화자당 30분 미만의 데이터로도 고품질 다화자 합성이 가능하며, 이 화자 임베딩 기법을 Tacotron에도 적용해 범용성을 입증했습니다.

### 핵심 기여
- 화자 임베딩을 RNN 초기 상태·게이트·특징 등 모델 여러 지점에 주입하는 다화자 TTS 구조 제안
- VCTK(108명), LibriSpeech(2,400+명) 규모에서 단일 모델로 다화자 합성 입증
- Deep Voice 1 대비 개선된 파이프라인(분리된 duration/F0 모델, WaveNet 보코더 결합)
- 화자 임베딩 공간이 성별·억양 등 의미 있는 구조를 학습함을 분석

### 이 논문이 중요한 이유
오늘날 모든 보이스 클로닝의 기본 전제 — "목소리 = 하나의 벡터" — 를 처음으로 대규모로 실증한 논문입니다. 이후 Neural Voice Cloning(2018), SV2TTS(2018), YourTTS, 그리고 오늘의 MiniMax-Speech에 이르는 **스피커 인코더 계보의 출발점**이라 흐름을 이해하는 데 필수입니다.

### 사전 지식
- Seq2seq/어텐션 기반 TTS(Tacotron) 구조
- 임베딩 테이블(lookup embedding) 개념
- WaveNet 보코더의 역할

### 관련 논문
- [Deep Voice: Real-time Neural Text-to-Speech (Arık et al., 2017)](https://arxiv.org/abs/1702.07825)
- [Neural Voice Cloning with a Few Samples (Arık et al., 2018)](https://arxiv.org/abs/1802.06006)
- [Transfer Learning from Speaker Verification to Multispeaker TTS / SV2TTS (Jia et al., 2018)](https://arxiv.org/abs/1806.04558)
- [Deep Voice 3: Scaling Text-to-Speech with Convolutional Sequence Learning (Ping et al., 2017)](https://arxiv.org/abs/1710.07654)

### 실무 적용
다화자 TTS 한 벌로 여러 성우/캐릭터 음성을 서비스하는 구조(단일 모델 + 화자 ID)는 서빙 비용을 크게 줄입니다. 화자 임베딩 공간을 보간해 "새로운 가상 목소리"를 만들거나, 화자 ID 단위로 사용 권한·워터마크를 관리하는 등 제품 설계의 기본 단위가 됩니다.

---

## Paper 3 (Recent): MiniMax-Speech: Intrinsic Zero-Shot Text-to-Speech with a Learnable Speaker Encoder
- **Authors:** Bowen Zhang, Congchao Guo, Geng Yang, et al. (MiniMax, 20 authors)
- **Year:** 2025
- **arXiv:** https://arxiv.org/abs/2505.07916
- **PDF:** [./minimax-speech-zhang-2025.pdf](./minimax-speech-zhang-2025.pdf)
- **Citation Count:** ~50+ (2025년 5월 공개, TTS Arena 1위 기록)

### 요약
MiniMax-Speech는 자기회귀 Transformer 기반 TTS로, **TTS 모델과 함께 공동 학습되는 스피커 인코더**를 통해 참조 음성의 전사(transcript) 없이도 음색을 추출합니다. 이를 통해 진정한 의미의 제로샷("intrinsic zero-shot") 합성과, 참조 음성+전사를 함께 쓰는 원샷 보이스 클로닝을 모두 지원합니다. Flow-VAE로 음질을 높였고, 32개 언어에서 WER·화자 유사도 SOTA를 달성했습니다.

### 핵심 기여
- 사전학습된 화자 검증 모델 대신 **TTS 목적함수로 end-to-end 학습되는 스피커 인코더** — 합성에 필요한 음색 정보에 최적화
- 참조 음성의 전사 불필요 → 교차 언어(cross-lingual) 클로닝이 자연스러워짐
- VAE 잠재 공간에 Flow를 결합한 Flow-VAE로 기존 mel/코덱 대비 음질 향상
- 스피커 임베딩 확장성: LoRA 기반 감정 제어, 텍스트로 음색 기술(T2V), PVC(전문 보이스 클로닝) 파인튜닝

### 이 논문이 중요한 이유
VALL-E 계열은 "참조 음성을 프롬프트로 이어 붙이는" in-context 방식이라, 참조 음성의 전사가 필요하고 운율까지 복사되는 한계가 있었습니다. MiniMax-Speech는 Deep Voice 2가 시작한 **"화자 = 벡터"** 아이디어를 대규모 AR 코덱 LM 시대에 다시 가져와 결합한 사례로, 오늘 읽는 두 고전의 교차점에 있는 논문입니다.

### 사전 지식
- VALL-E 방식의 코덱 언어 모델 TTS
- VAE와 Normalizing Flow 기초
- 화자 유사도(SIM), WER 등 TTS 평가 지표

### 관련 논문
- [Seed-TTS: A Family of High-Quality Versatile Speech Generation Models (Anastassiou et al., 2024)](https://arxiv.org/abs/2406.02430)
- [CosyVoice 2: Scalable Streaming Speech Synthesis with LLMs (Du et al., 2024)](https://arxiv.org/abs/2412.10117)
- [CosyVoice 3: Towards In-the-wild Speech Generation (Du et al., 2025)](https://arxiv.org/abs/2505.17589)
- [IndexTTS2: Emotionally Expressive and Duration-Controlled AR Zero-Shot TTS (Zhou et al., 2025)](https://arxiv.org/abs/2506.21619)

### 실무 적용
AI 더빙에서 "원본 화자 목소리로 다른 언어를 말하게 하기"는 전사 없는 크로스링구얼 클로닝이 핵심입니다. 제로샷(빠른 온보딩, 무료 체험) → PVC 파인튜닝(유료 고품질 티어)으로 이어지는 제품 등급 설계, 감정 LoRA를 통한 연출 컨트롤 UI 등 그대로 제품 기능 로드맵으로 옮길 수 있는 구조입니다.

---

## 추천 읽기 순서
1. **Deep Voice 2 (2017)** — "목소리를 벡터로 표현한다"는 발상부터 이해
2. **EnCodec (2022)** — "음성을 토큰으로 표현한다"는 두 번째 축 이해
3. **MiniMax-Speech (2025)** — 두 축(화자 벡터 + 음성 토큰 LM)이 어떻게 결합되어 최신 SOTA가 되는지 확인

## 핵심 테이크어웨이
- **Q. 보이스 클로닝은 결국 무엇을 풀어야 하는 문제인가?** → "누가(화자)"와 "무엇을/어떻게(내용·운율)"를 분리해 표현하고 다시 합치는 문제입니다.
- **Q. 왜 코덱이 중요한가?** → 음성이 이산 토큰이 되면서 LLM의 스케일링 법칙이 음성 합성에 그대로 적용되었습니다. 코덱 품질·비트레이트가 음질과 비용의 상한을 정합니다.
- **Q. in-context 프롬프트 vs 스피커 인코더, 무엇이 나은가?** → 프롬프트 방식은 운율까지 복제하지만 전사가 필요하고, 학습형 스피커 인코더는 음색만 깔끔히 분리해 크로스링구얼·제어성에 강합니다. 최신 모델은 두 방식을 모두 지원하는 방향으로 수렴 중입니다.
- **Q. 제품 관점의 가설은?** → "3초 클로닝"은 이제 코모디티입니다. 차별화는 감정·연출 제어, 다국어 음색 유지, 동의/워터마크 기반 신뢰 설계에서 나옵니다.

## 다음 토픽과의 연결
다음은 **Module 6: LLM for Natural Language Generation — GPT Architecture and Scaling Laws**입니다. 오늘 본 EnCodec → 코덱 LM → MiniMax-Speech의 흐름은 "모든 모달리티를 토큰으로 바꾸고 Transformer를 스케일한다"는 GPT 패러다임의 음성 버전입니다. 다음 날 GPT-3와 Scaling Laws를 읽으면, 왜 음성 합성 모델들이 파라미터·데이터를 키우는 방향으로 발전했는지 근본 원리를 확인할 수 있습니다.
