# Daily AI Paper Recommendations

> **Date:** 2026-08-28
> **Module:** Module 5: TTS and STT Model Development
> **Topic:** Voice Cloning and Speech Synthesis

---

## Paper 1 (Classic): AudioLM: a Language Modeling Approach to Audio Generation
- **Authors:** Zalán Borsos, Raphaël Marinier, Damien Vincent, Eugene Kharitonov, Olivier Pietquin, Matt Sharifi, Dominik Roblek, Olivier Teboul, David Grangier, Marco Tagliasacchi, Neil Zeghidour
- **Year:** 2022 (TASLP 2023)
- **arXiv:** https://arxiv.org/abs/2209.03143
- **PDF:** [./audiolm-borsos-2022.pdf](./audiolm-borsos-2022.pdf)
- **Citation Count:** ~1,300+

### 요약
AudioLM은 오디오 생성을 "언어 모델링" 문제로 재정의한 논문이다. 원시 파형을 두 종류의 이산 토큰(w2v-BERT에서 뽑은 의미 토큰 + SoundStream 코덱의 음향 토큰)으로 변환한 뒤, 이 토큰 시퀀스를 GPT처럼 자기회귀적으로 예측한다. 텍스트 전사나 라벨 없이 학습했음에도 3초 프롬프트만으로 화자의 목소리, 억양, 녹음 환경까지 유지한 자연스러운 음성 연속을 생성한다.

### 핵심 기여
- 의미 토큰(장기 구조)과 음향 토큰(음질)을 분리한 **하이브리드 토크나이제이션** 설계 — 이후 거의 모든 음성 LM의 표준이 됨
- 텍스트 없이 오디오만으로 학습하는 **완전 비지도 생성 파이프라인** 제시
- 3단계 계층적 생성(의미 → 거친 음향 → 세밀한 음향)으로 장기 일관성과 고음질을 동시에 확보
- 피아노 음악까지 동일 프레임워크로 생성하며 접근법의 도메인 일반성을 입증

### 이 논문이 중요한 이유
오늘날 VALL-E, CosyVoice, Seed-TTS, MaskGCT 등 사실상 모든 최신 음성 클로닝 모델은 "음성을 토큰으로 바꾸고 LLM으로 예측한다"는 AudioLM의 문법 위에 서 있다. AI 엔지니어가 음성 생성 스택을 이해하려면 코덱 → 토큰 → LM이라는 계층 구조를 반드시 체화해야 하며, 그 원형이 이 논문이다. 또한 "왜 의미 토큰과 음향 토큰을 나누는가"라는 질문은 지금도 아키텍처 설계의 핵심 논쟁점이다.

### 사전 지식
- 신경망 오디오 코덱(SoundStream, EnCodec)과 RVQ(Residual Vector Quantization)의 동작 원리
- 자기지도 음성 표현 학습(wav2vec 2.0, w2v-BERT)
- Transformer 디코더의 자기회귀 생성 및 토큰화 개념
- 음성 평가 지표: WER, 화자 유사도(SIM), MOS

### 관련 논문
- [SoundStream: An End-to-End Neural Audio Codec (Zeghidour et al., 2021)](https://arxiv.org/abs/2107.03312)
- [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations (Baevski et al., 2020)](https://arxiv.org/abs/2006.11477)
- [Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers / VALL-E (Wang et al., 2023)](https://arxiv.org/abs/2301.02111)
- [AudioPaLM: A Large Language Model That Can Speak and Listen (Rubenstein et al., 2023)](https://arxiv.org/abs/2306.12925)

### 실무 적용
AI 더빙·오디오북 서비스에서 화자 일관성이 필요한 긴 발화를 생성할 때, AudioLM식 2단 토큰 구조는 "내용은 유지하되 목소리는 유지"라는 요구를 자연스럽게 만족시킨다. 실무에서는 의미 토큰 레벨에서 언어·내용을 제어하고 음향 토큰 레벨에서 화자·녹음 환경을 제어하는 방식으로 파이프라인을 분리하면, 다국어 더빙에서 목소리를 고정한 채 언어만 바꾸는 기능을 구현할 수 있다.

---

## Paper 2 (Classic): Voicebox: Text-Guided Multilingual Universal Speech Generation at Scale
- **Authors:** Matthew Le, Apoorv Vyas, Bowen Shi, Brian Karrer, Leda Sari, Rashel Moritz, Mary Williamson, Vimal Manohar, Yossi Adi, Jay Mahadeokar, Wei-Ning Hsu
- **Year:** 2023 (NeurIPS 2023)
- **arXiv:** https://arxiv.org/abs/2306.15687
- **PDF:** [./voicebox-le-2023.pdf](./voicebox-le-2023.pdf)
- **Citation Count:** ~600+

### 요약
Voicebox는 음성 생성을 자기회귀 토큰 예측이 아니라 **텍스트 조건부 음성 인페인팅(infilling)** 문제로 정의한 비자기회귀 모델이다. Flow Matching 목적함수로 5만 시간 이상의 정제되지 않은 음성에서 학습되어, 하나의 모델로 제로샷 TTS, 노이즈 제거, 콘텐츠 편집, 스타일 변환, 다양성 샘플링을 모두 수행한다. VALL-E 대비 WER과 화자 유사도가 우수하면서 최대 20배 빠르다.

### 핵심 기여
- 음성 생성에 **Conditional Flow Matching**을 도입해 비자기회귀 고속 생성과 고품질을 동시 달성
- "마스킹된 구간을 앞뒤 문맥으로 채운다"는 단일 인페인팅 목표로 **여러 태스크를 통합**(task-agnostic 학습)
- 미래 문맥까지 조건으로 쓸 수 있어 GPT식 인컨텍스트 학습보다 편집·수정 작업에 유리함을 입증
- 필터링하지 않은 대규모 in-the-wild 데이터로 스케일링이 가능함을 실증

### 이 논문이 중요한 이유
음성 생성의 두 축인 **이산 토큰 자기회귀(AudioLM/VALL-E 계열)**와 **연속 잠재 공간 비자기회귀(Voicebox/F5-TTS 계열)** 중 후자의 대표 논문이다. 지연시간(latency)과 실시간성이 제품 요구사항인 상황에서 어떤 패러다임을 고를지 판단하려면 두 계열을 모두 알아야 한다. 오늘날 F5-TTS, E2-TTS, Seed-TTS_DiT 등이 모두 이 flow matching 계보에 속한다.

### 사전 지식
- Flow Matching / Continuous Normalizing Flow와 확산 모델(diffusion)의 관계
- 마스킹 기반 학습(BERT식 인페인팅)과 비자기회귀 생성의 트레이드오프
- 멜 스펙트로그램 및 보코더(HiFi-GAN 등) 기반 파이프라인
- Classifier-free guidance 개념

### 관련 논문
- [Flow Matching for Generative Modeling (Lipman et al., 2022)](https://arxiv.org/abs/2210.02747)
- [E2 TTS: Embarrassingly Easy Fully Non-Autoregressive Zero-Shot TTS (Eskimez et al., 2024)](https://arxiv.org/abs/2406.18009)
- [NaturalSpeech 2: Latent Diffusion Models are Natural and Zero-Shot Speech Synthesizers (Shen et al., 2023)](https://arxiv.org/abs/2304.09116)
- [F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching (Chen et al., 2024)](https://arxiv.org/abs/2410.06885)

### 실무 적용
더빙 서비스에서 "이미 만들어진 음성의 특정 단어만 교체"하는 리테이크(retake) 기능은 Voicebox식 인페인팅이 정확히 겨냥하는 문제다. 전체를 재생성하면 앞뒤 톤이 어긋나지만, 인페인팅은 주변 문맥을 조건으로 삼아 경계가 매끄럽다. 또한 비자기회귀 구조라 배치 추론 시 GPU 활용률이 높아 대량 렌더링 파이프라인의 단가를 낮추는 데 유리하다.

---

## Paper 3 (Recent): Seed-TTS: A Family of High-Quality Versatile Speech Generation Models
- **Authors:** Philip Anastassiou, Jiawei Chen, Jitong Chen, Yuanzhe Chen, Zhuo Chen, Ziyi Chen, Jian Cong, Lelai Deng, Chuang Ding, Lu Gao, et al. (ByteDance Seed Team)
- **Year:** 2024
- **arXiv:** https://arxiv.org/abs/2406.02430
- **PDF:** [./seed-tts-anastassiou-2024.pdf](./seed-tts-anastassiou-2024.pdf)
- **Citation Count:** ~400+

### 요약
Seed-TTS는 ByteDance가 공개한 대규모 자기회귀 음성 생성 모델 패밀리로, 제로샷 인컨텍스트 학습만으로 사람과 구분하기 어려운 수준의 음성을 만들어낸다. 감정·억양·화자 유사도를 세밀하게 제어할 수 있고, 자기 증류(self-distillation)를 통한 음색 분리와 **강화학습 기반 후처리**로 강건성과 제어성을 끌어올렸다. 완전 확산 기반 비자기회귀 변형인 Seed-TTS_DiT도 함께 제시한다.

### 핵심 기여
- 자기회귀 LM 기반 음성 생성을 인간 수준(화자 유사도·자연스러움에서 실제 녹음과 통계적으로 구분 불가)으로 끌어올림
- 음성 생성에 **RLHF/RL 후처리**를 본격 적용해 발음 오류·환각을 줄이고 화자 유사도를 개선
- 자기 증류로 음색(timbre)과 내용을 분리해 음성 변환(voice conversion)·편집 성능 확보
- 자기회귀 버전과 확산 버전(Seed-TTS_DiT)을 동일 조건에서 비교해 두 패러다임의 장단점을 정리

### 이 논문이 중요한 이유
"제로샷 음성 클로닝이 실제 제품 수준에 도달했다"는 것을 보여준 기준점이다. 특히 LLM 학습에서 익숙한 SFT → RL 파이프라인이 음성 도메인에도 그대로 이식된다는 점은, AI 엔지니어가 텍스트 LLM의 정렬(alignment) 기법을 다른 모달리티로 전이시킬 때 참고할 강력한 사례다. 동시에 이 정도 품질은 딥페이크 음성 리스크를 실무 이슈로 만들며, 워터마킹·동의 관리 설계의 필요성을 제기한다.

### 사전 지식
- VALL-E 계열 뉴럴 코덱 언어 모델의 구조(오늘의 AudioLM 논문이 선행 지식)
- LLM 정렬 기법: RLHF, DPO, PPO의 기본 개념
- Diffusion Transformer(DiT)와 잠재 확산 모델
- 지식 증류(distillation) 및 표현 분리(disentanglement) 개념

### 관련 논문
- [CosyVoice 2: Scalable Streaming Speech Synthesis with Large Language Models (Du et al., 2024)](https://arxiv.org/abs/2412.10117)
- [MaskGCT: Zero-Shot Text-to-Speech with Masked Generative Codec Transformer (Wang et al., 2024)](https://arxiv.org/abs/2409.00750)
- [Direct Preference Optimization (Rafailov et al., 2023)](https://arxiv.org/abs/2305.18290)
- [Voice Cloning: Comprehensive Survey (2025)](https://arxiv.org/abs/2505.00579)

### 실무 적용
B2B 더빙/보이스 SaaS에서 "고객이 3초 샘플만 올려도 자기 목소리로 콘텐츠를 만든다"는 온보딩 경험을 설계할 때 Seed-TTS의 접근이 직접적인 레퍼런스가 된다. 특히 RL 후처리는 프로덕션에서 가장 아픈 지점인 **간헐적 발음 오류·문장 누락**을 줄이는 실용적 수단이다. 다만 제품화 시에는 화자 동의 검증 플로우, 오디오 워터마킹, 남용 탐지 로그를 함께 설계해야 한다.

---

## 추천 읽기 순서

1. **AudioLM (2022)** — 먼저 "음성을 토큰으로 바꿔 언어 모델로 푼다"는 기본 문법을 익힌다. 이후 모든 논문의 전제다.
2. **Voicebox (2023)** — 반대 패러다임인 비자기회귀 flow matching을 보며 두 계열의 트레이드오프를 대조한다.
3. **Seed-TTS (2024)** — 두 패러다임이 스케일과 RL을 만나 제품 수준에 도달하는 지점을 확인한다.

## 핵심 테이크어웨이

- **음성 클로닝의 본질은 표현 분리다.** 내용(무엇을 말하는가)과 음색(누가 말하는가)을 어떤 층위에서 나누느냐가 아키텍처를 결정한다.
- **두 패러다임은 우열이 아니라 제약의 문제다.** 자기회귀는 표현력·감정 제어에서, 비자기회귀는 속도·편집 가능성에서 강하다. 실시간 대화형이면 스트리밍 AR, 대량 배치 렌더링이면 NAR이 유리한 경우가 많다.
- **정렬 기법은 모달리티를 넘는다.** RLHF/RL 후처리가 텍스트에서 음성으로 이식되어 실질적 품질 개선을 냈다는 사실은, 다른 생성 도메인에도 같은 레시피가 통할 가능성을 시사한다.
- **품질이 임계점을 넘으면 문제는 기술에서 거버넌스로 이동한다.** 화자 동의, 워터마킹, 남용 탐지는 이제 부가 기능이 아니라 제품 요구사항이다.

## 다음 토픽과의 연결

다음 모듈은 **LLM for Natural Language Generation**으로 이어진다. 오늘 본 세 논문은 모두 "LLM의 방법론(토큰화, 자기회귀 예측, 인컨텍스트 학습, RLHF)을 음성에 그대로 적용한 결과물"이었다. 즉 음성 생성의 발전은 상당 부분 LLM 발전의 파생물이다. 다음 토픽에서 GPT 아키텍처와 스케일링 법칙, 그리고 InstructGPT/DPO 계열의 정렬 기법을 원류에서 다시 보면, 오늘 읽은 Seed-TTS의 RL 파이프라인이 왜 그렇게 설계되었는지가 명확해진다.
