# Daily AI Paper Recommendations

> **Date:** 2026-06-07
> **Module:** Module 5: TTS and STT Model Development
> **Topic:** Neural Text-to-Speech

---

## Paper 1 (Classic): Tacotron: Towards End-to-End Speech Synthesis
- **Authors:** Yuxuan Wang, RJ Skerry-Ryan, Daisy Stanton, Yonghui Wu, Ron J. Weiss, Navdeep Jaitly, Zongheng Yang, Ying Xiao, Zhifeng Chen, Samy Bengio, Quoc Le, Yannis Agiomyrgiannakis, Rob Clark, Rif A. Saurous
- **Year:** 2017
- **arXiv:** https://arxiv.org/abs/1703.10135
- **PDF:** [./tacotron-wang-2017.pdf](./tacotron-wang-2017.pdf)
- **Citation Count:** ~4,000+ (approximate)

### 요약
Tacotron은 문자(character) 시퀀스를 입력받아 스펙트로그램을 직접 생성하는 최초의 완전한 end-to-end 신경망 TTS 모델이다. 기존 TTS 파이프라인이 텍스트 분석, 음향 모델, 보코더 등 여러 단계로 분리되어 각 단계의 오류가 누적되던 문제를, seq2seq + attention 구조 하나로 통합해 <텍스트, 오디오> 쌍만으로 처음부터 학습할 수 있게 했다. 미국 영어 기준 MOS 3.82를 기록하며 당시 상용 파라메트릭 시스템을 자연스러움에서 앞섰다.

### 핵심 기여
- 문자 → 스펙트로그램을 단일 신경망으로 학습하는 end-to-end TTS 패러다임 제시
- CBHG 인코더, attention 기반 디코더, "reduction factor"(한 스텝에 여러 프레임 예측)로 수렴 속도와 안정성 확보
- 프레임 단위 생성으로 sample 단위 자기회귀 방식 대비 빠른 추론, 복잡한 언어/음향 피처 엔지니어링 제거

### 이 논문이 중요한 이유
현대 신경망 TTS의 출발점이다. Tacotron 2, FastSpeech, VITS 등 이후 거의 모든 TTS 연구가 이 attention 기반 seq2seq 프레임워크를 변형·발전시켰다. AI 엔지니어가 음성 합성 스택을 이해하려면 "왜 end-to-end로 갔는가"라는 질문의 원점을 반드시 알아야 한다.

### 사전 지식
- Seq2seq와 어텐션 메커니즘(Bahdanau attention)
- 멜 스펙트로그램, STFT 등 음성 신호 표현
- RNN/GRU, 그리고 Griffin-Lim 같은 위상 복원 보코더의 기본 개념

### 관련 논문
- [Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions / Tacotron 2 (Shen et al., 2018)](https://arxiv.org/abs/1712.05884)
- [Neural Machine Translation by Jointly Learning to Align and Translate / Bahdanau Attention (Bahdanau et al., 2014)](https://arxiv.org/abs/1409.0473)

### 실무 적용
음성 비서, 오디오북, 더빙, 내비게이션 안내 음성 등 합성 음성이 필요한 거의 모든 제품의 기초 아키텍처다. 실무에서는 Tacotron 계열로 멜 스펙트로그램을 만들고 별도 신경망 보코더(WaveNet/HiFi-GAN)로 파형을 복원하는 2단계 구성이 오랫동안 표준이었다.

---

## Paper 2 (Classic): Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search
- **Authors:** Jaehyeon Kim, Sungwon Kim, Jungil Kong, Sungroh Yoon
- **Year:** 2020
- **arXiv:** https://arxiv.org/abs/2005.11129
- **PDF:** [./glow-tts-kim-2020.pdf](./glow-tts-kim-2020.pdf)
- **Citation Count:** ~900+ (approximate)

### 요약
Glow-TTS는 flow 기반 생성 모델을 TTS에 도입해, 외부 정렬기(aligner) 없이도 병렬(parallel) 음성 합성을 가능하게 한 논문이다. 핵심은 Monotonic Alignment Search(MAS)로, flow의 가역성과 동적 계획법을 결합해 텍스트와 음성 잠재 표현 사이의 가장 가능성 높은 단조 정렬을 스스로 찾아낸다. 자기회귀 모델인 Tacotron 2 대비 한 자릿수 이상의 추론 속도 향상을 달성하면서도 비슷한 품질을 유지한다.

### 핵심 기여
- 외부 aligner나 사전학습된 자기회귀 모델 없이 학습 가능한 Monotonic Alignment Search(MAS) 제안
- 정규화 flow를 이용한 병렬 생성으로 빠르고 다양하며 제어 가능한(피치·속도) 음성 합성 구현
- 단조 정렬을 강제해 긴 문장에서도 단어 반복·누락이 적은 robust한 합성 달성

### 이 논문이 중요한 이유
TTS가 "느린 자기회귀"에서 "빠른 병렬 생성"으로 넘어가는 결정적 다리 역할을 했다. 여기서 제안된 MAS는 이후 VITS의 핵심 구성요소로 그대로 채택되었고, 정렬 문제를 학습 안에서 푸는 사고방식은 현대 비자기회귀 TTS 설계의 표준이 되었다.

### 사전 지식
- 정규화 flow(normalizing flow)와 변수 변환 공식, 가역 변환의 개념
- 동적 계획법(dynamic programming), 단조 정렬(monotonic alignment)
- 자기회귀 vs 비자기회귀 생성의 추론 속도/품질 트레이드오프

### 관련 논문
- [Conditional Variational Autoencoder with Adversarial Learning for End-to-End TTS / VITS (Kim et al., 2021)](https://arxiv.org/abs/2106.06103)
- [FastSpeech: Fast, Robust and Controllable Text to Speech (Ren et al., 2019)](https://arxiv.org/abs/1905.09263)

### 실무 적용
실시간 응답이 중요한 음성 에이전트, 대화형 IVR, 게임 NPC 음성 등 저지연 합성이 필요한 환경에 적합하다. 피치·발화 속도를 명시적으로 제어할 수 있어 캐릭터 음성 톤 조정이나 감정 표현 튜닝에도 활용된다.

---

## Paper 3 (Recent): CosyVoice 2: Scalable Streaming Speech Synthesis with Large Language Models
- **Authors:** Zhihao Du, Yuxuan Wang, Qian Chen, Xian Shi, Xiang Lv, Tianyu Zhao, et al. (Alibaba FunAudioLLM Team)
- **Year:** 2024
- **arXiv:** https://arxiv.org/abs/2412.10117
- **PDF:** [./cosyvoice2-du-2024.pdf](./cosyvoice2-du-2024.pdf)
- **Citation Count:** ~150+ (approximate, 빠르게 증가 중)

### 요약
CosyVoice 2는 LLM과 chunk-aware flow matching을 결합한 스트리밍 zero-shot TTS 모델이다. 텍스트를 입력받는 즉시 음성을 생성하는 양방향 스트리밍(bi-streaming)으로 초저지연 합성을 달성하면서도, 운율 자연스러움·내용 일관성·화자 유사도를 사람 수준(human-parity)까지 끌어올렸다. 기존 CosyVoice를 확장해 instruction 기반 제어와 zero-shot 복제 능력을 하나의 모델로 통합했다.

### 핵심 기여
- LLM + chunk-aware flow matching 구조로 스트리밍/비스트리밍을 통합, 양방향 저지연 실시간 합성 구현
- 유한 스칼라 양자화(FSQ) 기반 speech token으로 코드북 활용률과 안정성 개선
- instruction 제어와 zero-shot 화자 복제를 단일 모델에 통합해 더 다양하고 표현력 있는 합성 제공

### 이 논문이 중요한 이유
2024년 TTS 트렌드의 핵심인 "코덱 토큰 + LLM" 패러다임과 flow matching을 결합한 대표 사례다. VALL-E 계열의 LLM-TTS와 F5-TTS 계열의 flow matching을 한 시스템에서 묶어, 실시간 대화형 AI(음성 에이전트, 통역)에 바로 쓸 수 있는 수준의 지연·품질을 보여준다. 고전 TTS에서 현대 production 시스템으로 이어지는 흐름을 이해하는 데 필수적이다.

### 사전 지식
- 신경망 오디오 코덱과 discrete speech token(예: SoundStream, EnCodec)
- Flow matching / conditional flow matching 기반 생성
- LLM의 자기회귀 토큰 생성, zero-shot/in-context voice cloning 개념

### 관련 논문
- [F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching (Chen et al., 2024)](https://arxiv.org/abs/2410.06885)
- [Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers / VALL-E (Wang et al., 2023)](https://arxiv.org/abs/2301.02111)

### 실무 적용
실시간 음성 대화 에이전트, 라이브 통역/더빙, 콜센터 봇 등 지연이 사용자 경험을 좌우하는 제품에 직접 적용된다. zero-shot 복제로 소량의 참조 음성만으로 특정 화자 목소리를 만들 수 있어, 개인화 보이스나 다국어 더빙 파이프라인 구축 비용을 크게 낮춘다.

---

## 추천 읽기 순서
1. **Tacotron (2017)** — end-to-end TTS의 출발점. seq2seq+attention로 왜 통합했는지 먼저 이해한다.
2. **Glow-TTS (2020)** — 자기회귀의 속도 한계를 flow와 MAS로 푸는 병렬 생성으로의 전환을 본다.
3. **CosyVoice 2 (2024)** — 코덱 토큰·LLM·flow matching이 결합된 최신 production급 스트리밍 TTS로 마무리한다.

## 핵심 테이크어웨이
- TTS의 진화는 "분리된 파이프라인 → end-to-end → 병렬/비자기회귀 → LLM·코덱 기반 스트리밍" 흐름으로 정리된다.
- 정렬(alignment) 문제를 어떻게 푸느냐가 각 세대를 가르는 핵심 축이다(외부 정렬 → attention → MAS → 토큰 기반).
- 최신 TTS의 경쟁력은 단순 품질을 넘어 **지연(latency)과 zero-shot 복제 능력**으로 이동했다.

## 다음 토픽과의 연결
다음 토픽인 **Voice Cloning and Speech Synthesis**(Day 13)는 오늘 다룬 zero-shot 합성을 더 깊게 파고든다. CosyVoice 2의 코덱 토큰·LLM 접근은 VALL-E, WaveNet 등 음성 복제·생성 모델로 자연스럽게 이어진다.
