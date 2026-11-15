# Daily AI Paper Recommendations

> **Date:** 2026-11-16
> **Module:** Module 5: TTS and STT Model Development
> **Topic:** Neural Text-to-Speech

---

## Paper 1 (Classic): Deep Voice: Real-time Neural Text-to-Speech
- **Authors:** Sercan Ö. Arık, Mike Chrzanowski, Adam Coates, Gregory Diamos, Andrew Gibiansky, Yongguo Kang, Xian Li, John Miller, Andrew Ng, Jonathan Raiman, Shubho Sengupta, Mohammad Shoeybi
- **Year:** 2017
- **arXiv:** https://arxiv.org/abs/1702.07825
- **PDF:** [./deep-voice-arik-2017.pdf](./deep-voice-arik-2017.pdf)
- **Citation Count:** ~800+

### 요약
Baidu가 전통적인 TTS 파이프라인(문자→음소 변환, 음소 경계 분할, 음소 길이 예측, 기본 주파수(F0) 예측, 오디오 합성)의 모든 구성요소를 딥러닝 모듈로 대체한 시스템입니다. 특히 WaveNet 변형 보코더를 최적화된 CPU/GPU 커널로 구현해 최대 400배 빠른 실시간 추론을 달성했습니다.

### 핵심 기여
- 수작업 특징 공학 없이 TTS 파이프라인 5개 모듈을 모두 신경망으로 구성
- WaveNet 추론을 실시간으로 만드는 커널 수준 최적화(persistent kernel 등) 제시
- 새 데이터셋/화자에 대해 최소한의 도메인 지식으로 재학습 가능한 구조

### 이 논문이 중요한 이유
"신경망 TTS를 프로덕션에서 실시간으로 돌릴 수 있는가?"라는 질문에 처음으로 실증적인 답을 준 논문입니다. 모델 품질뿐 아니라 추론 비용·지연시간을 엔지니어링 문제로 다룬 관점은 지금의 스트리밍 TTS 설계에도 그대로 이어집니다.

### 사전 지식
전통적 파라메트릭/연결 합성(concatenative) TTS 구조, WaveNet의 dilated causal convolution, CTC 기반 음소 정렬 개념

### 관련 논문
- [WaveNet: A Generative Model for Raw Audio (van den Oord et al., 2016)](https://arxiv.org/abs/1609.03499)
- [Deep Voice 2: Multi-Speaker Neural Text-to-Speech (Arik et al., 2017)](https://arxiv.org/abs/1705.08947)
- [Deep Voice 3: Scaling Text-to-Speech with Convolutional Sequence Learning (Ping et al., 2017)](https://arxiv.org/abs/1710.07654)

### 실무 적용
모듈형 파이프라인은 디버깅·부분 교체가 쉬워, 발음 사전 커스터마이징이 중요한 B2B 더빙/내레이션 서비스에서 여전히 유효한 설계 패턴입니다. GPU 커널 최적화 접근은 추론 비용(COGS) 절감 전략의 원형입니다.

---

## Paper 2 (Classic): Efficient Neural Audio Synthesis (WaveRNN)
- **Authors:** Nal Kalchbrenner, Erich Elsen, Karen Simonyan, Seb Noury, Norman Casagrande, Edward Lockhart, Florian Stimberg, Aäron van den Oord, Sander Dieleman, Koray Kavukcuoglu
- **Year:** 2018
- **arXiv:** https://arxiv.org/abs/1802.08435
- **PDF:** [./wavernn-kalchbrenner-2018.pdf](./wavernn-kalchbrenner-2018.pdf)
- **Citation Count:** ~1,000+

### 요약
단일 레이어 RNN과 듀얼 소프트맥스(16비트 오디오를 상위/하위 8비트로 분할)로 WaveNet 수준의 음질을 유지하면서 샘플 생성 속도를 크게 높인 보코더입니다. 가중치 프루닝(sparse WaveRNN)과 subscale 생성 기법으로 모바일 CPU에서도 실시간 합성이 가능함을 보였습니다.

### 핵심 기여
- 샘플당 연산량(N × 연산 비용)을 줄이는 설계 원칙 정립 — 단일 GRU 기반 WaveRNN
- 대규모 희소화(sparsity 96%+)로 모바일 기기 실시간 오디오 합성 달성
- Subscale WaveRNN: 한 스텝에 여러 샘플을 병렬 생성하는 배치 샘플링 방식

### 이 논문이 중요한 이유
"고품질 생성 모델을 엣지 디바이스로 가져오는 방법"의 교과서입니다. 희소화·구조 단순화를 통한 효율화는 이후 LPCNet, 온디바이스 TTS, 나아가 LLM 경량화 논의와도 맞닿아 있습니다.

### 사전 지식
자기회귀(autoregressive) 생성 모델, GRU/RNN 구조, μ-law 양자화, 모델 프루닝 개념

### 관련 논문
- [LPCNet: Improving Neural Speech Synthesis Through Linear Prediction (Valin & Skoglund, 2019)](https://arxiv.org/abs/1810.11846)
- [Parallel WaveNet: Fast High-Fidelity Speech Synthesis (van den Oord et al., 2017)](https://arxiv.org/abs/1711.10433)
- [HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis (Kong et al., 2020)](https://arxiv.org/abs/2010.05646)

### 실무 적용
온디바이스 음성 비서, 오프라인 TTS, 저지연 실시간 통역 등 클라우드 비용·프라이버시 제약이 있는 제품에서 보코더 선택 기준(품질 vs 속도 vs 메모리)을 세우는 데 직접적인 참고가 됩니다.

---

## Paper 3 (Recent): E2 TTS: Embarrassingly Easy Fully Non-Autoregressive Zero-Shot TTS
- **Authors:** Sefik Emre Eskimez, Xiaofei Wang, Manthan Thakker, Canrun Li, Chung-Hsien Tsai, Zhen Xiao, Hemin Yang, Zirun Zhu, Min Tang, Xu Tan, Yanqing Liu, Sheng Zhao, Naoyuki Kanda
- **Year:** 2024 (SLT 2024)
- **arXiv:** https://arxiv.org/abs/2406.18009
- **PDF:** [./e2-tts-eskimez-2024.pdf](./e2-tts-eskimez-2024.pdf)
- **Citation Count:** ~200+

### 요약
텍스트를 필러 토큰으로 패딩해 멜 스펙트로그램 길이에 맞춘 문자 시퀀스로 만들고, 플로우 매칭 기반 생성기를 오디오 인필링(infilling) 과제로 학습하는 극단적으로 단순한 제로샷 TTS입니다. 음소 변환(G2P), 길이 예측 모델, 정렬 탐색(MAS) 없이도 Voicebox·NaturalSpeech 3와 비슷하거나 더 나은 성능을 보고했습니다.

### 핵심 기여
- G2P·duration model·alignment를 모두 제거한 완전 비자기회귀(NAR) 구조
- 플로우 매칭 + 인필링 학습으로 짧은 음성 프롬프트만으로 화자 복제(zero-shot)
- 변형 모델: E2 TTS X1(프롬프트 전사 불필요), X2(단어별 발음 지정 가능)

### 이 논문이 중요한 이유
"구성요소를 덜어낼수록 더 잘 된다"는 2024년 TTS 트렌드의 대표 사례이며, 이 커리큘럼에서 여러 번 다룬 F5-TTS의 직접적 전신입니다. Deep Voice(모듈 분해)→E2 TTS(모듈 제거)로 이어지는 설계 철학의 변화를 읽을 수 있습니다.

### 사전 지식
Flow Matching / Conditional Flow Matching, Voicebox의 인필링 학습 방식, 멜 스펙트로그램과 보코더의 역할

### 관련 논문
- [Voicebox: Text-Guided Multilingual Universal Speech Generation at Scale (Le et al., 2023)](https://arxiv.org/abs/2306.15687)
- [F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching (Chen et al., 2024)](https://arxiv.org/abs/2410.06885)
- [Flow Matching for Generative Modeling (Lipman et al., 2022)](https://arxiv.org/abs/2210.02747)

### 실무 적용
다국어 AI 더빙처럼 언어별 G2P 구축 비용이 큰 서비스에서 문자 기반 입력만으로 확장할 수 있다는 점이 핵심 가치입니다. 오픈소스 체크포인트(F5-TTS 레포)로 빠른 PoC가 가능하나, 화자 복제 악용 방지(동의·워터마킹) 정책을 함께 설계해야 합니다.

---

## 추천 읽기 순서
1. **Deep Voice** — 전통 TTS 파이프라인이 어떻게 신경망 모듈로 분해되는지 전체 그림 파악
2. **WaveRNN** — 파이프라인의 마지막 단계(보코더)를 효율화하는 방법 이해
3. **E2 TTS** — 앞의 모듈들을 대부분 걷어낸 최신 end-to-end 제로샷 접근과 비교

## 핵심 테이크어웨이
- TTS의 발전사는 "모듈 분해 → 신경망 대체 → 모듈 제거·단순화"의 흐름이다.
- 품질만큼 **추론 속도·비용·디바이스 제약**이 제품 성패를 가른다 (Deep Voice, WaveRNN의 공통 메시지).
- 플로우 매칭 기반 NAR 모델은 제로샷 화자 복제를 단순한 구조로 가능하게 했지만, 그만큼 윤리·보안 가드레일이 제품 요구사항이 된다.

## 다음 토픽과의 연결
다음 주제인 **Voice Cloning and Speech Synthesis**에서는 E2 TTS가 보여준 제로샷 화자 복제를 VALL-E 계열의 코덱 언어모델 접근과 비교합니다. "플로우 매칭(NAR) vs 코덱 LM(AR)" 중 어떤 방식이 표현력·안정성·지연시간에서 유리한지 질문을 가지고 넘어가 보세요.
