# Daily AI Paper Recommendations

> **Date:** 2026-06-05
> **Module:** Module 4: NLP and Speech Data
> **Topic:** Speech Recognition Fundamentals

---

## Paper 1 (Classic): Sequence Transduction with Recurrent Neural Networks
- **Authors:** Alex Graves
- **Year:** 2012
- **arXiv:** https://arxiv.org/abs/1211.3711
- **PDF:** [./rnn-transducer-graves-2012.pdf](./rnn-transducer-graves-2012.pdf)
- **Citation Count:** ~2,400+

### 요약
입력 시퀀스와 출력 시퀀스의 길이가 다르고 정렬(alignment) 정보가 주어지지 않은 변환 문제를 RNN만으로 종단간(end-to-end) 학습하는 방법을 제안한다. CTC가 출력 라벨 간 독립성을 가정하는 한계를 넘어, 별도의 예측 네트워크(prediction network)를 두어 출력들 사이의 의존성까지 모델링한 RNN Transducer(RNN-T) 구조를 처음으로 정식화했다.

### 핵심 기여
- 입력-출력 정렬을 사전에 알 필요 없이 임의 길이의 이산 출력 시퀀스를 생성하는 확률적 변환 프레임워크 제안
- Transcription network(인코더)와 Prediction network(언어모델 역할)를 결합해 CTC의 조건부 독립 가정을 제거
- TIMIT 음소 인식에서 경쟁력 있는 성능을 보이며 종단간 시퀀스 변환의 일반적 토대 마련

### 이 논문이 중요한 이유
RNN-T는 오늘날 구글, 애플 등 대부분의 온디바이스/스트리밍 음성 인식 시스템의 사실상 표준 디코딩 구조다. 프레임을 받자마자 출력을 낼 수 있는 스트리밍 특성 덕분에 실시간 ASR의 핵심 알고리즘이 되었고, AI 엔지니어가 음성 파이프라인의 지연(latency)과 정확도 트레이드오프를 이해하려면 반드시 알아야 한다.

### 사전 지식
- RNN/LSTM의 기본 구조와 시퀀스 모델링
- CTC(Connectionist Temporal Classification)의 blank 토큰과 정렬 마진화 개념
- 동적 프로그래밍(forward-backward) 기반 손실 계산

### 관련 논문
- [Connectionist Temporal Classification (Graves et al., 2006)](https://www.cs.toronto.edu/~graves/icml_2006.pdf)
- [Streaming End-to-end Speech Recognition for Mobile Devices (He et al., 2018)](https://arxiv.org/abs/1811.06621)

### 실무 적용
스트리밍 자막, 음성 비서(웨이크워드 이후 실시간 받아쓰기), 콜센터 실시간 STT 등 지연이 중요한 모든 서비스의 디코더로 쓰인다. NVIDIA NeMo, ESPnet, k2/icefall 등 주요 ASR 프레임워크가 RNN-T(및 변형 TDT) 손실을 기본 제공한다.

---

## Paper 2 (Classic): Attention-Based Models for Speech Recognition
- **Authors:** Jan Chorowski, Dzmitry Bahdanau, Dmitriy Serdyuk, Kyunghyun Cho, Yoshua Bengio
- **Year:** 2015
- **arXiv:** https://arxiv.org/abs/1506.07503
- **PDF:** [./attention-based-asr-chorowski-2015.pdf](./attention-based-asr-chorowski-2015.pdf)
- **Citation Count:** ~2,000+

### 요약
기계 번역에서 등장한 어텐션 메커니즘을 음성 인식에 적용하면서 발생하는 문제(긴 입력에서의 정렬 붕괴)를 해결하기 위해 위치 인식(location-aware) 어텐션을 제안한다. 이전 스텝의 어텐션 가중치를 컨볼루션으로 가공해 다음 정렬을 안내함으로써, 단조롭게(monotonic) 진행해야 하는 음성 정렬 특성을 반영한다.

### 핵심 기여
- 콘텐츠 기반 어텐션에 위치(location) 특징을 더한 하이브리드 어텐션 메커니즘 제안
- 긴 발화에서도 정렬이 무너지지 않도록 score normalization과 sharpening 기법 도입
- TIMIT 음소 인식에서 어텐션 기반 종단간 모델의 실용성을 입증, 이후 LAS/Transformer ASR의 직접적 토대

### 이 논문이 중요한 이유
어텐션이 "왜 음성에서는 그대로 쓰면 안 되는가"를 명확히 보여주고, 음성 고유의 단조 정렬 문제를 어텐션 설계로 푸는 첫 사례다. 트랜스포머·Conformer 기반 ASR과 음성 합성의 어텐션 설계 사고를 이해하는 출발점으로, 시퀀스-투-시퀀스 음성 모델을 다루는 엔지니어에게 필독이다.

### 사전 지식
- Bahdanau 어텐션(2014)과 인코더-디코더 구조
- 음성의 단조 정렬(입력 프레임 → 출력 토큰이 시간순으로 진행) 특성
- 1D 컨볼루션과 소프트맥스 기반 어텐션 가중치 계산

### 관련 논문
- [Neural Machine Translation by Jointly Learning to Align and Translate (Bahdanau et al., 2014)](https://arxiv.org/abs/1409.0473)
- [Listen, Attend and Spell (Chan et al., 2015)](https://arxiv.org/abs/1508.01211)

### 실무 적용
어텐션 정렬 안정화 기법은 종단간 TTS(Tacotron 계열)와 ASR 모두에서 발화가 길어질 때 단어 반복/누락을 막는 데 직접 활용된다. 위치 인식 어텐션 아이디어는 이후 monotonic attention, RNN-T 등 스트리밍 정렬 연구로 이어졌다.

---

## Paper 3 (Recent): Less is More: Accurate Speech Recognition & Translation without Web-Scale Data
- **Authors:** Krishna C. Puvvada, Piotr Żelasko, He Huang, Oleksii Hrinchuk, Nithin Rao Koluguri, Kunal Dhawan, Somshubra Majumdar, Elena Rastorgueva, Zhehuai Chen, Vitaly Lavrukhin, Jagadeesh Balam, Boris Ginsburg (NVIDIA)
- **Year:** 2024
- **arXiv:** https://arxiv.org/abs/2406.19674
- **PDF:** [./canary-less-is-more-puvvada-2024.pdf](./canary-less-is-more-puvvada-2024.pdf)
- **Citation Count:** ~70+ (2024 발표, 빠르게 인용 증가 중)

### 요약
NVIDIA Canary 모델을 소개하는 논문으로, Whisper·OWSM·SeamlessM4T 대비 한 자릿수 배 적은 데이터(약 86k 시간)만으로 영어·독일어·프랑스어·스페인어 ASR과 번역에서 SOTA를 달성한다. 핵심은 FastConformer 인코더-디코더 구조와 데이터 효율을 끌어올리는 학습 기법 조합이다.

### 핵심 기여
- 웹 스케일 데이터 없이도 합성 번역 데이터 + 정교한 학습 전략으로 SOTA 달성, "데이터 양보다 질·전략"임을 실증
- data-balancing, dynamic data blending, dynamic bucketing, noise-robust fine-tuning 등 실전 학습 레시피 공개
- ASR과 AST(음성 번역)를 단일 모델로 통합, 모델·가중치·학습 코드 오픈소스화

### 이 논문이 중요한 이유
거대 데이터 확보가 어려운 현실에서 "적은 데이터로 강한 다국어 음성 모델 만들기"의 구체적 청사진을 제시한다. 클래식(RNN-T, 어텐션 ASR)에서 발전한 FastConformer가 실제 프로덕션 다국어 ASR/번역으로 어떻게 수렴하는지 보여주는 2024년 대표작이다.

### 사전 지식
- Conformer/FastConformer 인코더 구조
- Whisper식 약지도(weak supervision) 대규모 ASR 학습 패러다임
- 합성 데이터(기계 번역 기반) 증강과 데이터 블렌딩 개념

### 관련 논문
- [Robust Speech Recognition via Large-Scale Weak Supervision / Whisper (Radford et al., 2022)](https://arxiv.org/abs/2212.04356)
- [Conformer: Convolution-augmented Transformer for Speech Recognition (Gulati et al., 2020)](https://arxiv.org/abs/2005.08100)

### 실무 적용
NVIDIA NeMo로 바로 사용 가능하며, 다국어 자막·실시간 번역·콜센터 분석 등에 적용된다. 데이터 효율 학습 레시피는 자체 도메인 데이터가 제한적인 스타트업이 자신만의 ASR을 파인튜닝할 때 직접 참고할 수 있는 실용 가이드다.

---

## 추천 읽기 순서
1. **Attention-Based Models for Speech Recognition (2015)** — 어텐션을 음성에 적용할 때의 근본 문제와 해법을 먼저 이해
2. **Sequence Transduction with RNNs / RNN-T (2012)** — 스트리밍·정렬 자유 변환의 또 다른 축을 학습 (논문은 더 이르지만 개념은 어텐션과 상보적)
3. **Less is More / Canary (2024)** — 위 두 흐름이 FastConformer로 수렴한 현대 다국어 프로덕션 모델로 마무리

### 핵심 테이크어웨이
- 음성 인식의 본질은 "정렬 없는 시퀀스 변환"이며, 이를 푸는 두 갈래가 (1) Transducer/CTC 계열의 단조 정렬과 (2) 어텐션 기반 정렬이다.
- 음성은 단조 정렬 특성이 있어, 일반 어텐션을 그대로 쓰면 깨진다 — 위치 인식·monotonic 설계가 필요하다.
- 2024년 현재는 두 흐름을 흡수한 FastConformer/Conformer 인코더가 표준이며, 경쟁력은 데이터의 양보다 학습 전략과 데이터 품질에서 갈린다.

### 다음 토픽과의 연결
다음 모듈(Module 5: TTS and STT Model Development)에서는 오늘 다룬 인식(STT) 기반을 토대로, 현대적 STT(Whisper, wav2vec 2.0)와 그 반대 방향인 신경망 TTS로 확장한다. RNN-T의 스트리밍 사고와 어텐션 정렬 안정화는 TTS의 음향 정렬 설계에서 다시 등장한다.
