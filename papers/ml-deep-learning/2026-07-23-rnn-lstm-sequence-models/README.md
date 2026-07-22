# Daily AI Paper Recommendations

> **Date:** 2026-07-23
> **Module:** Module 3: Machine Learning and Deep Learning
> **Topic:** RNN, LSTM and Sequence Models

---

## Paper 1 (Classic): Neural Machine Translation by Jointly Learning to Align and Translate
- **Authors:** Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio
- **Year:** 2014
- **arXiv:** https://arxiv.org/abs/1409.0473
- **PDF:** [./neural-machine-translation-align-translate-bahdanau-2014.pdf](./neural-machine-translation-align-translate-bahdanau-2014.pdf)
- **Citation Count:** ~40,000+ (Google Scholar)

### 요약
기존 인코더-디코더 기반 신경망 기계번역은 소스 문장 전체를 하나의 고정 길이 벡터로 압축했는데, 문장이 길어질수록 성능이 급격히 떨어지는 병목이 있었다. 이 논문은 디코더가 각 단어를 생성할 때 소스 문장에서 관련 있는 부분을 스스로 찾아 "정렬(align)"하도록 하는 어텐션(attention) 메커니즘을 최초로 제안했다. 그 결과 긴 문장에서도 성능이 유지되며, 당시 최고 수준의 구문 기반 통계 번역에 필적하는 결과를 얻었다.

### 핵심 기여
- 고정 길이 벡터 병목을 제거하고, 소스 토큰별 은닉 상태의 가중합(context vector)을 매 디코딩 스텝마다 동적으로 계산하는 어텐션 도입
- 정렬(alignment)을 별도 학습 없이 번역과 함께 end-to-end로 공동 학습(soft alignment)
- 긴 문장에서의 성능 저하 문제를 실증적으로 해결하고, 어텐션 가중치 시각화로 해석 가능성 제공

### 이 논문이 중요한 이유
어텐션은 이후 Transformer("Attention Is All You Need")의 직접적 뿌리이며, 오늘날 모든 LLM의 근간이다. AI 엔지니어라면 "왜 어텐션이 필요한가"를 이 논문의 고정 길이 벡터 병목 문제에서 출발해 이해해야 이후 self-attention, cross-attention, KV 캐시 같은 개념이 자연스럽게 연결된다.

### 사전 지식
- RNN/LSTM/GRU의 기본 구조와 인코더-디코더(seq2seq) 프레임워크
- 소프트맥스, 가중합, 그리고 미분 가능한 연산으로 "선택"을 표현하는 방법

### 관련 논문
- [Sequence to Sequence Learning with Neural Networks (Sutskever et al., 2014)](https://arxiv.org/abs/1409.3215)
- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)

### 실무 적용
번역, 요약, 음성인식 등 seq2seq 계열 서비스의 품질을 좌우하는 핵심 원리다. RAG의 retriever-reader, 멀티모달 모델의 cross-attention, 그리고 LLM의 컨텍스트 활용 방식 모두 이 "관련 부분에 집중" 아이디어의 확장으로 볼 수 있다.

---

## Paper 2 (Classic): An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling
- **Authors:** Shaojie Bai, J. Zico Kolter, Vladlen Koltun
- **Year:** 2018
- **arXiv:** https://arxiv.org/abs/1803.01271
- **PDF:** [./empirical-evaluation-tcn-sequence-modeling-bai-2018.pdf](./empirical-evaluation-tcn-sequence-modeling-bai-2018.pdf)
- **Citation Count:** ~5,500+ (Google Scholar)

### 요약
"시퀀스 모델링 = RNN"이라는 통념에 도전한 논문이다. 저자들은 causal + dilated convolution에 residual 연결을 결합한 단순한 구조인 TCN(Temporal Convolutional Network)을 정의하고, 다양한 시퀀스 벤치마크에서 LSTM/GRU와 정면 비교했다. 결과적으로 TCN이 대부분의 과제에서 표준 순환망을 능가하며, 더 긴 유효 메모리를 보였다.

### 핵심 기여
- causal convolution(미래 정보 누출 방지) + dilated convolution(지수적 수용영역 확장) + residual block을 결합한 범용 TCN 구조 정립
- 다양한 태스크(polyphonic music, 언어 모델링, copy/adding 문제 등)에서 RNN 대비 우위를 체계적으로 실증
- 병렬 연산 가능·안정적 그래디언트·유연한 수용영역 등 convolution의 시퀀스 모델링 장점을 명확히 정리

### 이 논문이 중요한 이유
"순차 데이터에는 무조건 RNN"이라는 가정을 데이터로 반박한 대표 사례다. 아키텍처를 선택할 때 통념이 아니라 벤치마크로 검증하는 태도, 그리고 병렬화·수용영역·메모리 관점에서 모델을 비교하는 사고 틀을 길러준다. 이후 Transformer, 그리고 최근 SSM 계열의 "RNN 대체" 흐름을 이해하는 디딤돌이 된다.

### 사전 지식
- CNN의 합성곱/패딩/dilation 개념과 residual 연결
- RNN 계열의 그래디언트 소실·병렬화 한계에 대한 감각

### 관련 논문
- [WaveNet: A Generative Model for Raw Audio (van den Oord et al., 2016)](https://arxiv.org/abs/1609.03499)
- [Long Short-Term Memory (Hochreiter & Schmidhuber, 1997)](https://www.bioinf.jku.at/publications/older/2604.pdf)

### 실무 적용
시계열 예측, 이상탐지, 오디오/센서 신호 처리 등에서 TCN은 지금도 강력하고 학습이 빠른 베이스라인이다. 순환 구조보다 병렬 학습이 쉬워 대규모 시계열 파이프라인이나 엣지 추론 환경에서 실용적 선택지가 된다.

---

## Paper 3 (Recent): Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models
- **Authors:** Soham De, Samuel L. Smith, Anushan Fernando, Aleksandar Botev, et al. (Google DeepMind)
- **Year:** 2024
- **arXiv:** https://arxiv.org/abs/2402.19427
- **PDF:** [./griffin-gated-linear-recurrences-de-2024.pdf](./griffin-gated-linear-recurrences-de-2024.pdf)
- **Citation Count:** ~350+ (Google Scholar)

### 요약
Transformer의 self-attention은 시퀀스 길이에 따라 연산과 KV 캐시가 선형으로 커져 긴 문맥에서 비효율적이다. 이 논문은 게이트가 있는 선형 순환(gated linear recurrence, RG-LRU)만 쓰는 Hawk와, 이를 지역 어텐션(local attention)과 섞은 하이브리드 Griffin을 제안한다. Griffin은 6배 적은 토큰으로도 Llama-2급 성능을 내고, 학습 시 Transformer 수준의 하드웨어 효율을, 추론 시 더 낮은 지연과 높은 처리량을 달성하며 14B까지 확장된다.

### 핵심 기여
- 안정적 학습을 위한 실수 대각 게이트 선형 순환 유닛(RG-LRU) 설계와 이를 쓰는 순수 RNN 모델 Hawk 제안
- Hawk가 다운스트림 태스크에서 Mamba를 능가함을 보이고, 지역 어텐션과 결합한 Griffin으로 Llama-2급 품질 달성
- 학습 시 고정 상태 크기로 메모리를 절약하고, 추론 시 KV 캐시 없이 긴 시퀀스로의 외삽(길이 일반화)과 높은 throughput 확보

### 이 논문이 중요한 이유
LSTM/GRU로 대표되던 "순환" 아이디어가 Mamba·Griffin 같은 현대적 선형 순환으로 부활하는 흐름의 핵심 사례다. Transformer 일변도에서 벗어나 "순환 + 어텐션 하이브리드"가 효율-성능 트레이드오프를 어떻게 재정의하는지 보여주며, 온디바이스·장문맥 LLM 설계에 실질적 시사점을 준다.

### 사전 지식
- self-attention의 O(n) KV 캐시·장문맥 비용 문제
- 선형 순환/상태공간모델(SSM)의 기본 직관, 그리고 게이팅(LSTM/GRU 게이트)의 역할

### 관련 논문
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Gu & Dao, 2023)](https://arxiv.org/abs/2312.00752)
- [Transformers are SSMs / Mamba-2 (Dao & Gu, 2024)](https://arxiv.org/abs/2405.21060)

### 실무 적용
장문맥 챗봇·문서 요약·에이전트처럼 긴 컨텍스트를 낮은 지연으로 처리해야 하는 서비스에서 KV 캐시 부담을 줄이는 대안 아키텍처다. 동일 예산으로 더 긴 입력을 저비용 추론하려는 팀, 특히 온디바이스/엣지 LLM을 검토하는 프로덕트에 직접적 참고가 된다.

---

## 추천 읽기 순서
1. **Bahdanau (2014)** — 어텐션의 출발점. "왜 고정 벡터가 문제인가"를 먼저 이해한다.
2. **Bai/TCN (2018)** — "RNN이 정답인가?"라는 질문으로 시퀀스 모델링 관점을 넓힌다.
3. **Griffin (2024)** — 순환의 현대적 부활. 어텐션과 순환을 섞는 최신 효율화 흐름을 본다.

## 핵심 테이크어웨이
- 시퀀스 모델링의 근본 과제는 "긴 의존성을 어떻게 효율적으로 담을 것인가"이며, 어텐션·합성곱·선형 순환은 각기 다른 답이다.
- 어텐션은 표현력을, TCN은 병렬성과 안정성을, Griffin류는 장문맥 효율을 얻는 서로 다른 트레이드오프를 취한다.
- 아키텍처 선택은 통념이 아니라 벤치마크와 배포 제약(지연·메모리·문맥 길이)으로 정당화해야 한다.

## 다음 토픽과의 연결
어텐션에서 시작된 흐름은 다음 모듈의 Transformer(Day 8)와 BERT/사전학습(Day 9)으로 직결된다. 오늘 본 "순환 vs 어텐션"의 트레이드오프는 이후 LLM 아키텍처·스케일링(Day 14)과 효율화·양자화(Day 17), 장문맥/메모리 관리(Day 23) 논의의 밑그림이 된다.
