# Daily AI Paper Recommendations

> **Date:** 2026-09-15
> **Module:** Module 3: Machine Learning and Deep Learning
> **Topic:** RNN, LSTM and Sequence Models

---

## Paper 1 (Classic): Speech Recognition with Deep Recurrent Neural Networks
- **Authors:** Alex Graves, Abdel-rahman Mohamed, Geoffrey Hinton
- **Year:** 2013
- **arXiv:** https://arxiv.org/abs/1303.5778
- **PDF:** [./speech-recognition-deep-rnn-graves-2013.pdf](./speech-recognition-deep-rnn-graves-2013.pdf)
- **Citation Count:** 약 12,000회 이상

### 요약
LSTM을 여러 층으로 쌓고(deep) 양방향(bidirectional)으로 구성한 뒤 CTC와 RNN Transducer로 학습해, 당시 HMM 기반 시스템이 지배하던 TIMIT 음소 인식에서 17.7% PER이라는 최고 성능을 달성한 논문이다. "RNN은 이미 시간 축으로 깊으니 층을 더 쌓을 필요가 없다"는 통념을 데이터로 반박하고, 깊이(depth)가 순환(recurrence)과 직교하는 별개의 표현력 축임을 보였다.

### 핵심 기여
- Deep Bidirectional LSTM 구조를 제안하고, 층 수를 늘리는 것이 셀 수를 늘리는 것보다 효율적임을 실험으로 입증
- End-to-end 시퀀스 학습(CTC / RNN Transducer)만으로 HMM-GMM 파이프라인을 능가할 수 있음을 최초로 설득력 있게 보임
- 정규화(가중치 노이즈)와 깊이의 상호작용을 분석해, 깊은 RNN의 과적합 제어 레시피를 제시

### 이 논문이 중요한 이유
오늘날 Whisper·wav2vec 2.0 같은 음성 모델, 나아가 모든 end-to-end 시퀀스 모델의 출발점이 되는 논문이다. "도메인 지식으로 쌓아올린 파이프라인(HMM, 발음 사전, 정렬)을 하나의 미분 가능한 네트워크로 대체한다"는 AI 엔지니어링의 핵심 패턴이 여기서 처음 명확하게 증명됐다. 또한 깊이를 늘리는 것이 왜 효과적인지에 대한 실험 설계가 매우 깔끔해서, 아키텍처 실험을 어떻게 설계해야 하는지 배우기에도 좋다.

### 사전 지식
- LSTM 셀의 게이트 구조(input/forget/output gate)와 BPTT
- CTC(Connectionist Temporal Classification) 손실의 개념 — 입력과 출력 길이가 다를 때 정렬 없이 학습하는 방법
- 음성 인식의 기본 평가 지표(PER, WER)와 HMM-GMM 파이프라인의 대략적인 구조

### 관련 논문
- [Long Short-Term Memory (Hochreiter & Schmidhuber, 1997)](https://doi.org/10.1162/neco.1997.9.8.1735)
- [Connectionist Temporal Classification (Graves et al., 2006)](https://doi.org/10.1145/1143844.1143891)
- [Sequence Transduction with Recurrent Neural Networks (Graves, 2012)](https://arxiv.org/abs/1211.3711)
- [Listen, Attend and Spell (Chan et al., 2015)](https://arxiv.org/abs/1508.01211)

### 실무 적용
스트리밍 STT 제품에서 RNN-T(RNN Transducer)는 지금도 현역이다. Whisper 같은 attention 기반 모델은 전체 오디오를 봐야 하지만, RNN-T 계열은 토큰 단위로 즉시 출력할 수 있어 실시간 자막·통역·회의록 제품의 지연 시간(latency) 요구를 만족시킨다. AI 더빙이나 아바타 제품에서 "말하는 동안 자막이 따라붙어야 하는" 요구사항이 있다면, Whisper가 아니라 RNN-T 계열을 선택하게 되는 근거가 이 논문 계보에 있다.

---

## Paper 2 (Classic): HiPPO: Recurrent Memory with Optimal Polynomial Projections
- **Authors:** Albert Gu, Tri Dao, Stefano Ermon, Atri Rudra, Christopher Ré
- **Year:** 2020
- **arXiv:** https://arxiv.org/abs/2008.07669
- **PDF:** [./hippo-recurrent-memory-polynomial-projections-gu-2020.pdf](./hippo-recurrent-memory-polynomial-projections-gu-2020.pdf)
- **Citation Count:** 약 1,200회 이상

### 요약
"순환 신경망의 hidden state는 결국 과거 입력을 압축한 것인데, 최적의 압축이란 무엇인가?"라는 질문을 수학적으로 정면 돌파한 논문이다. 과거 시점의 중요도를 나타내는 측도(measure)를 정하면, 과거 신호를 직교 다항식 기저에 사영(projection)하는 것이 최적 온라인 근사임을 보이고, 그 해가 자동으로 특정 형태의 선형 ODE/점화식으로 유도됨을 증명한다. 이 프레임워크에서 LMU가 자연스럽게 유도되고, GRU의 게이팅도 특수한 경우로 설명된다.

### 핵심 기여
- 메모리를 "함수 근사 문제"로 재정의하는 HiPPO 프레임워크를 제시하고, 최적해를 닫힌 형태의 행렬(HiPPO 행렬)로 유도
- HiPPO-LegS를 제안 — 시간 스케일에 대한 사전 가정 없이 전체 히스토리를 기억하며, gradient가 유계이고 업데이트가 빠름
- 기존 RNN 게이팅 메커니즘이 HiPPO의 특수 사례임을 보여, 경험적 설계였던 게이트에 이론적 근거를 부여
- 100만 스텝 길이의 궤적 분류 등 초장기 의존성 벤치마크에서 기존 RNN 대비 압도적 성능

### 이 논문이 중요한 이유
S4 → Mamba → Jamba → RWKV-7로 이어지는 현대 State Space Model 계보의 **수학적 뿌리**가 이 논문이다. Mamba 논문에서 갑자기 등장하는 A 행렬 초기화(HiPPO 행렬)가 왜 그렇게 생겼는지, 왜 선택적 상태 공간이 "기억을 선택한다"고 말할 수 있는지는 HiPPO를 읽지 않으면 그냥 외우는 수밖에 없다. AI 엔지니어에게 이 논문은 "long context를 어떻게 처리할 것인가"라는 2025년의 실무 문제를 원리 수준에서 이해하게 해준다.

### 사전 지식
- 선형대수: 직교 기저, 사영(projection), 내적 공간
- 미분방정식: 선형 ODE와 이산화(discretization, bilinear/ZOH)
- RNN의 vanishing/exploding gradient 문제 (Pascanu et al., 2013)
- 르장드르/라게르 다항식의 존재 정도만 알아도 충분 (세부 유도는 부록 참고)

### 관련 논문
- [Legendre Memory Units (Voelker et al., 2019)](https://papers.nips.cc/paper/2019/hash/952285b9b7e7a1be5aa7849f32ffff05-Abstract.html)
- [Efficiently Modeling Long Sequences with Structured State Spaces / S4 (Gu et al., 2021)](https://arxiv.org/abs/2111.00396)
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Gu & Dao, 2023)](https://arxiv.org/abs/2312.00752)
- [On the difficulty of training Recurrent Neural Networks (Pascanu et al., 2013)](https://arxiv.org/abs/1211.5063)

### 실무 적용
장문 문서 RAG, 긴 통화 녹취 분석, 멀티턴 에이전트 메모리처럼 "입력이 계속 길어지는" 제품에서 Transformer의 O(n²)는 결국 비용 문제로 돌아온다. HiPPO 계열 SSM은 상태 크기가 고정이라 토큰당 추론 비용이 일정하다. 즉 100턴 대화든 1,000턴 대화든 응답 지연과 GPU 비용이 선형으로만 증가한다. Agentic AI 제품에서 대화 히스토리를 어디까지 유지할지 결정할 때, "요약해서 버린다 vs 압축해서 상태로 들고 간다"는 선택지의 후자가 바로 이 논문의 아이디어다.

---

## Paper 3 (Recent): RWKV-7 "Goose" with Expressive Dynamic State Evolution
- **Authors:** Bo Peng, Ruichong Zhang, Daniel Goldstein, Eric Alcaide, Xingjian Du, Haowen Hou, Quentin Anthony, Janna Lu, et al. (RWKV Project)
- **Year:** 2025
- **arXiv:** https://arxiv.org/abs/2503.14456
- **PDF:** [./rwkv7-goose-peng-2025.pdf](./rwkv7-goose-peng-2025.pdf)
- **Citation Count:** 약 100회 이상 (2026년 기준, 빠르게 증가 중)

### 요약
토큰당 상수 메모리·상수 시간으로 동작하는 순환 아키텍처 RWKV-7을 제안한다. 핵심은 delta rule을 벡터값 게이팅과 in-context learning rate로 일반화한 것으로, 상태 전이 행렬이 입력에 따라 동적으로 바뀐다. 2.9B 모델이 훨씬 적은 토큰으로 학습하고도 다국어 태스크에서 3B급 SOTA를 달성했으며, 3.1조 토큰 다국어 코퍼스도 함께 공개했다.

### 핵심 기여
- 일반화된 delta rule + 동적 상태 전이(dynamic state evolution) — 기존 선형 어텐션/SSM의 고정 전이 구조를 넘어섬
- **표현력의 이론적 우위 증명**: RWKV-7은 state tracking이 가능하고 모든 정규 언어(regular language)를 인식할 수 있는데, 이는 표준 복잡도 가정 하에서 TC⁰에 갇힌 Transformer가 못하는 일이다
- 학습 병렬화를 유지하면서 순환 추론의 효율을 확보 (RNN의 고질적 trade-off를 완화)
- 0.19B~2.9B 4개 모델과 3.1T 다국어 코퍼스 오픈소스 공개

### 이 논문이 중요한 이유
"Transformer를 대체할 수 있는가"라는 질문이 성능 비교를 넘어 **계산 복잡도 클래스 논쟁**으로 올라섰다는 신호다. Transformer가 원리적으로 못 푸는 문제(상태 추적, 괄호 매칭 같은 순차적 상태 유지)가 존재하고 순환 구조는 그걸 푼다는 주장은, 에이전트처럼 긴 상태를 유지해야 하는 워크로드에 직접 맞닿아 있다. 동시에 "적은 학습 토큰으로 SOTA"라는 결과는 데이터 효율이 여전히 아키텍처로 개선 가능함을 보여준다.

### 사전 지식
- Linear attention과 delta rule (DeltaNet 계열)의 기본 아이디어
- Mamba의 selective state space와 입력 의존적 게이팅
- 계산 복잡도 클래스 TC⁰의 대략적 의미, 그리고 Transformer의 표현력 한계 논의
- 오늘의 Paper 2(HiPPO)를 먼저 읽으면 상태 전이 행렬 설계의 맥락이 훨씬 선명해진다

### 관련 논문
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Gu & Dao, 2023)](https://arxiv.org/abs/2312.00752)
- [Parallelizing Linear Transformers with the Delta Rule over Sequence Length (Yang et al., 2024)](https://arxiv.org/abs/2406.06484)
- [Gated Delta Networks (Yang et al., 2024)](https://arxiv.org/abs/2412.06464)
- [RWKV: Reinventing RNNs for the Transformer Era (Peng et al., 2023)](https://arxiv.org/abs/2305.13048)
- [The Illusion of State in State-Space Models (Merrill et al., 2024)](https://arxiv.org/abs/2404.08819)

### 실무 적용
온디바이스·엣지 추론이 필요한 제품에서 가장 현실적인 선택지다. 상태 크기가 고정이라 KV 캐시가 없고, 따라서 모바일이나 저사양 GPU에서 긴 대화를 유지해도 메모리가 터지지 않는다. 실시간 음성 대화 아바타처럼 "지연이 곧 UX"인 제품, 그리고 다국어 지원이 필요한 SaaS에서 검토할 가치가 크다. 다만 프로덕션 도입 전에는 툴 콜링·구조화 출력 같은 에이전트 워크로드에서의 안정성을 직접 벤치마크해야 한다 — 이 영역은 아직 Transformer 계열 생태계가 압도적으로 성숙하다.

---

## 추천 읽기 순서

1. **Graves 2013 (Deep RNN)** — 4페이지짜리 짧은 논문이다. "RNN을 실제로 깊게 쌓으면 어떻게 되는가"라는 경험적 출발점을 먼저 잡는다.
2. **HiPPO 2020** — 본문과 Section 2까지만 정독하고, 다항식 유도는 결론(HiPPO-LegS 행렬)만 받아들여도 된다. 여기서 "hidden state = 과거의 압축"이라는 관점을 획득한다.
3. **RWKV-7 2025** — 앞의 두 논문이 각각 제공한 경험적 직관과 이론적 틀 위에서, 2025년의 순환 아키텍처가 어디까지 왔는지 확인한다.

시간이 부족하다면 HiPPO → RWKV-7 두 편만 읽어도 흐름은 잡힌다. Graves 2013은 통근길에 읽을 분량이다.

## 핵심 테이크어웨이

- **깊이와 순환은 별개의 축이다.** RNN이 "시간 축으로 깊다"는 것과 "표현이 깊다"는 것은 다른 문제이며, 후자를 늘리는 것이 대개 더 효율적이다 (Graves 2013).
- **Hidden state는 메모리가 아니라 압축된 함수 근사다.** 무엇을 기억할지는 "어떤 측도로 과거를 가중할 것인가"의 문제로 환원되고, 그 최적해는 수학적으로 유도 가능하다 (HiPPO).
- **순환 구조의 부활은 성능이 아니라 복잡도 때문이다.** 토큰당 상수 비용과 KV 캐시 부재가 핵심 가치이고, 표현력 측면에서도 Transformer가 못 하는 영역(state tracking)이 존재한다 (RWKV-7).
- **실무 판단 기준:** 입력 길이가 계속 늘어나고 지연이 UX를 좌우하면 순환/SSM 계열을, 생태계 성숙도와 툴 사용 안정성이 우선이면 Transformer를 택한다. 요즘 현실적인 답은 Jamba·Samba 같은 **하이브리드**다.

## 다음 토픽과의 연결

다음은 **Optimization and Regularization**(Adam, AdamW)이다. 오늘 본 RNN·SSM 계열이 왜 학습이 까다로운지 — gradient가 시간 축으로 곱해지며 폭발하거나 소멸하는 문제 — 는 결국 옵티마이저와 정규화 설계로 다뤄진다. HiPPO가 "bounded gradient"를 핵심 장점으로 내세운 이유를, 다음 논문들을 읽고 나면 훨씬 구체적으로 이해하게 된다.

또한 이 토픽은 Module 4의 **Attention Mechanism and Transformer**와 직접 이어진다. 오늘 읽은 순환 계열의 한계(병렬화 어려움, 고정 상태의 정보 병목)가 어텐션이 등장한 이유이고, RWKV-7은 그 어텐션의 한계를 다시 순환으로 되받아친 시도다. 두 계보를 왕복하며 읽는 것이 시퀀스 모델링을 이해하는 가장 빠른 길이다.
