# Daily AI Paper Recommendations

> **Date:** 2026-08-11
> **Module:** Module 8: LangChain and LLM Orchestration
> **Topic:** Memory and Long-Context Management

---

## Paper 1 (Classic): End-To-End Memory Networks
- **Authors:** Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, Rob Fergus
- **Year:** 2015
- **arXiv:** [https://arxiv.org/abs/1503.08895](https://arxiv.org/abs/1503.08895)
- **PDF:** [./end-to-end-memory-networks-sukhbaatar-2015.pdf](./end-to-end-memory-networks-sukhbaatar-2015.pdf)
- **Citation Count:** ~3,600회 (approx.)

### 요약
외부 메모리에 대한 순환적(attention) 접근을 신경망에 결합한 초기 메모리 증강 아키텍처다. 기존 Memory Network와 달리 전 과정을 end-to-end로 학습하기 때문에 중간 단계에 대한 강한 지도(supervision)가 필요 없고, 여러 번의 "hop"을 통해 메모리를 반복 조회하며 추론을 수행한다. 질의응답과 언어 모델링 모두에서 다중 hop이 성능을 끌어올린다는 것을 보였다.

### 핵심 기여
- 외부 메모리 슬롯을 soft attention으로 읽어들이는 미분 가능한(end-to-end 학습 가능) 메모리 접근 메커니즘 제안
- 다중 hop(multi-hop) 구조로 한 번의 출력마다 여러 번의 메모리 조회·추론 단계를 수행
- bAbI QA와 Penn Treebank/Text8 언어 모델링에서 RNN/LSTM 대비 경쟁력 있는 성능을 적은 지도만으로 달성

### 이 논문이 중요한 이유
"파라미터에 지식을 다 넣는" 방식의 한계를 넘어, 모델이 외부 메모리를 명시적으로 읽고 참조한다는 개념을 대중화한 논문이다. 오늘날 RAG, 에이전트 장기 메모리, key-value 메모리 구조의 사상적 뿌리로, AI 엔지니어가 "왜 컨텍스트를 외부화하는가"를 이해하는 출발점이 된다.

### 사전 지식
어텐션(soft attention)과 임베딩, RNN/LSTM의 기본 개념, softmax 기반 가중합, 언어 모델링과 QA 태스크의 평가 방식.

### 관련 논문
- [Memory Networks (Weston et al., 2014)](https://arxiv.org/abs/1410.3916)
- [Neural Turing Machines (Graves et al., 2014)](https://arxiv.org/abs/1410.5401)
- [Neural Machine Translation by Jointly Learning to Align and Translate (Bahdanau et al., 2014)](https://arxiv.org/abs/1409.0473)

### 실무 적용
챗봇/에이전트에서 대화 이력이나 문서를 외부 메모리에 저장하고 필요할 때 조회하는 구조의 원형이다. 다중 hop 아이디어는 오늘날 multi-step retrieval, iterative RAG, 그리고 에이전트의 반복적 추론 루프 설계에 그대로 응용된다.

---

## Paper 2 (Classic): Big Bird: Transformers for Longer Sequences
- **Authors:** Manzil Zaheer, Guru Guruganesh, Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontañón, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, Amr Ahmed
- **Year:** 2020
- **arXiv:** [https://arxiv.org/abs/2007.14062](https://arxiv.org/abs/2007.14062)
- **PDF:** [./big-bird-zaheer-2020.pdf](./big-bird-zaheer-2020.pdf)
- **Citation Count:** ~2,900회 (approx.)

### 요약
표준 트랜스포머의 어텐션은 시퀀스 길이에 대해 O(n²) 비용을 갖는데, Big Bird는 이를 희소(sparse) 어텐션으로 근사해 O(n) 수준으로 낮춘다. 슬라이딩 윈도우(로컬), 랜덤 어텐션, 글로벌 토큰이라는 세 가지 패턴을 결합해 동일 하드웨어에서 기존 대비 최대 8배 긴 시퀀스를 처리하며, 이 희소 구조가 이론적으로 표준 어텐션의 표현력(universal approximator, Turing complete)을 보존함을 증명했다.

### 핵심 기여
- 로컬 윈도우 + 랜덤 + 글로벌 토큰을 조합한 선형 복잡도 희소 어텐션 설계
- 희소 어텐션이 전역 어텐션의 이론적 성질(범용 근사·튜링 완전성)을 유지함을 수학적으로 증명
- 긴 문서 QA, 요약, 유전체 서열 등 long-context 태스크에서 SOTA급 성능 달성

### 이 논문이 중요한 이유
"컨텍스트 윈도우를 어떻게 늘릴 것인가"라는 문제에 대한 대표적 아키텍처 해법으로, 효율적 트랜스포머(Efficient Transformer) 계열을 이해하는 핵심 레퍼런스다. 오늘날 long-context LLM, 희소/선형 어텐션 연구의 기반 지식을 제공한다.

### 사전 지식
트랜스포머와 self-attention의 O(n²) 비용 구조, attention 패턴(로컬/글로벌) 개념, 그래프 스파스화 관점, QA·요약 벤치마크.

### 관련 논문
- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [Longformer: The Long-Document Transformer (Beltagy et al., 2020)](https://arxiv.org/abs/2004.05150)
- [Efficient Transformers: A Survey (Tay et al., 2020)](https://arxiv.org/abs/2009.06732)

### 실무 적용
긴 문서(계약서, 논문, 로그) 처리, RAG에서 큰 청크를 한 번에 인코딩해야 하는 인코더, 유전체·시계열 등 초장문 입력 파이프라인에 적용된다. 희소 어텐션 개념은 서빙 비용을 낮추는 long-context 모델 선택 기준으로도 실무에서 유용하다.

---

## Paper 3 (Recent): Titans: Learning to Memorize at Test Time
- **Authors:** Ali Behrouz, Peilin Zhong, Vahab Mirrokni (Google Research)
- **Year:** 2025
- **arXiv:** [https://arxiv.org/abs/2501.00663](https://arxiv.org/abs/2501.00663)
- **PDF:** [./titans-behrouz-2025.pdf](./titans-behrouz-2025.pdf)
- **Citation Count:** ~300회 (approx., NeurIPS 2025)

### 요약
Titans는 어텐션(단기 메모리)과 새로운 신경 장기 메모리 모듈(Long-term Memory Module, LMM)을 결합한 하이브리드 아키텍처다. 핵심은 테스트 시점(추론 중)에도 자신의 가중치를 갱신하며 무엇을 기억하고 무엇을 잊을지 학습하는 메타 인컨텍스트 학습기다. 그래디언트 기반 "놀라움(surprise)" 신호와 모멘텀으로 중요한 사건을 저장하고, 적응적 망각으로 메모리 과부하를 방지한다.

### 핵심 기여
- 추론 중 forward pass에서 자기 파라미터를 최적화하는 심층 비선형 순환 장기 메모리 모듈(LMM) 제안
- "surprise(그래디언트 크기) + 모멘텀 + 적응적 망각"으로 장기 정보를 선택적으로 기억/삭제하는 메커니즘
- 메모리를 컨텍스트/레이어/게이트 방식으로 통합하는 세 가지 변형을 제시하고, 2M 토큰 이상 초장문에서 트랜스포머·현대 선형 순환모델 대비 우수한 성능 확인

### 이 논문이 중요한 이유
"컨텍스트 윈도우를 늘리는 것"과 "진짜 장기 메모리를 갖는 것"은 다르다는 문제의식을 정면으로 다룬다. 테스트 타임 학습(test-time training)과 메모리를 결합해 무한 컨텍스트·에이전트 장기 기억 문제에 대한 최신 방향을 제시하며, 앞으로의 에이전트 메모리 아키텍처 설계에 큰 영향을 준다.

### 사전 지식
선형 어텐션/상태공간모델(SSM, Mamba)의 순환 관점, 메타러닝과 test-time training 개념, 그래디언트 기반 업데이트, 앞서 다룬 메모리 네트워크(Paper 1)와 long-context 어텐션(Paper 2).

### 관련 논문
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Gu & Dao, 2023)](https://arxiv.org/abs/2312.00752)
- [Leave No Context Behind: Infini-attention (Munkhdalai et al., 2024)](https://arxiv.org/abs/2404.07143)
- [Test-time Regression: A Unifying Framework for Sequence Models with Associative Memory (2025)](https://arxiv.org/abs/2501.12352)

### 실무 적용
장시간 세션을 유지하는 에이전트(개인 비서, 코딩 에이전트)에서 대화·작업 이력을 파라미터 메모리로 압축·유지하는 접근에 직접 연결된다. RAG로 다 감당하기 어려운 "지속적으로 진화하는 사용자별 장기 기억"을 모델 내부에 두는 차세대 설계 옵션을 제시한다.

---

## 추천 읽기 순서
1. **End-To-End Memory Networks (2015)** — "외부 메모리를 읽는다"는 개념의 원형부터 이해한다.
2. **Big Bird (2020)** — 메모리를 늘리는 또 다른 축인 "긴 컨텍스트를 효율적으로 어텐션하기"를 본다.
3. **Titans (2025)** — 두 흐름(메모리 조회 + 장기 컨텍스트)을 test-time 학습으로 통합한 최신 결론을 확인한다.

## 핵심 테이크어웨이
- 장기 정보를 다루는 방법은 크게 두 갈래다: (a) 외부 메모리를 읽는다(Memory Networks·RAG), (b) 어텐션 자체를 길게/효율적으로 만든다(Big Bird). Titans는 이 둘을 학습형 장기 메모리로 합친다.
- 다중 hop, 희소 어텐션, 적응적 망각 등은 서로 다른 시대의 해법이지만 모두 "무엇을 얼마나 오래 기억할지"라는 동일한 질문에 답한다.
- 컨텍스트 윈도우 확장 ≠ 진짜 메모리. 실무에서는 비용·지연·기억 지속성의 트레이드오프로 접근해야 한다.

## 다음 토픽과의 연결
다음 모듈(RAG)에서는 외부 지식을 검색해 생성에 결합하는 방법을 다룬다. 오늘 본 "메모리 조회"의 사상(End-to-End Memory Networks의 hop, Titans의 장기 메모리)이 Dense Retrieval과 RAG 아키텍처로 어떻게 확장되는지 자연스럽게 이어진다.
