# Daily AI Paper Recommendations

> **Date:** 2026-08-13
> **Module:** Module 9: RAG (Retrieval-Augmented Generation)
> **Topic:** RAG Architecture and Optimization

---

## Paper 1 (Classic): Generalization through Memorization: Nearest Neighbor Language Models (kNN-LM)
- **Authors:** Urvashi Khandelwal, Omer Levy, Dan Jurafsky, Luke Zettlemoyer, Mike Lewis
- **Year:** 2019 (ICLR 2020)
- **arXiv:** https://arxiv.org/abs/1911.00172
- **PDF:** [./knn-lm-khandelwal-2019.pdf](./knn-lm-khandelwal-2019.pdf)
- **Citation Count:** ~1,500+

### 요약
kNN-LM은 사전학습된 신경망 언어모델을 재학습 없이 확장하는 방법으로, 다음 토큰 예측 분포를 훈련 데이터에서 검색한 k개의 최근접 이웃(kNN) 분포와 선형 보간한다. 모델의 은닉 표현을 키로, 다음 토큰을 값으로 하는 대규모 데이터스토어를 구축하고, 추론 시 유사한 문맥을 검색해 예측을 보정한다. 이 방식만으로 WikiText-103에서 perplexity를 2.9포인트 개선하며 당시 SOTA를 달성했다.

### 핵심 기여
- 파라미터를 학습하지 않고도(training-free) 검색 기반 메모리를 결합해 언어모델 성능을 끌어올리는 단순하고 강력한 프레임워크 제시
- "시퀀스 간 유사도를 학습하는 것이 다음 단어를 직접 예측하는 것보다 쉽다"는 통찰 — 검색이 파라메트릭 예측의 한계를 보완함을 입증
- 희귀 패턴(사실적 지식, 롱테일 표현) 예측에서 특히 큰 이득을 보이며, 도메인 적응을 데이터스토어 교체만으로 달성 가능함을 제시

### 이 논문이 중요한 이유
kNN-LM은 오늘날 RAG의 핵심 아이디어 — "파라미터에 지식을 넣는 대신 외부 메모리에서 검색해 보강한다" — 를 언어모델 수준에서 가장 순수하게 구현한 초기 논문이다. RAG 아키텍처를 설계하는 AI 엔지니어라면, 검색 보강이 왜 작동하는지에 대한 근본 원리(비파라메트릭 메모리, 롱테일 보정)를 이 논문에서 얻을 수 있다.

### 사전 지식
- 신경망 언어모델과 토큰 확률 분포(softmax)
- 근사 최근접 이웃 탐색(ANN, 예: FAISS)과 임베딩 유사도
- Perplexity 평가 지표의 의미

### 관련 논문
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (Lewis et al., 2020)](https://arxiv.org/abs/2005.11401)
- [Why do Nearest Neighbor Language Models Work? (Xu et al., 2023)](https://arxiv.org/abs/2301.02828)

### 실무 적용
검색 기반 보정 아이디어는 도메인 특화 챗봇에서 재학습 없이 지식을 갱신하는 데 활용된다. 예를 들어 고객사별 문서를 데이터스토어로 교체하면 동일 모델로 여러 도메인에 대응할 수 있고, 이는 현대 RAG 파이프라인에서 벡터DB 인덱스만 교체해 지식을 업데이트하는 운영 방식의 원형이다.

---

## Paper 2 (Classic): In-Context Retrieval-Augmented Language Models (In-Context RALM)
- **Authors:** Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay, Amnon Shashua, Kevin Leyton-Brown, Yoav Shoham
- **Year:** 2023 (TACL)
- **arXiv:** https://arxiv.org/abs/2302.00083
- **PDF:** [./in-context-ralm-ram-2023.pdf](./in-context-ralm-ram-2023.pdf)
- **Citation Count:** ~700+

### 요약
In-Context RALM은 언어모델 구조나 파라미터를 전혀 바꾸지 않고, 검색한 관련 문서를 입력 프롬프트 앞에 그대로 이어붙이는(prepend) 가장 단순한 형태의 RAG를 제안한다. 놀랍게도 이 단순한 방식만으로 큰 성능 향상이 가능하며, 범용 리트리버를 그대로 써도 6.7B OPT 모델이 66B OPT 모델 수준의 언어모델링 성능에 도달했다. 나아가 리트리버를 LM 신호로 재순위화(reranking)하면 추가 이득을 얻는다.

### 핵심 기여
- 모델을 수정하지 않는(model-agnostic) "in-context" RAG가 실용적으로 매우 강력함을 대규모 실험으로 입증
- 검색 문서를 프롬프트에 붙이는 방식으로 사실 정확도 향상과 출처 귀속(attribution)을 동시에 확보
- 검색 빈도(retrieval stride)와 쿼리 구성, LM 기반 리랭킹 등 실무에서 바로 조정 가능한 최적화 레버들을 체계적으로 분석

### 이 논문이 중요한 이유
오늘날 대부분의 프로덕션 RAG 시스템은 별도 학습 없이 "검색 → 프롬프트에 붙이기 → 생성"으로 동작한다. 이 논문은 바로 그 표준 아키텍처가 왜 효과적인지, 그리고 어떤 하이퍼파라미터(검색 주기, 문서 선택, 리랭킹)가 성능을 좌우하는지를 정량적으로 정리한 실무 지침서 같은 논문이다. AI 엔지니어가 RAG를 튜닝할 때 가장 먼저 참고할 만하다.

### 사전 지식
- 프롬프트 기반 in-context learning의 개념
- BM25/DPR 등 리트리버의 기본 동작
- 언어모델의 컨텍스트 윈도우 제약

### 관련 논문
- [Improving language models by retrieving from trillions of tokens (RETRO, Borgeaud et al., 2022)](https://arxiv.org/abs/2112.04426)
- [Lost in the Middle: How Language Models Use Long Contexts (Liu et al., 2023)](https://arxiv.org/abs/2307.03172)

### 실무 적용
LangChain·LlamaIndex의 기본 RAG 체인이 바로 이 in-context 방식이다. 검색 주기·문서 개수·리랭커 도입 여부를 조정하는 실무 최적화가 모두 이 논문의 분석과 직결되며, 컨텍스트 윈도우 예산 안에서 어떤 문서를 어떻게 배치할지 결정하는 근거를 제공한다.

---

## Paper 3 (Recent): Adaptive-RAG: Learning to Adapt Retrieval-Augmented LLMs through Question Complexity
- **Authors:** Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju Hwang, Jong C. Park
- **Year:** 2024 (NAACL 2024)
- **arXiv:** https://arxiv.org/abs/2403.14403
- **PDF:** [./adaptive-rag-jeong-2024.pdf](./adaptive-rag-jeong-2024.pdf)
- **Citation Count:** ~400+

### 요약
Adaptive-RAG는 모든 질문에 동일한 검색 전략을 쓰는 기존 RAG의 비효율을 지적하고, 질문 복잡도에 따라 전략을 동적으로 선택하는 프레임워크를 제안한다. 작은 분류기(classifier)가 질문을 (1) 검색 불필요, (2) 단일 검색, (3) 다단계 반복 검색의 세 가지 복잡도로 분류하고, 그에 맞는 최적 파이프라인을 라우팅한다. 이로써 단순 질문의 불필요한 연산을 줄이고 복잡한 멀티홉 질문의 정확도를 높이는 효율-정확도 균형을 달성한다.

### 핵심 기여
- 질문 복잡도 기반의 동적 라우팅으로 "No-Retrieval / Single-step / Multi-step"을 자동 선택하는 적응형 RAG 프레임워크 제안
- 별도 레이블 없이 모델 예측 결과와 데이터셋의 편향을 활용해 복잡도 분류기를 학습하는 실용적 방법 제시
- 단일 QA 시스템 안에서 반복 검색과 단일 검색을 매끄럽게 오가며 전체 효율과 정확도를 동시에 개선

### 이 논문이 중요한 이유
프로덕션 RAG의 최대 과제는 "모든 쿼리에 무거운 파이프라인을 쓰면 비용·지연이 폭발하고, 가볍게 쓰면 어려운 질문을 놓친다"는 트레이드오프다. Adaptive-RAG는 이를 라우팅 문제로 재정의해 비용 대비 성능을 최적화하는 현실적 해법을 제시한다. Agentic RAG 설계의 핵심인 "언제 검색할지, 얼마나 검색할지"를 정량적으로 다룬다.

### 사전 지식
- 단일 검색 RAG와 반복(iterative/multi-hop) 검색의 차이
- 멀티홉 QA 벤치마크(HotpotQA, 2WikiMultihopQA 등)
- 라우팅/분류기 기반 파이프라인 오케스트레이션 개념

### 관련 논문
- [Modular RAG: Transforming RAG Systems into LEGO-like Reconfigurable Frameworks (Gao et al., 2024)](https://arxiv.org/abs/2407.21059)
- [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection (Asai et al., 2023)](https://arxiv.org/abs/2310.11511)

### 실무 적용
LangGraph·LlamaIndex에 Adaptive-RAG 라우팅이 실제로 통합되어 있으며, FAQ성 단순 질의는 검색 없이 즉답하고 복잡한 분석 질의만 멀티홉 검색으로 라우팅하는 식으로 API 비용과 응답 지연을 크게 절감한다. 대규모 트래픽의 B2B RAG 서비스에서 SLA와 비용을 동시에 관리하는 핵심 패턴이다.

---

## 추천 읽기 순서
1. **kNN-LM (Paper 1)** — 검색 보강이 "왜" 작동하는지 근본 원리를 먼저 이해한다.
2. **In-Context RALM (Paper 2)** — 그 원리가 오늘날 표준 RAG 아키텍처로 어떻게 구현·최적화되는지 익힌다.
3. **Adaptive-RAG (Paper 3)** — 표준 RAG를 넘어 쿼리별로 전략을 최적화하는 최신 방향으로 확장한다.

## 핵심 테이크어웨이
- RAG의 본질은 "파라메트릭 지식 + 비파라메트릭 외부 메모리"의 결합이며, 재학습 없이도 지식을 갱신·확장할 수 있다는 점이 핵심 가치다.
- 가장 단순한 in-context 방식도 강력하지만, 성능은 검색 주기·문서 선택·리랭킹 같은 최적화 레버에 크게 좌우된다.
- 다음 성숙 단계는 "모든 쿼리를 똑같이 처리하지 않는 것" — 질문 복잡도에 따라 검색 강도를 조절하는 적응형·에이전트형 RAG가 비용-정확도 균형의 열쇠다.

## 다음 토픽과의 연결
다음 토픽인 **Advanced RAG (Self-RAG, Corrective RAG)** 에서는 오늘 다룬 적응형 라우팅을 한 단계 더 발전시켜, 모델이 검색 결과의 품질을 스스로 평가하고(비평), 필요 시 재검색·수정하는 자기성찰형 RAG로 나아간다. 오늘의 "언제·얼마나 검색할지"에서 "검색한 것이 맞는지 어떻게 검증할지"로 확장된다.
