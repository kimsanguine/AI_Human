# Daily AI Paper Recommendations

> **Date:** 2026-06-20
> **Module:** Module 9: RAG (Retrieval-Augmented Generation)
> **Topic:** RAG Architecture and Optimization

---

## Paper 1 (Classic): Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering (Fusion-in-Decoder)
- **Authors:** Gautier Izacard, Edouard Grave
- **Year:** 2020
- **arXiv:** https://arxiv.org/abs/2007.01282
- **PDF:** [./fid-izacard-grave-2020.pdf](./fid-izacard-grave-2020.pdf)
- **Citation Count:** ~2,000+

### 요약
검색된 여러 개의 지문(passage)을 인코더에서 각각 독립적으로 처리한 뒤, 디코더 단계에서 한꺼번에 융합(fusion)하여 답을 생성하는 Fusion-in-Decoder(FiD) 구조를 제안한 논문이다. 검색기(DPR)로 가져온 다수의 근거 문서를 효율적으로 결합하는 방식으로, Natural Questions·TriviaQA에서 당시 SOTA를 달성했다.

### 핵심 기여
- 인코더는 각 지문을 따로 인코딩하고 디코더에서 모든 지문 표현을 cross-attention으로 통합하는 FiD 아키텍처 제안
- 검색 지문 수를 늘릴수록 성능이 선형적으로 향상됨을 실증 — 생성 모델이 다수 근거를 종합하는 데 강점이 있음을 보임
- 파라미터를 키우지 않고도(retrieval로) open-domain QA 성능을 끌어올릴 수 있음을 증명

### 이 논문이 중요한 이유
오늘날 대부분의 RAG 파이프라인이 "여러 청크를 검색해 LLM 컨텍스트에 넣는다"는 패턴을 따른다. FiD는 이 패턴의 원형으로, 검색 지문 수와 답변 품질의 관계, 그리고 융합 위치(인코더 vs 디코더)에 대한 핵심 직관을 제공한다. RAG의 retriever-reader 분리 설계를 이해하는 출발점이다.

### 사전 지식
- Transformer encoder-decoder(T5) 구조와 cross-attention
- Open-domain QA와 DPR(Dense Passage Retrieval) 기본 개념
- Extractive QA vs Generative QA의 차이

### 관련 논문
- [Dense Passage Retrieval for Open-Domain QA (Karpukhin et al., 2020)](https://arxiv.org/abs/2004.04906)
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (Lewis et al., 2020)](https://arxiv.org/abs/2005.11401)

### 실무 적용
RAG 시스템에서 "top-k 청크 수를 몇 개로 할지", "각 청크를 따로 임베딩할지 합칠지"를 결정할 때의 근거가 된다. LangChain/LlamaIndex의 retriever top_k 튜닝, 멀티 문서 종합형 QA 봇, 컨텍스트 윈도우 효율화 설계에 직접 연결된다.

---

## Paper 2 (Classic): REPLUG: Retrieval-Augmented Black-Box Language Models
- **Authors:** Weijia Shi, Sewon Min, Michihiro Yasunaga, Minjoon Seo, Rich James, Mike Lewis, Luke Zettlemoyer, Wen-tau Yih
- **Year:** 2023
- **arXiv:** https://arxiv.org/abs/2301.12652
- **PDF:** [./replug-shi-2023.pdf](./replug-shi-2023.pdf)
- **Citation Count:** ~700+

### 요약
LLM을 블랙박스(가중치 접근 불가)로 두고, 검색된 문서를 단순히 입력 앞에 붙이는(prepend) 방식으로 보강하는 프레임워크를 제안한다. 핵심은 LLM을 고정한 채 검색기만 학습시키되, LLM의 출력 확률을 감독 신호로 사용해 "LLM이 더 잘 맞히게 돕는 문서"를 찾도록 retriever를 튜닝한다는 점이다.

### 핵심 기여
- 특수한 cross-attention 학습 없이, 기존 어떤 retriever·LLM에도 적용 가능한 plug-and-play 방식 제시
- LM 출력을 supervision으로 활용해 retriever를 학습하는 REPLUG LSR(LM-Supervised Retrieval) 기법
- GPT-3 등 API-only 모델에서도 retrieval로 언어모델링·QA 성능을 향상시킴을 입증

### 이 논문이 중요한 이유
실무 RAG의 대부분은 GPT/Claude처럼 가중치를 만질 수 없는 폐쇄형 모델을 사용한다. REPLUG는 "모델을 못 건드릴 때 검색 쪽을 어떻게 최적화하는가"라는 질문에 정면으로 답한 대표 연구로, 현대 RAG 최적화의 표준 사고방식을 제시한다.

### 사전 지식
- LLM의 토큰 확률(logits)과 ensemble 개념
- 검색기 fine-tuning과 contrastive learning 기초
- Frozen LM vs fine-tuned LM의 트레이드오프

### 관련 논문
- [In-Context Retrieval-Augmented Language Models (Ram et al., 2023)](https://arxiv.org/abs/2302.00083)
- [Atlas: Few-shot Learning with Retrieval Augmented Language Models (Izacard et al., 2022)](https://arxiv.org/abs/2208.03299)

### 실무 적용
폐쇄형 LLM API 위에 올리는 RAG에서 retriever를 LLM 피드백으로 개선하는 패턴의 근거가 된다. 검색 품질 평가(어떤 문서가 실제 답변 정확도를 높이는가)를 LLM 출력 기반으로 측정·최적화하는 RAG 평가/튜닝 파이프라인 설계에 활용된다.

---

## Paper 3 (Recent): Modular RAG: Transforming RAG Systems into LEGO-like Reconfigurable Frameworks
- **Authors:** Yunfan Gao, Yun Xiong, Meng Wang, Haofen Wang
- **Year:** 2024
- **arXiv:** https://arxiv.org/abs/2407.21059
- **PDF:** [./modular-rag-gao-2024.pdf](./modular-rag-gao-2024.pdf)
- **Citation Count:** ~200+

### 요약
초기의 단순한 "검색→생성" 선형 RAG를 넘어, RAG 시스템을 독립적인 모듈과 연산자(operator)의 조합으로 재구성하는 Modular RAG 프레임워크를 제안한다. 라우팅·스케줄링·융합 메커니즘을 도입해 RAG를 레고처럼 재조합 가능한 구조로 추상화하고, 실제 시스템에서 나타나는 RAG 패턴들을 체계적으로 정리한다.

### 핵심 기여
- RAG를 모듈(module)·서브모듈·연산자 계층으로 분해하는 3계층 추상화 제시
- 실무에서 반복되는 RAG 흐름을 linear / conditional / branching / looping 4가지 패턴으로 유형화
- 라우팅·스케줄링·융합 등 오케스트레이션 메커니즘을 명시화하여 복잡한 파이프라인 설계를 체계화

### 이 논문이 중요한 이유
2024년 현재 RAG는 단순 검색-생성을 넘어 query rewriting, reranking, 반복 검색, adaptive retrieval 등이 얽힌 복합 시스템으로 진화했다. 이 논문은 그 복잡성을 다루는 공통 설계 언어를 제공하여, 엔지니어가 RAG 아키텍처를 모듈 단위로 설계·교체·디버깅할 수 있게 한다.

### 사전 지식
- 기본 RAG(Naive RAG)와 Advanced RAG의 차이
- query rewriting, reranking, fusion 등 RAG 구성 요소
- LangGraph/LlamaIndex 같은 오케스트레이션 프레임워크의 흐름 제어 개념

### 관련 논문
- [Retrieval-Augmented Generation for Large Language Models: A Survey (Gao et al., 2023)](https://arxiv.org/abs/2312.10997)
- [Searching for Best Practices in Retrieval-Augmented Generation (Wang et al., 2024)](https://arxiv.org/abs/2407.01219)

### 실무 적용
프로덕션 RAG를 모듈식으로 설계할 때의 청사진을 제공한다. 검색 전략을 조건부로 분기하거나(branching), 답이 부족하면 반복 검색하는(looping) 파이프라인을 LangGraph 등으로 구현할 때 참조 아키텍처로 쓰인다. RAG 시스템의 A/B 테스트와 컴포넌트 교체형 개선에 적합하다.

---

## 추천 읽기 순서
1. **FiD (2020)** — 다수 지문을 결합해 답을 만드는 RAG의 기본 reader 구조를 먼저 이해한다.
2. **REPLUG (2023)** — 블랙박스 LLM 환경에서 retriever를 최적화하는 실무형 사고를 익힌다.
3. **Modular RAG (2024)** — 위 요소들이 어떻게 복합 시스템으로 조립되는지 큰 그림으로 정리한다.

## 핵심 테이크어웨이
- RAG 성능은 retriever와 reader(생성기)의 **분리·최적화**에서 나온다. FiD는 reader 측, REPLUG는 retriever 측 최적화의 원형이다.
- 폐쇄형 LLM 시대에는 모델을 못 건드리므로 **검색 쪽 최적화**가 RAG 엔지니어링의 핵심 레버다.
- 현대 RAG는 선형 파이프라인이 아니라 **모듈식·재조합 가능한 시스템**이다. 패턴(linear/conditional/branching/looping)으로 사고하면 설계와 디버깅이 쉬워진다.

## 다음 토픽과의 연결
다음 토픽은 **Advanced RAG (Self-RAG, Corrective RAG)** 이다. 오늘 다룬 Modular RAG의 conditional/looping 패턴은 Self-RAG의 자기검증·재검색, CRAG의 검색 결과 보정으로 자연스럽게 이어진다. 검색의 "언제·무엇을·다시" 결정 문제로 심화된다.
