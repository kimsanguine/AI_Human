# Daily AI Paper Recommendations

> **Date:** 2026-06-21
> **Module:** Module 9: RAG (Retrieval-Augmented Generation)
> **Topic:** Advanced RAG — Self-RAG, Corrective RAG

---

## Paper 1 (Classic): Corrective Retrieval Augmented Generation (CRAG)
- **Authors:** Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, Zhen-Hua Ling
- **Year:** 2024
- **arXiv:** https://arxiv.org/abs/2401.15884
- **PDF:** [./crag-yan-2024.pdf](./crag-yan-2024.pdf)
- **Citation Count:** approximately 600+ (2026년 6월 기준)

### 요약
CRAG는 검색된 문서의 품질이 낮을 때 RAG가 잘못된 생성을 하는 문제를 해결하기 위해, 검색 결과의 신뢰도를 평가하는 경량 "검색 평가기(retrieval evaluator)"를 도입한다. 평가 결과에 따라 Correct/Incorrect/Ambiguous 세 가지 행동을 트리거하고, 필요 시 웹 검색으로 지식을 보강하며 decompose-then-recompose 알고리즘으로 핵심 정보만 추출한다. 어떤 RAG 시스템에도 plug-and-play로 붙일 수 있도록 설계된 것이 특징이다.

### 핵심 기여
- 검색 문서의 관련성을 점수화하는 경량 검색 평가기를 제안하고, 신뢰도 등급에 따라 행동을 분기하는 corrective 메커니즘 설계
- 검색 실패에 대비해 대규모 웹 검색을 보조 지식 소스로 활용하여 정적 코퍼스의 한계를 보완
- 검색된 문서에서 노이즈를 걷어내는 decompose-then-recompose 방식으로 핵심 지식만 선별
- 기존 RAG 및 self-RAG 파이프라인에 손쉽게 결합 가능한 범용성 입증

### 이 논문이 중요한 이유
실무 RAG의 가장 큰 실패 원인은 "잘못된 문서를 검색했는데 LLM이 그대로 믿고 답하는 것"이다. CRAG는 검색-생성 사이에 품질 게이트를 넣는 가장 직관적이고 널리 채택된 패턴을 제시한다. AI 엔지니어가 production RAG의 hallucination을 줄이려면 반드시 이해해야 하는 corrective 루프의 표준 레퍼런스다.

### 사전 지식
기본 RAG 파이프라인(retriever–generator), dense retrieval과 임베딩 유사도, hallucination 개념, 그리고 Self-RAG의 self-reflection 아이디어를 알고 있으면 비교가 쉽다.

### 관련 논문
- [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection (Asai et al., 2023)](https://arxiv.org/abs/2310.11511)
- [Active Retrieval Augmented Generation / FLARE (Jiang et al., 2023)](https://arxiv.org/abs/2305.06983)

### 실무 적용
고객 지원 챗봇, 사내 문서 QA, 검색 기반 코파일럿에서 "검색 신뢰도 낮음 → 웹 검색 fallback 또는 답변 보류"로직으로 직접 적용된다. LangGraph/LlamaIndex의 corrective-RAG 워크플로우 템플릿이 이 논문 구조를 그대로 따른다.

---

## Paper 2 (Classic): RAPTOR — Recursive Abstractive Processing for Tree-Organized Retrieval
- **Authors:** Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh Khanna, Anna Goldie, Christopher D. Manning
- **Year:** 2024 (ICLR 2024)
- **arXiv:** https://arxiv.org/abs/2401.18059
- **PDF:** [./raptor-sarthi-2024.pdf](./raptor-sarthi-2024.pdf)
- **Citation Count:** approximately 700+ (2026년 6월 기준)

### 요약
RAPTOR는 문서를 평면적인 청크 집합으로 다루는 기존 RAG의 한계를 넘어, 청크를 재귀적으로 클러스터링하고 요약하여 트리 구조를 구성한다. 트리의 하위 노드는 원문 청크, 상위 노드는 점점 더 추상화된 요약을 담는다. 질의 시 여러 추상화 레벨에서 검색하여 세부 사실과 전체 맥락(long-context, multi-hop)을 동시에 확보할 수 있다.

### 핵심 기여
- 임베딩 기반 soft clustering(가우시안 혼합) + LLM 요약으로 계층적 retrieval 트리를 자동 구축
- 서로 다른 추상화 수준의 노드를 함께 검색하는 collapsed-tree retrieval 방식 제안
- 복잡한 multi-step 추론이 필요한 QA(QuALITY 등)에서 기존 검색 대비 큰 폭의 성능 향상 입증
- 긴 문서·전체 코퍼스 차원의 "주제 수준" 질문에 강한 retrieval 패러다임 제시

### 이 논문이 중요한 이유
단순 top-k 청크 검색은 "이 문서 전체의 주제가 뭐야?" 같은 글로벌 질문에 약하다. RAPTOR는 이 문제를 계층적 요약으로 푸는 대표 해법으로, 이후 GraphRAG 계열 글로벌 검색 연구의 출발점이 되었다. 검색 단위 설계(chunking 그 이상)를 고민하는 엔지니어의 필독서다.

### 사전 지식
임베딩과 클러스터링(특히 GMM), 텍스트 요약을 위한 LLM 호출, 청킹 전략, 그리고 multi-hop QA의 어려움을 이해하면 좋다.

### 관련 논문
- [Dense Passage Retrieval for Open-Domain QA / DPR (Karpukhin et al., 2020)](https://arxiv.org/abs/2004.04906)
- [From Local to Global: A Graph RAG Approach to Query-Focused Summarization (Edge et al., 2024)](https://arxiv.org/abs/2404.16130)

### 실무 적용
긴 보고서·법률 문서·기술 매뉴얼 QA에서 "요약 노드"를 통해 문서 전체를 아우르는 답변을 생성할 때 사용된다. LlamaIndex의 RAPTOR pack으로 제공되어 사내 지식베이스의 계층적 인덱싱에 바로 적용 가능하다.

---

## Paper 3 (Recent): LightRAG — Simple and Fast Retrieval-Augmented Generation
- **Authors:** Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, Chao Huang
- **Year:** 2024 (v3, 2025-04 업데이트)
- **arXiv:** https://arxiv.org/abs/2410.05779
- **PDF:** [./lightrag-guo-2024.pdf](./lightrag-guo-2024.pdf)
- **Citation Count:** approximately 400+ (2026년 6월 기준, 빠르게 증가 중)

### 요약
LightRAG는 GraphRAG의 높은 비용·복잡도를 줄이면서도 그래프 기반 검색의 장점을 살린 경량 프레임워크다. 텍스트에서 엔티티·관계를 추출해 지식 그래프를 만들고, low-level(구체적 엔티티)과 high-level(주제·개념) 검색을 결합한 dual-level retrieval을 수행한다. 증분 업데이트(incremental update)를 지원해 새 문서 추가 시 전체 인덱스를 재구축할 필요가 없다.

### 핵심 기여
- 그래프 구조와 벡터 검색을 통합한 dual-level retrieval 패러다임 제안
- 전체 재인덱싱 없이 새 데이터를 반영하는 incremental update 알고리즘으로 운영 비용 절감
- GraphRAG 대비 토큰·API 호출 비용을 크게 낮추면서 동등 이상의 검색 품질 달성
- 오픈소스로 공개되어 실무 채택이 빠르게 확산

### 이 논문이 중요한 이유
GraphRAG는 강력하지만 인덱싱 비용이 과도해 production 도입이 어렵다는 비판이 있었다. LightRAG는 "그래프 RAG를 실제로 쓸 수 있게" 만든 2024–2025년의 대표적 실용화 연구로, 비용 대비 성능을 중시하는 엔지니어에게 현재 가장 핫한 선택지 중 하나다.

### 사전 지식
지식 그래프 기초, 엔티티·관계 추출(IE), 벡터 검색, 그리고 GraphRAG의 글로벌 검색 개념을 알면 차이를 명확히 이해할 수 있다.

### 관련 논문
- [From Local to Global: A Graph RAG Approach (Edge et al., 2024)](https://arxiv.org/abs/2404.16130)
- [Corrective Retrieval Augmented Generation / CRAG (Yan et al., 2024)](https://arxiv.org/abs/2401.15884)

### 실무 적용
대규모·지속적으로 갱신되는 지식베이스(사내 위키, 제품 문서, 뉴스 피드)에서 비용 효율적인 그래프 RAG가 필요할 때 적용된다. 오픈소스 LightRAG 라이브러리로 빠르게 PoC를 구축하고, 증분 업데이트로 운영 단계까지 이어가기 좋다.

---

## 추천 읽기 순서
1. **CRAG** — 검색 품질을 평가하고 교정하는 corrective 루프의 기본 개념부터 잡는다.
2. **RAPTOR** — 검색 단위를 계층화하는 retrieval 구조 설계로 시야를 넓힌다.
3. **LightRAG** — 위 두 흐름(품질·구조)을 그래프 기반으로 통합한 최신 실용 프레임워크로 마무리한다.

## 핵심 테이크어웨이
- Advanced RAG의 발전은 "더 좋은 검색"을 넘어 **검색 결과를 평가·교정(CRAG)**, **검색 단위를 재구성(RAPTOR)**, **지식 구조를 그래프화(LightRAG)** 하는 방향으로 진화한다.
- 세 논문 모두 "top-k 청크를 그대로 LLM에 넣는다"는 순진한 RAG의 한계를 각기 다른 각도에서 공격한다.
- production 관점에서는 정확도뿐 아니라 **비용·증분 업데이트·plug-and-play 결합성**이 핵심 평가 축임을 보여준다.

## 다음 토픽과의 연결
다음 토픽인 **Vector Databases and Indexing(Day 27)**은 오늘 다룬 advanced RAG 기법들이 실제로 의존하는 검색 인프라를 다룬다. CRAG의 fallback 검색, RAPTOR의 계층적 노드, LightRAG의 그래프+벡터 하이브리드 모두 효율적인 ANN 인덱싱(HNSW, FAISS)과 하이브리드 검색 위에서 동작하므로, 인덱싱 계층을 이해하면 오늘의 기법들을 더 잘 최적화할 수 있다.
