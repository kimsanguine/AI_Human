# Daily AI Paper Recommendations

> **Date:** 2026-05-25
> **Module:** Module 9 - RAG (Retrieval-Augmented Generation)
> **Topic:** Advanced RAG - HyDE, IRCoT, and Graph RAG

---

## Paper 1 (Classic): Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE)
- **Authors:** Luyu Gao, Xueguang Ma, Jimmy Lin, Jamie Callan
- **Year:** 2022
- **arXiv:** https://arxiv.org/abs/2212.10496
- **PDF:** [./hyde-gao-2022.pdf](./hyde-gao-2022.pdf)
- **Citation Count:** 600+ (2026년 기준)

### 요약
HyDE(Hypothetical Document Embeddings)는 관련성 라벨(relevance labels) 없이도 강력한 검색 성능을 달성하는 제로샷 검색 방식이다. 쿼리에 직접 임베딩을 적용하는 대신, LLM이 먼저 "가상의 답변 문서"를 생성하고 그 가상 문서의 임베딩으로 실제 코퍼스에서 유사 문서를 검색한다. 이 접근법은 별도의 학습 데이터 없이도 fine-tuned retriever 수준의 성능을 보여준다.


### 핵심 기여
- **쿼리-문서 비대칭 문제 해결:** 쿼리 임베딩과 문서 임베딩의 분포 차이를 가상 문서 생성으로 메움
- **제로샷 검색의 새로운 패러다임:** 관련성 라벨 없이 InstructGPT + Contriever 만으로 supervised baseline에 근접
- **다국어/다도메인 일반화:** 웹 검색, QA, 도메인 특화 검색까지 광범위하게 작동함을 실험으로 입증
- **LLM-as-retriever-helper 개념:** 검색에서 생성 모델의 역할을 재정의

### 이 논문이 중요한 이유
AI 엔지니어가 RAG 시스템을 설계할 때 가장 큰 난관 중 하나가 "도메인 특화 retriever를 어떻게 학습시킬 것인가"이다. HyDE는 학습 없이도 LLM의 사전 지식만으로 검색 품질을 끌어올리는 실용적 해법을 제시한다. 특히 신규 도메인 진입 시 cold-start 문제를 해결하는 표준 기법으로 자리잡았으며, LangChain·LlamaIndex 등 주요 프레임워크에 기본 모듈로 통합되어 있다.

### 사전 지식
- Dense retrieval과 sparse retrieval(BM25)의 차이
- Sentence/Passage embedding 모델(예: Contriever, SBERT) 동작 원리
- LLM의 instruction-following 능력과 zero-shot 추론
- Cosine similarity 기반 벡터 검색

### 관련 논문
- [Dense Passage Retrieval for Open-Domain QA (Karpukhin et al., 2020)](https://arxiv.org/abs/2004.04906)
- [Contriever: Unsupervised Dense Information Retrieval (Izacard et al., 2021)](https://arxiv.org/abs/2112.09118)
- [Query2doc: Query Expansion with LLMs (Wang et al., 2023)](https://arxiv.org/abs/2303.07678)

### 실무 적용
- **B2B SaaS 검색 기능:** 도메인 학습 데이터가 부족한 초기 단계에서 HyDE로 검색 품질을 빠르게 확보
- **고객지원 챗봇 RAG:** 모호한 사용자 질문(예: "이거 환불 되나요?")을 LLM이 구체화한 뒤 검색하여 정확도 향상
- **법률·의료 도메인 검색:** 전문 용어를 LLM이 가상 답변에 풀어넣어 일반 임베딩 모델의 도메인 갭을 보완
- **AI Agent의 retrieval tool:** Agent가 검색 전 가상 답변을 생성해 retrieval intent를 명확히 함

---


## Paper 2 (Classic): Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions (IRCoT)
- **Authors:** Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, Ashish Sabharwal
- **Year:** 2022
- **arXiv:** https://arxiv.org/abs/2212.10509
- **PDF:** [./ircot-trivedi-2022.pdf](./ircot-trivedi-2022.pdf)
- **Citation Count:** 700+ (2026년 기준)

### 요약
IRCoT는 검색(Retrieval)과 Chain-of-Thought(CoT) 추론을 단순히 순차적으로 수행하는 대신, **상호 교차(interleaving)** 방식으로 결합한 advanced RAG 기법이다. 매 CoT 단계에서 생성된 문장을 다음 retrieval의 쿼리로 사용하고, 새로 검색된 문서를 다시 다음 추론 단계에 투입한다. 이를 통해 다단계 추론(multi-hop QA)에서 단일 검색만으로는 도달할 수 없는 정보 경로를 풀어낸다.

### 핵심 기여
- **Retrieval-Reasoning 결합 패러다임:** "한 번 검색하고 답한다"는 단일 RAG의 한계를 극복
- **Multi-hop QA 성능 대폭 향상:** HotpotQA, MuSiQue, 2WikiMultihopQA 등에서 SOTA 갱신
- **모델 크기 의존성 완화:** Flan-T5-large 같은 작은 모델에서도 효과 입증
- **ReAct·Self-Ask 등 후속 연구의 기반:** Agentic RAG의 직접적인 선조

### 이 논문이 중요한 이유
Agentic AI 제품을 만들 때 "한 번에 답이 안 나오는 질문"은 거의 모든 도메인에서 발생한다. IRCoT는 retrieval을 추론의 일부로 통합하는 첫 번째 본격적인 시도였고, 오늘날 LangGraph·OpenAI Agents SDK에서 사용되는 retrieve-think-retrieve 루프의 이론적 토대를 제공한다. AI 엔지니어가 멀티스텝 RAG를 설계할 때 반드시 이해해야 하는 핵심 논문이다.

### 사전 지식
- Chain-of-Thought prompting (Wei et al., 2022)
- Multi-hop Question Answering 개념과 데이터셋(HotpotQA 등)
- Few-shot in-context learning
- 기본 RAG(Lewis et al., 2020) 아키텍처

### 관련 논문
- [Chain-of-Thought Prompting (Wei et al., 2022)](https://arxiv.org/abs/2201.11903)
- [Self-Ask: Measuring and Narrowing the Compositionality Gap (Press et al., 2022)](https://arxiv.org/abs/2210.03350)
- [ReAct: Synergizing Reasoning and Acting (Yao et al., 2022)](https://arxiv.org/abs/2210.03629)
- [MuSiQue: Multihop Questions via Single-hop Question Composition (Trivedi et al., 2022)](https://arxiv.org/abs/2108.00573)

### 실무 적용
- **엔터프라이즈 검색·QA:** 여러 문서에 흩어진 정보를 종합해야 하는 사내 위키 검색
- **Deep Research 에이전트:** Anthropic·OpenAI의 deep research류 제품의 기본 동작 방식
- **컴플라이언스·법무 자동화:** 규제 문서에서 단계적 reasoning으로 적용 가능한 조항 추출
- **AI Avatar/Dubbing 워크플로우:** 다국어·다문서 맥락에서 일관된 응답을 위한 reasoning chain 구성

---


## Paper 3 (Recent): From Local to Global: A Graph RAG Approach to Query-Focused Summarization
- **Authors:** Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, Jonathan Larson (Microsoft Research)
- **Year:** 2024
- **arXiv:** https://arxiv.org/abs/2404.16130
- **PDF:** [./graphrag-edge-2024.pdf](./graphrag-edge-2024.pdf)
- **Citation Count:** 1000+ (2026년 기준)

### 요약
Microsoft Research의 GraphRAG는 전통적 vector-only RAG가 잘 풀지 못하는 **"전역적(global) 질문"**(예: "이 코퍼스의 핵심 주제는?")을 해결하기 위해 LLM으로 코퍼스에서 엔티티-관계 그래프를 추출하고, Leiden 등 커뮤니티 탐지 알고리즘으로 계층적 요약을 생성한 뒤, 질문 시점에 community summary들을 map-reduce 방식으로 결합한다. 로컬·전역 질문 모두에서 baseline RAG를 큰 격차로 능가한다.

### 핵심 기여
- **Global vs Local query 분리:** RAG가 풀어야 할 질문 유형을 새롭게 정의
- **LLM 기반 지식 그래프 자동 구축:** 도메인 지식 그래프 수작업 부담 제거
- **Community summarization 파이프라인:** Leiden 알고리즘 + LLM 요약으로 계층적 코퍼스 표현
- **오픈소스 공개(GraphRAG library):** Microsoft가 PyPI 패키지로 공개하여 사실상 표준화

### 이 논문이 중요한 이유
2024년 이후 RAG 진영의 가장 큰 변곡점은 GraphRAG였다. 기존 vector RAG는 "이 문서에서 X에 대해 알려줘" 같은 local 질문엔 강하지만, "이 회사 전체 전략 방향은?" 같은 global 질문에선 무력하다. CPO/PM 관점에서 RAG 제품의 한계와 차세대 방향성을 이해하려면 반드시 읽어야 한다. 또한 2025년에 등장한 LightRAG, KAG, HippoRAG 등 후속 연구의 기준점이다.

### 사전 지식
- 기본 RAG 아키텍처 (Lewis et al., 2020)
- 임베딩 기반 vector retrieval과 한계점
- 그래프 알고리즘 기초 (community detection, Leiden 알고리즘)
- Map-reduce 패턴
- LLM의 정보 추출(entity/relation extraction) 능력

### 관련 논문
- [LightRAG: Simple and Fast Retrieval-Augmented Generation (Guo et al., 2024)](https://arxiv.org/abs/2410.05779)
- [HippoRAG: Neurobiologically Inspired Long-Term Memory for LLMs (Gutierrez et al., 2024)](https://arxiv.org/abs/2405.14831)
- [KAG: Boosting LLMs in Professional Domains via Knowledge Augmented Generation (Liang et al., 2024)](https://arxiv.org/abs/2409.13731)
- [RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval (Sarthi et al., 2024)](https://arxiv.org/abs/2401.18059)

### 실무 적용
- **엔터프라이즈 Knowledge Hub:** 회사 전체 문서를 그래프화해 "전사 전략 동향" 같은 거시 질문 응답
- **Customer Insight 분석:** 수천 건의 인터뷰/리뷰에서 테마와 페르소나를 자동 추출
- **Agentic AI 제품 설계:** Agent가 작업 도메인 그래프를 미리 구축해 long-horizon planning에 활용
- **데이터 드리븐 PM 워크플로우:** 사용자 피드백·세션 로그에서 community-level pattern 추출하여 가설 도출

---


## 추천 읽기 순서

1. **HyDE (Gao et al., 2022)** — 가장 가볍고 직관적이다. "왜 LLM이 retrieval을 도와줄 수 있는가"라는 advanced RAG의 출발점 사고를 잡는다.
2. **IRCoT (Trivedi et al., 2022)** — retrieval과 reasoning을 결합하는 방법론으로 시야를 확장한다. 이후 등장한 모든 Agentic RAG의 사고 모델을 이해할 수 있다.
3. **GraphRAG (Edge et al., 2024)** — 단일 쿼리·문서 관점을 넘어 "코퍼스 전체 구조"를 다루는 진화 단계로 마무리한다.

## 핵심 테이크어웨이

- **Vector-only RAG는 출발점일 뿐이다.** 실제 프로덕션 RAG는 ① 쿼리 보강(HyDE), ② 다단계 추론 통합(IRCoT), ③ 코퍼스 구조화(GraphRAG)라는 세 축으로 진화해왔다.
- **LLM은 retrieval의 보조 모델로도 강력하다.** 검색 학습 데이터가 없어도 LLM의 사전 지식만으로 retrieval 품질을 크게 끌어올릴 수 있다는 점은 cold-start 도메인 진입 전략의 핵심이다.
- **"Local vs Global" 질문 분류는 새로운 제품 디자인 축이다.** PM 관점에서 사용자의 질문 패턴을 두 유형으로 나누고 각각에 다른 retrieval 전략을 매핑하는 것이 차세대 RAG UX 설계의 핵심이 된다.
- **Retrieval은 한 번이 아니라 "loop"이다.** Agentic AI 시대에 retrieval은 단일 액션이 아닌 반복 가능한 도구로 다뤄야 하며, 이는 IRCoT 이후 표준이 되었다.
- **Knowledge Graph의 부활.** GraphRAG는 한때 verbose하다고 평가받던 KG를 LLM이 자동 구축하게 만들어 RAG 스택에 다시 끌어들였다.

## 다음 토픽과의 연결

- **Day 27: Vector Databases and Indexing** — HyDE의 가상 문서 임베딩, GraphRAG의 community embedding을 효율적으로 저장·검색하려면 vector DB의 indexing 기법(HNSW, IVF, hybrid search)을 이해해야 한다.
- **Module 8 복습 (LLM Orchestration / Agents)** — IRCoT의 retrieval-reasoning loop는 LangGraph·ReAct·Toolformer가 직접적으로 계승했다.
- **Module 7 (Prompt Engineering)** — HyDE의 hypothetical document 생성 프롬프트와 GraphRAG의 community summarization 프롬프트 모두 prompt 설계가 성능의 큰 비중을 차지한다.
- **Production Concerns:** 세 논문 모두 latency·cost 측면에서 추가 LLM 호출이 발생하므로, 다음 단계로 caching·distillation·routing 전략을 함께 고민해야 한다.
