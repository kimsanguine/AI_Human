# Daily AI Paper Recommendations

> **Date:** 2026-04-27
> **Module:** Module 9: RAG (Retrieval-Augmented Generation)
> **Topic:** RAG Architecture and Optimization

---

## Paper 1 (Classic): Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
- **Authors:** Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela
- **Year:** 2020 (NeurIPS 2020)
- **arXiv:** https://arxiv.org/abs/2005.11401
- **PDF:** [./rag-lewis-2020.pdf](./rag-lewis-2020.pdf)
- **Citation Count:** ~6,000+ (RAG라는 용어를 처음 정립한 논문)

### 요약
이 논문은 사전학습된 seq2seq 모델(BART)과 비파라메트릭 메모리(Wikipedia DPR 인덱스)를 결합하여 "RAG"라는 패러다임을 처음으로 정의한 작품이다. 두 가지 변형(RAG-Sequence, RAG-Token)을 제안하고, Open Domain QA, Fact Verification, Question Generation 등 지식 집약적 태스크에서 SOTA를 달성하면서 환각(hallucination)을 줄이고 사실 기반 응답을 생성할 수 있음을 증명했다.

### 핵심 기여
- 파라메트릭 메모리(LLM weights)와 비파라메트릭 메모리(외부 지식 인덱스)를 통합한 end-to-end 학습 가능한 RAG 프레임워크 제안
- RAG-Sequence(전체 답변에 같은 문서 사용) vs RAG-Token(토큰마다 다른 문서 사용) 두 가지 marginal 디코딩 방식 비교
- DPR retriever + BART generator의 조합으로 Natural Questions, TriviaQA, WebQuestions 등에서 폐쇄형 QA 모델 대비 큰 성능 향상
- 외부 지식을 swap-in/swap-out 가능한 "non-parametric memory"로 다룸으로써, 재학습 없이 지식 업데이트가 가능하다는 점을 보임

### 이 논문이 중요한 이유
오늘날 모든 LLM 제품에 들어가는 RAG의 원형이자 명명자다. ChatGPT 검색 기능, Perplexity, Notion AI, 사내 지식 챗봇 등 거의 모든 enterprise AI 제품의 backbone 패러다임이 여기서 출발했다. AI 엔지니어라면 이 논문의 retriever-generator 분리 설계, marginalization 수식, 그리고 "왜 retrieval이 fine-tuning보다 지식 주입에 효과적인가"에 대한 직관을 반드시 이해하고 있어야 한다.

### 사전 지식
- Seq2seq 모델 구조 (BART, T5)
- DPR(Dense Passage Retrieval) 및 dual-encoder 검색 구조
- Top-k retrieval과 marginalization (확률적 latent 문서 모델링)
- Open-Domain QA 벤치마크 (Natural Questions, TriviaQA)

### 관련 논문
- [REALM: Retrieval-Augmented Language Model Pre-Training (Guu et al., 2020)](https://arxiv.org/abs/2002.08909)
- [Dense Passage Retrieval for Open-Domain QA (Karpukhin et al., 2020)](https://arxiv.org/abs/2004.04906)
- [FiD: Leveraging Passage Retrieval with Generative Models (Izacard & Grave, 2020)](https://arxiv.org/abs/2007.01282)

### 실무 적용
사내 문서 챗봇, 고객 지원 자동화, 법률·의료 AI 어시스턴트 등 "기업 내 정확한 지식 기반 응답"이 필요한 모든 제품에서 1차 아키텍처로 채택된다. 실무에서는 이 논문의 RAG-Token 방식을 직접 쓰진 않지만, retriever→reranker→generator 파이프라인과 "검색 결과를 context로 stuffing 후 생성"하는 prompting 방식이 사실상 산업 표준이 되었다. LangChain, LlamaIndex, Haystack 같은 프레임워크의 기본 구조도 이 논문에서 직접 영감을 받았다.

---

## Paper 2 (Classic): REALM: Retrieval-Augmented Language Model Pre-Training
- **Authors:** Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, Ming-Wei Chang
- **Year:** 2020 (ICML 2020)
- **arXiv:** https://arxiv.org/abs/2002.08909
- **PDF:** [./realm-guu-2020.pdf](./realm-guu-2020.pdf)
- **Citation Count:** ~2,500+

### 요약
REALM은 사전학습 단계부터 retrieval을 명시적으로 통합한 최초의 언어 모델이다. Masked Language Modeling 동안 신경망 retriever(knowledge retriever)를 함께 학습하여, 모델이 매 입력마다 Wikipedia에서 관련 문서를 검색·참조하는 방식으로 학습된다. Open-Domain QA에서 T5처럼 더 큰 폐쇄형 모델을 압도하며, "지식을 weight에 박아넣지 않고 외부에서 가져오는" 학습 패러다임을 제시했다.

### 핵심 기여
- Pre-training 단계에서 retriever와 encoder를 jointly end-to-end로 학습하는 최초의 시도 (latent variable로 retrieved doc을 marginalize)
- Inverse Cloze Task 없이 MLM 신호만으로 retriever를 학습할 수 있음을 보임
- Asynchronous index refresh: 거대 corpus를 매 step 임베딩하지 않고, 주기적으로 인덱스를 갱신하여 효율적인 학습 구현
- Open-Domain QA(NQ, WebQuestions, CuratedTrec)에서 T5-11B를 크게 상회하는 성능 — 작은 모델 + retrieval이 큰 모델 단독보다 효과적임을 증명

### 이 논문이 중요한 이유
RAG가 "추론 시점(inference-time) retrieval"이라면, REALM은 "사전학습 시점 retrieval"이라는 더 근본적인 패러다임을 제시했다. 최근 Atlas, RETRO, In-Context RALM, Self-RAG 같은 retrieval-aware pre-training/post-training 흐름의 출발점이며, "스케일이 클수록 retrieval은 의미가 없다"는 통념을 뒤집은 핵심 증거다. 또한 retriever를 LLM과 함께 학습하는 alignment 문제를 처음으로 정면으로 다룬 작품이다.

### 사전 지식
- BERT의 Masked Language Modeling 목적함수
- Latent variable model과 EM, marginal likelihood
- ANN 인덱스(MIPS, Maximum Inner Product Search)와 비동기 인덱스 갱신 전략
- Open-Domain QA의 단일 retrieval-then-read 파이프라인

### 관련 논문
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP (Lewis et al., 2020)](https://arxiv.org/abs/2005.11401)
- [Atlas: Few-shot Learning with Retrieval Augmented Language Models (Izacard et al., 2022)](https://arxiv.org/abs/2208.03299)
- [Improving Language Models by Retrieving from Trillions of Tokens / RETRO (Borgeaud et al., 2021)](https://arxiv.org/abs/2112.04426)

### 실무 적용
실무에서 REALM을 그대로 쓰는 일은 드물지만, 이 논문이 만든 영향은 깊다. 첫째, 도메인 특화 LLM을 만들 때 "전체 corpus로 fine-tuning할 것인가 vs retrieval로 외재화할 것인가"의 의사결정 프레임은 REALM에서 출발한다. 둘째, vector DB의 ANN 인덱스를 주기적으로 재구성하는 운영 패턴(asynchronous reindexing)도 REALM의 학습 트릭과 동일한 발상이다. 셋째, retrieval을 학습 signal로 쓰는 최근의 RA-DIT, SAIL, Self-RAG 같은 기법을 이해하려면 REALM의 marginalization 수식 이해가 필수다.

---

## Paper 3 (Recent): Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG
- **Authors:** Aditi Singh, Abul Ehtesham, Saket Kumar, Tala Talaei Khoei
- **Year:** 2025 (arXiv 2025년 1월)
- **arXiv:** https://arxiv.org/abs/2501.09136
- **PDF:** [./agentic-rag-survey-singh-2025.pdf](./agentic-rag-survey-singh-2025.pdf)
- **Citation Count:** 200+ (2025년 기준 빠르게 인용 증가 중)

### 요약
이 서베이는 Naïve RAG → Advanced RAG → Modular RAG → Graph RAG → Agentic RAG로 이어지는 RAG 아키텍처의 진화 궤적을 정리한 2025년 핵심 문헌이다. 특히 자율 에이전트가 RAG 파이프라인 안에서 reflection, planning, tool use, multi-agent collaboration을 수행하는 Agentic RAG의 분류 체계(single-agent / multi-agent / hierarchical / corrective / adaptive)를 제시하고, 의료·법률·금융·교육 분야의 적용 사례와 LangGraph, LlamaIndex, CrewAI, AutoGen 등 구현 프레임워크를 연결지어 다룬다.

### 핵심 기여
- RAG 패러다임의 5세대 진화 분류(Naïve → Advanced → Modular → Graph → Agentic)를 명확한 비교 표로 정리
- Agentic RAG의 5가지 아키텍처 패턴 분류: Single-Agent, Multi-Agent, Hierarchical, Corrective, Adaptive RAG
- 산업별 적용 사례(헬스케어, 법률, 핀테크, 교육)와 도메인별 베스트 프랙티스 매핑
- 주요 구현 프레임워크(LangGraph, LlamaIndex, CrewAI, OpenAI Swarm, Haystack Agents)와 평가 벤치마크 정리
- adaptive retrieval, query rewriting, dynamic tool selection 등 modular RAG 최신 기법의 분류 체계 제공

### 이 논문이 중요한 이유
2024-2025년 RAG 분야의 가장 큰 변화는 "정적 retrieval 파이프라인"에서 "에이전트 기반 동적 retrieval"으로의 이동이다. 이 서베이는 그 변화의 큰 그림을 한눈에 보여주는 가장 최신의 지도다. AI 엔지니어가 production에서 RAG를 설계할 때 "Naïve RAG로 시작 → Advanced/Modular로 확장 → 필요 시 Agentic으로 고도화"라는 의사결정 트리를 갖추는 데 직접적인 도움이 된다. 또한 Anthropic MCP, OpenAI Agents SDK 같은 최신 에이전트 프레임워크가 RAG와 어떻게 결합되는지에 대한 이론적 위치를 제공한다.

### 사전 지식
- Naïve RAG의 retrieve-then-read 파이프라인
- Tool use / Function calling LLM (ReAct, Toolformer)
- LLM 에이전트 기본 개념 (planner, executor, memory, reflection)
- Self-RAG, Corrective RAG (CRAG), FLARE 등 reflective RAG 기법

### 관련 논문
- [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection (Asai et al., 2023)](https://arxiv.org/abs/2310.11511)
- [Retrieval-Augmented Generation for Large Language Models: A Survey (Gao et al., 2023)](https://arxiv.org/abs/2312.10997)
- [A Comprehensive Survey of Retrieval-Augmented Generation (RAG): Evolution, Current Landscape and Future Directions (Gupta et al., 2024)](https://arxiv.org/abs/2410.12837)

### 실무 적용
B2B SaaS 제품에서 RAG를 도입할 때 가장 흔히 마주치는 문제 — "복잡한 질의에 대해 단일 retrieval로 충분치 않다", "정확도와 비용/지연의 tradeoff를 어떻게 관리할 것인가" — 에 대한 실용적 디자인 패턴을 제시한다. 예를 들어 Adaptive RAG는 쿼리 복잡도에 따라 0/1/N-step retrieval을 동적으로 선택해 단순 질의는 빠르게, 복잡 질의는 깊게 처리한다. Multi-Agent RAG는 도메인별 specialist agent(예: SQL agent, doc agent, web search agent)를 분리해 사내 데이터·문서·실시간 정보를 통합 응답한다. Customer support, legal-tech, FP&A copilot 같은 실제 production 시스템에서 ROI가 큰 패턴들이다.

---

## 추천 읽기 순서
1. **Lewis et al. (RAG, 2020)** — RAG 패러다임의 정의와 retriever-generator 통합 학습의 기본기를 잡는다.
2. **Guu et al. (REALM, 2020)** — Pre-training 단계 retrieval과 marginalization의 수학적 기반을 익힌다. RAG보다 먼저 읽어도 무방하지만, 수식이 다소 무거우므로 RAG 논문으로 직관을 잡은 뒤 보는 편이 효율적이다.
3. **Singh et al. (Agentic RAG Survey, 2025)** — 2020-2025년의 발전 궤적을 한 번에 조망하고, 실무 디자인 패턴 카탈로그로 활용한다.

## 핵심 테이크어웨이
- **RAG의 본질은 "지식의 외재화"다.** 모델 weight에 모든 지식을 넣는 대신, 외부 인덱스에서 동적으로 가져온다. 이는 정확성, 업데이트 용이성, 출처 추적 가능성이라는 세 가지 압도적 이점을 준다.
- **Retriever와 Generator는 분리되지만 정렬되어야 한다.** REALM은 jointly train, RAG는 retriever 고정 + generator 학습이라는 두 가지 정렬 전략을 제시했다. 실무에서는 보통 후자 + reranker 추가가 가장 비용 효율적이다.
- **2024-2025의 화두는 Modular RAG와 Agentic RAG.** 단일 파이프라인이 아닌, 라우팅/리쿼리/리플렉션/자기수정을 포함한 "동적 RAG"로 패러다임이 이동하고 있다.
- **Adaptive retrieval이 곧 비용 최적화 전략이다.** 모든 쿼리에 retrieval을 호출하면 지연·비용이 폭증한다. 쿼리 복잡도에 따라 retrieval 여부와 깊이를 동적으로 결정하는 능력이 production-grade RAG의 차별점이다.

## 다음 토픽과의 연결
내일은 Day 26: **Advanced RAG — Self-RAG, Corrective RAG, FLARE**로 이어진다. 오늘 학습한 RAG/REALM의 기본 파이프라인 위에서, 모델이 스스로 retrieval 여부와 결과 품질을 판단하고 수정하는 reflective RAG 기법들이 어떻게 작동하는지 살펴본다. 또한 Day 27의 **Vector DB와 인덱싱(FAISS, HNSW)**으로 가면 오늘 다룬 retrieval의 인프라 레이어가 어떻게 구성되는지를 깊게 이해하게 된다. Module 9 전체를 관통하는 질문은 "production-grade RAG는 정확도, 지연, 비용, 신선도 사이에서 어떻게 균형을 잡는가?"이다.
