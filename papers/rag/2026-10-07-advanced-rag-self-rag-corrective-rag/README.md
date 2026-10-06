# Daily AI Paper Recommendations

> **Date:** 2026-10-07
> **Module:** Module 9: RAG (Retrieval-Augmented Generation)
> **Topic:** Advanced RAG Self-RAG Corrective RAG

> 이번 사이클에서는 이전 사이클에서 다룬 Self-RAG, FLARE, CRAG, Speculative RAG, GraphRAG, Adaptive-RAG, RAPTOR 등과 겹치지 않도록 **"질의를 고치는 RAG(Query Rewriting)" → "RAG가 어디서 실패하는지 측정하는 벤치마크(RGB)" → "검색 자체를 RL로 학습하는 에이전틱 RAG(Search-R1)"** 흐름으로 구성했습니다.

---

## Paper 1 (Classic): Query Rewriting for Retrieval-Augmented Large Language Models
- **Authors:** Xinbei Ma, Yeyun Gong, Pengcheng He, Hai Zhao, Nan Duan
- **Year:** 2023 (EMNLP 2023)
- **arXiv:** https://arxiv.org/abs/2305.14283
- **PDF:** [./query-rewriting-rewrite-retrieve-read-ma-2023.pdf](./query-rewriting-rewrite-retrieve-read-ma-2023.pdf)
- **Citation Count:** ~500+ (approximate)

### 요약
기존 RAG 연구가 "검색기(retriever)"나 "리더(reader, LLM)"를 고치는 데 집중했다면, 이 논문은 **사용자의 원래 질의와 실제로 필요한 지식 사이의 간극**에 주목해 *질의 자체*를 고치는 Rewrite-Retrieve-Read 프레임워크를 제안합니다. LLM 프롬프팅으로 질의를 재작성하는 방식과, 작은 LM(T5)을 재작성기로 두고 블랙박스 LLM 리더의 피드백을 보상으로 강화학습하는 방식 두 가지를 보여줍니다.

### 핵심 기여
- Retrieve-then-Read 파이프라인 앞단에 **Rewrite 단계**를 추가한 Rewrite-Retrieve-Read 구조 정식화
- 블랙박스 LLM(API)을 건드리지 않고, **작은 학습 가능한 rewriter를 LLM 리더의 정답 여부를 보상으로 RL(PPO) 학습**하는 방법 제시
- 웹 검색 엔진(Bing)을 retriever로 사용해 오픈도메인 QA(HotpotQA, AmbigNQ 등)와 객관식 QA(MMLU)에서 일관된 성능 향상 입증

### 이 논문이 중요한 이유
실무 RAG 실패의 상당수는 "검색이 엉뚱한 문서를 가져온다"이고, 그 원인은 종종 **사용자 질문이 검색에 적합하지 않은 형태**라는 데 있습니다. 이 논문은 오늘날 거의 모든 RAG 프레임워크(LangChain MultiQueryRetriever, LlamaIndex Query Transform 등)에 들어간 "쿼리 재작성/확장" 기법의 학술적 출발점 중 하나이며, 이후 Search-R1 같은 "검색 행동을 RL로 학습"하는 흐름의 전조이기도 합니다.

### 사전 지식
- 기본 RAG 구조(Retrieve-then-Read)와 BM25/웹 검색 개념
- 강화학습 기초(PPO, 보상 함수, KL 페널티)
- T5 같은 seq2seq 모델과 few-shot 프롬프팅

### 관련 논문
- [Precise Zero-Shot Dense Retrieval without Relevance Labels / HyDE (Gao et al., 2022)](https://arxiv.org/abs/2212.10496)
- [Query2doc: Query Expansion with Large Language Models (Wang et al., 2023)](https://arxiv.org/abs/2303.07678)
- [Measuring and Narrowing the Compositionality Gap / Self-Ask (Press et al., 2022)](https://arxiv.org/abs/2210.03350)
- [RQ-RAG: Learning to Refine Queries for Retrieval Augmented Generation (Chan et al., 2024)](https://arxiv.org/abs/2404.00610)

### 실무 적용
- 챗봇형 서비스에서 대화 맥락이 섞인 질문("그거 가격은?")을 **독립 검색 질의로 재작성(condense question)**하는 단계로 바로 적용
- 사내 검색/고객지원 RAG에서 하나의 질문을 여러 하위 질의로 쪼개 병렬 검색(Multi-Query) 후 결과를 합치는 패턴
- 비싼 LLM은 그대로 두고 **작은 rewriter만 로그 기반으로 튜닝**하는 비용 효율적 개선 전략

---

## Paper 2 (Classic): Benchmarking Large Language Models in Retrieval-Augmented Generation
- **Authors:** Jiawei Chen, Hongyu Lin, Xianpei Han, Le Sun
- **Year:** 2023 (AAAI 2024)
- **arXiv:** https://arxiv.org/abs/2309.01431
- **PDF:** [./rgb-rag-benchmark-chen-2023.pdf](./rgb-rag-benchmark-chen-2023.pdf)
- **Citation Count:** ~800+ (approximate)

### 요약
RAG가 환각을 줄여준다고 하지만, "LLM이 검색 결과를 얼마나 잘 다루는가"는 체계적으로 측정된 적이 없었습니다. 이 논문은 RAG에 필요한 4가지 기본 능력 — **노이즈 강건성(Noise Robustness), 거부 능력(Negative Rejection), 정보 통합(Information Integration), 반사실 강건성(Counterfactual Robustness)** — 을 정의하고, 영어·중국어 벤치마크 RGB를 구축해 여러 LLM을 평가했습니다. 결과적으로 LLM은 노이즈에는 어느 정도 버티지만, 답이 없을 때 거부하기·여러 문서 통합·잘못된 정보 걸러내기에서 크게 실패함을 보였습니다.

### 핵심 기여
- RAG 성능을 **4개의 독립된 능력 축**으로 분해한 평가 프레임워크 제시
- 최신 뉴스 기반으로 구성해 사전학습 지식 누출을 줄인 이중언어 벤치마크 RGB 공개
- ChatGPT, ChatGLM, Vicuna 등 LLM의 RAG 병목을 정량화 — 특히 "문서가 틀렸을 때 그대로 믿는" 반사실 취약성 확인

### 이 논문이 중요한 이유
Self-RAG(자기 비판), CRAG(검색 결과 교정) 같은 Advanced RAG 기법이 **"왜 필요한가"를 데이터로 보여주는 논문**입니다. RAG 시스템을 만드는 엔지니어에게 "정확도 하나"가 아니라 거부율·통합 능력·오정보 탐지율을 따로 봐야 한다는 평가 관점을 제공하며, 이후 RAGAS, CRUD-RAG 등 RAG 평가 연구의 기준점이 되었습니다.

### 사전 지식
- RAG 파이프라인과 top-k 검색 결과가 프롬프트에 들어가는 방식
- LLM 환각(hallucination)과 평가 지표(Accuracy, Rejection Rate, Error Detection Rate) 개념
- (선택) Self-RAG, CRAG 논문 — 이 벤치마크가 드러낸 문제를 해결하려는 시도들

### 관련 논문
- [RAGAS: Automated Evaluation of Retrieval Augmented Generation (Es et al., 2023)](https://arxiv.org/abs/2309.15217)
- [Making Retrieval-Augmented Language Models Robust to Irrelevant Context (Yoran et al., 2023)](https://arxiv.org/abs/2310.01558)
- [Corrective Retrieval Augmented Generation / CRAG (Yan et al., 2024)](https://arxiv.org/abs/2401.15884)
- [Lost in the Middle: How Language Models Use Long Contexts (Liu et al., 2023)](https://arxiv.org/abs/2307.03172)

### 실무 적용
- 사내 RAG 평가셋을 만들 때 4개 축(노이즈 섞기, 정답 문서 제거, 다문서 질문, 오정보 주입)으로 **테스트 케이스를 설계**하는 템플릿으로 활용
- "모르면 모른다고 답하기(negative rejection)"를 KPI로 별도 추적 — B2B 고객지원/법무/의료 도메인에서 신뢰도 핵심
- 모델 교체(예: GPT → 오픈소스) 시 회귀 테스트 기준으로 사용

---

## Paper 3 (Recent): Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning
- **Authors:** Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Ö. Arık, Dong Wang, Hamed Zamani, Jiawei Han
- **Year:** 2025 (COLM 2025)
- **arXiv:** https://arxiv.org/abs/2503.09516
- **PDF:** [./search-r1-jin-2025.pdf](./search-r1-jin-2025.pdf)
- **Code:** https://github.com/PeterGriffinJin/Search-R1
- **Citation Count:** ~700+ (approximate)

### 요약
DeepSeek-R1 스타일의 결과 기반 강화학습을 RAG로 확장해, LLM이 **추론 도중 스스로 검색 질의를 생성하고(`<search>`), 결과를 읽고(`<information>`), 다시 추론**하는 다중 턴 검색 행동을 학습하게 합니다. 검색으로 가져온 토큰은 손실 계산에서 마스킹해 학습을 안정화하고, 보상은 최종 정답 일치(EM)만 사용하는 단순한 구조입니다. 7개 QA 데이터셋에서 동일 조건의 RAG 베이스라인 대비 Qwen2.5-7B 기준 약 24%, 3B 기준 약 20% 성능 향상을 보고했습니다.

### 핵심 기여
- "언제, 무엇을, 몇 번 검색할지"를 프롬프트 규칙이 아닌 **RL로 학습되는 정책**으로 전환
- **Retrieved token masking**: 외부 문서 토큰에는 그래디언트를 주지 않아 RL 학습 안정성 확보
- 과정 보상 없이 **결과 기반 보상(outcome reward)**만으로도 다중 홉 검색 추론이 창발함을 실험으로 입증 (PPO/GRPO 비교 포함)

### 이 논문이 중요한 이유
Self-RAG(반성 토큰), FLARE(불확실할 때 검색), CRAG(검색 결과 평가)가 **사람이 설계한 규칙/토큰으로 "적응형 검색"**을 구현했다면, Search-R1은 이를 **RL로 end-to-end 학습**하는 방향으로 패러다임을 옮겼습니다. 2025년 이후 Deep Research류 에이전트, R1-Searcher, ZeroSearch, ReSearch 등 "에이전틱 RAG + RL" 연구 붐의 출발점이며, AI 에이전트 개발자가 반드시 이해해야 할 설계 관점입니다.

### 사전 지식
- ReAct, IRCoT 같은 추론-검색 교차(interleaving) 방식
- RL for LLM 기초: PPO, GRPO, 결과 기반 보상(RLVR)
- DeepSeek-R1의 "추론을 RL로 끌어내기" 개념

### 관련 논문
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning (DeepSeek-AI, 2025)](https://arxiv.org/abs/2501.12948)
- [R1-Searcher: Incentivizing the Search Capability in LLMs via Reinforcement Learning (Song et al., 2025)](https://arxiv.org/abs/2503.05592)
- [ZeroSearch: Incentivize the Search Capability of LLMs without Searching (Sun et al., 2025)](https://arxiv.org/abs/2505.04588)
- [Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG (Singh et al., 2025)](https://arxiv.org/abs/2501.09136)

### 실무 적용
- Deep Research / 리서치 에이전트처럼 **여러 번 검색하며 답을 좁혀가는 제품**의 백본 학습 레시피
- 도메인 검색 도구(사내 위키, 상품 DB)를 붙인 소형 모델(3B~7B)을 RL로 튜닝해 **대형 API 모델 대비 비용 절감**
- 검색 호출 횟수·지연시간을 보상에 반영해 품질-비용 트레이드오프를 정책 수준에서 조절하는 아이디어로 확장 가능

---

## 추천 읽기 순서
1. **RGB (Chen et al., 2023)** — 먼저 "RAG는 어디서 실패하는가"를 4가지 능력 축으로 이해합니다. 문제 정의가 선행되어야 해결책이 보입니다.
2. **Rewrite-Retrieve-Read (Ma et al., 2023)** — 실패 원인 중 "질의-지식 간극"을 질의 재작성으로 푸는 가장 실용적인 해법과, rewriter를 RL로 학습하는 아이디어를 익힙니다.
3. **Search-R1 (Jin et al., 2025)** — 질의 재작성과 적응형 검색을 하나의 RL 정책으로 통합한 최신 접근을 읽으며, 2번의 RL 아이디어가 어떻게 확장되었는지 비교합니다.

## 핵심 테이크어웨이
- **Q. Advanced RAG의 본질은?** → "한 번 검색해서 붙인다"를 넘어 *무엇을(질의), 언제(적응형), 어떻게 믿을지(검증)*를 제어하는 것.
- **Q. RAG 품질은 어떻게 측정해야 하나?** → 단일 정확도가 아니라 노이즈 강건성·거부·통합·오정보 탐지를 분리해서 봐야 병목이 보인다(RGB).
- **Q. 가장 싸고 빠른 개선 레버는?** → 질의 재작성. LLM·인덱스를 바꾸지 않고도 검색 적중률을 끌어올릴 수 있다.
- **Q. 앞으로의 방향은?** → 규칙 기반 Self-RAG/CRAG → **RL로 학습되는 에이전틱 검색 정책(Search-R1)**. RAG와 에이전트의 경계가 사라지고 있다.

## 다음 토픽과의 연결
다음 토픽은 **Vector Databases and Indexing**입니다. 오늘 본 Rewrite-Retrieve-Read와 Search-R1은 **질의 수가 1회 → 다회로 늘어나는 구조**라, 검색 백엔드의 지연시간·처리량이 곧 제품 UX와 비용이 됩니다. 내일은 HNSW, IVF-PQ, 하이브리드(sparse+dense) 검색 등 이 "다회 검색"을 감당하는 인덱싱 기술을 살펴보며, Advanced RAG 설계가 인프라 선택에 어떤 요구사항을 주는지 연결해 보세요.
