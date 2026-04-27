# Daily AI Paper Recommendations

> **Date:** 2026-04-28
> **Module:** Module 9 — RAG (Retrieval-Augmented Generation)
> **Topic:** Advanced RAG — Self-RAG, Corrective RAG, and Speculative RAG

---

## Paper 1 (Classic): Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection
- **Authors:** Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, Hannaneh Hajishirzi
- **Year:** 2023
- **arXiv:** https://arxiv.org/abs/2310.11511
- **PDF:** [./self-rag-asai-2023.pdf](./self-rag-asai-2023.pdf)
- **Citation Count:** 1,500+ (ICLR 2024 Oral)

### 요약
Self-RAG는 LLM이 "검색이 필요한가?", "검색 결과가 관련 있는가?", "내가 생성한 답변이 근거에 부합하는가?"를 스스로 판단하도록 학습시키는 프레임워크다. 핵심은 'Reflection Token'이라는 특수 토큰을 도입해, 모델이 추론 과정에서 능동적으로 검색을 트리거하고 결과를 비평하며 출력을 검증하도록 만드는 것이다. 결과적으로 hallucination을 줄이고 사실성을 유의미하게 끌어올린다.

### 핵심 기여
- **Reflection Token 도입**: `Retrieve`, `IsRel`, `IsSup`, `IsUse` 4종의 특수 토큰으로 검색·관련성·근거성·유용성을 단일 디코딩 루프 안에서 판단
- **On-demand Retrieval**: 매 토큰/매 문장마다 무조건 검색하지 않고, 모델이 필요할 때만 검색을 호출 — latency와 noise를 동시에 줄임
- **Self-Critique 학습 파이프라인**: GPT-4로 reflection 라벨을 생성한 뒤 7B/13B 모델에 distill, 추론 시 외부 critic 없이 모델 자체가 판단

### 이 논문이 중요한 이유
일반적인 RAG는 "검색 → 무조건 컨텍스트에 주입 → 생성"이라는 정적 파이프라인이라 noise에 취약하다. Self-RAG는 RAG를 **에이전트적 의사결정**으로 전환한 첫 대표 논문으로, 이후 Corrective RAG, Adaptive RAG, Agentic RAG 등 거의 모든 advanced RAG 연구의 출발점이 된다. AI 엔지니어가 RAG 품질 문제를 마주쳤을 때 가장 먼저 검토해야 할 baseline이다.

### 사전 지식
- 기본 RAG 파이프라인 (Lewis et al., 2020)과 dense retrieval 개념
- Instruction tuning 및 supervised fine-tuning의 데이터 구성 방식
- LLM의 special token / control token 메커니즘

### 관련 논문
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (Lewis et al., 2020)](https://arxiv.org/abs/2005.11401)
- [Active Retrieval Augmented Generation / FLARE (Jiang et al., 2023)](https://arxiv.org/abs/2305.06983)
- [Corrective Retrieval Augmented Generation (Yan et al., 2024)](https://arxiv.org/abs/2401.15884)

### 실무 적용
사내 지식 QA 챗봇에서 retrieval noise로 답변 품질이 망가질 때 Self-RAG 패턴이 강력하다. 실전에서는 (1) "검색 필요 여부 판단"을 별도 small classifier로 분리하거나, (2) `IsSup`(근거 일치도) 점수를 응답 confidence로 노출해 UI에서 "답변 근거 보기/낮은 확신" 같은 신뢰성 시그널로 변환하는 식으로 응용된다. Notion AI Q&A, Perplexity, Glean 같은 엔터프라이즈 RAG가 모두 유사한 self-reflection 단계를 운영 중이다.

---

## Paper 2 (Classic): Active Retrieval Augmented Generation (FLARE)
- **Authors:** Zhengbao Jiang, Frank F. Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie Callan, Graham Neubig
- **Year:** 2023
- **arXiv:** https://arxiv.org/abs/2305.06983
- **PDF:** [./flare-jiang-2023.pdf](./flare-jiang-2023.pdf)
- **Citation Count:** 700+ (EMNLP 2023)

### 요약
FLARE(Forward-Looking Active REtrieval)는 long-form 생성 중 모델이 **다음에 쓸 문장을 미리 예측**한 뒤, 그 예측에서 confidence가 낮은 토큰이 발견되면 그 문장을 쿼리로 변환해 즉석 검색을 수행한다. 즉, 검색을 한 번에 끝내지 않고 **생성 중간중간 능동적으로** 다시 검색하는 iterative RAG의 대표 방식이다.

### 핵심 기여
- **Look-ahead 기반 retrieval 트리거**: 다음 문장을 임시로 생성해보고, 그 안의 low-confidence 토큰이 임계값을 넘으면 검색 호출
- **쿼리 자동 재구성**: 생성 중인 문장 자체를 쿼리로 사용하여 사용자의 원 질문을 넘어선 sub-topic 검색이 가능
- **학습 없이 동작 (training-free)**: Self-RAG와 달리 추가 fine-tuning 없이 GPT-3.5/3.5-turbo 같은 black-box 모델 위에서도 그대로 적용 가능

### 이 논문이 중요한 이유
RAG의 핵심 난제 중 하나는 "한 번 검색한 컨텍스트가 long-form 답변 후반부에는 더 이상 충분하지 않다"는 점이다. FLARE는 이 문제를 명시적으로 정의하고 **active retrieval**이라는 패러다임을 도입했다. RAG 연구를 single-shot에서 multi-step / iterative으로 확장한 분기점이며, 이후 Self-RAG, IRCoT, Adaptive-RAG의 사상적 토대가 된다.

### 사전 지식
- LLM의 token-level log-probability / confidence 해석
- Iterative / multi-hop QA 개념 (예: HotpotQA, 2WikiMultiHopQA)
- 기본 RAG와 dense retriever (DPR, Contriever) 동작 원리

### 관련 논문
- [Self-RAG (Asai et al., 2023)](https://arxiv.org/abs/2310.11511)
- [Interleaving Retrieval with Chain-of-Thought / IRCoT (Trivedi et al., 2022)](https://arxiv.org/abs/2212.10509)
- [Adaptive-RAG (Jeong et al., 2024)](https://arxiv.org/abs/2403.14403)

### 실무 적용
긴 보고서·법률 의견서·기술 문서를 LLM이 생성해야 하는 케이스에서 FLARE 패턴이 효과적이다. 운영 환경에서는 매 문장마다 retrieval을 호출하는 비용이 부담되므로, (1) 토큰 confidence threshold를 도메인별로 튜닝하거나, (2) "사실 진술 문장 vs 연결 문장"을 분류해 사실 진술일 때만 active retrieval을 발동시키는 변형을 쓴다. Perplexity의 long-form mode, Anthropic Claude의 web-search 모드가 본질적으로 같은 아이디어를 사용한다.

---

## Paper 3 (Recent): Speculative RAG: Enhancing Retrieval Augmented Generation through Drafting
- **Authors:** Zilong Wang, Zifeng Wang, Long Le, Huaixiu Steven Zheng, Swaroop Mishra, Vincent Perot, Yuwei Zhang, Anush Mattapalli, Ankur Taly, Jingbo Shang, Chen-Yu Lee, Tomas Pfister
- **Year:** 2024 (ICLR 2025)
- **arXiv:** https://arxiv.org/abs/2407.08223
- **PDF:** [./speculative-rag-wang-2024.pdf](./speculative-rag-wang-2024.pdf)
- **Citation Count:** 200+ (Google Research, ICLR 2025)

### 요약
Speculative RAG는 **작은 specialist LM이 다수의 답변 초안(draft)을 병렬 생성**하고, **큰 generalist LM이 그 초안들을 한 번에 검증(verify)**하는 두 단계 구조를 제안한다. 각 draft는 검색 결과의 서로 다른 부분집합을 사용하므로 다양한 관점을 반영하면서도 draft당 입력 토큰을 줄여 long-context 위치 편향을 완화한다. PubHealth에서 정확도 +12.97%, latency −51%를 보고하며 정확도와 속도를 동시에 개선했다.

### 핵심 기여
- **Drafter–Verifier 분리 구조**: speculative decoding의 아이디어를 token-level이 아니라 **answer-level**로 끌어올림
- **Document subset 분할 전략**: 검색된 문서를 군집화해 각 draft에 다른 부분집합만 주입 → 다양한 가설 + position bias 완화
- **검증을 verifier LM의 단일 forward pass로 처리**: 답변 후보들의 perplexity·confidence를 기반으로 best draft 선택, generalist LM에 long context를 직접 강요하지 않음

### 이 논문이 중요한 이유
2024–2025년 RAG 연구의 핵심 축은 "정확도 vs 비용/지연" 트레이드오프다. Speculative RAG는 small/large 모델 협업이라는 실용적 패러다임을 RAG에 이식하면서 두 축을 동시에 개선한 보기 드문 결과를 냈다. AI 엔지니어가 production RAG의 latency를 잡으면서 hallucination도 줄여야 할 때 가장 먼저 시도해볼 만한 아키텍처다.

### 사전 지식
- Self-RAG와 Corrective RAG의 reflection/critique 메커니즘
- Speculative decoding (Leviathan et al., 2023) — token-level draft & verify
- Long-context "lost in the middle" 현상과 position bias

### 관련 논문
- [Self-RAG (Asai et al., 2023)](https://arxiv.org/abs/2310.11511)
- [Corrective Retrieval Augmented Generation / CRAG (Yan et al., 2024)](https://arxiv.org/abs/2401.15884)
- [Speculative Decoding (Leviathan et al., 2023)](https://arxiv.org/abs/2211.17192)

### 실무 적용
검색 결과가 많고 노이즈가 큰 엔터프라이즈 RAG (예: 사내 위키 + 티켓 + 코드)에서 특히 효과적이다. 실전 배치에서는 (1) drafter로 7B 클래스 distilled 모델, verifier로 GPT-4o-mini/Claude Haiku 같은 fast 대형 모델을 두고, (2) draft를 K=3~5개로 병렬 생성한 뒤 logprob 기반 ranking + verifier 검증을 결합해 사용한다. 코드 RAG, 의료 RAG처럼 정확도가 중요한 도메인에서 self-consistency를 대체하는 새로운 default가 되어가는 추세다.

---

## 추천 읽기 순서
1. **FLARE (Paper 2)** — 먼저 "왜 RAG가 한 번 검색으로 끝나면 안 되는가?"라는 문제 의식을 잡는다. 가장 가볍고 training-free라 멘탈 모델을 빠르게 형성하기 좋다.
2. **Self-RAG (Paper 1)** — FLARE의 active retrieval 아이디어를 모델 내부 reflection token으로 끌어올린 흐름을 이해한다. Reflection token 설계가 이후 모든 advanced RAG의 공용어가 된다.
3. **Speculative RAG (Paper 3)** — 위 두 논문이 만든 reflective/iterative RAG를 production 관점(정확도 + 지연)에서 어떻게 풀어냈는지 본다. 2024–2025 트렌드인 small+large 협업 구조의 실제 사례.

## 핵심 테이크어웨이
- Advanced RAG의 본질은 "검색을 **언제, 무엇을, 어떻게** 다시 할지를 모델이 스스로 결정"하게 만드는 것이다. 정적인 retrieve-then-generate는 더 이상 충분하지 않다.
- **세 개의 직교 축**이 존재한다: (1) **언제 검색할지** 결정(FLARE의 confidence trigger / Self-RAG의 Retrieve token), (2) **검색 결과 신뢰도** 평가(Self-RAG의 IsRel/IsSup, CRAG의 evaluator), (3) **여러 후보를 어떻게 종합**할지(Speculative RAG의 drafter–verifier).
- 운영 관점에서 Reflection / Critique / Verification은 결국 **별도 작은 모델로 분리**되는 추세다. 이는 latency와 비용을 잡으면서도 self-consistency급 정확도를 얻기 위한 자연스러운 진화 방향이다.
- 모든 advanced RAG는 결국 **에이전트적**이다. RAG가 "검색 도구를 가진 reasoning agent"로 수렴 중이며, 이는 다음 토픽인 vector DB 인프라와 그 위의 agent 오케스트레이션으로 자연스럽게 이어진다.

## 다음 토픽과의 연결
다음 Day 27 토픽인 **"Vector Databases and Indexing"**(FAISS, HNSW)은 오늘 다룬 advanced RAG가 production에서 동작하기 위한 **인프라 계층**이다. Self-RAG·FLARE·Speculative RAG 모두 retrieval을 여러 번 호출하는 구조이므로 ANN 검색의 latency·recall·메모리 트레이드오프가 시스템 성능을 직접 결정한다. 또한 Speculative RAG의 draft별 document subset 분할은 vector DB의 클러스터링/필터링 기능과 직접 맞물리므로, 두 토픽을 함께 보면 production-grade RAG 스택의 전체 그림이 완성된다.
