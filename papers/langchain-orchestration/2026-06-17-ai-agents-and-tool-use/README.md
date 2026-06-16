# Daily AI Paper Recommendations

> **Date:** 2026-06-17
> **Module:** Module 8: LangChain and LLM Orchestration
> **Topic:** AI Agents and Tool Use

---

## Paper 1 (Classic): ReAct: Synergizing Reasoning and Acting in Language Models
- **Authors:** Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, Yuan Cao
- **Year:** 2022 (ICLR 2023)
- **arXiv:** https://arxiv.org/abs/2210.03629
- **PDF:** [./react-yao-2022.pdf](./react-yao-2022.pdf)
- **Citation Count:** approx. 4,500+

### 요약
ReAct는 LLM이 "추론(Reasoning)"과 "행동(Acting)"을 분리하지 않고 번갈아 가며 생성하도록 만든 프롬프팅 패러다임이다. 추론 흔적(reasoning trace)은 모델이 행동 계획을 세우고 추적·수정하게 해주고, 행동(action)은 외부 도구나 지식원(API, 검색 등)과 연결되어 추가 정보를 가져오게 한다. 이 둘의 상호작용으로 환각(hallucination)을 줄이고, 사람이 해석 가능한 의사결정 경로를 만든다.

### 핵심 기여
- "생각 → 행동 → 관찰(thought–action–observation)" 루프를 도입해 추론과 도구 사용을 하나의 흐름으로 통합했다.
- 외부 지식(위키 검색 등)과 결합해 질문응답(HotpotQA, FEVER)에서 환각과 오류 전파를 줄였다.
- ALFWorld, WebShop 같은 의사결정 태스크에서 모방학습/강화학습 대비 적은 예시만으로 큰 성능 향상을 보였다.

### 이 논문이 중요한 이유
오늘날 거의 모든 LLM 에이전트 프레임워크(LangChain, LangGraph, AutoGPT 등)의 기본 실행 루프가 ReAct에서 출발한다. "도구를 호출하고 결과를 관찰한 뒤 다음 행동을 추론한다"는 패턴은 에이전트의 사실상 표준이 되었으므로, AI 엔지니어라면 반드시 원리를 이해해야 한다.

### 사전 지식
- Chain-of-Thought(CoT) 프롬프팅과 few-shot in-context learning
- LLM의 환각(hallucination) 문제와 외부 도구 연동의 필요성
- 기본적인 에이전트/환경 상호작용(action, observation) 개념

### 관련 논문
- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models (Wei et al., 2022)](https://arxiv.org/abs/2201.11903)
- [Reflexion: Language Agents with Verbal Reinforcement Learning (Shinn et al., 2023)](https://arxiv.org/abs/2303.11366)

### 실무 적용
RAG 챗봇, 코드 에이전트, 고객지원 자동화에서 "검색 → 추론 → 행동"의 ReAct 루프가 그대로 쓰인다. LangChain의 ReAct Agent, OpenAI/Anthropic의 function calling 루프, MCP 기반 툴 호출 모두 이 구조를 따른다.

---

## Paper 2 (Classic): Gorilla: Large Language Model Connected with Massive APIs
- **Authors:** Shishir G. Patil, Tianjun Zhang, Xin Wang, Joseph E. Gonzalez
- **Year:** 2023
- **arXiv:** https://arxiv.org/abs/2305.15334
- **PDF:** [./gorilla-patil-2023.pdf](./gorilla-patil-2023.pdf)
- **Citation Count:** approx. 1,300+

### 요약
Gorilla는 LLaMA 기반으로 파인튜닝되어, 수천 개의 ML API(HuggingFace, TorchHub, TensorHub 등)에 대해 정확한 API 호출을 작성하는 모델이다. 문서 검색기(retriever)와 결합하면 테스트 시점에 바뀐 API 문서에도 적응하며, 인자(argument)를 잘못 생성하거나 존재하지 않는 API를 호출하는 환각을 크게 줄인다. API 호출 정확도에서 GPT-4를 능가했다.

### 핵심 기여
- API 문서를 self-instruct 방식으로 대량 생성해 "API 호출 작성" 능력을 학습시키는 데이터셋(APIBench)을 구축했다.
- 검색기 인지 학습(retriever-aware training)으로 문서가 바뀌어도 최신 API에 맞춰 호출을 생성하도록 했다.
- 호출의 정확성을 측정하기 위해 AST(추상 구문 트리) 기반 평가로 환각을 정량화하는 방법을 제시했다.

### 이 논문이 중요한 이유
도구 사용(tool use)의 핵심 난제는 "어떤 도구를, 어떤 인자로 호출할 것인가"이며, Gorilla는 이를 정면으로 다룬 대표적 연구다. 오늘날 function calling / tool calling의 신뢰성 문제와 직결되며, 에이전트가 수백~수천 개 도구를 다룰 때의 확장성 문제를 미리 보여준다.

### 사전 지식
- LLM의 function/tool calling과 JSON 스키마 기반 인자 생성
- 검색 증강(retrieval-augmented) 기법의 기본 개념
- 파인튜닝과 instruction tuning

### 관련 논문
- [Toolformer: Language Models Can Teach Themselves to Use Tools (Schick et al., 2023)](https://arxiv.org/abs/2302.04761)
- [ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs (Qin et al., 2023)](https://arxiv.org/abs/2307.16789)

### 실무 적용
플러그인/툴 레지스트리가 수백 개로 늘어나는 실제 에이전트 시스템에서, "도구 검색 후 정확한 호출 생성"은 Gorilla식 retriever + 호출 검증 패턴으로 구현된다. MCP 서버가 많아질수록 이 접근의 가치가 커진다.

---

## Paper 3 (Recent): τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains
- **Authors:** Shunyu Yao, Noah Shinn, Pedram Razavi, Karthik Narasimhan
- **Year:** 2024
- **arXiv:** https://arxiv.org/abs/2406.12045
- **PDF:** [./tau-bench-yao-2024.pdf](./tau-bench-yao-2024.pdf)
- **Citation Count:** approx. 250+

### 요약
τ-bench는 에이전트가 도메인 특화 API 도구와 정책 가이드라인을 갖춘 상태에서, LLM으로 시뮬레이션된 사용자와 동적으로 대화하며 작업을 수행하도록 만든 벤치마크다. 대화 종료 시점의 데이터베이스 상태를 정답 상태와 비교해 충실하게 평가하며, 여러 번 시도했을 때의 일관성을 재는 pass^k 지표를 제안한다. 최신 function calling 에이전트(gpt-4o)조차 과제 성공률이 50% 미만이고 일관성도 낮음을 보였다.

### 핵심 기여
- 사용자-에이전트-도구의 3자 상호작용을 실제 도메인(retail, airline)에서 시뮬레이션하는 평가 환경을 제시했다.
- 결과(데이터베이스 상태) 기반의 객관적 채점과, 신뢰성을 재는 새로운 pass^k 지표를 도입했다.
- 최신 에이전트의 낮은 성공률·낮은 일관성·정책 준수 실패를 드러내 도구 에이전트 평가의 빈틈을 보였다.

### 이 논문이 중요한 이유
대부분의 에이전트 벤치마크는 단발성 과제만 측정하지만, 실서비스에서는 "사용자와 여러 턴 대화하며 규칙을 지키고 일관되게 행동하는가"가 핵심이다. τ-bench는 바로 그 격차를 측정하므로, 에이전트를 프로덕션에 배포하려는 엔지니어에게 직접적인 평가 기준을 제공한다.

### 사전 지식
- LLM function calling과 멀티턴 대화 상태 관리
- 에이전트 평가 지표(success rate)와 신뢰성/일관성 개념
- ReAct 등 기본 에이전트 실행 루프

### 관련 논문
- [ReAct: Synergizing Reasoning and Acting in Language Models (Yao et al., 2022)](https://arxiv.org/abs/2210.03629)
- [AgentBench: Evaluating LLMs as Agents (Liu et al., 2023)](https://arxiv.org/abs/2308.03688)

### 실무 적용
고객지원·예약·결제처럼 정책 준수와 일관성이 중요한 에이전트 제품을 출시하기 전, τ-bench식 시뮬레이션 사용자 + 상태 기반 채점으로 회귀 테스트를 구성할 수 있다. pass^k는 SLA 신뢰성 지표 설계에 응용 가능하다.

---

## 추천 읽기 순서
1. **ReAct (2022)** — 에이전트의 기본 실행 루프(추론+행동)를 먼저 이해한다.
2. **Gorilla (2023)** — 도구를 "정확하게" 호출하는 문제와 확장성(수많은 API) 해법을 본다.
3. **τ-bench (2024)** — 만든 에이전트가 실서비스 수준인지 "어떻게 평가"할지 배운다.

## 핵심 테이크어웨이
- 에이전트의 본질은 "추론 → 도구 호출 → 관찰"의 반복 루프이며, ReAct가 그 표준을 세웠다.
- 도구 사용의 신뢰성은 "올바른 도구·올바른 인자" 생성에 달려 있고, 검색 결합과 호출 검증이 환각을 줄인다.
- 단발 성공률이 아니라 멀티턴·정책 준수·일관성(pass^k)으로 평가해야 프로덕션 준비 여부를 알 수 있다.

## 다음 토픽과의 연결
다음 토픽(Memory and Long-Context Management)은 에이전트가 긴 대화·다단계 작업에서 상태와 기억을 어떻게 유지하는지를 다룬다. τ-bench가 드러낸 "멀티턴 일관성" 문제는 곧 메모리 아키텍처의 필요성으로 이어진다.
