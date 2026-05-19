# Daily AI Paper Recommendations

> **Date:** 2026-05-20
> **Module:** Module 8 — LangChain and LLM Orchestration
> **Topic:** LLM Application Frameworks and Orchestration

---

## Paper 1 (Classic): Gorilla: Large Language Model Connected with Massive APIs
- **Authors:** Shishir G. Patil, Tianjun Zhang, Xin Wang, Joseph E. Gonzalez
- **Year:** 2023
- **arXiv:** [https://arxiv.org/abs/2305.15334](https://arxiv.org/abs/2305.15334)
- **PDF:** [./gorilla-patil-2023.pdf](./gorilla-patil-2023.pdf)
- **Citation Count:** 약 1,000+ (Semantic Scholar 기준)

### 요약
Gorilla는 LLaMA를 파인튜닝하여 1,600개 이상의 API 호출을 정확하게 생성할 수 있도록 학습시킨 모델로, GPT-4보다 API 호출 정확도에서 우수한 성능을 보였다. 문서 리트리버와 결합 시 테스트 타임에 변경되는 API 명세에 적응 가능하며, LLM이 도구를 사용할 때 흔히 발생하는 할루시네이션 문제를 크게 완화한다.

### 핵심 기여
- **APIBench 벤치마크 공개**: HuggingFace, TorchHub, TensorHub의 1,600+ API를 포괄하는 대규모 도구 호출 평가 데이터셋 구축
- **Retriever-Aware Training**: 학습 시점에 문서 리트리버 결과를 함께 컨디셔닝하여, 런타임에 API 명세가 바뀌어도 적응 가능한 학습 패러다임 제시
- **Hallucination 완화**: 잘못된 API 호출(존재하지 않는 함수, 잘못된 인자) 비율을 GPT-4 대비 큰 폭으로 감소
- **AST Sub-tree Matching 평가 지표**: 함수 호출 정확도를 함수명·인자 단위로 정밀 측정하는 자동 평가 방법론

### 이 논문이 중요한 이유
AI Engineer가 다루는 거의 모든 Agentic 시스템은 결국 "LLM이 외부 도구·API를 정확히 호출할 수 있는가"라는 문제로 귀결된다. Gorilla는 Function Calling이 OpenAI의 공식 기능이 되기 이전부터 이 문제를 체계적으로 정의하고, 학습 데이터·평가 벤치마크·리트리버 통합 학습이라는 풀스택을 처음으로 제안했다. 오늘날의 Tool-use, MCP, Function Calling, Agent SDK가 모두 이 논문의 문제 정의 위에서 발전했다.

### 사전 지식
- LLaMA / 인스트럭션 튜닝 기본 개념
- Retrieval-Augmented Generation (RAG) 파이프라인
- API 스키마(JSON Schema, OpenAPI)와 함수 호출 형식
- AST(추상 구문 트리) 비교 기반 평가 방식

### 관련 논문
- [Toolformer: Language Models Can Teach Themselves to Use Tools (Schick et al., 2023)](https://arxiv.org/abs/2302.04761)
- [ReAct: Synergizing Reasoning and Acting in Language Models (Yao et al., 2022)](https://arxiv.org/abs/2210.03629)
- [ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs (Qin et al., 2023)](https://arxiv.org/abs/2307.16789)
- [Berkeley Function Calling Leaderboard (BFCL) — Gorilla 후속작](https://gorilla.cs.berkeley.edu/leaderboard.html)

### 실무 적용
B2B SaaS에서 "고객사 API에 자동 통합되는 Agent"를 만들 때 Gorilla의 retriever-aware 학습 방식이 직접적으로 적용된다. 즉, 새로운 고객사 API 명세가 추가될 때마다 재학습하는 대신, API 문서를 인덱싱해 두고 런타임에 검색해 호출하게 만들면 된다. 또한 Function Calling 정확도를 평가할 때 단순 string match가 아닌 AST 단위 비교를 도입하면, 모델 비교(GPT-4 vs Claude vs Qwen)와 회귀 테스트의 신뢰도가 크게 올라간다.

---


## Paper 2 (Classic): AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation
- **Authors:** Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Beibin Li, Erkang Zhu, Li Jiang, Xiaoyun Zhang, Shaokun Zhang, Jiale Liu, Ahmed Hassan Awadallah, Ryen W. White, Doug Burger, Chi Wang
- **Year:** 2023
- **arXiv:** [https://arxiv.org/abs/2308.08155](https://arxiv.org/abs/2308.08155)
- **PDF:** [./autogen-wu-2023.pdf](./autogen-wu-2023.pdf)
- **Citation Count:** 약 1,500+ (Semantic Scholar 기준)

### 요약
AutoGen은 멀티 에이전트가 자연어로 대화하며 협업해 작업을 완수하도록 만드는 오픈소스 프레임워크다. 각 에이전트는 LLM·도구·휴먼 인풋의 임의 조합으로 구성 가능하고, 개발자가 대화 패턴(group chat, hierarchical, sequential 등)을 선언적으로 정의해 복잡한 워크플로우를 구현한다. 수학·코드·QA·운영 의사결정 등 6개 도메인에서 단일 LLM 호출 대비 우수한 성능을 보였다.

### 핵심 기여
- **"Conversable Agent" 추상화**: 모든 에이전트가 동일한 인터페이스(메시지 송수신)를 가지며, LLM/도구/사람을 모두 동일한 방식으로 다룰 수 있는 통합 모델
- **Programmable Conversation Patterns**: 2-agent chat부터 그룹 챗·계층 구조까지 코드로 자유롭게 설계 가능한 대화 토폴로지
- **Code-as-Action**: 에이전트가 Python 코드를 생성→실행→결과 관찰의 루프를 자연스럽게 수행, 실질적인 도구 사용을 단순화
- **검증 가능한 사례 연구**: 수학 문제·코드 생성·온라인 의사결정 등 6개 응용에서 정량적 성능 향상 입증

### 이 논문이 중요한 이유
2026년 시점에서 LangGraph, CrewAI, OpenAI Agents SDK, Claude Code 등 거의 모든 Agent 프레임워크는 "에이전트는 메시지로 소통하는 자율 단위"라는 AutoGen의 추상화를 그대로 계승했다. 또한 멀티 에이전트 시스템 설계 시 "역할(Role) + 대화 토폴로지(Topology) + 종료 조건(Termination)"이라는 3축 분해는 사실상 산업 표준이 되었다. Agentic AI 제품을 리딩하는 입장에서, AutoGen 논문은 "왜 멀티 에이전트인가"에 대한 가장 영향력 있는 근거 자료다.

### 사전 지식
- LLM 함수 호출 / Tool-use 기본
- ReAct (Reasoning + Acting) 패러다임
- Async / event loop 기반 메시지 패싱 개념
- Python 환경에서의 코드 실행 샌드박싱(예: Docker, Pyodide)

### 관련 논문
- [ReAct: Synergizing Reasoning and Acting in Language Models (Yao et al., 2022)](https://arxiv.org/abs/2210.03629)
- [Generative Agents: Interactive Simulacra of Human Behavior (Park et al., 2023)](https://arxiv.org/abs/2304.03442)
- [MetaGPT: Meta Programming for Multi-Agent Collaborative Framework (Hong et al., 2023)](https://arxiv.org/abs/2308.00352)
- [CAMEL: Communicative Agents for Mind Exploration of Large Language Model Society (Li et al., 2023)](https://arxiv.org/abs/2303.17760)

### 실무 적용
AI Avatar/Dubbing 같은 멀티스텝 콘텐츠 생성 파이프라인에서 (a) 스크립트 작성 에이전트, (b) 보이스 캐스팅 에이전트, (c) 립싱크 검증 에이전트로 역할을 분리하면 단일 거대 프롬프트로 처리할 때보다 디버깅·관측 가능성·재시도 정책이 명확해진다. 또한 Customer Support Agent를 만들 때 "Triage Agent → Specialist Agent → QA Agent"의 hierarchical pattern은 AutoGen의 GroupChatManager 패턴을 직접 차용한 구조다. 핵심 운영 관점은 종료 조건(termination condition)과 LLM 호출 비용 상한선을 명시적으로 설계하는 것이다.

---


## Paper 3 (Recent): A Survey of Agent Interoperability Protocols — MCP, ACP, A2A, ANP
- **Authors:** Abul Ehtesham, Aditi Singh, Gaurang Bansal, Saket Kumar 외
- **Year:** 2025
- **arXiv:** [https://arxiv.org/abs/2505.02279](https://arxiv.org/abs/2505.02279)
- **PDF:** [./agent-interoperability-protocols-ehtesham-2025.pdf](./agent-interoperability-protocols-ehtesham-2025.pdf)
- **Citation Count:** 신규 논문 — 현재 빠르게 인용 증가 중

### 요약
2024-2025년에 LLM 에이전트 산업이 폭발적으로 성장하면서 등장한 4대 상호운용 프로토콜(MCP, ACP, A2A, ANP)을 비교 분석한 첫 체계적 서베이다. 각 프로토콜의 메시지 모델·인증·디스커버리·실행 의미론을 표준화된 프레임워크로 정리하고, "MCP → ACP → A2A → ANP" 순으로 채택을 확장하는 단계별 로드맵을 제시한다. 컴파운드 AI 시스템 설계자에게 어떤 프로토콜을 언제 채택할지의 의사결정 가이드라인을 제공한다.

### 핵심 기여
- **4대 프로토콜의 통합 분류 체계**: MCP(도구 접근, JSON-RPC), ACP(RESTful 멀티모달 메시지), A2A(에이전트 간 위임, Agent Card), ANP(분산 에이전트 마켓플레이스)를 동일한 축으로 비교
- **단계적 채택 로드맵**: 조직이 도구 통합 → 다중 에이전트 협업 → 분산 네트워크로 진화하는 현실적 경로 제시
- **보안·관측·디스커버리 설계 패턴**: 인증(OAuth, mTLS), Agent Card 기반 capability discovery, async 메시지 라우팅 등 운영 관점에서의 베스트 프랙티스 정리
- **공백 영역 식별**: 4개 프로토콜 간 브리징, 신원·신뢰 모델, governance 표준 부재 등 향후 연구 과제 명시

### 이 논문이 중요한 이유
2024년 말 Anthropic이 발표한 MCP가 사실상 표준으로 자리 잡으면서, 2026년 현재 모든 Agent 플랫폼은 MCP 호환성을 1차 요구사항으로 받아들이고 있다. 그러나 MCP만으로는 다중 에이전트·다중 조직 협업이 불가능하기에 A2A·ACP가 등장했다. 이 서베이는 CPO 관점에서 "우리 제품이 어떤 프로토콜 레이어에 위치해야 하는지, 어떤 호환성을 우선 지원해야 하는지"를 판단하는 가장 빠른 출발점이다.

### 사전 지식
- JSON-RPC, RESTful API, gRPC 등 RPC 패턴
- OAuth 2.0 / mTLS 인증 흐름
- LLM Function Calling / Tool-use 추상화 (Gorilla, AutoGen 등 선행 논문 참고)
- Service Discovery 패러다임 (DNS-SD, Kubernetes Service, Agent Card 등)

### 관련 논문
- [Model Context Protocol (MCP) Official Spec](https://modelcontextprotocol.io/)
- [A Survey on Large Language Model based Autonomous Agents (Wang et al., 2023)](https://arxiv.org/abs/2308.11432)
- [From Standalone LLMs to Integrated Intelligence: A Survey of Compound AI Systems (Lin et al., 2025)](https://arxiv.org/abs/2506.04565)
- [Model Context Protocol (MCP): Landscape, Security Threats, and Future Research Directions](https://arxiv.org/abs/2503.23278)

### 실무 적용
Agentic AI 제품 로드맵을 그릴 때 즉시 적용 가능한 의사결정:
- **단기(0-3개월)**: MCP 서버/클라이언트를 1순위로 구현 — 외부 SaaS·내부 DB·파일 시스템 접근의 표준화된 진입점 확보
- **중기(3-9개월)**: A2A를 도입해 자사 에이전트가 외부(파트너사) 에이전트와 capability 단위로 협업 가능하도록 설계 → B2B 통합 영업 기회 확대
- **장기**: ACP/ANP는 멀티모달 워크플로우와 분산 에이전트 마켓플레이스가 필요할 때 검토. 단, 표준 성숙도가 아직 낮으므로 PoC 단계로 한정
- **데이터 드리븐 관점**: 프로토콜 채택률(MCP 호출 / 전체 도구 호출), Agent Card 발견 성공률, 멀티 에이전트 협업 성공률을 핵심 지표로 트래킹

---

## 추천 읽기 순서

1. **Gorilla (2023)** — "LLM이 정확히 도구를 호출한다"는 가장 기본 문제부터 출발한다. Function Calling의 학습·평가 메서드론을 먼저 익혀야 멀티 에이전트나 프로토콜 설계의 의미가 분명해진다.
2. **AutoGen (2023)** — 단일 에이전트의 Tool-use에서 멀티 에이전트 협업으로 추상화 레벨을 올린다. 대화 토폴로지·코드 실행 루프·종료 조건 등 운영 설계의 핵심 어휘를 학습한다.
3. **Agent Interoperability Protocols Survey (2025)** — 마지막으로 산업 표준 레이어(MCP/A2A/ACP/ANP)에서 위 두 논문의 아이디어가 어떻게 표준화되고 있는지 조망한다. 자사 제품을 어느 레이어에서 차별화할지 판단하는 데 사용한다.

## 핵심 테이크어웨이

- **Tool-use는 모델 능력이 아니라 시스템 설계 문제다.** Gorilla가 보여준 것은 "더 큰 모델"이 아니라 "도구 명세를 학습/검색 가능하게 만드는 데이터·평가 파이프라인"이 본질이라는 점.
- **멀티 에이전트의 본질은 대화 토폴로지다.** AutoGen 이후 "어떤 에이전트가 누구에게, 언제, 어떤 형식으로 말하는가"가 LLM 호출 자체보다 시스템 품질을 좌우한다.
- **2026년의 차별화 포인트는 프로토콜 호환성과 관측 가능성이다.** MCP/A2A 호환은 이미 기본 요구사항이며, 자사 Agent의 신뢰도·재현성·디버깅 가능성이 진짜 경쟁 우위가 된다.
- **CPO 의사결정 프레임**: ① 도구 통합 표준(MCP) → ② 에이전트 간 협업(A2A) → ③ 사용자 워크플로우 자동화 → ④ 평가·관측 인프라. 이 순서로 투자해야 ROI가 명확해진다.

## 다음 토픽과의 연결

내일(Day 22 / 사이클 2)은 **"AI Agents and Tool Use"** 토픽으로 이어진다. 오늘 다룬 Gorilla(도구 호출)·AutoGen(멀티 에이전트)·Interoperability Protocols(표준 레이어)는 모두 "자율 에이전트"라는 상위 개념의 구성 요소다. 내일은 이 위에 **Planning, Reflection, Self-Improvement, Long-Horizon Task Execution** 같은 에이전트 행동 패턴을 본격적으로 다룬다. 특히 OpenAI Agents SDK·Claude MCP·LangGraph 최신 동향과의 매핑이 핵심 포인트가 될 것이다.
