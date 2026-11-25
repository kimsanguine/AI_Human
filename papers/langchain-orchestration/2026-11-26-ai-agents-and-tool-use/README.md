# Daily AI Paper Recommendations

> **Date:** 2026-11-26
> **Module:** Module 8: LangChain and LLM Orchestration
> **Topic:** AI Agents and Tool Use

---

## Paper 1 (Classic): Cognitive Architectures for Language Agents (CoALA)
- **Authors:** Theodore R. Sumers, Shunyu Yao, Karthik Narasimhan, Thomas L. Griffiths
- **Year:** 2023 (TMLR 2024)
- **arXiv:** https://arxiv.org/abs/2309.02427
- **PDF:** [./coala-cognitive-architectures-sumers-2023.pdf](./coala-cognitive-architectures-sumers-2023.pdf)
- **Citation Count:** ~600+

### 요약
인지과학의 고전적 "인지 아키텍처"(Soar, ACT-R) 개념을 LLM 에이전트에 접목해, 에이전트를 **메모리(작업/에피소드/의미/절차) · 행동 공간(내부/외부 행동) · 의사결정 루프**라는 세 축으로 구조화한 개념 프레임워크입니다. 이 틀로 ReAct, Reflexion, Voyager, Generative Agents 등 기존 에이전트들을 하나의 언어로 정리하고 비교합니다.

### 핵심 기여
- LLM 에이전트를 설계하기 위한 공통 어휘(모듈형 메모리, 구조화된 행동 공간, 계획-실행 사이클) 제시
- 행동을 "외부 행동(grounding: 도구·API·환경)"과 "내부 행동(retrieval, reasoning, learning)"으로 구분
- 기존 에이전트 논문들을 CoALA 프레임 위에 매핑하여 설계 공백(예: 학습 행동, 절차 메모리 갱신)을 식별

### 이 논문이 중요한 이유
LangGraph·OpenAI Agents SDK·Claude Agent SDK처럼 프레임워크가 쏟아지는 상황에서, "이 프레임워크가 결국 무엇을 추상화하는가?"를 판단할 기준점이 됩니다. 프레임워크가 바뀌어도 살아남는 **설계 사고의 뼈대**를 제공하므로 AI 엔지니어의 필독서입니다.

### 사전 지식
LLM 프롬프팅 기본, ReAct(추론+행동 루프), RAG의 개념, 단기/장기 메모리 구분에 대한 감각

### 관련 논문
- [ReAct: Synergizing Reasoning and Acting in Language Models (Yao et al., 2022)](https://arxiv.org/abs/2210.03629)
- [Generative Agents: Interactive Simulacra of Human Behavior (Park et al., 2023)](https://arxiv.org/abs/2304.03442)
- [Voyager: An Open-Ended Embodied Agent with LLMs (Wang et al., 2023)](https://arxiv.org/abs/2305.16291)

### 실무 적용
에이전트 PRD/아키텍처 문서를 쓸 때 "어떤 메모리를 두는가, 어떤 외부 도구를 허용하는가, 결정 루프는 몇 단계인가"를 CoALA 체크리스트로 점검할 수 있습니다. 예: 고객지원 에이전트 = 에피소드 메모리(티켓 이력) + 의미 메모리(KB) + 외부 행동(CRM API) + 계획/검증 루프.

---

## Paper 2 (Classic): API-Bank: A Comprehensive Benchmark for Tool-Augmented LLMs
- **Authors:** Minghao Li, Yingxiu Zhao, Bowen Yu, Feifan Song, Hangyu Li, Haiyang Yu, Zhoujun Li, Fei Huang, Yongbin Li
- **Year:** 2023 (EMNLP 2023)
- **arXiv:** https://arxiv.org/abs/2304.08244
- **PDF:** [./api-bank-tool-augmented-llms-li-2023.pdf](./api-bank-tool-augmented-llms-li-2023.pdf)
- **Citation Count:** ~400+

### 요약
도구 사용 LLM을 체계적으로 평가하기 위한 초기 대표 벤치마크입니다. 73개의 실행 가능한 API와 314개의 도구 사용 대화(753회 API 호출)로 평가 세트를 구성하고, 1,888개 대화의 학습 세트로 Alpaca 기반 Lynx 모델을 파인튜닝해 도구 사용 능력을 비교했습니다.

### 핵심 기여
- 도구 사용 능력을 **Call(호출) → Retrieve+Call(검색 후 호출) → Plan+Retrieve+Call(계획 포함)** 3단계로 분해
- 실제로 실행 가능한 API 환경 기반 평가로 "말로만 맞는 호출"이 아닌 실제 결과를 검증
- GPT-4는 계획 능력, GPT-3.5는 GPT-3 대비 도구 사용 개선을 보이며, 실패 원인(API 환각, 인자 오류 등)을 분석

### 이 논문이 중요한 이유
오늘날 Function Calling / MCP 도구 호출의 품질을 평가할 때 쓰는 "도구 선택 → 인자 채우기 → 다단계 계획"이라는 분해 방식의 원형입니다. 에이전트 평가(eval) 설계를 배우는 출발점입니다.

### 사전 지식
OpenAI Function Calling 또는 tool_use 개념, JSON 스키마, 지시 튜닝(instruction tuning) 기본

### 관련 논문
- [Toolformer: Language Models Can Teach Themselves to Use Tools (Schick et al., 2023)](https://arxiv.org/abs/2302.04761)
- [ToolLLM: Facilitating LLMs to Master 16000+ Real-world APIs (Qin et al., 2023)](https://arxiv.org/abs/2307.16789)
- [Gorilla: Large Language Model Connected with Massive APIs (Patil et al., 2023)](https://arxiv.org/abs/2305.15334)

### 실무 적용
자사 에이전트의 도구 호출 eval을 만들 때, 실패를 "잘못된 도구 선택 / 인자 오류 / 계획 실패 / 환각 API"로 라벨링하면 개선 우선순위가 명확해집니다. MCP 서버를 늘릴수록 Retrieve 단계(도구 검색)의 정확도가 병목이 된다는 점도 이 프레임으로 설명됩니다.

---

## Paper 3 (Recent): TheAgentCompany: Benchmarking LLM Agents on Consequential Real World Tasks
- **Authors:** Frank F. Xu, Yufan Song, Boxuan Li, Yuxuan Tang, Kritanjali Jain, Mengxue Bao, Zora Z. Wang, Xuhui Zhou, Zhitong Guo, Murong Cao, Mingyang Yang, Hao Yang Lu, Amaad Martin, Zhe Su, Leander Melroy Maben, Raj Mehta, Wayne Chi, Lawrence Jang, Yiqing Xie, Shuyan Zhou, Graham Neubig
- **Year:** 2024 (NeurIPS 2025 Datasets & Benchmarks)
- **arXiv:** https://arxiv.org/abs/2412.14161
- **PDF:** [./theagentcompany-xu-2024.pdf](./theagentcompany-xu-2024.pdf)
- **Citation Count:** ~150+

### 요약
가상의 소프트웨어 회사를 통째로 시뮬레이션(사내 위키, GitLab, 메신저, 프로젝트 관리 툴, 동료 NPC)하고, 에이전트가 개발·PM·재무·HR 등 실제 직무 175개 과제를 수행하게 하는 벤치마크입니다. 체크포인트 기반 부분 점수로 평가하며, 최고 모델(v3 기준 Gemini 2.5 Pro)도 완전 자율 완료율이 약 30% 수준에 그칩니다.

### 핵심 기여
- 웹 브라우징 + 코드 실행 + **동료와의 커뮤니케이션**을 동시에 요구하는 "디지털 직원" 시나리오 벤치마크
- 셀프 호스팅 가능한 오픈소스 환경과 체크포인트 기반 부분 점수 평가 체계
- 실패 분석: 사회적 상호작용(동료에게 묻기), 복잡한 웹 UI, 장기 과제에서 에이전트가 특히 약함을 정량화

### 이 논문이 중요한 이유
"에이전트가 업무를 얼마나 대체할 수 있나?"라는 비즈니스 질문에 데이터로 답하는 몇 안 되는 연구입니다. 데모 수준의 성공과 실제 업무 완결률 사이의 간극을 보여 주어, 에이전트 제품의 기대치와 휴먼-인-더-루프 설계를 현실적으로 잡게 해 줍니다.

### 사전 지식
OpenHands/SWE-agent 같은 코딩 에이전트 구조, 웹 에이전트 벤치마크(WebArena) 개념, 부분 점수 평가 방식

### 관련 논문
- [WebArena: A Realistic Web Environment for Building Autonomous Agents (Zhou et al., 2023)](https://arxiv.org/abs/2307.13854)
- [OpenHands: An Open Platform for AI Software Developers as Generalist Agents (Wang et al., 2024)](https://arxiv.org/abs/2407.16741)
- [τ-bench: A Benchmark for Tool-Agent-User Interaction (Yao et al., 2024)](https://arxiv.org/abs/2406.12045)

### 실무 적용
B2B 에이전트 제품의 KPI를 "완전 자동화율"이 아니라 **체크포인트 단위 진척도 + 사람 개입 지점**으로 설계하는 근거가 됩니다. 자사 도메인 버전의 미니 "AgentCompany"(내부 툴 목업 + 과제 + 체크포인트)를 만들어 릴리스마다 회귀 평가하는 방식으로 응용할 수 있습니다.

---

## 추천 읽기 순서
1. **CoALA** — 에이전트를 바라보는 설계 언어부터 잡기 (메모리·행동·결정)
2. **API-Bank** — 그중 "외부 행동 = 도구 호출"을 어떻게 측정하는지 이해
3. **TheAgentCompany** — 도구 호출이 모여 실제 업무가 될 때 어디서 무너지는지 확인

## 핵심 테이크어웨이
- 에이전트 = LLM + 메모리 + 행동 공간 + 결정 루프. 프레임워크는 이 네 요소의 구현 차이일 뿐입니다.
- 도구 사용 능력은 "선택-인자-계획"으로 분해해 평가해야 개선 포인트가 보입니다.
- 실제 업무 환경에서 최고 에이전트도 완결률 ~30%: 제품은 부분 자동화 + 사람 검수 구조로 설계하는 것이 현실적입니다.
- 사회적 상호작용(질문하기, 확인 받기)이 에이전트의 다음 큰 약점이자 기회 영역입니다.

## 다음 토픽과의 연결
다음 토픽 **"Memory and Long-Context Management"**에서는 CoALA가 정의한 메모리 모듈(에피소드·의미·절차 메모리)을 MemGPT, Generative Agents 등이 실제로 어떻게 구현하는지 살펴봅니다. TheAgentCompany에서 드러난 장기 과제 실패의 상당 부분이 메모리/컨텍스트 관리 문제라는 점에서 자연스럽게 이어집니다.
