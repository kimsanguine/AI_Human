# Daily AI Paper Recommendations

> **Date:** 2026-09-06
> **Module:** Module 8: LangChain and LLM Orchestration
> **Topic:** AI Agents and Tool Use

---

## Paper 1 (Classic): WebGPT: Browser-assisted question-answering with human feedback
- **Authors:** Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu, Long Ouyang, Christina Kim, Christopher Hesse, Shantanu Jain, Vineet Kosaraju, William Saunders, et al. (OpenAI)
- **Year:** 2021
- **arXiv:** https://arxiv.org/abs/2112.09332
- **PDF:** [./webgpt-nakano-2021.pdf](./webgpt-nakano-2021.pdf)
- **Citation Count:** 약 1,800+

### 요약
WebGPT는 GPT-3를 텍스트 기반 웹 브라우저 환경에 연결해, 모델이 직접 검색·클릭·스크롤·인용 수집을 수행하며 장문 질문에 답하도록 만든 초기 대표 연구다. 사람의 시연(behavior cloning)으로 브라우징 행동을 학습시킨 뒤, 답변 선호 데이터로 보상 모델을 학습하고 rejection sampling과 RL로 최적화했다. 결과적으로 ELI5 질문에서 사람이 쓴 답변보다 선호되는 응답을 생성했고, 모든 주장에 출처를 인용하도록 만들어 사실성 검증을 가능하게 했다.

### 핵심 기여
- LLM에게 "행동 가능한 환경(text-based browser)"을 부여하고, 검색·인용을 하나의 액션 공간으로 정의한 최초 수준의 실증
- 시연 학습 → 보상 모델 → RL/rejection sampling으로 이어지는 에이전트 학습 파이프라인 제시 (이후 InstructGPT/RLHF 계보와 직결)
- 인용(citation) 기반 검증 가능성을 답변 품질의 1급 요건으로 승격 — 현대 RAG·Deep Research 제품의 원형
- 툴 사용 실패 모드(잘못된 출처 신뢰, 검색 쿼리 품질 의존성)를 정량적으로 드러냄

### 이 논문이 중요한 이유
오늘날 "AI 에이전트"라 부르는 모든 제품(Deep Research, Perplexity, Claude/ChatGPT의 웹 검색)의 설계 원형이 이 논문에 있다. 특히 AI 엔지니어 관점에서 중요한 것은 결과가 아니라 구조다: 액션 공간을 어떻게 정의할 것인가, 사람의 시연 데이터를 어떻게 확보할 것인가, 그리고 "정답"이 아니라 "선호"로 평가해야 하는 열린 태스크를 어떻게 최적화할 것인가. 이 세 질문은 2026년 현재 에이전트 개발에서도 그대로 반복된다.

### 사전 지식
- GPT-3 / few-shot prompting 기본 개념
- 행동 복제(behavior cloning), 보상 모델, PPO 등 RLHF 기초 용어
- rejection sampling(best-of-n) 추론 방식
- 정보 검색(IR)에서의 relevance와 근거(grounding) 개념

### 관련 논문
- [Training language models to follow instructions with human feedback (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155)
- [Toolformer: Language Models Can Teach Themselves to Use Tools (Schick et al., 2023)](https://arxiv.org/abs/2302.04761)
- [ReAct: Synergizing Reasoning and Acting in Language Models (Yao et al., 2022)](https://arxiv.org/abs/2210.03629)
- [Teaching language models to support answers with verified quotes / GopherCite (Menick et al., 2022)](https://arxiv.org/abs/2203.11147)

### 실무 적용
검색형 에이전트를 만들 때 이 논문의 3가지가 그대로 이식된다. (1) 액션 스키마 설계 — search / click / quote / scroll처럼 좁고 검증 가능한 툴셋이 넓은 툴셋보다 안정적이다. (2) 인용 강제 — 근거 스니펫을 응답과 함께 반환시키면 환각률과 CS 문의가 함께 떨어진다. (3) 선호 기반 평가 — 정답이 없는 장문 응답은 accuracy가 아니라 pairwise 선호나 LLM-as-judge로 측정해야 하며, 이 평가 파이프라인을 제품 출시 전에 먼저 만들어야 한다.

---

## Paper 2 (Classic): MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework
- **Authors:** Sirui Hong, Mingchen Zhuge, Jonathan Chen, Xiawu Zheng, Yuheng Cheng, Ceyao Zhang, Jinlin Wang, Zili Wang, Steven Ka Shing Yau, Zijuan Lin, Liyang Zhou, Chenyu Ran, Lingfeng Xiao, Chenglin Wu, Jürgen Schmidhuber
- **Year:** 2023 (ICLR 2024 Oral)
- **arXiv:** https://arxiv.org/abs/2308.00352
- **PDF:** [./metagpt-hong-2023.pdf](./metagpt-hong-2023.pdf)
- **Citation Count:** 약 1,000+

### 요약
MetaGPT는 인간 조직의 SOP(표준 운영 절차)를 프롬프트로 인코딩해 다중 에이전트 협업의 안정성을 끌어올린 프레임워크다. Product Manager, Architect, Engineer, QA 같은 역할을 부여하고, 각 역할이 자연어 대화가 아니라 PRD·설계 문서·인터페이스 명세 같은 **구조화된 산출물(structured artifact)** 을 주고받게 한다. 여기에 게시판(publish-subscribe) 방식의 공유 메시지 풀과 구독 메커니즘을 결합해, 대화가 길어질수록 정보가 왜곡·증폭되는 다중 에이전트 특유의 실패를 억제했다.

### 핵심 기여
- SOP를 에이전트 워크플로에 명시적으로 주입 — "역할 부여"를 넘어 "산출물 규격 부여"로 전환
- 자연어 대화 대신 문서 기반 통신을 채택해 halucination 전파와 컨텍스트 폭발을 완화
- 공유 메시지 풀 + 구독 기반 라우팅으로 N개 에이전트 간 O(N²) 통신 비용 문제를 구조적으로 해결
- 실행 피드백(코드 실행/테스트 결과)을 루프에 포함시켜 다중 에이전트 코드 생성 SOTA 달성

### 이 논문이 중요한 이유
멀티 에이전트를 처음 시도한 팀이 반드시 겪는 실패 — "에이전트끼리 서로 칭찬하다가 아무것도 못 만드는 현상" — 의 원인과 처방을 명확히 보여준다. 핵심 통찰은 에이전트를 더 똑똑하게 만드는 게 아니라 **에이전트 사이의 인터페이스를 좁히는 것**이 답이라는 점이다. 이는 마이크로서비스 설계나 조직 설계와 동일한 원리이며, PM/CPO 관점에서도 곧바로 재사용 가능한 사고 틀이다.

### 사전 지식
- 단일 에이전트 루프(ReAct, plan-and-execute)의 기본 구조
- 소프트웨어 개발 SOP(PRD, 설계 문서, 테스트 케이스) 흐름
- pub/sub 메시징과 컨텍스트 윈도우 관리의 기본 개념
- LangGraph 등 그래프형 오케스트레이션의 노드/엣지 개념(있으면 이해가 빠름)

### 관련 논문
- [AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation (Wu et al., 2023)](https://arxiv.org/abs/2308.08155)
- [CAMEL: Communicative Agents for "Mind" Exploration of Large Language Model Society (Li et al., 2023)](https://arxiv.org/abs/2303.17760)
- [ChatDev / Communicative Agents for Software Development (Qian et al., 2023)](https://arxiv.org/abs/2307.07924)
- [Reflexion: Language Agents with Verbal Reinforcement Learning (Shinn et al., 2023)](https://arxiv.org/abs/2303.11366)

### 실무 적용
멀티 에이전트를 도입할 때의 체크리스트로 그대로 쓸 수 있다. (1) 각 에이전트의 출력에 JSON 스키마나 문서 템플릿을 강제해 free-form 대화를 금지한다. (2) 에이전트 수를 늘리기 전에 산출물 검증 게이트(테스트 실행, 스키마 검증)를 먼저 만든다. (3) 전체 히스토리를 모두에게 broadcast하지 말고 구독 기반으로 필요한 정보만 전달해 토큰 비용을 선형에 가깝게 유지한다. 실무에서 멀티 에이전트가 단일 에이전트보다 나은 구간은 "역할별 산출물이 명확히 분리되는 태스크"뿐이라는 판단 기준도 여기서 나온다.

---

## Paper 3 (Recent): The Evolution of Tool Use in LLM Agents: From Single-Tool Call to Multi-Tool Orchestration
- **Authors:** Haoyuan Xu et al. (14 authors)
- **Year:** 2026
- **arXiv:** https://arxiv.org/abs/2603.22862
- **PDF:** [./tool-use-evolution-xu-2026.pdf](./tool-use-evolution-xu-2026.pdf)
- **Citation Count:** 신규 논문 (2026), 인용 축적 중

### 요약
툴 사용 연구의 중심 질문이 "단일 툴을 올바르게 호출할 수 있는가"에서 "긴 궤적(long-horizon) 위에서 여러 툴을 어떻게 조율하는가"로 이동했음을 정리한 서베이다. 저자들은 single-call tool use와 multi-tool orchestration의 문제 정의를 분리하고, 중간 상태·실행 피드백·변화하는 환경·안전성/비용/검증가능성 같은 실전 제약을 포함한 통합 프레임을 제시한다. 또한 MCP 등 표준 프로토콜 등장 이후의 툴 생태계, 오케스트레이션 학습 방법, 궤적 단위 평가 벤치마크를 함께 조망한다.

### 핵심 기여
- 단일 호출과 다중 툴 오케스트레이션의 태스크 정형화(task formulation)를 명시적으로 구분
- 툴 선택·계획·실행 피드백 반영·실패 복구를 하나의 파이프라인으로 묶는 분류 체계 제시
- 안전성, 비용, 검증가능성 같은 배포 제약을 연구 축으로 승격 — 벤치마크 점수 중심 논의의 한계 지적
- 궤적 기반 평가(trajectory-aware evaluation)와 대규모 툴 환경에서의 툴 검색/선택 문제를 정리

### 이 논문이 중요한 이유
2026년 현재 실무에서 에이전트가 실패하는 지점은 툴 호출 정확도가 아니라 **여러 툴을 순차적으로 엮을 때의 상태 관리와 복구**다. 이 서베이는 그 문제를 독립된 연구 대상으로 승격시키고, 흩어져 있던 개념(MCP, 툴 검색, 재계획, 궤적 평가)을 하나의 지도 위에 배치한다. 최신 논문 하나를 깊게 읽기보다, 이 지도를 먼저 확보한 뒤 필요한 가지로 내려가는 편이 학습 효율이 높다.

### 사전 지식
- ReAct / function calling / MCP의 기본 동작 방식
- 에이전트 벤치마크(τ-bench, OSWorld, SWE-bench 계열)의 평가 방식
- 계획(planning)과 재계획(replanning), 상태 저장 툴 호출의 개념
- RAG와 툴 검색(tool retrieval)의 차이

### 관련 논문
- [A Survey of AI Agent Protocols (Yang et al., 2025)](https://arxiv.org/abs/2504.16736)
- [A survey of agent interoperability protocols: MCP, ACP, A2A, ANP (Ehtesham et al., 2025)](https://arxiv.org/abs/2505.02279)
- [TRAJECT-Bench: A Trajectory-Aware Benchmark for Evaluating Agentic Tool Use (2025)](https://arxiv.org/abs/2510.04550)
- [Dynamic ReAct: Scalable Tool Selection for Large-Scale MCP Environments (2025)](https://arxiv.org/abs/2509.20386)

### 실무 적용
툴이 20개를 넘어가는 순간 "모든 툴을 프롬프트에 넣는" 방식은 무너진다. 이 논문의 처방을 제품에 옮기면: (1) 툴 검색 레이어를 두어 태스크마다 후보 툴을 5~10개로 좁힌다. (2) 툴 실행 결과를 그대로 컨텍스트에 붙이지 말고 요약·정규화해 상태 객체로 관리한다. (3) 실패 시 전체 재시작이 아니라 특정 스텝만 재계획하는 복구 경로를 설계한다. (4) 평가를 최종 정답률이 아니라 궤적 단위(어느 스텝에서 어떤 이유로 실패했는지)로 계측해야 개선 우선순위가 나온다.

---

## 추천 읽기 순서
1. **WebGPT (2021)** — 툴 사용 에이전트의 원형. 액션 공간 설계와 인용 기반 검증이라는 두 축을 먼저 잡는다.
2. **MetaGPT (2023)** — 단일 에이전트에서 다중 에이전트로 확장할 때 생기는 통신·산출물 문제와 그 해법을 본다.
3. **Tool Use Evolution 서베이 (2026)** — 앞의 두 논문이 남긴 문제들이 2026년에 어떤 연구 축으로 정리됐는지 지도를 확보한다.

## 핵심 테이크어웨이
- 에이전트 성능은 모델 지능보다 **액션 공간과 인터페이스 설계**에 더 크게 좌우된다. 좁고 검증 가능한 툴셋이 넓고 모호한 툴셋을 이긴다.
- 다중 에이전트의 핵심은 대화가 아니라 **구조화된 산출물 교환**이다. 자유 대화는 컨텍스트를 오염시키고 비용만 늘린다.
- 2026년의 병목은 단일 툴 호출 정확도가 아니라 **긴 궤적에서의 상태 관리·실패 복구·툴 검색**이다.
- 평가 설계가 곧 제품 로드맵이다. 궤적 단위 계측이 없으면 어디를 고쳐야 할지 알 수 없다.

## 다음 토픽과의 연결
다음 주제인 **Memory and Long-Context Management**는 오늘 확인한 병목의 직접적 후속이다. 툴 호출이 길어질수록 컨텍스트가 폭발하고, 그 순간 필요한 것이 메모리 아키텍처(MemGPT의 계층적 메모리, Generative Agents의 reflection)다. 오늘 읽은 "산출물 기반 통신"과 "상태 객체 관리"는 내일의 메모리 설계 논의로 그대로 이어진다.
