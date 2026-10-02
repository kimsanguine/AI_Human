# Daily AI Paper Recommendations

> **Date:** 2026-10-03
> **Module:** Module 8: LangChain and LLM Orchestration
> **Topic:** AI Agents and Tool Use

> 이번 사이클에서는 기존에 다룬 Wang Survey·HuggingGPT·Reflexion·AutoGen·ReAct·Gorilla·ToolLLM·Voyager·WebGPT·MetaGPT와 겹치지 않도록, "에이전트끼리 어떻게 협업시키는가(CAMEL) → 에이전트 능력을 어떻게 측정하는가(AgentBench) → 실제 제품 수준의 에이전트 플랫폼은 어떻게 생겼는가(OpenHands)"라는 세 가지 질문 축으로 논문을 골랐습니다.

---

## Paper 1 (Classic): CAMEL: Communicative Agents for "Mind" Exploration of Large Language Model Society
- **Authors:** Guohao Li, Hasan Abed Al Kader Hammoud, Hani Itani, Dmitrii Khizbullin, Bernard Ghanem
- **Year:** 2023 (NeurIPS 2023)
- **arXiv:** https://arxiv.org/abs/2303.17760
- **PDF:** [./camel-li-2023.pdf](./camel-li-2023.pdf)
- **Citation Count:** ~1,000+

### 요약
두 개의 LLM 에이전트에게 각각 "AI 사용자(지시자)"와 "AI 어시스턴트(수행자)" 역할을 주고, 사람의 개입 없이 대화만으로 과제를 끝까지 수행하게 하는 **역할 놀이(Role-Playing) 프레임워크**를 제안한 논문입니다. 핵심은 "Inception Prompting"이라는 초기 시스템 프롬프트 설계로, 에이전트가 역할을 뒤바꾸거나, 같은 말을 반복하거나, 대화를 멋대로 끝내는 문제를 줄였습니다.

### 핵심 기여
- **Role-Playing 프레임워크:** Task Specifier → AI User ↔ AI Assistant 구조로, 사람은 처음 아이디어만 주고 나머지는 에이전트끼리 협업
- **Inception Prompting:** 역할 전환(role flipping), 지시 반복, 무한 루프, 가짜 응답 등 멀티에이전트 대화의 실패 패턴을 정리하고 프롬프트 수준의 해결책 제시
- **대규모 대화 데이터셋 공개:** AI Society, Code, Math, Science 등 에이전트 간 대화 데이터를 만들어 공개하고, 이를 파인튜닝 데이터로 활용 가능함을 보임

### 이 논문이 중요한 이유
AutoGen, CrewAI, LangGraph의 멀티에이전트 패턴은 결국 "누가 지시하고, 누가 수행하며, 언제 멈추는가"라는 질문에 답하는 구조입니다. CAMEL은 이 질문을 가장 단순한 2-에이전트 구조로 처음 체계화했고, 특히 **멀티에이전트가 실패하는 방식**을 구체적으로 분류했다는 점이 실무자에게 큰 가치가 있습니다. "에이전트를 여러 개 붙이면 더 똑똑해진다"는 직관이 왜 자주 깨지는지 이해하는 출발점입니다.

### 사전 지식
- 시스템 프롬프트와 역할(Role) 부여의 개념
- ReAct, Chain-of-Thought 등 기본 프롬프팅 기법
- 멀티턴 대화에서 컨텍스트가 쌓이는 방식

### 관련 논문
- [AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation (Wu et al., 2023)](https://arxiv.org/abs/2308.08155)
- [MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework (Hong et al., 2023)](https://arxiv.org/abs/2308.00352)
- [Communicative Agents for Software Development / ChatDev (Qian et al., 2023)](https://arxiv.org/abs/2307.07924)

### 실무 적용
- **Planner–Executor 분리:** 하나의 에이전트에 모든 걸 맡기기보다, 계획 에이전트와 실행 에이전트로 나누고 종료 조건(`<TASK_DONE>` 같은 토큰)을 명시하면 루프 폭주를 줄일 수 있습니다.
- **합성 데이터 생성:** 역할 놀이로 도메인 대화 데이터를 대량 생성해 소형 모델 파인튜닝에 활용 (예: 고객 상담봇 학습 데이터)
- **실패 모니터링 체크리스트:** 역할 전환·지시 반복·조기 종료를 운영 로그에서 탐지하는 지표로 활용

---

## Paper 2 (Classic): AgentBench: Evaluating LLMs as Agents
- **Authors:** Xiao Liu, Hao Yu, Hanchen Zhang, Yifan Xu, Xuanyu Lei, Hanyu Lai, Yu Gu, Hangliang Ding, Kaiwen Men, Kejuan Yang, Shudan Zhang, Xiang Deng, Aohan Zeng, Zhengxiao Du, Chenhui Zhang, Sheng Shen, Tianjun Zhang, Yu Su, Huan Sun, Minlie Huang, Yuxiao Dong, Jie Tang
- **Year:** 2023 (ICLR 2024)
- **arXiv:** https://arxiv.org/abs/2308.03688
- **PDF:** [./agentbench-liu-2023.pdf](./agentbench-liu-2023.pdf)
- **Citation Count:** ~900+

### 요약
LLM을 "대화 상대"가 아니라 "환경 안에서 행동하는 에이전트"로 평가하기 위한 최초의 대규모 종합 벤치마크입니다. 운영체제(OS), 데이터베이스, 지식 그래프, 디지털 카드 게임, 수평적 사고 퍼즐, 가정 환경(ALFWorld), 웹 쇼핑, 웹 브라우징 등 **8개 환경**에서 수십 개의 LLM을 멀티턴으로 평가했고, 상용 모델과 오픈소스 모델 사이의 큰 격차를 드러냈습니다.

### 핵심 기여
- **8개 이질적 환경을 하나의 평가 프레임워크로 통합:** Code 기반(OS, DB, KG), Game 기반, Web 기반 과제를 같은 인터페이스로 평가
- **실패 원인 분류:** Context Limit Exceeded, Invalid Format, Invalid Action, Task Limit Exceeded 등으로 에이전트 실패 이유를 정량화
- **핵심 인사이트:** 장기 추론·의사결정·지시 따르기 능력이 에이전트 성능의 병목이며, 코드 학습과 고품질 정렬(alignment) 데이터가 에이전트 능력에 도움이 된다는 분석 제시

### 이 논문이 중요한 이유
에이전트 제품을 만들 때 가장 어려운 질문은 "어떤 모델을 써야 하나?"와 "우리 에이전트가 정말 좋아졌나?"입니다. AgentBench는 **단일 응답 정확도가 아니라 멀티턴 행동의 성공률**로 모델을 비교해야 한다는 관점을 확립했습니다. 이후 τ-bench, OSWorld, MCP-Universe 같은 후속 벤치마크는 모두 이 문제의식 위에 서 있습니다.

### 사전 지식
- ReAct 스타일의 Thought–Action–Observation 루프
- MMLU, HumanEval 같은 정적 벤치마크와의 차이
- SQL, Bash 등 기본적인 도구 사용 환경에 대한 이해

### 관련 논문
- [τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains (Yao et al., 2024)](https://arxiv.org/abs/2406.12045)
- [WebArena: A Realistic Web Environment for Building Autonomous Agents (Zhou et al., 2023)](https://arxiv.org/abs/2307.13854)
- [ALFWorld: Aligning Text and Embodied Environments for Interactive Learning (Shridhar et al., 2020)](https://arxiv.org/abs/2010.03768)

### 실무 적용
- **사내 에이전트 평가 세트 설계:** 우리 제품의 핵심 시나리오를 "환경 + 목표 + 성공 조건"으로 정의해 미니 AgentBench를 만들면 모델 교체·프롬프트 변경 시 회귀 테스트가 가능합니다.
- **실패 유형 대시보드:** Invalid Format / Invalid Action / 턴 초과를 별도 지표로 추적하면 "모델 문제인지, 도구 스키마 문제인지, 컨텍스트 문제인지"를 빠르게 분리할 수 있습니다.
- **모델 라우팅 근거:** 과제 유형별 성공률을 측정해 비싼 모델이 꼭 필요한 구간만 라우팅

---

## Paper 3 (Recent): OpenHands: An Open Platform for AI Software Developers as Generalist Agents
- **Authors:** Xingyao Wang, Boxuan Li, Yufan Song, Frank F. Xu, Xiangru Tang, Mingchen Zhuge, Jiayi Pan, Yueqi Song, Bowen Li, Jaskirat Singh, Hoang H. Tran, Fuqiang Li, Ren Ma, Mingzhang Zheng, Bill Qian, Yanjun Shao, Niklas Muennighoff, Yizhe Zhang, Binyuan Hui, Junyang Lin, Robert Brennan, Hao Peng, Heng Ji, Graham Neubig
- **Year:** 2024 (ICLR 2025)
- **arXiv:** https://arxiv.org/abs/2407.16741
- **PDF:** [./openhands-wang-2024.pdf](./openhands-wang-2024.pdf)
- **Citation Count:** ~400+

### 요약
사람 개발자처럼 코드를 작성하고, 터미널 명령을 실행하고, 웹을 탐색하는 범용 AI 에이전트를 만들기 위한 **오픈소스 플랫폼**(구 OpenDevin)을 소개한 논문입니다. Docker 샌드박스 런타임, 이벤트 스트림 기반 아키텍처, 에이전트 간 위임(delegation), 그리고 SWE-bench·WebArena 등 15개 벤치마크 통합 평가를 제공합니다.

### 핵심 기여
- **이벤트 스트림 아키텍처:** 에이전트의 모든 Action과 Observation을 하나의 이벤트 로그로 관리해 재현·디버깅·사람 개입을 쉽게 함
- **CodeAct 기반 행동 공간:** JSON 함수 호출 대신 Python/Bash 코드를 "행동"으로 실행하게 하여, 복잡한 도구 조합을 하나의 행동으로 표현
- **안전한 샌드박스 + 멀티에이전트 위임:** Docker 런타임에서 격리 실행, `AgentDelegateAction`으로 전문 에이전트에게 하위 작업 위임
- **통합 평가 하네스:** SWE-bench, HumanEvalFix, WebArena, GAIA 등 15개 벤치마크를 동일한 플랫폼에서 평가

### 이 논문이 중요한 이유
2025–2026년 에이전트 제품(Claude Code, Codex, Devin, Cursor Agent 등)의 공통 구조인 **"샌드박스 + 코드 실행 + 이벤트 로그 + 사람 개입 지점"**을 오픈소스로 가장 투명하게 보여주는 레퍼런스입니다. CAMEL이 "대화로 협업"을, AgentBench가 "평가"를 다뤘다면, OpenHands는 이를 **실제로 돌아가는 제품 아키텍처**로 엮어냈습니다.

### 사전 지식
- ReAct 및 함수 호출(Function Calling)의 기본 구조
- Docker 컨테이너와 샌드박스 개념
- SWE-bench 등 소프트웨어 엔지니어링 에이전트 벤치마크에 대한 이해

### 관련 논문
- [Executable Code Actions Elicit Better LLM Agents / CodeAct (Wang et al., 2024)](https://arxiv.org/abs/2402.01030)
- [SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering (Yang et al., 2024)](https://arxiv.org/abs/2405.15793)
- [SWE-bench: Can Language Models Resolve Real-World GitHub Issues? (Jimenez et al., 2023)](https://arxiv.org/abs/2310.06770)

### 실무 적용
- **에이전트 런타임 설계 레퍼런스:** 자체 에이전트를 만들 때 이벤트 스트림(Action/Observation 로그)을 1급 객체로 두면 Human-in-the-loop 승인, 재실행, 감사(audit) 로그를 한 번에 해결할 수 있습니다.
- **Function Calling vs Code Action 선택:** 도구가 많고 조합이 복잡한 경우 코드 실행 방식이 턴 수와 비용을 줄여줍니다(단, 샌드박스 필수).
- **MCP와의 연결:** 외부 도구는 MCP 서버로, 실행은 샌드박스 런타임으로 분리하는 구조가 현재 에이전트 제품의 표준 패턴으로 자리 잡고 있습니다.

---

## 추천 읽기 순서
1. **CAMEL** → 에이전트 2개가 대화로 협업하는 가장 단순한 구조와, 그것이 실패하는 방식부터 이해합니다.
2. **AgentBench** → "에이전트가 잘한다"를 어떻게 측정하는지, 그리고 실패 유형을 어떻게 분류하는지 배웁니다.
3. **OpenHands** → 위 두 관점이 실제 제품급 플랫폼(런타임·이벤트 로그·위임·평가 하네스)으로 어떻게 구현되는지 확인합니다.

## 핵심 테이크어웨이
- **Q. 에이전트를 여러 개 붙이면 더 좋아지나?** → 자동으로 좋아지지 않습니다. CAMEL이 보여주듯 역할 전환·반복·조기 종료 같은 실패 모드를 프롬프트와 종료 조건으로 통제해야 합니다.
- **Q. 우리 에이전트가 좋아졌는지 어떻게 아나?** → 단일 응답 정확도가 아니라, 환경 안에서의 **멀티턴 과제 성공률 + 실패 유형 분포**로 봐야 합니다(AgentBench).
- **Q. 제품으로 만들려면 무엇이 필요한가?** → 모델보다 **런타임 설계**(샌드박스, 이벤트 로그, 사람 개입 지점, 평가 하네스)가 신뢰성을 좌우합니다(OpenHands).
- **가설 하나:** 에이전트 제품의 경쟁력은 "어떤 모델을 쓰느냐"에서 "실패를 얼마나 빨리 관측·재현·수정할 수 있느냐"로 옮겨가고 있습니다.

## 다음 토픽과의 연결
다음 토픽은 **Memory and Long-Context Management**입니다. 오늘 본 세 논문 모두 결국 "멀티턴이 길어질수록 컨텍스트가 터진다"는 문제에 부딪힙니다 — AgentBench의 Context Limit Exceeded 실패, CAMEL의 긴 대화 루프, OpenHands의 이벤트 스트림 누적이 그 예입니다. 다음 날에는 에이전트가 무엇을 기억하고 무엇을 잊어야 하는지, 즉 **메모리 계층과 컨텍스트 압축** 전략을 다룹니다.
