# Daily AI Paper Recommendations

> **Date:** 2026-10-30
> **Module:** Module 8: LangChain and LLM Orchestration
> **Topic:** AI Agents and Tool Use

---

## Paper 1 (Classic): The Rise and Potential of Large Language Model Based Agents: A Survey
- **Authors:** Zhiheng Xi, Wenxiang Chen, Xin Guo, Wei He, Yiwen Ding, Boyang Hong, Ming Zhang, Junzhe Wang, Senjie Jin, Enyu Zhou, Rui Zheng, Xiaoran Fan, Xiao Wang, Limao Xiong, Yuhao Zhou, Weiran Wang, Changhao Jiang, Yicheng Zou, Xiangyang Liu, Zhangyue Yin, Shihan Dou, Rongxiang Weng, Wensen Cheng, Qi Zhang, Wenjuan Qin, Yongyan Zheng, Xipeng Qiu, Xuanjing Huang, Tao Gui
- **Year:** 2023
- **arXiv:** https://arxiv.org/abs/2309.07864
- **PDF:** [./rise-of-llm-based-agents-survey-xi-2023.pdf](./rise-of-llm-based-agents-survey-xi-2023.pdf)
- **Citation Count:** ~1,500+

### 요약
푸단대 NLP 그룹이 정리한 LLM 에이전트 서베이로, 에이전트를 **Brain(두뇌) – Perception(지각) – Action(행동)** 세 모듈로 나누는 개념 프레임워크를 제시합니다. 단일 에이전트, 다중 에이전트, 인간-에이전트 협업 시나리오를 폭넓게 다루고, 에이전트 사회(Agent Society)의 사회적 행동·성격까지 논의합니다.

### 핵심 기여
- Brain(지식·기억·추론·계획) / Perception(텍스트·비전·오디오 입력) / Action(텍스트 출력·도구 사용·체화 행동)이라는 일반화된 에이전트 아키텍처 제시
- 응용 시나리오를 Single-agent, Multi-agent(협력/경쟁), Human-Agent 협업으로 체계화
- 에이전트 사회 시뮬레이션, 신뢰성·안전성·평가 등 열린 문제를 정리한 연구 로드맵 제공

### 이 논문이 중요한 이유
"에이전트란 무엇인가?"에 대한 공통 어휘를 제공합니다. 팀에서 에이전트를 설계할 때 '이 기능은 Brain 문제인가, Action 문제인가?'처럼 문제를 분해하는 사고 틀이 되며, Wang et al.(2023)의 서베이(Profile–Memory–Planning–Action)와 함께 읽으면 에이전트 설계 공간 전체를 조망할 수 있습니다.

### 사전 지식
- LLM 기본 개념(프롬프트, In-Context Learning, Chain-of-Thought)
- ReAct·Toolformer 등 도구 사용 기초 논문에 대한 감각
- 강화학습의 Agent–Environment 개념(선택)

### 관련 논문
- [A Survey on Large Language Model based Autonomous Agents (Wang et al., 2023)](https://arxiv.org/abs/2308.11432)
- [Generative Agents: Interactive Simulacra of Human Behavior (Park et al., 2023)](https://arxiv.org/abs/2304.03442)
- [Cognitive Architectures for Language Agents / CoALA (Sumers et al., 2023)](https://arxiv.org/abs/2309.02427)

### 실무 적용
에이전트 제품의 PRD·아키텍처 문서 템플릿으로 활용하기 좋습니다. 예: AI 더빙/아바타 서비스에서 "Perception = 원본 영상·음성 분석, Brain = 번역·톤 계획, Action = TTS/립싱크 툴 호출"로 모듈을 나누면 각 모듈별 KPI와 실패 원인을 독립적으로 추적할 수 있습니다.

---

## Paper 2 (Classic): WebArena: A Realistic Web Environment for Building Autonomous Agents
- **Authors:** Shuyan Zhou, Frank F. Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Tianyue Ou, Yonatan Bisk, Daniel Fried, Uri Alon, Graham Neubig
- **Year:** 2023 (ICLR 2024)
- **arXiv:** https://arxiv.org/abs/2307.13854
- **PDF:** [./webarena-zhou-2023.pdf](./webarena-zhou-2023.pdf)
- **Citation Count:** ~800+

### 요약
이커머스, 커뮤니티 포럼(Reddit 클론), 협업 개발(GitLab), CMS 등 **실제로 동작하는 셀프호스팅 웹사이트**로 구성된 재현 가능한 에이전트 평가 환경입니다. 812개의 장기(long-horizon) 과제를 제공하며, 발표 당시 GPT-4 기반 에이전트의 성공률은 약 14%로 인간(약 78%)과 큰 격차를 보였습니다.

### 핵심 기여
- 실제 오픈소스 웹앱을 도커로 묶어 재현 가능하고 현실적인 웹 에이전트 환경 구축
- 결과 문자열이 아니라 **최종 상태(functional correctness)**를 검증하는 평가 방식 도입
- 지도·계산기·스크래치패드 같은 도구와 문서(매뉴얼)까지 포함해 실제 업무 맥락 재현

### 이 논문이 중요한 이유
"데모는 되는데 실서비스에선 안 된다"는 에이전트의 본질적 문제를 숫자로 보여준 기준점입니다. 이후 VisualWebArena, WorkArena, OSWorld 등 수많은 벤치마크의 원형이 되었고, 브라우저 에이전트(Computer Use, Operator 류) 성능 비교 시 지금도 기본으로 인용됩니다.

### 사전 지식
- ReAct 형태의 관찰–행동 루프
- 웹 DOM/접근성 트리(Accessibility Tree) 기본 개념
- 벤치마크 평가 지표(성공률, exact match vs. functional evaluation)

### 관련 논문
- [Mind2Web: Towards a Generalist Agent for the Web (Deng et al., 2023)](https://arxiv.org/abs/2306.06070)
- [VisualWebArena (Koh et al., 2024)](https://arxiv.org/abs/2401.13649)
- [OSWorld (Xie et al., 2024)](https://arxiv.org/abs/2404.07972)

### 실무 적용
사내 에이전트 QA 체계를 만들 때 "최종 상태 검증" 철학을 그대로 가져올 수 있습니다. 예: SaaS 백오피스 자동화 에이전트라면 응답 텍스트 대신 DB 레코드/설정 값이 기대대로 바뀌었는지를 체크하는 회귀 테스트 세트를 구성합니다. 스테이징 환경을 도커로 복제해 에이전트 릴리즈마다 성공률을 추적하면 그로스 실험처럼 데이터 기반으로 개선할 수 있습니다.

---

## Paper 3 (Recent): Executable Code Actions Elicit Better LLM Agents (CodeAct)
- **Authors:** Xingyao Wang, Yangyi Chen, Lifan Yuan, Yizhe Zhang, Yunzhu Li, Hao Peng, Heng Ji
- **Year:** 2024 (ICML 2024)
- **arXiv:** https://arxiv.org/abs/2402.01030
- **PDF:** [./codeact-executable-code-actions-wang-2024.pdf](./codeact-executable-code-actions-wang-2024.pdf)
- **Citation Count:** ~400+

### 요약
에이전트의 행동을 JSON/텍스트 형식의 툴 콜 대신 **실행 가능한 Python 코드**로 통일하자는 제안입니다. Python 인터프리터와 결합해 에이전트가 실행 결과(에러 포함)를 보고 다음 행동을 동적으로 수정하며, 17개 LLM 대상 실험(API-Bank 및 신규 벤치마크 M³ToolEval)에서 최대 약 20% 높은 성공률을 보였습니다.

### 핵심 기여
- 코드 = 통합 행동 공간: 반복문·조건문·변수로 여러 툴을 한 번에 조합(multi-tool composition)
- 실행 피드백(stdout, 에러 메시지)을 활용한 자기 디버깅 멀티턴 상호작용
- 7k 상호작용 데이터셋 CodeActInstruct 및 Llama2/Mistral 기반 CodeActAgent 공개

### 이 논문이 중요한 이유
오늘날 OpenHands(구 OpenDevin), smolagents의 CodeAgent, 코드 실행 기반 MCP 패턴 등 "코드로 행동하는 에이전트" 흐름의 이론적 근거입니다. JSON 툴 콜이 많아질수록 왕복 횟수와 토큰 비용이 폭증하는 문제에 대해, 코드 한 블록으로 여러 툴을 오케스트레이션하는 대안을 정량적으로 보여줍니다.

### 사전 지식
- Function Calling / JSON 툴 스키마 방식
- ReAct 루프와 멀티턴 에이전트 상호작용
- 샌드박스 코드 실행 환경(Docker, Jupyter 커널)의 기본 개념

### 관련 논문
- [Program-aided Language Models / PAL (Gao et al., 2022)](https://arxiv.org/abs/2211.10435)
- [OpenHands: An Open Platform for AI Software Developers as Generalist Agents (Wang et al., 2024)](https://arxiv.org/abs/2407.16741)
- [API-Bank: A Comprehensive Benchmark for Tool-Augmented LLMs (Li et al., 2023)](https://arxiv.org/abs/2304.08244)

### 실무 적용
데이터 분석·리포트 생성·배치 작업처럼 툴 호출이 많은 워크플로우에서 JSON 툴 콜 대신 "샌드박스에서 코드 실행" 툴 하나를 제공하면 지연 시간과 토큰 비용을 크게 줄일 수 있습니다. 단, 보안(샌드박스 격리, 네트워크 제한, 권한 최소화)과 관측성(실행 로그 저장)을 반드시 함께 설계해야 합니다.

---

## 추천 읽기 순서
1. **Xi et al. 서베이** — Brain/Perception/Action 프레임으로 큰 그림을 먼저 잡기 (Section 3 중심으로 훑기)
2. **CodeAct** — "Action을 어떤 형식으로 표현할 것인가"라는 구체적 설계 질문으로 내려가기
3. **WebArena** — 설계한 에이전트를 "어떻게 현실적으로 평가할 것인가"로 마무리

## 핵심 테이크어웨이
- **Q. 에이전트를 어떻게 분해해서 설계할까?** → Brain(추론·기억·계획) / Perception / Action으로 나누면 실패 원인을 모듈 단위로 진단할 수 있다.
- **Q. 툴 호출은 JSON이 최선일까?** → 아니다. 다중 툴 조합·제어 흐름이 필요하면 코드 행동(CodeAct)이 성공률과 효율 면에서 유리하다. 대신 샌드박스 보안 비용이 생긴다.
- **Q. 에이전트가 "잘 된다"는 걸 어떻게 증명할까?** → 텍스트 출력이 아니라 환경의 최종 상태를 검증하는 functional evaluation이 신뢰할 수 있는 지표다.
- **가설 거리:** 우리 서비스의 에이전트 실패 중 몇 %가 Brain(계획 오류) 문제이고 몇 %가 Action(툴 형식·호출 오류) 문제일까? 이 비율에 따라 투자 우선순위가 달라진다.

## 다음 토픽과의 연결
다음 토픽은 **Memory and Long-Context Management**입니다. WebArena의 장기 과제에서 에이전트가 실패하는 주요 원인 중 하나가 긴 상호작용 기록을 잃어버리는 것이며, Xi et al. 서베이의 Brain 모듈 중 "Memory"가 바로 다음 시간의 주제(MemGPT, Generative Agents의 메모리 스트림)로 이어집니다.
