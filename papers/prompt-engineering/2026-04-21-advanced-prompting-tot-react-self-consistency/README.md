# Daily AI Paper Recommendations

> **Date:** 2026-04-21
> **Module:** Module 7: Prompt Engineering
> **Topic:** Advanced Prompting — ToT, ReAct, Self-Consistency

---

## Paper 1 (Classic): Tree of Thoughts: Deliberate Problem Solving with Large Language Models
- **Authors:** Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, Karthik Narasimhan
- **Year:** 2023
- **arXiv:** [https://arxiv.org/abs/2305.10601](https://arxiv.org/abs/2305.10601)
- **PDF:** [./tree-of-thoughts-yao-2023.pdf](./tree-of-thoughts-yao-2023.pdf)
- **Citation Count:** 약 3,500회 이상 (2026년 4월 기준)

### 요약
기존 Chain-of-Thought가 한 방향으로만 추론을 이어가는 한계를 넘어, LLM이 여러 "생각(thought)"을 트리 형태로 탐색하도록 하는 프롬프팅 프레임워크를 제안한다. 각 중간 상태를 self-evaluation으로 채점하고 BFS/DFS 같은 고전적인 탐색 알고리즘과 결합하여, 24 게임·창작 글쓰기·미니 크로스워드 같은 비선형 탐색 문제에서 성능을 극적으로 끌어올렸다(예: 24 게임에서 CoT 4% → ToT 74%).

### 핵심 기여
- LLM 추론을 "탐색 가능한 트리 구조"로 재정의한 최초의 일반화된 프레임워크 제시
- Thought generation + Thought evaluation + Search algorithm의 3단 구성으로 추론 과정을 분해
- Lookahead와 Backtracking이 필요한 문제군에서 CoT/Self-Consistency 대비 수십 배 성능 향상을 실증

### 이 논문이 중요한 이유
AI 엔지니어라면 단순 CoT 프롬프트로는 풀리지 않는 '진짜 문제'를 만나게 되는데, ToT는 "LLM을 어떻게 시스템 레벨에서 reasoning engine으로 쓸 것인가"에 대한 설계 원형(design primitive)을 제공한다. 오늘날 추론형 모델(o1, DeepSeek-R1)과 agent framework(LangGraph, AutoGen)의 사고 설계에 직접적 영향을 준 기초 논문이므로 반드시 읽어야 한다.

### 사전 지식
- Chain-of-Thought prompting과 Self-Consistency decoding의 개념
- 기본 탐색 알고리즘(BFS, DFS, Beam Search)
- LLM의 in-context learning과 few-shot prompting 기본 원리

### 관련 논문
- [Chain-of-Thought Prompting Elicits Reasoning (Wei et al., 2022)](https://arxiv.org/abs/2201.11903)
- [Self-Consistency Improves Chain of Thought Reasoning (Wang et al., 2022)](https://arxiv.org/abs/2203.11171)
- [Graph of Thoughts (Besta et al., 2024)](https://arxiv.org/abs/2308.09687)

### 실무 적용
복잡한 의사결정(예: 코드 디버깅 에이전트, 법률 문서 추론, 수학 문제 풀이) 파이프라인에서 "여러 후보 해를 생성 → self-rating → 가지치기 → 확장"이라는 ToT 루프는 Claude·GPT 기반 에이전트의 planner 모듈로 자주 활용된다. 특히 Agentic AI 제품에서 tool 호출 전 단계의 plan search, RAG의 query expansion, multi-step task decomposition에 그대로 적용 가능하다.

---

## Paper 2 (Classic): ReAct: Synergizing Reasoning and Acting in Language Models
- **Authors:** Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, Yuan Cao
- **Year:** 2022 (ICLR 2023)
- **arXiv:** [https://arxiv.org/abs/2210.03629](https://arxiv.org/abs/2210.03629)
- **PDF:** [./react-reasoning-acting-yao-2022.pdf](./react-reasoning-acting-yao-2022.pdf)
- **Citation Count:** 약 4,500회 이상 (2026년 4월 기준)

### 요약
LLM의 reasoning(생각)과 acting(도구 호출·환경 관찰)을 하나의 인터리빙 궤적으로 결합한 프롬프팅 패러다임을 제안한다. "Thought → Action → Observation"을 반복하여, 검색·계산·네비게이션 같은 외부 도구를 활용해 환각을 줄이고, HotpotQA·FEVER·ALFWorld·WebShop 등에서 순수 CoT나 순수 Act-only 방식을 모두 능가한다.

### 핵심 기여
- LLM이 "생각"과 "행동"을 번갈아 수행하는 표준 프롬프트 템플릿(ReAct trace)을 정립
- Knowledge-intensive QA에서 chain-of-thought의 hallucination을 외부 관찰로 교정 가능함을 입증
- 강화학습 없이도 few-shot 프롬프팅만으로 interactive decision-making에서 SOTA에 근접

### 이 논문이 중요한 이유
오늘날 모든 AI 에이전트 프레임워크(LangChain, LlamaIndex, OpenAI Assistants, Claude의 tool use 등)의 핵심 실행 루프가 ReAct 패턴을 그대로 또는 변형하여 사용한다. Agentic AI 제품을 설계하려면 이 논문의 Thought/Action/Observation 구조와 한계(recovery 부재, long-horizon 문제)를 이해하는 것이 필수적이다.

### 사전 지식
- Chain-of-Thought prompting의 장단점
- Tool use / function calling의 기초 개념
- LLM을 policy로 본 interactive decision-making(RL-lite) 관점

### 관련 논문
- [Chain-of-Thought Prompting (Wei et al., 2022)](https://arxiv.org/abs/2201.11903)
- [Toolformer: Language Models Can Teach Themselves to Use Tools (Schick et al., 2023)](https://arxiv.org/abs/2302.04761)
- [Reflexion: Language Agents with Verbal Reinforcement Learning (Shinn et al., 2023)](https://arxiv.org/abs/2303.11366)

### 실무 적용
ReAct는 "LLM + tools" 제품의 기본 설계 템플릿이다. 사내 지식검색 에이전트, 고객지원 에이전트, 코딩 에이전트, 브라우저 자동화(Operator/Agent) 등에서 tool routing과 recovery 로직을 구현할 때 Thought/Action/Observation trace를 로그·디버깅·트레이싱의 단위로 삼는 설계가 표준이 되었다. Self-consistency, Reflexion, Plan-and-Execute 등 후속 패턴도 모두 ReAct 위에 쌓인다.

---

## Paper 3 (Recent): Graph of Thoughts: Solving Elaborate Problems with Large Language Models
- **Authors:** Maciej Besta, Nils Blach, Ales Kubicek, Robert Gerstenberger, Michal Podstawski, Lukas Gianinazzi, Joanna Gajda, Tomasz Lehmann, Hubert Niewiadomski, Piotr Nyczyk, Torsten Hoefler
- **Year:** 2024 (AAAI 2024 / arXiv 2023-08 최초 공개, AAAI 공식 출판 2024)
- **arXiv:** [https://arxiv.org/abs/2308.09687](https://arxiv.org/abs/2308.09687)
- **DOI (AAAI):** [https://doi.org/10.1609/aaai.v38i16.29720](https://doi.org/10.1609/aaai.v38i16.29720)
- **PDF:** [./graph-of-thoughts-besta-2024.pdf](./graph-of-thoughts-besta-2024.pdf)
- **Citation Count:** 약 1,200회 이상 (2026년 4월 기준)

### 요약
ToT의 트리 구조를 임의의 유향 그래프(DAG)로 일반화하여, LLM의 "thought"들을 노드로 두고 aggregation(여러 생각 병합), refinement(자기 수정 루프), backtracking을 엣지로 모델링한다. Sorting, Set Intersection, Keyword Counting, Document Merging 같은 과제에서 ToT 대비 품질 +62%, 비용 -31%를 동시에 달성하여 "추론 그래프를 조작하는 프롬프팅"이라는 새로운 추상화를 제시한다.

### 핵심 기여
- Thought를 그래프의 vertex, 변환(generate/aggregate/refine)을 edge로 정의한 일반화된 framework
- Aggregation 연산을 통해 여러 후보 해의 정보를 합성하는 "thought composition" 개념 도입
- 품질과 비용(토큰 수)을 동시에 개선 가능함을 수학적·실험적으로 정량화하고 오픈소스 구현 공개

### 이 논문이 중요한 이유
CoT → Self-Consistency → ToT로 이어진 추론 구조의 진화에서 "그래프"가 자연스러운 상위 일반화임을 보여준 이정표 논문이다. 2024~2026년 등장한 reasoning model(o1, DeepSeek-R1), LangGraph 같은 graph-based agent orchestration 프레임워크의 이론적 근거를 제공하며, AI 엔지니어가 복잡한 agent 설계 시 "어떤 위상으로 생각을 구성할 것인가"를 결정하는 관점을 준다.

### 사전 지식
- Tree of Thoughts (Yao et al., 2023)의 thought generation/evaluation/search 구조
- Chain-of-Thought와 Self-Consistency
- DAG 기반 워크플로/DSPy, LangGraph 같은 graph-orchestration 개념에 대한 친숙도

### 관련 논문
- [Tree of Thoughts (Yao et al., 2023)](https://arxiv.org/abs/2305.10601)
- [Self-Consistency Improves CoT Reasoning (Wang et al., 2022)](https://arxiv.org/abs/2203.11171)
- [Skeleton-of-Thought (Ning et al., 2023)](https://arxiv.org/abs/2307.15337)
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL (DeepSeek-AI, 2025)](https://arxiv.org/abs/2501.12948)

### 실무 적용
LangGraph, CrewAI, AutoGen 같은 그래프 기반 에이전트 오케스트레이션 도구에서 사용하는 "노드=LLM 호출, 엣지=데이터 흐름" 설계의 개념적 기반이 된다. 실무에서는 (1) RAG에서 여러 소스의 부분 답변을 aggregation 노드로 합성하거나, (2) 코드 생성 에이전트에서 여러 후보를 생성 → self-critic refinement → merge 루프를 돌릴 때, (3) multi-agent 시스템의 토픽/서브태스크 분해와 결과 취합에 직접 쓰인다.

---

## 추천 읽기 순서
1. **ReAct (Yao et al., 2022)** — 가장 단순하고 오늘날 에이전트의 표준 루프이므로 먼저 읽어 "Thought/Action/Observation" 감각을 체화한다.
2. **Tree of Thoughts (Yao et al., 2023)** — 이어서 "생각을 탐색한다"는 관점을 확장. ReAct의 선형 trace를 탐색 트리로 재구성하는 과정에 집중.
3. **Graph of Thoughts (Besta et al., 2024)** — 마지막으로 트리를 그래프로 일반화하는 상위 추상을 학습. 이 순서로 읽으면 CoT → ReAct → ToT → GoT로 이어지는 추론 프롬프팅의 계보를 자연스럽게 익힐 수 있다.

## 핵심 테이크어웨이
- **추론은 구조(topology)의 문제다.** Linear CoT → Tree(ToT) → Graph(GoT)로의 확장은 곧 "문제에 맞는 탐색 구조 선택"이 핵심 설계 결정이라는 점을 시사한다.
- **Reasoning과 Acting은 분리할 수 없다.** ReAct가 보여주듯, tool 호출과 관찰이 결합되어야 환각을 줄이고 실제 제품 수준 신뢰도를 확보할 수 있다.
- **Self-evaluation은 1급 시민이다.** ToT·GoT는 모두 LLM이 자기 후보를 채점하는 과정을 명시적 구성요소로 둔다. 제품에서는 verifier/critic 모델을 별도 구성하는 패턴으로 이어진다.
- **비용-품질 트레이드오프를 공학으로 다뤄라.** GoT의 정량화는 "생각을 많이 펼칠수록 좋다"가 아니라 "어떤 위상에서 펼쳐야 가장 효율적인가"를 질문하게 만든다.

## 다음 토픽과의 연결
다음 Day(Module 7 Day 20: Automatic Prompt Optimization)에서는 APE(Zhou et al., 2022)와 Prompt Tuning(Lester et al., 2021), 그리고 DSPy/TextGrad 같은 현대적 자동 프롬프트 최적화 기법을 다룬다. 오늘 배운 ToT/ReAct/GoT의 "구조적 프롬프트"가 이제는 "사람이 일일이 작성"에서 "자동으로 탐색·최적화"되는 단계로 넘어가는 흐름을 이해하게 될 것이다. 또한 Module 8의 LLM Orchestration(LangGraph·compound AI system)과 Module 9의 Agentic RAG가 모두 오늘의 논문들을 실행 인프라로 전제하므로, 프롬프트 엔지니어링에서 시스템 엔지니어링으로 사고를 확장하는 전환점이 된다.
