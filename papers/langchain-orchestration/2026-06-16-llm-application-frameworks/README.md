# Daily AI Paper Recommendations

> **Date:** 2026-06-16
> **Module:** Module 8: LangChain and LLM Orchestration
> **Topic:** LLM Application Frameworks and Orchestration

---

## Paper 1 (Classic): DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines
- **Authors:** Omar Khattab, Arnav Singhvi, Paridhi Maheshwari, Zhiyuan Zhang, Keshav Santhanam, Sri Vardhamanan, Saiful Haq, Ashutosh Sharma, Thomas T. Joshi, Hanna Moazam, Heather Miller, Matei Zaharia, Christopher Potts
- **Year:** 2023
- **arXiv:** https://arxiv.org/abs/2310.03714
- **PDF:** [./dspy-khattab-2023.pdf](./dspy-khattab-2023.pdf)
- **Citation Count:** ~900+ (꾸준히 증가 중)

### 요약
DSPy는 LLM 파이프라인을 손으로 깎은 프롬프트 문자열의 나열이 아니라, 선언적(declarative) 모듈로 구성된 "텍스트 변환 그래프"로 추상화한다. 각 모듈은 파라미터화되어 있어, 컴파일러가 주어진 지표(metric)를 최대화하도록 few-shot 예시 생성·프롬프트·파인튜닝 전략을 자동으로 최적화한다. 몇 줄의 DSPy 코드만으로 GPT-3.5와 llama2-13b가 수작업 프롬프트 대비 25~65% 이상 성능을 끌어올린다.

### 핵심 기여
- 프롬프트를 직접 작성하는 방식에서 벗어나, "시그니처(Signature) → 모듈(Module) → 옵티마이저(Optimizer/Teleprompter)"라는 프로그래밍 추상화를 제시
- 파이프라인을 자동으로 부트스트랩(self-bootstrap)하여 demonstration을 생성·선택하는 컴파일러 도입
- 멀티홉 검색, 수학 문제, 에이전트 루프 등 복잡한 LM 파이프라인을 간결한 코드로 표현·최적화 가능함을 실증

### 이 논문이 중요한 이유
AI 엔지니어가 마주하는 가장 큰 고통 중 하나는 "프롬프트 깎기(prompt tinkering)"의 비재현성과 유지보수 부담이다. DSPy는 프롬프트 엔지니어링을 코드 컴파일 문제로 재정의하여, 모델이 바뀌어도 파이프라인을 재컴파일하면 되는 구조를 제안한다. 오케스트레이션을 "수작업 마법"에서 "엔지니어링 규율"로 끌어올린 분기점 같은 논문이다.

### 사전 지식
- few-shot / in-context learning, Chain-of-Thought 프롬프팅의 기본 개념
- 검색 증강(RAG)과 멀티홉 QA의 기본 흐름
- 컴파일러·옵티마이저 비유(목표 지표를 향한 탐색)에 대한 직관

### 관련 논문
- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models (Wei et al., 2022)](https://arxiv.org/abs/2201.11903)
- [Demonstrate-Search-Predict: Composing retrieval and language models (Khattab et al., 2022)](https://arxiv.org/abs/2212.14024)

### 실무 적용
LangChain/LlamaIndex로 손수 프롬프트를 조립하던 RAG·에이전트 파이프라인을 DSPy로 옮기면, 평가 데이터셋과 지표만 정의해 두고 프롬프트/예시를 자동 최적화할 수 있다. 모델 교체(GPT → Llama → Qwen) 시마다 프롬프트를 다시 깎지 않고 재컴파일로 대응하는 MLOps 친화적 워크플로우를 구축할 수 있다.

---

## Paper 2 (Classic): ReWOO: Decoupling Reasoning from Observations for Efficient Augmented Language Models
- **Authors:** Binfeng Xu, Zhiyuan Peng, Bowen Lei, Subhabrata Mukherjee, Yuchen Liu, Dongkuan Xu
- **Year:** 2023
- **arXiv:** https://arxiv.org/abs/2305.18323
- **PDF:** [./rewoo-xu-2023.pdf](./rewoo-xu-2023.pdf)
- **Citation Count:** ~400+

### 요약
기존 ReAct류 도구 증강(ALM) 시스템은 "추론 → 도구 호출 → 관측 → 다시 추론"을 교차(interleaved)로 반복하기 때문에, 매 단계 이전 프롬프트 전체를 다시 넣어 토큰이 폭증한다. ReWOO(Reasoning WithOut Observation)는 추론 과정을 도구 관측으로부터 분리(decouple)하여, 먼저 전체 계획(Plan)을 한 번에 세우고 도구는 Worker가 실행한 뒤 Solver가 종합한다. HotpotQA에서 5배 토큰 효율과 4% 정확도 향상을 달성한다.

### 핵심 기여
- Planner / Worker / Solver로 모듈을 분리한 오케스트레이션 패러다임 제시
- 관측을 기다리지 않고 추론 계획을 선(先) 생성함으로써 반복 프롬프트로 인한 토큰 낭비 제거
- 도구 실패(tool-failure) 상황에서의 강건성 확보, 그리고 175B GPT-3.5의 추론 능력을 7B LLaMA로 오프로딩(instruction fine-tuning)할 수 있음을 실증

### 이 논문이 중요한 이유
에이전트의 비용과 지연시간(latency)은 대부분 "교차 실행으로 인한 프롬프트 반복"에서 발생한다. ReWOO는 추론과 관측을 분리한다는 단순하지만 강력한 아이디어로, 토큰·비용 효율을 구조적으로 개선한다. LangGraph 등 현대 워크플로우 엔진의 "계획 후 실행(plan-then-execute)" 패턴의 이론적 토대를 제공한다.

### 사전 지식
- ReAct(Reasoning + Acting) 패러다임과 그 토큰 비효율 문제
- 도구 호출(tool/function calling)과 멀티스텝 에이전트 루프의 기본 구조
- LLM 추론 비용이 입력 토큰 길이에 비례한다는 점

### 관련 논문
- [ReAct: Synergizing Reasoning and Acting in Language Models (Yao et al., 2022)](https://arxiv.org/abs/2210.03629)
- [Toolformer: Language Models Can Teach Themselves to Use Tools (Schick et al., 2023)](https://arxiv.org/abs/2302.04761)

### 실무 적용
지연·비용이 중요한 프로덕션 에이전트에서 "계획-실행 분리(plan-then-execute)" 구조를 채택할 때 직접적인 설계 근거가 된다. LangGraph의 Plan-and-Execute 에이전트, 또는 다단계 도구 호출이 잦은 RAG 에이전트에서 토큰 사용량을 수 배 줄이고, 작은 모델로 추론을 오프로딩해 운영 비용을 절감할 수 있다.

---

## Paper 3 (Recent): AFlow: Automating Agentic Workflow Generation
- **Authors:** Jiayi Zhang, Jinyu Xiang, Zhaoyang Yu, Fengwei Teng, Xionghui Chen, Jiaqi Chen, Mingchen Zhuge, Xin Cheng, Sirui Hong, Jinlin Wang, Bingnan Zheng, Bang Liu, Yuyu Luo, Chenglin Wu
- **Year:** 2024 (ICLR 2025 Oral)
- **arXiv:** https://arxiv.org/abs/2410.10762
- **PDF:** [./aflow-zhang-2024.pdf](./aflow-zhang-2024.pdf)
- **Citation Count:** ~200+ (빠르게 증가 중)

### 요약
에이전트 워크플로우는 강력하지만, 사람이 직접 설계해야 해서 확장성과 일반화가 떨어진다. AFlow는 워크플로우 최적화를 "코드로 표현된 워크플로우 공간에서의 탐색 문제"로 재정의하고, Monte Carlo Tree Search(MCTS)로 LLM 호출 노드를 엣지로 연결한 효과적인 워크플로우를 자동 탐색한다. 재사용 가능한 빌딩 블록인 Operator 개념으로 탐색 효율을 높인다.

### 핵심 기여
- 에이전트 워크플로우를 코드로 표현된 노드-엣지 그래프로 형식화하고, 자동 생성·최적화 문제로 재정의
- MCTS 기반 탐색 + Operator(재사용 가능한 연산 블록)로 사람 개입 없이 워크플로우를 자동 구성
- HumanEval, MBPP, GSM8K, MATH, HotpotQA, DROP 6개 벤치마크에서 수작업 워크플로우를 능가하며, 작은 모델로도 더 큰 모델 성능에 근접함을 실증

### 이 논문이 중요한 이유
DSPy가 "프롬프트/예시"를 자동 최적화했다면, AFlow는 한 단계 더 나아가 "워크플로우 구조 자체"를 자동 생성한다. 컴파운드 AI 시스템(compound AI system)의 설계를 사람의 직관에서 자동 탐색으로 옮기는 흐름의 최신 사례로, 오케스트레이션 자동화의 프론티어를 보여준다.

### 사전 지식
- DSPy/ReWOO 등 프롬프트·파이프라인 최적화의 기본 아이디어
- Monte Carlo Tree Search(MCTS)의 탐색-활용 트레이드오프
- 컴파운드 AI 시스템과 에이전트 워크플로우(노드/엣지) 개념

### 관련 논문
- [DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines (Khattab et al., 2023)](https://arxiv.org/abs/2310.03714)
- [GPTSwarm: Language Agents as Optimizable Graphs (Zhuge et al., 2024)](https://arxiv.org/abs/2402.16823)

### 실무 적용
새로운 태스크마다 에이전트 그래프를 수작업으로 설계하던 과정을, 평가 지표를 정의해 두면 AFlow가 자동으로 후보 워크플로우를 탐색·개선하도록 대체할 수 있다. 코드 생성·수학·QA 등 정량 평가가 가능한 도메인에서, 비싼 모델 대신 저렴한 모델 + 최적화된 워크플로우로 비용을 낮추는 전략에 직접 활용된다.

---

## 추천 읽기 순서
1. **ReWOO** — 먼저 "추론과 관측을 분리한다"는 오케스트레이션의 핵심 직관을 잡는다. ReAct 대비 무엇이 비효율이었는지 체감하기 좋다.
2. **DSPy** — 프롬프트 깎기를 컴파일 문제로 바꾸는 프로그래밍 추상화로 한 단계 추상화 레벨을 올린다.
3. **AFlow** — 프롬프트를 넘어 워크플로우 "구조"까지 자동화하는 최신 흐름으로 마무리한다.

## 핵심 테이크어웨이
- 오케스트레이션의 진화는 "프롬프트 수작업 → 프롬프트 자동 최적화(DSPy) → 워크플로우 구조 자동 생성(AFlow)"의 방향으로 진행되고 있다.
- 에이전트의 비용·지연 문제는 대부분 구조(교차 실행 vs. 계획-실행 분리)에서 비롯되며, ReWOO는 이를 설계로 해결한다.
- 평가 지표를 명확히 정의하는 것이 모든 자동 최적화(DSPy/AFlow)의 전제 조건이다 — "최적화할 대상을 측정할 수 없으면 자동화할 수 없다."

## 다음 토픽과의 연결
다음 토픽인 **AI Agents and Tool Use(Day 22)**에서는 오늘 배운 오케스트레이션 패턴(계획-실행 분리, 워크플로우 그래프) 위에서 실제 에이전트가 도구를 자율적으로 선택·호출하는 메커니즘을 다룬다. 오늘의 ReWOO·AFlow가 "어떻게 흐름을 구성하는가"였다면, 다음은 "에이전트가 무엇을 어떻게 호출하는가"로 이어진다.
