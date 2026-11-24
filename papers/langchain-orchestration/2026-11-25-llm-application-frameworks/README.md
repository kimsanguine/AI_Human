# Daily AI Paper Recommendations

> **Date:** 2026-11-25
> **Module:** Module 8: LangChain and LLM Orchestration
> **Topic:** LLM Application Frameworks and Orchestration

> 이번 사이클에서는 이전 사이클에서 다룬 Toolformer, MRKL, Gorilla, AutoGen, DSPy, ReWOO, MetaGPT, PAL, LM Cascades, LLMCompiler, Chameleon, ART와 겹치지 않도록, "도구 호출의 기원(TALM)"과 "프롬프트를 프로그램으로 다루는 언어(LMQL)"라는 두 축의 고전을 골랐습니다. 최신 논문은 컴파운드 AI 시스템의 구성 자체를 자동 탐색하는 Archon입니다.

---

## Paper 1 (Classic): TALM: Tool Augmented Language Models
- **Authors:** Aaron Parisi, Yao Zhao, Noah Fiedel (Google)
- **Year:** 2022
- **arXiv:** https://arxiv.org/abs/2205.12255
- **PDF:** [./talm-tool-augmented-language-models-parisi-2022.pdf](./talm-tool-augmented-language-models-parisi-2022.pdf)
- **Citation Count:** ~300+ (approx., Google Scholar)

### 요약
TALM은 언어 모델이 텍스트만으로 "도구 입력 → 도구 결과 → 최종 답변" 형식의 시퀀스를 생성하도록 학습시켜, 외부 API(검색, 계산기 등)를 호출할 수 있게 만든 초기 연구입니다. 소수의 도구 사용 예시에서 시작해, 모델이 스스로 생성한 성공 사례를 다시 학습 데이터로 넣는 반복적 self-play 방식으로 도구 사용 능력을 키웁니다. Toolformer보다 앞서 "LLM + 도구"를 텍스트 인터페이스로 통일한 원형으로 평가받습니다.

### 핵심 기여
- 도구 호출을 `|tool-call ... |result ... |output` 같은 순수 텍스트 포맷으로 정의해, 별도 아키텍처 변경 없이 LM이 도구를 쓰게 함
- 적은 시드 데이터에서 출발해 성공한 도구 사용 궤적만 골라 재학습하는 iterative self-play 부트스트래핑 제안
- 지식 집약형 QA(Natural Questions)와 수학 추론(MathQA)에서, 도구를 쓴 작은 모델이 도구 없는 큰 모델을 능가할 수 있음을 보임

### 이 논문이 중요한 이유
오늘날 Function Calling, Tool Use, MCP는 모두 "모델이 구조화된 텍스트로 도구 호출을 출력하고, 오케스트레이터가 실행 결과를 다시 컨텍스트에 넣는다"는 루프 위에 서 있습니다. TALM은 이 루프를 가장 단순한 형태로 보여주기 때문에, LangChain의 Tool/Agent 추상화가 왜 그렇게 생겼는지 이해하는 출발점이 됩니다. 또한 "성공 궤적만 남겨 재학습"이라는 아이디어는 이후 Toolformer, ToolLLM, 에이전트 RL 데이터 파이프라인의 공통 패턴이 되었습니다.

### 사전 지식
- T5 계열 seq2seq 모델과 파인튜닝의 기본 개념
- 오픈 도메인 QA(검색 기반 질의응답) 문제 설정
- Self-training / bootstrapping(모델 출력으로 데이터 증강) 개념

### 관련 논문
- [Toolformer: Language Models Can Teach Themselves to Use Tools (Schick et al., 2023)](https://arxiv.org/abs/2302.04761)
- [MRKL Systems (Karpas et al., 2022)](https://arxiv.org/abs/2205.00445)
- [WebGPT: Browser-assisted question-answering with human feedback (Nakano et al., 2021)](https://arxiv.org/abs/2112.09332)
- [ReAct: Synergizing Reasoning and Acting in Language Models (Yao et al., 2022)](https://arxiv.org/abs/2210.03629)

### 실무 적용
사내 API(재고 조회, CRM 검색, 요금 계산 등)를 LLM에 붙일 때, "도구 호출 포맷을 명확히 정의 → 성공한 호출 로그만 모아 few-shot 또는 파인튜닝 데이터로 재사용"하는 TALM식 루프를 그대로 적용할 수 있습니다. 특히 작은 오픈소스 모델을 도메인 에이전트로 만들 때, 프로덕션 트레이스에서 정답 궤적만 필터링해 SFT하는 방식은 비용 대비 효과가 매우 큽니다.

---

## Paper 2 (Classic): Prompting Is Programming: A Query Language for Large Language Models (LMQL)
- **Authors:** Luca Beurer-Kellner, Marc Fischer, Martin Vechev (ETH Zurich)
- **Year:** 2022 (PLDI 2023)
- **arXiv:** https://arxiv.org/abs/2212.06094
- **PDF:** [./lmql-prompting-is-programming-beurer-kellner-2022.pdf](./lmql-prompting-is-programming-beurer-kellner-2022.pdf)
- **Citation Count:** ~250+ (approx., Google Scholar)

### 요약
LMQL은 프롬프트를 문자열이 아니라 "프로그램"으로 다루자는 관점에서, 텍스트 템플릿·제어 흐름(Python 문법)·출력 제약조건을 결합한 쿼리 언어를 제안합니다. 디코딩 단계에서 제약조건을 미리 평가해 불가능한 토큰을 마스킹(eager partial evaluation)함으로써, 형식이 보장된 출력을 얻으면서 API 호출 횟수와 비용을 크게 줄입니다. "Language Model Programming(LMP)"이라는 개념을 정식화한 대표 논문입니다.

### 핵심 기여
- 프롬프트 + 스크립팅 + 출력 제약(타입, 길이, 선택지, 정규식 등)을 하나로 묶은 LMQL 언어와 런타임 제안
- 제약조건을 토큰 단위로 미리 계산해 디코딩 마스크로 변환하는 부분 평가(partial evaluation) 기법
- 체인형 프롬프트(CoT, ReAct, 산술 등)에서 정확도를 유지하면서 모델 호출 수와 비용을 26~85% 절감

### 이 논문이 중요한 이유
LangChain이 "체인을 조립하는 라이브러리"라면, LMQL은 "LLM 호출 자체를 프로그래밍 언어의 1급 연산으로 만드는" 시도입니다. 오늘날 Structured Outputs, JSON mode, Guidance/Outlines 같은 constrained decoding, 그리고 SGLang의 프런트엔드 언어까지 모두 이 계보에 있습니다. AI 엔지니어가 "프롬프트 문자열 관리"에서 "LLM 프로그램 설계"로 사고를 전환하는 데 가장 좋은 교재입니다.

### 사전 지식
- 자기회귀 디코딩(greedy, beam search, sampling)과 토큰 마스킹 개념
- 프롬프트 체이닝(CoT, ReAct)의 기본 구조
- 간단한 프로그래밍 언어 개념(인터프리터, 제어 흐름, 타입 제약)

### 관련 논문
- [SGLang: Efficient Execution of Structured Language Model Programs (Zheng et al., 2023)](https://arxiv.org/abs/2312.07104)
- [Efficient Guided Generation for Large Language Models / Outlines (Willard & Louf, 2023)](https://arxiv.org/abs/2307.09702)
- [DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines (Khattab et al., 2023)](https://arxiv.org/abs/2310.03714)
- [Language Model Cascades (Dohan et al., 2022)](https://arxiv.org/abs/2207.10342)

### 실무 적용
에이전트가 도구 인자를 JSON으로 내보내거나, 분류 결과를 정해진 라벨 중 하나로만 출력해야 할 때, 사후 파싱·재시도 대신 디코딩 단계에서 형식을 강제하면 실패율과 토큰 비용이 동시에 줄어듭니다. 실무에서는 OpenAI Structured Outputs, vLLM/SGLang의 guided decoding, Pydantic 스키마 기반 출력으로 LMQL의 아이디어를 그대로 쓰고 있으며, 파이프라인 설계 시 "어디까지 프롬프트로, 어디서부터 제약으로 처리할지"를 결정하는 기준이 됩니다.

---

## Paper 3 (Recent): Archon: An Architecture Search Framework for Inference-Time Techniques
- **Authors:** Jon Saad-Falcon, Adrian Gamarra Lafuente, Shlok Natarajan, Nahum Maru, Hristo Todorov, Etash Guha, E. Kelly Buchanan, Mayee Chen, Neel Guha, Christopher Ré, Azalia Mirhoseini (Stanford)
- **Year:** 2024 (ICML 2025)
- **arXiv:** https://arxiv.org/abs/2409.15254
- **PDF:** [./archon-inference-time-architecture-search-saad-falcon-2024.pdf](./archon-inference-time-architecture-search-saad-falcon-2024.pdf)
- **Citation Count:** ~60+ (approx., Google Scholar)

### 요약
Archon은 여러 LLM과 추론 시점 기법(생성 앙상블, 반복 샘플링, 랭킹, 퓨전, 비평, 검증, 단위 테스트)을 레이어로 쌓아 "컴파운드 AI 시스템"을 만들고, 그 구성(어떤 모델을 몇 개, 어떤 순서로)을 자동으로 탐색하는 프레임워크입니다. 베이지안 최적화 기반의 ITAS(Inference-Time Architecture Search)로 주어진 추론 예산 안에서 최적 아키텍처를 찾으며, 오픈소스 모델만으로 GPT-4o, Claude 3.5 Sonnet 단일 호출 대비 여러 벤치마크(MT-Bench, AlpacaEval 2.0, Arena-Hard, MixEval, MATH, CodeContests)에서 평균 15% 이상 높은 성능을 보고합니다.

### 핵심 기여
- 추론 시점 기법들을 조합 가능한 "레이어"로 모듈화한 컴파운드 시스템 설계 공간 정의
- 각 기법의 효용 분석: 예) Ranker·Fuser는 모델 다양성이 클수록, Verifier·Unit Test는 추론/코드 과제에서 효과적
- 예산 제약 하에서 아키텍처를 자동 탐색하는 ITAS 알고리즘과 오픈소스 프레임워크 공개

### 이 논문이 중요한 이유
LangGraph나 LlamaIndex Workflows로 파이프라인을 짤 때 가장 어려운 질문은 "몇 번 샘플링하고, 어떤 모델로 검증하고, 어디서 합칠까?"입니다. Archon은 이 질문을 감(感)이 아니라 탐색·최적화 문제로 바꿨다는 점에서, 오케스트레이션 설계가 "프롬프트 튜닝"을 넘어 "시스템 아키텍처 최적화"로 이동하고 있음을 보여줍니다. Test-time compute 스케일링 흐름과 컴파운드 AI 시스템을 연결해 주는 핵심 최신 논문입니다.

### 사전 지식
- Best-of-N, Self-Consistency, LLM-as-a-Judge 등 추론 시점 기법
- Mixture-of-Agents 같은 다중 모델 앙상블 개념
- 베이지안 최적화(하이퍼파라미터 탐색)의 기본 아이디어

### 관련 논문
- [Mixture-of-Agents Enhances Large Language Model Capabilities (Wang et al., 2024)](https://arxiv.org/abs/2406.04692)
- [Large Language Monkeys: Scaling Inference Compute with Repeated Sampling (Brown et al., 2024)](https://arxiv.org/abs/2407.21787)
- [Are More LLM Calls All You Need? (Chen et al., 2024)](https://arxiv.org/abs/2403.02419)
- [Automated Design of Agentic Systems (Hu et al., 2024)](https://arxiv.org/abs/2408.08435)

### 실무 적용
고품질 답변이 중요한 B2B 기능(계약서 검토, 리포트 생성, 코드 생성)에서 단일 프런티어 모델 호출 대신 "여러 오픈소스 모델 생성 → 랭킹 → 퓨전 → 검증" 파이프라인을 구성하고, 비용·지연 예산 안에서 구성을 자동 탐색하는 방식으로 활용할 수 있습니다. 반대로 Archon의 분석은 "레이어를 늘릴수록 지연과 비용이 선형 이상으로 늘어난다"는 점도 보여주므로, 제품 PM 관점에서는 품질 향상폭과 응답 시간 SLA 사이의 트레이드오프를 정량화하는 실험 설계 템플릿으로도 쓸 수 있습니다.

---

## 추천 읽기 순서
1. **TALM (2022)** — 6페이지의 짧은 논문으로, "LLM이 텍스트로 도구를 부르고 결과를 다시 읽는" 오케스트레이션의 최소 단위를 먼저 이해합니다.
2. **LMQL (2022)** — 그 루프를 사람이 어떻게 "프로그래밍"하고 출력 형식을 제어할지, 즉 프레임워크 레이어의 설계 관점을 익힙니다.
3. **Archon (2024)** — 여러 모델·기법을 조합한 컴파운드 시스템을 어떻게 자동으로 설계·최적화하는지, 오케스트레이션의 최전선을 봅니다.

## 핵심 테이크어웨이
- **오케스트레이션의 본질은 "텍스트 인터페이스 위의 루프"**: TALM이 보여준 도구 호출 루프는 지금의 Function Calling·MCP까지 그대로 이어집니다.
- **프롬프트는 문자열이 아니라 프로그램**: LMQL은 제어 흐름과 출력 제약을 코드로 다룰 때 안정성과 비용이 함께 개선된다는 것을 증명했습니다.
- **구성 자체가 최적화 대상**: Archon은 "어떤 모델을 어떻게 엮을지"를 탐색 문제로 바꿨고, 이는 LangGraph 같은 그래프 프레임워크를 설계할 때도 실험·평가 기반으로 접근해야 함을 시사합니다.
- **생각해볼 질문**: 우리 제품의 LLM 파이프라인에서 (1) 사후 파싱으로 처리하는 부분 중 제약 디코딩으로 옮길 수 있는 곳은? (2) 단일 대형 모델 호출을 작은 모델 앙상블+검증으로 대체했을 때 품질/비용/지연은 어떻게 바뀔까?

## 다음 토픽과의 연결
다음 토픽은 **AI Agents and Tool Use**입니다. 오늘 다룬 TALM의 도구 호출 루프와 LMQL의 제어 흐름, Archon의 다중 모델 조합은 모두 "에이전트"의 부품입니다. 내일은 이 부품들이 계획(planning), 반복 행동, 환경 피드백과 결합해 자율 에이전트로 확장되는 과정과, 이를 평가하는 에이전트 벤치마크를 살펴봅니다.
