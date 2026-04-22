# Daily AI Paper Recommendations

> **Date:** 2026-04-23
> **Module:** Module 8 - LangChain and LLM Orchestration
> **Topic:** LLM Application Frameworks and Orchestration

---

## Paper 1 (Classic): Toolformer: Language Models Can Teach Themselves to Use Tools
- **Authors:** Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, Thomas Scialom
- **Year:** 2023
- **arXiv:** [https://arxiv.org/abs/2302.04761](https://arxiv.org/abs/2302.04761)
- **PDF:** [./toolformer-schick-2023.pdf](./toolformer-schick-2023.pdf)
- **Citation Count:** 약 2,000+회

### 요약
Toolformer는 LLM이 스스로 외부 도구(검색 엔진, 계산기, 번역기, 달력, Q&A 시스템 등)를 언제·어떻게 호출해야 하는지를 자기지도학습(self-supervised) 방식으로 학습할 수 있음을 보인 논문이다. 소수의 데모만으로 API 호출 위치를 샘플링하고, 그 호출이 모델의 perplexity를 실제로 낮추는지 필터링하여 학습 데이터를 자동 생성하는 접근을 제시했다.

### 핵심 기여
- LLM이 "도구 호출 토큰"을 일반 텍스트 토큰처럼 다루도록 하는 API-in-text 포맷을 정의하여, 도구 사용을 사후적으로 붙이는 것이 아닌 언어모델링 목표 자체에 통합함
- 인간 라벨링 없이 API 호출의 유용성을 측정하는 self-supervised filtering 기법 제안 (loss reduction 기준)
- 6.7B 파라미터 Toolformer가 훨씬 큰 GPT-3(175B)를 여러 다운스트림 태스크에서 능가하며 도구 사용이 성능의 핵심 레버리지임을 실증

### 이 논문이 중요한 이유
현대 LLM 에이전트와 Function Calling, Claude MCP, OpenAI Tools, LangChain Agent 등 모든 "도구를 쓰는 LLM"의 개념적 뿌리에 해당한다. LLM을 단순 생성기가 아니라 외부 시스템과 상호작용하는 reasoning engine으로 보는 패러다임 전환을 기술적으로 증명한 첫 번째 대표 논문이다. AI 엔지니어라면 지금 쓰는 모든 tool-use/function-calling API의 동작 원리를 이해하기 위해 반드시 한 번은 읽어야 한다.

### 사전 지식
- Transformer decoder 구조와 causal LM 학습 방식
- Self-supervised learning과 perplexity 개념
- Few-shot / in-context learning (GPT-3 스타일 프롬프팅)
- API/함수 호출의 기본 개념

### 관련 논문
- [ReAct: Synergizing Reasoning and Acting in Language Models (Yao et al., 2022)](https://arxiv.org/abs/2210.03629)
- [Language Models are Few-Shot Learners / GPT-3 (Brown et al., 2020)](https://arxiv.org/abs/2005.14165)
- [WebGPT: Browser-assisted question-answering (Nakano et al., 2021)](https://arxiv.org/abs/2112.09332)

### 실무 적용
- OpenAI Function Calling, Anthropic Tool Use, Google Gemini Function Calling API의 설계 철학이 Toolformer의 API-in-text 포맷과 직결됨
- LangChain/LangGraph의 ToolNode, LlamaIndex의 ReActAgent가 내부적으로 채택한 "언어모델이 도구를 호출 → 결과를 받아 다시 생성" 루프의 원형
- Agentic RAG에서 검색 툴 호출 시점과 파라미터를 LLM이 자체 결정하도록 하는 구조에 그대로 적용
- Claude MCP(Model Context Protocol)로 외부 서비스를 LLM에 연결할 때, 어떤 툴을 언제 호출할지를 LLM이 판단하는 기반 메커니즘

---

## Paper 2 (Classic): MRKL Systems: A modular, neuro-symbolic architecture that combines large language models, external knowledge sources and discrete reasoning
- **Authors:** Ehud Karpas, Omri Abend, Yonatan Belinkov, Barak Lenz, Opher Lieber, Nir Ratner, Yoav Shoham, Hofit Bata, Yoav Levine, Kevin Leyton-Brown, Dor Muhlgay, Noam Rozen, Erez Schwartz, Gal Shachaf, Shai Shalev-Shwartz, Amnon Shashua, Moshe Tenenholtz
- **Year:** 2022
- **arXiv:** [https://arxiv.org/abs/2205.00445](https://arxiv.org/abs/2205.00445)
- **PDF:** [./mrkl-systems-karpas-2022.pdf](./mrkl-systems-karpas-2022.pdf)
- **Citation Count:** 약 500+회

### 요약
MRKL(Modular Reasoning, Knowledge and Language, "미라클")은 LLM을 중심 라우터로 두고, 계산기·DB·검색엔진·도메인 특화 모듈 등 이산적(discrete)이고 심볼릭한 "전문가 모듈"들을 플러그인처럼 결합하는 신경-심볼릭 아키텍처를 제안한 AI21 Labs의 논문이다. LLM의 약점(산술, 최신 지식, 정확성)을 외부 모듈로 보완하고, LLM은 "어떤 모듈에 무엇을 위임할지"를 결정하는 오케스트레이터 역할을 맡는다.

### 핵심 기여
- LLM 단일 모델에 모든 능력을 밀어넣는 대신, "LLM + 특화 모듈" 조합(compound system)으로 설계하는 아키텍처 원칙 정립
- Router(LLM)가 입력을 파싱하고 적절한 Expert 모듈로 분기시키는 패턴 — 오늘날 multi-tool agent 설계의 청사진
- 수치 계산, 환율 변환 등 LLM이 체계적으로 약한 영역을 외부 모듈로 위임하면 정확도가 극적으로 개선됨을 실증

### 이 논문이 중요한 이유
"LLM 하나로 모든 걸 해결한다"는 접근의 한계를 최초로 체계화하고, 오늘의 "Compound AI Systems" 패러다임을 선언한 논문이다. LangChain, LangGraph, LlamaIndex, Semantic Kernel, Haystack 등 모든 LLM 오케스트레이션 프레임워크의 철학적·구조적 기반이다. AI 제품을 LLM 단독이 아닌 "시스템"으로 설계해야 하는 이유를 이해하려면 필수.

### 사전 지식
- 대형 언어모델의 기본 능력과 한계 (환각, 산술 약점, knowledge cutoff)
- 신경망(neural)과 심볼릭(symbolic) AI의 차이
- Router/dispatcher 패턴, modular architecture 개념

### 관련 논문
- [Toolformer: Language Models Can Teach Themselves to Use Tools (Schick et al., 2023)](https://arxiv.org/abs/2302.04761)
- [Retrieval-Augmented Generation (Lewis et al., 2020)](https://arxiv.org/abs/2005.11401)
- [The Shift from Models to Compound AI Systems (Zaharia et al., 2024, BAIR Blog)](https://bair.berkeley.edu/blog/2024/02/18/compound-ai-systems/)

### 실무 적용
- 엔터프라이즈 AI 제품 설계 시, "단일 거대 모델"이 아닌 "작은 LLM + 다수의 툴/서비스" 구조로 가는 현재 트렌드의 이론적 근거
- Agentic AI 제품에서 Router Agent → 특화 Sub-agent(검색, 코드 실행, DB 쿼리, 결제 API) 구조의 프로토타입
- AI21의 Jurassic 모델 제품군, Salesforce Agentforce, Microsoft Copilot Studio의 Skill/Plugin 구조가 모두 MRKL 패턴의 변형
- AI Dubbing/Avatar 제품에서도 "언어 이해 LLM + 음성 합성 모듈 + 얼굴 렌더링 모듈"의 분리된 구조가 정확히 이 철학

---

## Paper 3 (Recent): LLM-based Optimization of Compound AI Systems: A Survey
- **Authors:** Matthieu Lin, Jenny Sheng, Andrew Zhao, Shenzhi Wang, Yang Yue, Victor Shea Jay Huang, Huan Liu, Jun Liu, Gao Huang, Yong-Jin Liu
- **Year:** 2024
- **arXiv:** [https://arxiv.org/abs/2410.16392](https://arxiv.org/abs/2410.16392)
- **PDF:** [./compound-ai-systems-survey-lin-2024.pdf](./compound-ai-systems-survey-lin-2024.pdf)
- **Citation Count:** 약 50+회 (신규 서베이)

### 요약
Compound AI System(LLM 호출, 검색기, 에이전트, 도구 사용이 결합된 다단계 시스템)의 "파라미터"를 어떻게 최적화할 것인가를 통합적으로 정리한 2024년 서베이다. 전통적 ML의 gradient-based 학습과 달리, 프롬프트·툴 선택·라우팅·에이전트 구성 등 비미분 가능한(non-differentiable) 요소를 LLM 자체를 optimizer로 써서 튜닝하는 DSPy, TextGrad, Trace 같은 최신 방법론을 체계화한다.

### 핵심 기여
- Compound AI System 최적화 문제를 "archetypes(원형)" 관점에서 분류: RAG, Multi-Agent, Self-Refine, Program-of-Thoughts 등
- LLM-as-Optimizer 패러다임 정리: DSPy의 MIPRO, TextGrad의 "textual gradient", OPRO 등을 통일된 프레임으로 비교
- End-to-end 자동 최적화를 위한 미래 연구 방향 제시 — 토큰 효율성, 신뢰성, 멀티모달 확장

### 이 논문이 중요한 이유
2024~2025년 LLM 애플리케이션 구축의 가장 큰 변화는 "프롬프트를 사람이 손으로 튜닝"에서 "시스템이 스스로 프롬프트와 구조를 최적화"로의 전환이다. 이 서베이는 그 흐름을 한 번에 따라잡게 해주며, DSPy나 TextGrad를 실무에 도입할지 결정할 때 반드시 참고해야 한다. Agentic AI 제품을 리딩하는 PM/CPO 관점에서는 향후 2~3년 아키텍처 선택의 지도(map)에 해당한다.

### 핵심 기여
- 비미분 가능한 AI 시스템을 LLM 자체로 최적화하는 방법론 총정리
- 프롬프트·모델 선택·라우팅을 단일 최적화 루프 안에 통합하는 관점 제시
- 실제 벤치마크 상에서 개별 기법(DSPy, TextGrad, Trace, ADAS)을 비교 평가

### 사전 지식
- Toolformer, ReAct, Chain-of-Thought 등 기본 prompting/tool-use 기법
- RAG, Multi-agent 시스템의 기본 구조
- 최적화(optimization) 기본 개념 — gradient descent, black-box optimization

### 관련 논문
- [DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines (Khattab et al., 2023)](https://arxiv.org/abs/2310.03714)
- [TextGrad: Automatic "Differentiation" via Text (Yuksekgonul et al., 2024)](https://arxiv.org/abs/2406.07496)
- [Large Language Models as Optimizers / OPRO (Yang et al., 2023)](https://arxiv.org/abs/2309.03409)
- [The Shift from Models to Compound AI Systems (Zaharia et al., 2024, BAIR Blog)](https://bair.berkeley.edu/blog/2024/02/18/compound-ai-systems/)

### 실무 적용
- DSPy/TextGrad를 도입하여 Agentic AI 제품의 프롬프트·체인 구조를 수동 튜닝에서 자동 최적화로 전환
- 멀티 에이전트 시스템에서 "어느 노드에 어떤 LLM(큰/작은, 고가/저가)을 쓸지" 자동 라우팅 설계에 참고
- LLM 비용 최적화: 파이프라인 단계별로 GPT-4급 vs Haiku급을 자동 선택하는 LLMSelector류 기법
- 프로덕션 Agent의 회귀 방지: 프롬프트 변경 시 metric 기반으로 자동 롤백/선택하는 CI/CD 파이프라인

---

## 추천 읽기 순서
1. **MRKL Systems (Karpas, 2022)** — 왜 "LLM 단독"이 아니라 "LLM + 모듈"이어야 하는지, 모든 오케스트레이션의 why부터 잡는다.
2. **Toolformer (Schick, 2023)** — 그렇다면 LLM에게 "어떻게" 도구를 쓰게 할 것인가. self-supervised로 학습하는 첫 성공 사례.
3. **LLM-based Optimization of Compound AI Systems (Lin, 2024)** — why와 how가 잡힌 상태에서, 이제 "전체 시스템을 어떻게 자동 최적화할 것인가"로 시야를 넓힌다.

이 순서는 "철학 → 단일 메커니즘 → 시스템 전체"의 계층 구조를 따른다.

## 핵심 테이크어웨이
- **LLM은 모델이 아니라 시스템의 한 구성요소다.** MRKL이 선언하고 Compound AI Systems 서베이가 재확인한 전제 — 제품을 만들 때 "모델을 고르는 일"보다 "시스템을 설계하는 일"이 더 중요해진다.
- **도구 사용은 후처리가 아니라 학습 목표다.** Toolformer는 tool-use를 LM loss에 통합함으로써 지금의 Function Calling API 생태계를 열었다. 실무에서는 "툴 설명서 품질"이 모델 성능을 크게 좌우한다.
- **프롬프트 엔지니어링의 종말과 프롬프트 컴파일러의 시작.** 2024~2025 서베이는 수동 프롬프트 튜닝에서 DSPy/TextGrad 같은 자동 최적화로의 이행을 기록한다. AI 제품팀의 프로세스도 여기에 맞게 바뀌어야 한다.
- **라우팅과 오케스트레이션이 차별화 포인트.** 모든 팀이 비슷한 파운데이션 모델을 쓰는 시대에, 어떤 컴포넌트를 어떻게 엮고 어떻게 최적화하는지가 제품 경쟁력의 핵심이 된다.

## 다음 토픽과의 연결
다음 주제인 **"AI Agents and Tool Use"(Day 22)**는 오늘의 Toolformer/MRKL이 깔아놓은 기반 위에서, LLM이 자율적으로 행동하는 에이전트 패러다임으로 확장된다. 특히 HuggingGPT와 Autonomous Agents Survey는 오늘 본 "compound system + tool use"를 "목표 주도적 자율 실행"으로 끌어올리는 논문들이다. 또한 Day 23의 **Memory and Long-Context Management**는 오늘 서베이가 지적한 "compound system의 상태/메모리 관리" 문제를 깊게 다룬다.
