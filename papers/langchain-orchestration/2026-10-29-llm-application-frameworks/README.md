# Daily AI Paper Recommendations

> **Date:** 2026-10-29
> **Module:** Module 8: LangChain and LLM Orchestration
> **Topic:** LLM Application Frameworks and Orchestration

> 이번 사이클에서는 기존에 다룬 Toolformer·MRKL·DSPy·ReWOO·Gorilla·AutoGen·MetaGPT·PAL·Cascades·LLMCompiler와 겹치지 않도록, **"누가 워크플로를 설계하는가?"** 라는 질문 하나로 논문을 골랐습니다. LLM이 도구 조합을 계획하고(Chameleon) → 과거 예시로 프로그램을 만들고(ART) → 아예 에이전트 시스템 자체를 코드로 설계하는(ADAS) 흐름입니다.

---

## Paper 1 (Classic): Chameleon: Plug-and-Play Compositional Reasoning with Large Language Models
- **Authors:** Pan Lu, Baolin Peng, Hao Cheng, Michel Galley, Kai-Wei Chang, Ying Nian Wu, Song-Chun Zhu, Jianfeng Gao
- **Year:** 2023 (NeurIPS 2023)
- **arXiv:** https://arxiv.org/abs/2304.09842
- **PDF:** [./chameleon-lu-2023.pdf](./chameleon-lu-2023.pdf)
- **Citation Count:** ~400+

### 요약
GPT-4 같은 LLM을 **플래너(Planner)** 로 두고, 질문마다 필요한 도구(LLM, 비전 모델, 웹 검색, Python 실행기, 규칙 기반 모듈 등)를 골라 순서대로 엮은 "자연어 프로그램"을 생성해 실행하는 프레임워크입니다. 도구를 플러그인처럼 끼우고 빼는 구조로, ScienceQA(86.54%)와 TabMWP(98.78%)에서 당시 최고 수준의 few-shot 성능을 냈습니다.

### 핵심 기여
- **Planner–Executor 분리**: LLM은 "어떤 모듈을 어떤 순서로 쓸지"만 결정하고, 실제 처리는 각 모듈이 담당
- **Plug-and-Play 모듈 구조**: 새 도구를 모듈 목록과 설명만 추가하면 재학습 없이 사용 가능
- 생성된 도구 호출 순서(프로그램)를 분석해, 플래너가 **작업 유형별로 다른 도구 패턴**을 학습한다는 것을 보여줌

### 이 논문이 중요한 이유
LangGraph, LlamaIndex의 Router, OpenAI Agents SDK의 handoff 등 현재 프레임워크의 "라우터/플래너 노드"가 하는 일을 가장 깔끔하게 보여준 초기 논문입니다. "모델 하나가 다 한다"가 아니라 "모델은 지휘자, 도구는 연주자"라는 설계 감각을 잡는 데 좋습니다.

### 사전 지식
- 프롬프트 기반 few-shot 학습
- ReAct·MRKL 같은 도구 사용 LLM의 기본 개념
- 멀티모달 QA 벤치마크(ScienceQA) 정도의 배경

### 관련 논문
- [MRKL Systems (Karpas et al., 2022)](https://arxiv.org/abs/2205.00445)
- [Visual Programming: Compositional visual reasoning without training / VisProg (Gupta & Kembhavi, 2022)](https://arxiv.org/abs/2211.11559)
- [HuggingGPT (Shen et al., 2023)](https://arxiv.org/abs/2303.17580)

### 실무 적용
- 고객 문의 에이전트에서 "검색 → 표 계산 → 요약" 같은 도구 순서를 LLM 플래너가 생성하도록 하고, 실행은 결정적인 코드로 처리
- 생성된 도구 시퀀스를 로그로 남기면 **"어떤 질문 유형에서 어떤 경로가 실패하는가"** 를 데이터로 볼 수 있어, 프롬프트 개선 우선순위를 정하기 쉬움
- 가설 예시: *"플래너가 웹 검색을 건너뛰는 질문 유형에서 오답률이 가장 높을 것이다"* → 로그 분석으로 검증

---

## Paper 2 (Classic): ART: Automatic multi-step reasoning and tool-use for large language models
- **Authors:** Bhargavi Paranjape, Scott Lundberg, Sameer Singh, Hannaneh Hajishirzi, Luke Zettlemoyer, Marco Tulio Ribeiro
- **Year:** 2023
- **arXiv:** https://arxiv.org/abs/2303.09014
- **PDF:** [./art-automatic-reasoning-tool-use-paranjape-2023.pdf](./art-automatic-reasoning-tool-use-paranjape-2023.pdf)
- **Citation Count:** ~200+

### 요약
**작업 라이브러리(Task Library)** 에 저장된 "다단계 추론 + 도구 호출" 예시 프로그램을 새 작업과 비슷한 것으로 골라 프롬프트에 넣고, LLM이 새 작업용 프로그램을 생성하게 합니다. 생성 중 도구 호출 지점이 나오면 생성을 멈추고 실제 도구(검색, 코드 실행 등)를 실행한 뒤 결과를 붙여 이어서 생성합니다. BigBench·MMLU의 처음 보는 작업에서 일반 few-shot과 자동 CoT보다 크게 앞섰습니다.

### 핵심 기여
- **예시 검색 기반 프로그램 생성**: 사람이 작업마다 프롬프트를 쓰지 않아도, 비슷한 작업의 예시를 자동으로 가져와 활용
- **생성 일시정지 → 도구 실행 → 재개** 루프를 프레임워크 수준에서 구조화 (Guidance 라이브러리 기반)
- 사람이 라이브러리의 예시나 도구를 조금 고치는 것만으로 성능이 크게 오르는 **Human-in-the-loop 확장성**을 보여줌

### 이 논문이 중요한 이유
오늘날 "프롬프트/예시 저장소 + 동적 few-shot 선택 + 도구 호출" 구조(LangChain의 Example Selector, DSPy의 데모 최적화 등)의 원형입니다. 에이전트 품질이 결국 **"좋은 예시 라이브러리를 얼마나 잘 관리하는가"** 로 귀결된다는 실무 교훈을 줍니다.

### 사전 지식
- Chain-of-Thought와 few-shot 프롬프팅
- 임베딩 기반 유사도 검색(예시 검색)
- 도구 호출(Function Calling)의 기본 흐름

### 관련 논문
- [Toolformer (Schick et al., 2023)](https://arxiv.org/abs/2302.04761)
- [Automatic Chain of Thought Prompting / Auto-CoT (Zhang et al., 2022)](https://arxiv.org/abs/2210.03493)
- [PAL: Program-aided Language Models (Gao et al., 2022)](https://arxiv.org/abs/2211.10435)

### 실무 적용
- 성공한 에이전트 실행 기록을 "작업 라이브러리"로 쌓고, 새 요청이 오면 유사한 성공 사례를 동적으로 few-shot에 넣기
- 실패 사례를 사람이 고쳐 라이브러리에 다시 넣는 운영 루프 = 저비용 지속 개선 장치
- 가설 예시: *"동적 예시 선택을 도입하면 고정 예시 대비 신규 유형 요청의 성공률이 10%p 이상 오를 것이다"*

---

## Paper 3 (Recent): Automated Design of Agentic Systems (ADAS)
- **Authors:** Shengran Hu, Cong Lu, Jeff Clune
- **Year:** 2024 (ICLR 2025)
- **arXiv:** https://arxiv.org/abs/2408.08435
- **PDF:** [./automated-design-agentic-systems-hu-2024.pdf](./automated-design-agentic-systems-hu-2024.pdf)
- **Citation Count:** ~150+

### 요약
CoT, Self-Refine, 멀티 에이전트 토론 같은 에이전트 설계를 사람이 손으로 만드는 대신, **메타 에이전트(Meta Agent)가 에이전트를 Python 코드로 직접 설계**하게 하는 연구입니다. "Meta Agent Search" 알고리즘은 지금까지 발견한 에이전트들을 아카이브에 쌓아두고, 이를 참고해 더 나은 새 에이전트를 반복적으로 프로그래밍합니다. 발견된 에이전트는 사람이 설계한 에이전트보다 성능이 좋았고, 다른 도메인·다른 모델로 옮겨도 성능이 유지되었습니다.

### 핵심 기여
- **ADAS라는 새 연구 영역 정의**: 에이전트 시스템 설계 자체를 자동 탐색 문제로 정식화
- **코드 공간에서의 탐색**: 프롬프트만이 아니라 워크플로, 도구 사용, 제어 흐름까지 코드로 표현해 탐색 범위를 넓힘
- 수학·독해·과학 QA 등 여러 벤치마크에서 수작업 에이전트 대비 성능 향상과 **도메인/모델 간 전이성**을 확인

### 이 논문이 중요한 이유
Chameleon이 "LLM이 도구 순서를 계획"하고, ART가 "LLM이 예시로 프로그램을 생성"했다면, ADAS는 **"LLM이 오케스트레이션 프레임워크 위의 에이전트 구조 자체를 설계"** 하는 단계입니다. AFlow, MaAS 등 후속 연구와 함께, 앞으로 LangGraph 그래프를 사람이 그리는 일이 점점 자동화될 것임을 보여줍니다.

### 사전 지식
- Self-Refine, LLM Debate, Self-Consistency 같은 대표 에이전트 패턴
- 진화 탐색·Open-endedness 개념(아카이브 기반 탐색)
- 평가 데이터셋으로 에이전트를 자동 채점하는 방법

### 관련 논문
- [AFlow: Automating Agentic Workflow Generation (Zhang et al., 2024)](https://arxiv.org/abs/2410.10762)
- [Promptbreeder: Self-Referential Self-Improvement via Prompt Evolution (Fernando et al., 2023)](https://arxiv.org/abs/2309.16797)
- [Self-Refine: Iterative Refinement with Self-Feedback (Madaan et al., 2023)](https://arxiv.org/abs/2303.17651)

### 실무 적용
- 자체 평가셋(골든셋)이 있다면, 에이전트 워크플로 후보를 LLM이 생성 → 자동 평가 → 상위 후보를 아카이브에 저장하는 **오프라인 탐색 파이프라인**을 만들 수 있음
- 핵심 전제는 **신뢰할 수 있는 평가 함수**: 평가셋이 없으면 자동 설계는 과적합된 이상한 에이전트를 만들 위험이 큼
- 비용 관리: 탐색 비용(수백~수천 회 평가 호출)과 얻는 성능 향상을 비교해 ROI 판단 필요

---

## 추천 읽기 순서
1. **Chameleon** — "플래너가 도구를 조합한다"는 가장 직관적인 구조부터 시작합니다. 그림 1~2와 생성된 프로그램 예시 위주로 보세요.
2. **ART** — 같은 구조를 "예시 라이브러리 + 도구 호출 루프"로 일반화하는 방법을 봅니다. 사람 피드백 실험 부분이 실무에 특히 유용합니다.
3. **ADAS** — 마지막으로 "설계자까지 LLM으로 바꾸면?"이라는 질문으로 확장합니다. Meta Agent Search 프롬프트(부록)를 직접 읽어보길 추천합니다.

## 핵심 테이크어웨이
- **Q. 오케스트레이션에서 LLM의 역할은?** → 실행자가 아니라 **계획자/설계자**로 갈수록 시스템이 유연해집니다. 실행은 가능한 한 결정적인 코드에 맡기세요.
- **Q. 에이전트 품질은 무엇이 좌우하나?** → 모델보다 **예시 라이브러리와 평가셋의 품질**입니다. ART와 ADAS 모두 "좋은 기록과 좋은 평가"가 성능의 원천입니다.
- **Q. 워크플로 설계는 자동화될까?** → ADAS·AFlow가 그 방향을 보여줍니다. 단, 평가 함수가 없으면 자동 설계는 불가능하므로 **평가 인프라가 먼저**입니다.
- **Q. PM/엔지니어가 가져갈 질문은?** → "우리 워크플로의 계획 단계와 실행 단계는 분리되어 있는가? 성공 사례를 재사용하고 있는가? 자동으로 개선할 수 있는 평가셋이 있는가?"

## 다음 토픽과의 연결
다음 토픽은 **AI Agents and Tool Use**입니다. 오늘은 "도구와 LLM 호출을 어떤 구조로 엮을 것인가"라는 오케스트레이션 설계를 다뤘다면, 다음에는 그 구조 안에서 에이전트가 스스로 목표를 세우고 도구를 고르며 환경과 상호작용하는 **자율 에이전트**로 넘어갑니다. Chameleon의 플래너와 ADAS의 메타 에이전트가 에이전트 아키텍처에서 어떻게 반복되는지 연결해서 보세요.
