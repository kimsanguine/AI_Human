# Daily AI Paper Recommendations

> **Date:** 2026-10-02
> **Module:** Module 8: LangChain and LLM Orchestration
> **Topic:** LLM Application Frameworks and Orchestration

> 이번 사이클에서는 기존에 다룬 Toolformer·MRKL·DSPy·ReWOO·Gorilla·AutoGen·MetaGPT·PAL과 겹치지 않도록, "체인을 어떻게 정의하고(Cascades) → 어떻게 빠르게 실행하며(LLMCompiler) → 몇 번 호출해야 최적인가(More LLM Calls)"라는 오케스트레이션의 세 가지 질문 축으로 논문을 골랐습니다.

---

## Paper 1 (Classic): Language Model Cascades
- **Authors:** David Dohan, Winnie Xu, Aitor Lewkowycz, Jacob Austin, David Bieber, Raphael Gontijo Lopes, Yuhuai Wu, Henryk Michalewski, Rif A. Saurous, Jascha Sohl-Dickstein, Kevin Murphy, Charles Sutton
- **Year:** 2022
- **arXiv:** https://arxiv.org/abs/2207.10342
- **PDF:** [./language-model-cascades-dohan-2022.pdf](./language-model-cascades-dohan-2022.pdf)
- **Citation Count:** ~200+

### 요약
LLM을 여러 번 호출하고 그 결과를 이어 붙이는 방식(Chain-of-Thought, Verifier, STaR, Selection-Inference, Tool use 등)을 "문자열 위의 확률적 프로그램(Probabilistic Program)"이라는 하나의 틀로 통합한 논문입니다. 저자들은 이런 다단계 LLM 호출 구조를 "Language Model Cascade"라고 부르며, 각 LLM 호출을 확률 변수로, 전체 파이프라인을 그래프 모델로 표현합니다.

### 핵심 기여
- CoT, Scratchpad, Verifier, Tool use 같은 개별 기법을 **하나의 확률적 프로그래밍 언어(PPL) 관점**으로 형식화
- LLM 호출을 "샘플링 가능한 함수"로 보고, 조건부 추론(conditioning)·사후 분포 추론으로 체인을 설계할 수 있음을 제시
- 이후 LangChain, DSPy 등 "LLM 호출을 조합하는 프로그래밍 모델"의 이론적 토대를 마련

### 이 논문이 중요한 이유
LangChain/LangGraph를 쓰다 보면 "체인", "노드", "엣지"라는 용어가 나오지만 그 본질이 무엇인지 설명하는 문서는 드뭅니다. 이 논문은 "LLM 파이프라인 = 확률 변수의 그래프"라는 관점을 처음 명확히 정리했습니다. 이 관점을 가지면 프레임워크가 바뀌어도 설계 원리를 그대로 가져갈 수 있습니다.

### 사전 지식
- Chain-of-Thought 프롬프팅의 기본 개념
- 확률 그래프 모델(베이지안 네트워크)과 조건부 확률의 기초
- 샘플링(temperature, self-consistency)에 대한 이해

### 관련 논문
- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models (Wei et al., 2022)](https://arxiv.org/abs/2201.11903)
- [STaR: Bootstrapping Reasoning With Reasoning (Zelikman et al., 2022)](https://arxiv.org/abs/2203.14465)
- [DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines (Khattab et al., 2023)](https://arxiv.org/abs/2310.03714)

### 실무 적용
- LangGraph에서 상태 그래프를 설계할 때, 각 노드를 "어떤 입력에 조건부인 확률적 출력"으로 정의하면 재시도·검증·분기 로직이 자연스럽게 정리됩니다.
- "생성 → 검증(Verifier) → 재생성" 루프는 이 논문에서 말하는 rejection sampling의 실무 버전입니다. 예: 에이전트가 만든 SQL을 실행 결과로 검증 후 재생성.
- 질문해 볼 것: *우리 파이프라인에서 어떤 노드가 "확률적"이고 어떤 노드가 "결정적"인가? 검증 노드는 어디에 둬야 가장 효과적인가?*

---

## Paper 2 (Classic): An LLM Compiler for Parallel Function Calling (LLMCompiler)
- **Authors:** Sehoon Kim, Suhong Moon, Ryan Tabrizi, Nicholas Lee, Michael W. Mahoney, Kurt Keutzer, Amir Gholami
- **Year:** 2023 (ICML 2024)
- **arXiv:** https://arxiv.org/abs/2312.04511
- **PDF:** [./llm-compiler-parallel-function-calling-kim-2023.pdf](./llm-compiler-parallel-function-calling-kim-2023.pdf)
- **Code:** https://github.com/SqueezeAILab/LLMCompiler
- **Citation Count:** ~250+

### 요약
ReAct처럼 "생각 → 도구 호출 → 관찰"을 한 번에 하나씩 순차 실행하면 느리고 비쌉니다. LLMCompiler는 컴파일러의 아이디어를 빌려, LLM이 먼저 도구 호출들의 **의존성 그래프(DAG)**를 계획하고, 의존성이 없는 호출들은 **병렬로 실행**하도록 합니다. 그 결과 ReAct 대비 최대 3.7배 지연 감소, 6.7배 비용 절감, 약 9% 정확도 향상을 보고했습니다.

### 핵심 기여
- **Function Calling Planner**: 사용자 질문을 도구 호출 태스크의 DAG로 분해
- **Task Fetching Unit**: 의존성이 해결된 태스크를 즉시 디스패치하고, 이전 결과로 변수를 치환
- **Executor**: 독립 태스크를 병렬 실행 / 필요 시 재계획(Replanning) 지원

### 이 논문이 중요한 이유
에이전트 제품에서 사용자가 가장 먼저 체감하는 문제는 "느리다"입니다. 이 논문은 에이전트 오케스트레이션을 **정확도 문제가 아니라 시스템/스케줄링 문제**로도 바라봐야 한다는 점을 보여줍니다. LangGraph 공식 예제에도 LLMCompiler 패턴이 포함될 정도로 실무 프레임워크에 큰 영향을 줬습니다.

### 사전 지식
- ReAct 패턴 (Thought-Action-Observation 루프)
- Function calling / Tool use API의 동작 방식
- DAG와 위상 정렬, 병렬 처리의 기초

### 관련 논문
- [ReAct: Synergizing Reasoning and Acting in Language Models (Yao et al., 2022)](https://arxiv.org/abs/2210.03629)
- [ReWOO: Decoupling Reasoning from Observations for Efficient Augmented Language Models (Xu et al., 2023)](https://arxiv.org/abs/2305.18323)
- [Plan-and-Solve Prompting (Wang et al., 2023)](https://arxiv.org/abs/2305.04091)

### 실무 적용
- "A 회사와 B 회사의 매출을 비교해줘" 같은 요청은 두 검색이 독립적이므로 병렬 실행 → 응답 시간 단축
- LangGraph의 `Send` API, 병렬 브랜치, OpenAI/Anthropic의 parallel tool calls와 직결되는 개념
- PM 관점 질문: *우리 에이전트의 p95 지연에서 도구 호출 대기 시간이 차지하는 비율은? 순차 실행 중인 호출 중 실제로 의존성이 있는 것은 몇 개인가?*

---

## Paper 3 (Recent): Are More LLM Calls All You Need? Towards Scaling Laws of Compound Inference Systems
- **Authors:** Lingjiao Chen, Jared Quincy Davis, Boris Hanin, Peter Bailis, Ion Stoica, Matei Zaharia, James Zou
- **Year:** 2024 (NeurIPS 2024)
- **arXiv:** https://arxiv.org/abs/2403.02419
- **PDF:** [./more-llm-calls-scaling-compound-inference-chen-2024.pdf](./more-llm-calls-scaling-compound-inference-chen-2024.pdf)
- **Citation Count:** ~100+

### 요약
여러 번 LLM을 호출한 뒤 다수결(Vote)이나 필터 후 다수결(Filter-Vote)로 답을 합치는 "Compound AI System"에서, 호출 횟수를 늘리면 성능이 계속 오를까요? 이 논문은 **성능이 처음엔 오르다가 오히려 떨어질 수 있다(비단조성)**는 것을 실험과 이론으로 보여줍니다. 원인은 쿼리 난이도의 다양성으로, 호출을 늘리면 쉬운 문제는 더 잘 맞히지만 어려운 문제는 오히려 더 틀리게 됩니다.

### 핵심 기여
- Vote / Filter-Vote 시스템에서 LLM 호출 수에 따른 **비단조적 성능 곡선**을 실증
- "쉬운 쿼리 vs 어려운 쿼리" 비율로 이를 설명하는 이론적 분석 제시
- 소수의 샘플로 **최적 호출 횟수를 예측하는 분석적 스케일링 모델** 제안

### 이 논문이 중요한 이유
Berkeley의 Compound AI Systems 흐름(Zaharia, Stoica 공저)에서 나온 논문으로, "호출을 많이 할수록 좋다"는 직관이 틀릴 수 있음을 정량적으로 보여줍니다. 오케스트레이션 설계는 결국 **비용·지연·품질의 트레이드오프**이며, 이 논문은 그 최적점을 찾는 방법론을 제공합니다. 최근 test-time compute 스케일링 논의의 출발점 중 하나이기도 합니다.

### 사전 지식
- Self-Consistency (다수결 기반 샘플링)
- Compound AI System 개념 (Berkeley BAIR 블로그, 2024)
- 기초 확률(이항 분포, 다수결 정확도)

### 관련 논문
- [Self-Consistency Improves Chain of Thought Reasoning in Language Models (Wang et al., 2022)](https://arxiv.org/abs/2203.11171)
- [Large Language Monkeys: Scaling Inference Compute with Repeated Sampling (Brown et al., 2024)](https://arxiv.org/abs/2407.21787)
- [Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters (Snell et al., 2024)](https://arxiv.org/abs/2408.03314)

### 실무 적용
- "N번 생성 후 다수결" 로직을 넣을 때, N을 감으로 정하지 말고 쿼리 난이도별로 소규모 실험 후 최적 N을 추정
- 난이도 라우팅: 쉬운 질문은 다중 호출, 어려운 질문은 더 강한 모델 1회 호출 또는 사람 검토로 분기
- 가설 예시: *"우리 고객지원 에이전트에서 5회 투표를 3회로 줄이면 비용은 40% 줄고 정확도 손실은 1%p 이내일 것이다"* → A/B 테스트로 검증

---

## 추천 읽기 순서
1. **Language Model Cascades** — 먼저 "LLM 파이프라인을 어떻게 추상화할 것인가"라는 개념 틀을 잡습니다. (수식은 훑어보고 그림과 예시 위주로)
2. **LLMCompiler** — 그 파이프라인을 "어떻게 빠르고 싸게 실행할 것인가"라는 시스템 관점으로 확장합니다. GitHub 코드와 함께 읽는 것을 추천합니다.
3. **Are More LLM Calls All You Need?** — 마지막으로 "몇 번 호출하는 것이 최적인가"라는 정량적 의사결정 문제로 마무리합니다.

## 핵심 테이크어웨이
- **Q. LLM 오케스트레이션의 본질은?** → 확률적 함수(LLM 호출)들을 조합한 프로그램입니다. 프레임워크(LangChain, LangGraph, DSPy)는 이를 표현하는 문법일 뿐입니다.
- **Q. 에이전트가 느린 이유는?** → 많은 경우 모델 성능이 아니라 순차 실행 구조 때문입니다. 의존성 그래프를 만들면 병렬화할 여지가 보입니다.
- **Q. 호출을 늘리면 항상 좋아지나?** → 아닙니다. 쿼리 난이도 분포에 따라 성능이 오히려 떨어질 수 있으므로, 최적 호출 수는 데이터로 결정해야 합니다.
- **Q. PM/엔지니어가 가져갈 질문은?** → "이 노드는 왜 존재하는가? 병렬화할 수 있는가? 호출 수를 줄여도 품질이 유지되는가?"

## 다음 토픽과의 연결
다음 토픽은 **AI Agents and Tool Use**입니다. 오늘은 "여러 LLM 호출을 어떻게 조합·실행·최적화하는가"라는 오케스트레이션 레이어를 다뤘다면, 다음에는 그 위에서 LLM이 스스로 도구를 선택하고 목표를 달성하는 **자율 에이전트**로 넘어갑니다. LLMCompiler의 Planner-Executor 구조는 에이전트 아키텍처에서 그대로 반복되는 패턴이니 연결해서 보세요.
