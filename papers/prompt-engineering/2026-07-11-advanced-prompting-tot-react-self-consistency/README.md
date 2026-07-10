# Daily AI Paper Recommendations

> **Date:** 2026-07-11
> **Module:** Module 7: Prompt Engineering
> **Topic:** Advanced Prompting ToT ReAct Self-Consistency

---

## Paper 1 (Classic): PAL: Program-aided Language Models
- **Authors:** Luyu Gao, Aman Madaan, Shuyan Zhou, Uri Alon, Pengfei Liu, Yiming Yang, Jamie Callan, Graham Neubig
- **Year:** 2022
- **arXiv:** [https://arxiv.org/abs/2211.10435](https://arxiv.org/abs/2211.10435)
- **PDF:** [./pal-program-aided-language-models-gao-2022.pdf](./pal-program-aided-language-models-gao-2022.pdf)
- **Citation Count:** ~1,900+

### 요약
LLM이 자연어로 추론 과정을 서술하는 대신, 문제를 파이썬 코드로 분해해 작성하고 실제 계산은 인터프리터에 위임하는 프롬프팅 기법을 제안한다. 추론(분해)과 계산(실행)을 분리함으로써 CoT가 자주 틀리는 산술·기호 연산 오류를 구조적으로 제거한다. GSM8K 등 13개 벤치마크에서 당시 CoT 대비 큰 폭의 성능 향상을 보였다.

### 핵심 기여
- 자연어 추론 단계를 실행 가능한 프로그램으로 대체하는 Program-aided 프롬프팅 패러다임 제시
- LLM의 역할을 "문제 분해"로 한정하고 정확한 계산은 인터프리터에 위임하는 역할 분리 설계
- 수학·기호·알고리즘 추론 13개 태스크에서 CoT 대비 일관된 성능 우위 실증

### 이 논문이 중요한 이유
오늘날 AI 에이전트의 핵심 패턴인 "코드 실행 도구 사용(tool use)"의 이론적 출발점 중 하나다. ChatGPT의 Code Interpreter, 각종 에이전트 프레임워크의 코드 실행 노드가 모두 PAL의 아이디어 위에 서 있다. LLM의 약점(정확한 계산)을 외부 도구로 보완하는 설계 원칙을 익힐 수 있다.

### 사전 지식
Chain-of-Thought 프롬프팅(Wei et al., 2022), Few-shot 프롬프팅 기본기, 파이썬 기초. LLM이 왜 산술 연산에 약한지(토큰 기반 생성의 한계)를 이해하고 있으면 좋다.

### 관련 논문
- [Program of Thoughts Prompting (Chen et al., 2022)](https://arxiv.org/abs/2211.12588)
- [Chain-of-Thought Prompting Elicits Reasoning (Wei et al., 2022)](https://arxiv.org/abs/2201.11903)
- [Toolformer (Schick et al., 2023)](https://arxiv.org/abs/2302.04761)

### 실무 적용
데이터 분석 코파일럿, 금융 계산 챗봇 등 정확한 수치가 필요한 제품에서 LLM에게 직접 계산시키지 않고 코드 생성 → 샌드박스 실행 → 결과 반환 파이프라인을 구성하는 표준 패턴으로 쓰인다. 에이전트 설계 시 "무엇을 모델에게, 무엇을 도구에게 맡길지" 결정하는 기준이 된다.

---

## Paper 2 (Classic): Measuring and Narrowing the Compositionality Gap in Language Models (Self-Ask)
- **Authors:** Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt, Noah A. Smith, Mike Lewis
- **Year:** 2022
- **arXiv:** [https://arxiv.org/abs/2210.03350](https://arxiv.org/abs/2210.03350)
- **PDF:** [./self-ask-compositionality-gap-press-2022.pdf](./self-ask-compositionality-gap-press-2022.pdf)
- **Citation Count:** ~1,000+

### 요약
LLM이 하위 질문 각각은 맞히면서 이를 조합한 복합 질문은 틀리는 "구성성 격차(compositionality gap)"를 정량화하고, 모델이 스스로 후속 질문을 생성·답변하며 최종 답에 도달하는 Self-Ask 프롬프팅을 제안한다. 후속 질문을 검색 엔진에 위임하면 성능이 추가로 향상됨을 보였다.

### 핵심 기여
- 모델 규모가 커져도 좁혀지지 않는 구성성 격차를 측정하는 벤치마크(Compositional Celebrities, Bamboogle) 제시
- 복합 질문을 명시적 후속 질문으로 분해하는 Self-Ask 프롬프팅 구조 제안
- 후속 질문을 검색 엔진과 연결해 LLM + 검색 결합의 초기 실용 패턴 제시

### 이 논문이 중요한 이유
Self-Ask의 "스스로 묻고 → 외부에서 답을 찾고 → 종합한다" 구조는 ReAct와 함께 검색 결합형 에이전트(Agentic RAG)의 원형이다. 멀티홉 질문 처리, 질의 분해(query decomposition) 등 현대 RAG 시스템의 핵심 기법이 이 논문에서 출발했다.

### 사전 지식
Few-shot/CoT 프롬프팅, 멀티홉 QA 개념(HotpotQA류), 검색 엔진 API 연동에 대한 기본 이해.

### 관련 논문
- [ReAct: Synergizing Reasoning and Acting (Yao et al., 2022)](https://arxiv.org/abs/2210.03629)
- [Decomposed Prompting (Khot et al., 2022)](https://arxiv.org/abs/2210.02406)
- [Active Retrieval Augmented Generation / FLARE (Jiang et al., 2023)](https://arxiv.org/abs/2305.06983)

### 실무 적용
RAG 제품에서 복합 질문을 하위 질의로 분해해 각각 검색한 뒤 종합하는 query decomposition 파이프라인, 그리고 Perplexity류 검색 챗봇의 후속 질문 생성 로직에 직접 적용되는 패턴이다. 사용자 질문이 복잡할수록 답변 품질이 떨어지는 문제를 진단하고 개선할 때 유용한 프레임이다.

---

## Paper 3 (Recent): Scaling LLM Test-Time Compute Optimally Can Be More Effective than Scaling Model Parameters
- **Authors:** Charlie Snell, Jaehoon Lee, Kelvin Xu, Aviral Kumar
- **Year:** 2024
- **arXiv:** [https://arxiv.org/abs/2408.03314](https://arxiv.org/abs/2408.03314)
- **PDF:** [./scaling-test-time-compute-snell-2024.pdf](./scaling-test-time-compute-snell-2024.pdf)
- **Citation Count:** ~1,000+ (ICLR 2025)

### 요약
추론 시점(test-time)에 컴퓨트를 더 쓰는 것(검증자 기반 탐색, 순차적 수정 등)이 모델 파라미터를 키우는 것보다 효과적일 수 있음을 체계적으로 분석한다. 문제 난이도에 따라 최적의 컴퓨트 배분 전략이 달라짐을 보이고, "compute-optimal" 배분 시 best-of-N 대비 4배 이상 효율을 달성했다.

### 핵심 기여
- 순차 수정(revision)과 병렬 탐색(search)이라는 두 축으로 test-time 컴퓨트 확장 전략을 정식화
- 문제 난이도별 최적 전략이 다름을 실증하고, 난이도 추정 기반 compute-optimal 배분법 제안
- 쉬운·중간 난이도 문제에서 작은 모델 + 추론 컴퓨트가 14배 큰 모델을 능가할 수 있음을 입증

### 이 논문이 중요한 이유
Self-Consistency, ToT 같은 고전적 다중 샘플링 기법을 "추론 컴퓨트 스케일링"이라는 통합 관점으로 일반화한 논문이다. o1, DeepSeek-R1 등 추론 모델 시대의 이론적 배경을 제공하며, "더 큰 모델 vs 더 많은 추론"이라는 제품 차원의 비용-품질 트레이드오프를 정량적으로 사고하게 해준다.

### 사전 지식
Self-Consistency(다수결 샘플링), best-of-N 샘플링, 검증자(verifier/PRM) 개념, 기본적인 스케일링 법칙(Kaplan et al., 2020)에 대한 이해.

### 관련 논문
- [Self-Consistency Improves Chain of Thought Reasoning (Wang et al., 2022)](https://arxiv.org/abs/2203.11171)
- [s1: Simple test-time scaling (Muennighoff et al., 2025)](https://arxiv.org/abs/2501.19393)
- [Let's Verify Step by Step (Lightman et al., 2023)](https://arxiv.org/abs/2305.20050)

### 실무 적용
AI 제품에서 응답 품질·지연시간·비용의 트레이드오프를 설계할 때 직접 적용된다. 예: 쉬운 질문은 1회 생성, 어려운 질문만 다중 샘플링 + 검증자 선택으로 라우팅하는 적응형 추론 파이프라인. LLM API 비용 최적화와 reasoning effort 파라미터 튜닝의 이론적 근거가 된다.

---

## 추천 읽기 순서
1. **PAL (Gao et al., 2022)** — 추론과 계산의 역할 분리라는 가장 직관적인 아이디어부터 시작
2. **Self-Ask (Press et al., 2022)** — 질문 분해와 외부 지식 결합으로 확장
3. **Test-Time Compute Scaling (Snell et al., 2024)** — 지금까지의 고급 프롬프팅 기법들을 컴퓨트 배분이라는 통합 관점에서 재해석

## 핵심 테이크어웨이
- 고급 프롬프팅의 본질은 LLM 단일 호출의 한계를 구조로 극복하는 것이다: 계산은 도구에(PAL), 지식은 검색에(Self-Ask), 신뢰도는 다중 샘플링에(Self-Consistency) 위임한다.
- 복합 문제는 분해가 핵심이다. PAL은 코드로, Self-Ask는 후속 질문으로 분해하며, 분해 단위가 명시적일수록 디버깅과 개선이 쉬워진다.
- 추론 시점 컴퓨트는 새로운 스케일링 축이다. 문제 난이도에 따라 컴퓨트를 적응적으로 배분하면 더 큰 모델 없이도 품질을 끌어올릴 수 있다.
- 이 세 논문의 아이디어는 각각 도구 사용 에이전트, Agentic RAG, 추론 모델(o1/R1)로 진화했다 — 프롬프팅 기법이 곧 시스템 아키텍처의 씨앗이다.

## 다음 토픽과의 연결
다음 토픽은 **Automatic Prompt Optimization**이다. 오늘 다룬 기법들은 모두 사람이 설계한 프롬프트 구조인데, 다음 단계에서는 APE, 프롬프트 튜닝, DSPy류 컴파일러처럼 프롬프트 자체를 자동으로 탐색·최적화하는 방법을 다룬다. "사람이 만든 추론 구조"에서 "기계가 찾는 추론 구조"로의 전환이다.
