# Daily AI Paper Recommendations

> **Date:** 2026-06-13
> **Module:** Module 7: Prompt Engineering
> **Topic:** Chain-of-Thought and Few-Shot Prompting

---

## Paper 1 (Classic): Least-to-Most Prompting Enables Complex Reasoning in Large Language Models
- **Authors:** Denny Zhou, Nathanael Schärli, Le Hou, Jason Wei, Nathan Scales, Xuezhi Wang, Dale Schuurmans, Claire Cui, Olivier Bousquet, Quoc Le, Ed Chi
- **Year:** 2022
- **arXiv:** https://arxiv.org/abs/2205.10625
- **PDF:** [./least-to-most-prompting-zhou-2022.pdf](./least-to-most-prompting-zhou-2022.pdf)
- **Citation Count:** approx. 2,000+

### 요약
복잡한 문제를 더 작은 하위 문제들로 분해한 뒤, 앞서 푼 하위 문제의 답을 활용해 순차적으로 해결하도록 유도하는 프롬프팅 기법을 제안한다. 일반 Chain-of-Thought(CoT)가 프롬프트 예시보다 어려운 문제에서 일반화에 실패하는 한계를 "쉬운 것 → 어려운 것(least-to-most)" 분해로 극복한다.

### 핵심 기여
- 문제 분해(decomposition)와 하위 문제 순차 풀이(sequential solving)를 분리한 2단계 프롬프팅 구조를 정식화
- CoT의 약점인 easy-to-hard 일반화(예시보다 어려운 문제로의 외삽) 문제를 명확히 규정하고 해결책 제시
- SCAN 조합 일반화 벤치마크에서 단 14개 예시로 99% 이상 정확도 달성 (동일 조건 CoT는 16%)

### 이 논문이 중요한 이유
오늘날 에이전트 워크플로우의 핵심인 "작업 분해(task decomposition)" 사고의 원형이다. 단순히 추론 과정을 길게 쓰는 것을 넘어, 문제를 구조적으로 쪼개 단계적으로 푸는 패턴은 LangChain/LangGraph의 플래너-실행자 구조, ReAct, Plan-and-Solve 등으로 이어진다. AI 엔지니어가 복잡한 태스크를 LLM에게 안정적으로 위임하는 설계 원칙을 이해하는 출발점이다.

### 사전 지식
- Chain-of-Thought 프롬프팅의 기본 개념(중간 추론 단계 생성)
- Few-shot in-context learning과 exemplar(예시)의 역할
- 조합 일반화(compositional generalization), SCAN/GSM8K 같은 추론 벤치마크의 개념

### 관련 논문
- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models (Wei et al., 2022)](https://arxiv.org/abs/2201.11903)
- [Plan-and-Solve Prompting (Wang et al., 2023)](https://arxiv.org/abs/2305.04091)
- [Decomposed Prompting: A Modular Approach (Khot et al., 2022)](https://arxiv.org/abs/2210.02406)

### 실무 적용
복잡한 사용자 요청(예: 다단계 데이터 분석, 멀티홉 질의응답, 코드 생성 파이프라인)을 단일 프롬프트로 처리하면 실패율이 높다. Least-to-Most 패턴을 적용해 "먼저 하위 질문 목록을 생성 → 순서대로 해결 → 최종 답 종합" 구조로 만들면 정확도와 디버깅 용이성이 모두 향상된다. 실제 RAG 파이프라인의 query decomposition, 에이전트의 sub-task planning에 그대로 쓰인다.

---

## Paper 2 (Classic): Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?
- **Authors:** Sewon Min, Xinxi Lyu, Ari Holtzman, Mikel Artetxe, Mike Lewis, Hannaneh Hajishirzi, Luke Zettlemoyer
- **Year:** 2022
- **arXiv:** https://arxiv.org/abs/2202.12837
- **PDF:** [./rethinking-role-of-demonstrations-min-2022.pdf](./rethinking-role-of-demonstrations-min-2022.pdf)
- **Citation Count:** approx. 2,000+

### 요약
Few-shot in-context learning에서 "예시의 정답 레이블"이 실제로 성능에 얼마나 기여하는지를 실증적으로 분해한다. 놀랍게도 예시의 레이블을 무작위로 바꿔도(틀린 레이블을 줘도) GPT-3를 포함한 12개 모델에서 성능 저하가 거의 없었다. 진짜 중요한 것은 레이블의 정확성이 아니라 (1) 레이블 공간, (2) 입력 텍스트 분포, (3) 시퀀스 전체 포맷이라는 점을 밝혔다.

### 핵심 기여
- "정답 데모가 필요하다"는 통념을 반증 — 레이블 정확성보다 형식·분포가 핵심임을 규명
- in-context learning이 작동하는 4가지 요인(label space, input distribution, format, input-label pairing)을 분리 분석
- 12개 모델·다양한 태스크에 걸친 대규모 ablation으로 강건한 결론 제시

### 이 논문이 중요한 이유
프롬프트에 예시를 넣을 때 "무엇을, 왜 넣어야 하는가"에 대한 가장 영향력 있는 실증 연구다. AI 엔지니어가 few-shot 예시를 설계할 때 정답의 완벽함보다 포맷 일관성·레이블 분포·도메인 대표성에 투자해야 한다는 실천적 지침을 준다. 프롬프트 엔지니어링을 직관이 아닌 검증 가능한 원리로 다루게 한 전환점.

### 사전 지식
- In-context learning / few-shot prompting의 기본 메커니즘
- 데모(demonstration) 예시가 프롬프트에서 차지하는 구조
- ablation study(요인 제거 실험)의 개념과 통계적 해석

### 관련 논문
- [Language Models are Few-Shot Learners / GPT-3 (Brown et al., 2020)](https://arxiv.org/abs/2005.14165)
- [An Explanation of In-context Learning as Implicit Bayesian Inference (Xie et al., 2021)](https://arxiv.org/abs/2111.02080)
- [Calibrate Before Use: Improving Few-Shot Performance of Language Models (Zhao et al., 2021)](https://arxiv.org/abs/2102.09690)

### 실무 적용
few-shot 예시 큐레이션 전략에 직접 적용된다. 정답을 일일이 검수하는 데 과도한 비용을 쓰기보다, 예시의 포맷을 일관되게 통일하고 레이블 분포를 실제 분포와 맞추며 입력 도메인을 대표성 있게 샘플링하는 것이 효율적이다. 동적 few-shot(예: 임베딩 기반 예시 검색) 시스템 설계 시 "무엇을 기준으로 예시를 선택할지"의 근거가 된다.

---

## Paper 3 (Recent): Meta-Prompting: Enhancing Language Models with Task-Agnostic Scaffolding
- **Authors:** Mirac Suzgun, Adam Tauman Kalai
- **Year:** 2024
- **arXiv:** https://arxiv.org/abs/2401.12954
- **PDF:** [./meta-prompting-suzgun-2024.pdf](./meta-prompting-suzgun-2024.pdf)
- **Citation Count:** approx. 300+

### 요약
하나의 LLM을 "지휘자(conductor)"이자 동시에 여러 "전문가(expert) 인스턴스"로 동작하게 하는 태스크 비종속적(task-agnostic) 스캐폴딩 기법을 제안한다. 고수준 지시만으로 LLM이 복잡한 문제를 하위 태스크로 분해하고, 각 하위 태스크를 맞춤 지시를 받은 별도 전문가 인스턴스에 위임한 뒤 결과를 종합한다. 제로샷이며 Python 인터프리터 같은 외부 도구도 통합한다.

### 핵심 기여
- 단일 모델이 오케스트레이터 + 다수 전문가 역할을 동시에 수행하는 meta-prompting 프레임워크 제안
- 태스크별 상세 프롬프트 없이도 작동하는 zero-shot·task-agnostic 구조로 사용자 부담 최소화
- 외부 도구(코드 실행기 등) 통합으로 적용 범위 확장, 다양한 벤치마크에서 표준 프롬프팅 대비 성능 향상

### 이 논문이 중요한 이유
2024년 멀티-에이전트/오케스트레이션 흐름을 단일 LLM 프롬프팅 수준에서 구현한 대표 사례다. 별도 에이전트 프레임워크 없이도 프롬프트 구조만으로 "전문가 패널" 효과를 내는 접근은, 비용·복잡도를 낮추면서 에이전트형 추론을 적용하려는 엔지니어에게 실용적 청사진을 제공한다. Module 7(프롬프팅)과 Module 8(오케스트레이션)을 잇는 가교.

### 사전 지식
- Least-to-Most / 작업 분해 기반 프롬프팅 개념
- 멀티-에이전트 시스템과 오케스트레이션의 기본 아이디어
- LLM의 도구 사용(tool use) 및 코드 실행 통합 패턴

### 관련 논문
- [Self-Discover: Large Language Models Self-Compose Reasoning Structures (Zhou et al., 2024)](https://arxiv.org/abs/2402.03620)
- [ReAct: Synergizing Reasoning and Acting in Language Models (Yao et al., 2022)](https://arxiv.org/abs/2210.03629)
- [Least-to-Most Prompting (Zhou et al., 2022)](https://arxiv.org/abs/2205.10625)

### 실무 적용
복잡한 문제를 하나의 모델로 풀되 품질을 높여야 할 때, 별도 에이전트 인프라 없이 meta-prompting 스캐폴드를 적용할 수 있다. 예를 들어 코드 생성·검증, 다관점 분석 리포트 작성, 복잡한 추론 QA에서 "전문가 호출 → 결과 비판/종합" 루프를 프롬프트로 구현하면 단일 패스 대비 정확도가 올라간다. 토큰 비용은 증가하므로 난이도 높은 태스크에 선택적으로 적용하는 것이 실무 포인트다.

---

## 추천 읽기 순서
1. **Rethinking the Role of Demonstrations (Min et al., 2022)** — few-shot 예시가 "왜" 작동하는지부터 이해하면 이후 분해·스캐폴딩 기법의 동기가 명확해진다.
2. **Least-to-Most Prompting (Zhou et al., 2022)** — 단순 예시 제공을 넘어 문제를 구조적으로 분해하는 사고로 확장한다.
3. **Meta-Prompting (Suzgun & Kalai, 2024)** — 분해 사고를 단일 모델 내 멀티-전문가 오케스트레이션으로 발전시킨 최신 형태로 마무리한다.

## 핵심 테이크어웨이
- Few-shot의 힘은 정답 레이블의 정확성보다 **포맷·분포·레이블 공간**에서 나온다 (Min et al.).
- 어려운 문제는 **분해 후 순차 해결(least-to-most)** 로 풀면 예시보다 어려운 케이스에도 일반화된다 (Zhou et al.).
- 2024년 프롬프팅은 단일 프롬프트를 넘어 **단일 모델이 스스로 전문가를 호출·종합하는 오케스트레이션 스캐폴드**로 진화 중이다 (Suzgun & Kalai).
- 공통 줄기는 "**구조화된 분해와 위임**" — 프롬프트 엔지니어링이 곧 추론 구조 설계임을 보여준다.

## 다음 토픽과의 연결
다음 토픽인 **Advanced Prompting (Tree-of-Thought, ReAct, Self-Consistency)** 은 오늘의 "분해·스캐폴딩" 아이디어를 탐색 트리·행동 루프·다중 샘플링으로 확장한다. Least-to-Most의 순차 분해는 ToT의 분기 탐색으로, Meta-Prompting의 전문가 위임은 ReAct의 행동-관찰 루프로 자연스럽게 이어진다.
