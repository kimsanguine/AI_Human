# Daily AI Paper Recommendations

> **Date:** 2026-08-07
> **Module:** Module 7: Prompt Engineering
> **Topic:** Advanced Prompting ToT ReAct Self-Consistency

---

## Paper 1 (Classic): Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models
- **Authors:** Lei Wang, Wanyu Xu, Yihuai Lan, Zhiqiang Hu, Yunshi Lan, Roy Ka-Wei Lee, Ee-Peng Lim
- **Year:** 2023 (ACL 2023)
- **arXiv:** https://arxiv.org/abs/2305.04091
- **PDF:** [./plan-and-solve-prompting-wang-2023.pdf](./plan-and-solve-prompting-wang-2023.pdf)
- **Citation Count:** approx. 1,300+

### 요약
Zero-shot-CoT("Let's think step by step")가 겪는 세 가지 문제(계산 오류, 추론 단계 누락, 의미 오해)를 진단하고, 이를 완화하는 Plan-and-Solve(PS) 프롬프팅을 제안한다. 모델에게 먼저 문제를 이해하고 "전체 계획을 세운 뒤 단계별로 실행"하도록 지시하고, 여기에 계산·변수 추출을 명시하는 PS+ 프롬프트를 더한다. GPT-3 기준으로 별도 예시(few-shot) 없이도 Zero-shot-CoT를 큰 폭으로 능가하고, 8-shot CoT에 필적하는 성능을 낸다.

### 핵심 기여
- Zero-shot-CoT의 실패 유형을 계산 오류(7%), 단계 누락(12%), 의미 오해(27%)로 정량 분류
- "계획 수립 → 단계 실행" 2단계 지시로 단계 누락 문제를 완화하는 PS 프롬프트 제안
- 변수/수식 추출을 명시적으로 유도하는 PS+ 프롬프트로 계산 오류까지 개선
- 예시 없이 zero-shot만으로 few-shot CoT에 근접하는 성능 달성(설계·유지비용 절감)

### 이 논문이 중요한 이유
프롬프트 한 줄로 추론 성능을 끌어올리는 대표적 zero-shot 기법으로, few-shot 예시 큐레이션 비용 없이 실무에서 바로 적용 가능하다. "계획 후 실행"이라는 아이디어는 이후 에이전트의 planning 모듈, ReAct·ToT 계열의 구조적 추론으로 이어지는 개념적 다리 역할을 한다.

### 사전 지식
- Chain-of-Thought(CoT) 프롬프팅과 Zero-shot-CoT("Let's think step by step")의 차이
- few-shot in-context learning의 개념과 예시 큐레이션 비용
- 산술/상식/기호 추론 벤치마크(GSM8K, SVAMP, AQuA 등)의 기본 개념

### 관련 논문
- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models (Wei et al., 2022)](https://arxiv.org/abs/2201.11903)
- [Large Language Models are Zero-Shot Reasoners (Kojima et al., 2022)](https://arxiv.org/abs/2205.11916)

### 실무 적용
고객 지원 봇의 다단계 문의 처리, 수식·수치 계산이 포함된 업무 자동화, 데이터 파이프라인의 규칙 기반 판단 등에서 few-shot 예시를 만들지 않고도 안정적인 추론 품질을 확보할 수 있다. 특히 프롬프트 템플릿에 "먼저 계획을 세우고 단계별로 실행하라"를 넣는 것만으로 누락·계산 오류가 줄어 QA 비용을 낮춘다.

---

## Paper 2 (Classic): Decomposed Prompting: A Modular Approach for Solving Complex Tasks
- **Authors:** Tushar Khot, Harsh Trivedi, Matthew Finlayson, Yao Fu, Kyle Richardson, Peter Clark, Ashish Sabharwal
- **Year:** 2022 (ICLR 2023)
- **arXiv:** https://arxiv.org/abs/2210.02406
- **PDF:** [./decomposed-prompting-khot-2022.pdf](./decomposed-prompting-khot-2022.pdf)
- **Citation Count:** approx. 800+

### 요약
복잡한 과제를 프롬프팅으로 더 단순한 하위 과제(sub-task)로 분해하고, 각 하위 과제를 전담하는 프롬프트 LLM들의 라이브러리에 위임하는 Decomposed Prompting(DecomP)을 제안한다. "분해기(decomposer)" 프롬프트가 과제를 여러 하위 호출로 쪼개면, 각 하위 프롬프트는 자신만의 예시로 최적화되고 필요 시 재귀적으로 더 분해되거나 검색·기호 함수로 교체될 수 있다. 이 모듈성 덕분에 GPT-3 기반에서 기존 few-shot·CoT를 능가한다.

### 핵심 기여
- 과제를 하위 과제 호출의 시퀀스로 표현하는 모듈형 프롬프팅 프레임워크 정립
- 하위 과제별 프롬프트를 독립적으로 최적화·교체·재귀 분해할 수 있는 유연한 구조 제시
- 하위 과제를 검색(retrieval)이나 기호 연산 같은 외부 도구로 연결 가능함을 실증
- 길이 일반화(예: 긴 리스트 처리)와 조합적 일반화에서 단일 프롬프트 대비 강한 견고성

### 이 논문이 중요한 이유
"복잡한 문제를 잘게 나눠 각각 전문화된 호출로 처리한다"는 발상은 오늘날 LLM 오케스트레이션·에이전트 워크플로우(LangGraph, 서브에이전트, 도구 호출)의 직접적 원형이다. 모놀리식 프롬프트의 한계를 구조로 극복하는 사고방식을 제공한다.

### 사전 지식
- few-shot 프롬프팅과 그 한계(과제 복잡도 증가 시 성능 저하)
- CoT 및 Least-to-Most 등 분해 기반 추론의 기본 개념
- 함수 호출/도구 사용(tool use)과 모듈화 소프트웨어 설계의 기본 직관

### 관련 논문
- [Least-to-Most Prompting Enables Complex Reasoning in Large Language Models (Zhou et al., 2022)](https://arxiv.org/abs/2205.10625)
- [ReAct: Synergizing Reasoning and Acting in Language Models (Yao et al., 2022)](https://arxiv.org/abs/2210.03629)

### 실무 적용
문서 처리 파이프라인(추출→정규화→검증), 복잡한 RAG(질의 분해 후 다중 검색), 멀티스텝 에이전트 설계에서 각 단계를 독립 프롬프트/도구로 분리하면 디버깅·평가·교체가 쉬워진다. 실제로 서브에이전트 아키텍처와 프롬프트 라우팅 설계의 이론적 근거로 활용된다.

---

## Paper 3 (Recent): Self-Discover: Large Language Models Self-Compose Reasoning Structures
- **Authors:** Pei Zhou, Jay Pujara, Xiang Ren, Xinyun Chen, Heng-Tze Cheng, Quoc V. Le, Ed H. Chi, Denny Zhou, Swaroop Mishra, Huaixiu Steven Zheng
- **Year:** 2024 (Google DeepMind / USC)
- **arXiv:** https://arxiv.org/abs/2402.03620
- **PDF:** [./self-discover-zhou-2024.pdf](./self-discover-zhou-2024.pdf)
- **Citation Count:** approx. 400+

### 요약
LLM이 과제 자체의 고유한 추론 구조를 스스로 발견(self-discover)하도록 하는 일반 프레임워크를 제안한다. 모델은 "비판적 사고", "단계별 사고" 같은 원자적 추론 모듈(atomic reasoning module)들을 선택·적응·조합해 명시적 추론 구조(예: JSON 형태의 key-value 계획)를 만들고, 이를 문제 풀이에 적용한다. GPT-4·PaLM 2에서 BigBench-Hard, MATH 등 난이도 높은 벤치마크를 CoT 대비 최대 32% 개선하고, CoT-Self-Consistency보다 추론 연산을 10~40배 적게 쓰면서 20% 이상 능가한다.

### 핵심 기여
- 과제별 추론 구조를 모델이 스스로 구성하는 self-discovery 2단계(구조 발견 → 구조 적용) 제안
- SELECT / ADAPT / IMPLEMENT의 메타-추론 단계로 39개 추론 모듈을 조합하는 방법 제시
- Self-Consistency 대비 10~40배 적은 추론 비용으로 더 높은 성능(효율적 test-time reasoning)
- 발견된 추론 구조가 모델 계열을 넘어 전이 가능함을 입증(PaLM 2-L→GPT-4, GPT-4→Llama2)

### 이 논문이 중요한 이유
샘플을 여러 번 뽑아 다수결하는 Self-Consistency의 높은 비용 문제를 "구조를 한 번 잘 설계"하는 방향으로 전환한다. 프롬프트 엔지니어링을 사람이 수작업으로 하던 데서 모델이 과제에 맞는 추론 스키마를 자동 조립하는 방향으로 이동시키는 최신 흐름을 대표한다.

### 사전 지식
- CoT, Self-Consistency, ToT 등 기존 추론 프롬프팅 기법과 그 연산 비용 특성
- test-time compute(추론 시점 연산) 확장의 개념
- 구조화 출력(JSON) 및 메타-프롬프팅의 기초

### 관련 논문
- [Self-Consistency Improves Chain of Thought Reasoning in Language Models (Wang et al., 2022)](https://arxiv.org/abs/2203.11171)
- [Tree of Thoughts: Deliberate Problem Solving with Large Language Models (Yao et al., 2023)](https://arxiv.org/abs/2305.10601)

### 실무 적용
난이도가 높고 유형이 다양한 태스크(복합 분석, 에이전트 계획 수립)에서 태스크별 추론 템플릿을 모델이 자동 생성하게 하면, 값비싼 다중 샘플링 없이 비용 대비 정확도를 크게 높일 수 있다. 발견된 구조를 재사용·전이하면 프롬프트 자산화(prompt asset) 관점에서 운영 효율도 개선된다.

---

## 추천 읽기 순서
1. **Plan-and-Solve (2023)** — 단일 프롬프트로 "계획 후 실행"이라는 구조적 추론의 기초 직관을 익힌다.
2. **Decomposed Prompting (2022)** — 하나의 프롬프트를 넘어 과제를 모듈로 분해하는 오케스트레이션 사고로 확장한다.
3. **Self-Discover (2024)** — 사람이 짜던 추론 구조를 모델이 스스로 조립하는 최신 방향을 확인하며, 비용-성능 트레이드오프를 이해한다.

## 핵심 테이크어웨이
- 고급 프롬프팅의 본질은 "추론을 구조화"하는 것이다: 계획 수립(PS), 과제 분해(DecomP), 구조 자동 조립(Self-Discover)로 정교해진다.
- 성능뿐 아니라 **추론 비용**이 핵심 지표다. Self-Discover는 Self-Consistency의 다수결 비용을 구조 설계로 대체해 효율을 얻는다.
- 분해·모듈화 아이디어는 프롬프팅에서 시작해 에이전트/오케스트레이션 아키텍처로 자연스럽게 확장된다.

## 다음 토픽과의 연결
오늘의 "추론 구조화·과제 분해" 개념은 Module 7의 마지막 토픽인 **Automatic Prompt Optimization**(APE, DSPy 등)으로 이어진다. 사람이 구조를 설계하던 것을 넘어 프롬프트 자체를 자동 탐색·컴파일하는 흐름으로 확장되며, 이후 Module 8의 **LLM 오케스트레이션·에이전트** 설계의 이론적 토대가 된다.
