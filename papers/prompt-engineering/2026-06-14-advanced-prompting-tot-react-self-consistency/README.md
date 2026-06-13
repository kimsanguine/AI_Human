# Daily AI Paper Recommendations

> **Date:** 2026-06-14
> **Module:** Module 7: Prompt Engineering
> **Topic:** Advanced Prompting (ToT, ReAct, Self-Consistency)

---

## Paper 1 (Classic): Least-to-Most Prompting Enables Complex Reasoning in Large Language Models
- **Authors:** Denny Zhou, Nathanael Schärli, Le Hou, Jason Wei, Nathan Scales, Xuezhi Wang, Dale Schuurmans, Olivier Bousquet, Quoc Le, Ed H. Chi
- **Year:** 2022
- **arXiv:** [https://arxiv.org/abs/2205.10625](https://arxiv.org/abs/2205.10625)
- **PDF:** [./least-to-most-prompting-zhou-2022.pdf](./least-to-most-prompting-zhou-2022.pdf)
- **Citation Count:** ~2,000+

### 요약
복잡한 문제를 먼저 더 쉬운 하위 문제(sub-problem)들로 분해한 뒤, 앞에서 푼 하위 문제의 답을 다음 하위 문제의 입력으로 넘기며 순차적으로 풀어 나가는 프롬프팅 전략이다. Chain-of-Thought가 한 번에 추론을 전개하는 것과 달리, "분해 → 순차 해결"이라는 2단계 구조로 더 어려운 문제로의 일반화(easy-to-hard generalization)를 가능하게 한다.

### 핵심 기여
- 문제 분해(decomposition)와 하위 문제 순차 해결(sequential solving)을 분리한 2단계 프롬프팅 패러다임 제시
- 프롬프트에 등장한 예시보다 더 복잡한 문제로 일반화되는 능력(length/compositional generalization)을 실증
- SCAN 조합 일반화 벤치마크에서 단 14개 예시만으로 99% 이상 정확도 달성 (CoT는 16% 수준)

### 이 논문이 중요한 이유
복잡한 작업을 LLM에게 통째로 던지면 실패하지만, 사람처럼 "작게 쪼개서 차례로" 풀게 하면 성능이 급격히 올라간다는 점을 명확히 보여준다. 오늘날 에이전트의 task decomposition, planning 단계 설계의 직접적인 사고적 뿌리이며, 모든 AI 엔지니어가 프롬프트/워크플로 설계 시 가장 먼저 떠올려야 할 원칙이다.

### 사전 지식
- Chain-of-Thought(CoT) 프롬프팅과 few-shot in-context learning의 기본 개념
- 조합 일반화(compositional generalization)와 SCAN 같은 벤치마크의 취지
- LLM이 프롬프트 예시로부터 패턴을 추론하는 방식

### 관련 논문
- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models (Wei et al., 2022)](https://arxiv.org/abs/2201.11903)
- [Decomposed Prompting: A Modular Approach for Solving Complex Tasks (Khot et al., 2022)](https://arxiv.org/abs/2210.02406)

### 실무 적용
긴 문서 처리, 다단계 데이터 변환, 복잡한 코드 생성 등에서 작업을 명시적 단계로 분해해 파이프라인으로 구성할 때 활용된다. LangGraph/에이전트의 planner-executor 구조, RAG의 multi-hop 질의 분해 등이 모두 이 아이디어의 실무 구현체다.

---

## Paper 2 (Classic): Self-Refine: Iterative Refinement with Self-Feedback
- **Authors:** Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Shashank Gupta, Bodhisattwa Prasad Majumder, Katherine Hermann, Sean Welleck, Amir Yazdanbakhsh, Peter Clark
- **Year:** 2023
- **arXiv:** [https://arxiv.org/abs/2303.17651](https://arxiv.org/abs/2303.17651)
- **PDF:** [./self-refine-madaan-2023.pdf](./self-refine-madaan-2023.pdf)
- **Citation Count:** ~1,800+

### 요약
하나의 LLM이 생성자(generator), 비평자(critic), 수정자(refiner) 역할을 모두 맡아, 자신이 만든 출력에 스스로 피드백을 주고 그 피드백으로 출력을 반복 개선하는 방법이다. 추가 학습, 강화학습, 지도 데이터 없이 프롬프팅만으로 동작한다.

### 핵심 기여
- 별도 학습 없이 "생성 → 자기 피드백 → 수정"의 반복 루프만으로 출력 품질을 개선하는 범용 프레임워크 제시
- 대화 응답, 코드 최적화, 수학 추론 등 7개 과제에서 단발 생성 대비 사람·자동 지표 모두 약 20% 향상
- 과거 피드백·출력 이력을 프롬프트에 누적해 같은 실수를 반복하지 않게 하는 메모리 메커니즘

### 이 논문이 중요한 이유
"한 번에 잘 쓰기"가 아니라 "고쳐 쓰기"로 품질을 끌어올리는 self-correction 패턴의 대표작이다. 오늘날 reflection 기반 에이전트, LLM-as-a-judge, 자동 평가-수정 루프의 핵심 사고 모델이며, 별도 모델 없이 단일 LLM만으로 신뢰성을 높이는 실용적 기법이다.

### 사전 지식
- LLM이 자기 출력을 평가(self-evaluation)할 수 있다는 전제와 그 한계
- few-shot 프롬프팅으로 피드백 형식을 유도하는 방법
- 반복 추론에서의 컨텍스트 길이/비용 트레이드오프

### 관련 논문
- [Reflexion: Language Agents with Verbal Reinforcement Learning (Shinn et al., 2023)](https://arxiv.org/abs/2303.11366)
- [Constitutional AI: Harmlessness from AI Feedback (Bai et al., 2022)](https://arxiv.org/abs/2212.08073)

### 실무 적용
코드 생성 후 자체 리뷰-수정, 초안 작성 후 자기 교정, RAG 답변의 사실성 자체 점검 등에 적용된다. 다만 반복마다 토큰 비용과 지연이 늘어나므로, 종료 조건과 최대 반복 횟수를 명확히 설계하는 것이 실무 핵심이다.

---

## Paper 3 (Recent): Buffer of Thoughts: Thought-Augmented Reasoning with Large Language Models
- **Authors:** Ling Yang, Zhaochen Yu, Tianjun Zhang, Shiyi Cao, Minkai Xu, Wentao Zhang, Joseph E. Gonzalez, Bin Cui
- **Year:** 2024 (NeurIPS 2024 Spotlight)
- **arXiv:** [https://arxiv.org/abs/2406.04271](https://arxiv.org/abs/2406.04271)
- **PDF:** [./buffer-of-thoughts-yang-2024.pdf](./buffer-of-thoughts-yang-2024.pdf)
- **Citation Count:** ~300+

### 요약
다양한 문제 풀이 과정에서 추출한 고수준 "사고 템플릿(thought-template)"을 메타 버퍼(meta-buffer)에 저장해 두고, 새로운 문제를 만나면 관련 템플릿을 검색해 그 문제에 맞게 적응적으로 인스턴스화하여 추론하는 방법이다. 버퍼 매니저가 푸는 문제가 늘어날수록 메타 버퍼를 동적으로 갱신한다.

### 핵심 기여
- 과거 추론 구조를 재사용 가능한 템플릿으로 축적·검색하는 thought-augmented reasoning 패러다임 제시
- Game of 24(+11%), Geometric Shapes(+20%), Checkmate-in-One(+51%) 등 10개 난이도 높은 추론 과제에서 큰 성능 향상
- 다중 질의(multi-query) 프롬프팅 대비 평균 약 12% 비용만으로 더 높은 정확도·강건성 달성

### 이 논문이 중요한 이유
ToT/GoT 계열의 "그때그때 새로 탐색"하는 비용 문제를 "한 번 잘 푼 추론 패턴을 재사용"으로 해결한, 2024년 reasoning 연구의 중요한 전환점이다. 추론을 검색 가능한 자산으로 다루는 발상은 에이전트 메모리·경험 재사용 설계와 직접 맞닿아 있어 AI 엔지니어에게 시의성이 높다.

### 사전 지식
- Tree of Thoughts / Graph of Thoughts 등 multi-path 추론 기법의 비용 구조
- 임베딩 기반 검색(retrieval)과 템플릿 매칭의 기본 개념
- in-context learning에서 예시(exemplar) 선택이 성능에 미치는 영향

### 관련 논문
- [Tree of Thoughts: Deliberate Problem Solving with Large Language Models (Yao et al., 2023)](https://arxiv.org/abs/2305.10601)
- [Graph of Thoughts: Solving Elaborate Problems with Large Language Models (Besta et al., 2023)](https://arxiv.org/abs/2308.09687)

### 실무 적용
반복적으로 유사한 유형의 작업을 처리하는 에이전트에서, 성공한 추론 경로를 템플릿으로 저장해 두고 재사용하면 비용과 지연을 줄이면서 품질을 유지할 수 있다. 사내 워크플로 자동화, 반복 분석 작업, 도메인 특화 에이전트의 "경험 라이브러리" 구축에 적용 가능하다.

---

## 추천 읽기 순서
1. **Least-to-Most Prompting** — "복잡한 문제는 쪼개서 푼다"는 가장 기초적이고 강력한 원칙을 먼저 체득한다.
2. **Self-Refine** — 분해해서 푼 결과를 "고쳐 쓰며" 품질을 올리는 self-correction 루프를 이해한다.
3. **Buffer of Thoughts** — 분해·자기수정으로 얻은 좋은 추론 패턴을 "저장·재사용"하는 최신 발전을 본다.

## 핵심 테이크어웨이
- 고급 프롬프팅의 본질은 ① 분해(decomposition), ② 자기수정(self-correction), ③ 재사용(reuse)의 세 축이다.
- 추가 학습 없이 프롬프트 설계만으로도 추론 성능을 크게 끌어올릴 수 있으나, 토큰 비용·지연이라는 대가가 따른다.
- 2024년 흐름은 "매번 새로 탐색"에서 "잘 푼 추론을 자산으로 축적·재사용"하는 방향으로 이동하고 있다.

## 다음 토픽과의 연결
다음 토픽인 **Automatic Prompt Optimization(자동 프롬프트 최적화)**는 오늘 본 수작업 프롬프팅 전략들을 사람이 아닌 알고리즘이 자동으로 탐색·개선하도록 만든다. 분해·자기수정·재사용을 자동화하는 DSPy, APE 같은 도구로 자연스럽게 이어진다.
