# Daily AI Paper Recommendations

> **Date:** 2026-09-30
> **Module:** Module 7: Prompt Engineering
> **Topic:** Advanced Prompting ToT ReAct Self-Consistency

---

## Paper 1 (Classic): Chain-of-Verification Reduces Hallucination in Large Language Models
- **Authors:** Shehzaad Dhuliawala, Mojtaba Komeili, Jing Xu, Roberta Raileanu, Xian Li, Asli Celikyilmaz, Jason Weston
- **Year:** 2023 (Findings of ACL 2024)
- **arXiv:** https://arxiv.org/abs/2309.11495
- **PDF:** [./chain-of-verification-dhuliawala-2023.pdf](./chain-of-verification-dhuliawala-2023.pdf)
- **Citation Count:** ~700+

### 요약
LLM이 스스로 만든 답변을 "검증 질문"으로 다시 점검하게 하는 프롬프팅 기법 CoVe를 제안합니다. (1) 초안 작성 → (2) 초안을 검증할 질문 계획 → (3) 각 질문에 독립적으로 답변 → (4) 검증 결과를 반영한 최종 답변 생성의 4단계로 구성되며, 추가 학습 없이 환각(hallucination)을 크게 줄입니다.

### 핵심 기여
- 생성-검증-수정을 명시적 단계로 분리한 **Chain-of-Verification** 프레임워크 제시
- 검증 질문을 초안과 **분리된 컨텍스트에서 독립적으로(Factored)** 답하게 해, 모델이 자기 초안의 오류를 그대로 베끼는 "자기 복제 편향"을 줄임
- Wikidata 리스트형 질의에서 Llama 65B few-shot 대비 정밀도 0.17 → 0.36으로 2배 이상 개선, 장문 전기(biography) 생성에서도 FactScore 향상

### 이 논문이 중요한 이유
Self-Consistency가 "여러 번 풀어서 다수결"이라면, CoVe는 "한 번 풀고 스스로 팩트체크"하는 방식입니다. 추론 경로를 늘리는 ToT/Self-Consistency와 달리 **사실성(factuality)** 문제를 정면으로 다루며, 오늘날 에이전트의 self-critique·reflection 루프의 가장 깔끔한 프롬프트 레벨 원형입니다.

### 사전 지식
- Chain-of-Thought 프롬프팅, Few-shot 프롬프팅
- LLM 환각(hallucination)의 개념과 유형 (closed-book QA, 리스트 생성, 장문 생성)
- Self-Refine / Reflexion 등 자기 피드백 기법에 대한 기본 이해

### 관련 논문
- [Self-Refine: Iterative Refinement with Self-Feedback (Madaan et al., 2023)](https://arxiv.org/abs/2303.17651)
- [FActScore: Fine-grained Atomic Evaluation of Factual Precision (Min et al., 2023)](https://arxiv.org/abs/2305.14251)
- [Measuring and Narrowing the Compositionality Gap / Self-Ask (Press et al., 2022)](https://arxiv.org/abs/2210.03350)

### 실무 적용
- RAG 없이 답해야 하는 FAQ·요약 봇에서 "초안 → 검증 질문 → 독립 답변 → 최종본" 파이프라인을 LangGraph 노드로 구현하면 사실 오류를 줄일 수 있습니다.
- 검증 질문 단계에 검색 도구를 붙이면 곧바로 "도구 기반 팩트체크 에이전트"가 됩니다.
- 비용이 3~4배 늘어나므로, 고위험 응답(의료·법률·수치)에만 선택적으로 적용하는 라우팅 설계가 현실적입니다.

---

## Paper 2 (Classic): Universal Self-Consistency for Large Language Model Generation
- **Authors:** Xinyun Chen, Renat Aksitov, Uri Alon, Jie Ren, Kefan Xiao, Pengcheng Yin, Sushant Prakash, Charles Sutton, Xuezhi Wang, Denny Zhou
- **Year:** 2023
- **arXiv:** https://arxiv.org/abs/2311.17311
- **PDF:** [./universal-self-consistency-chen-2023.pdf](./universal-self-consistency-chen-2023.pdf)
- **Citation Count:** ~200+

### 요약
기존 Self-Consistency는 여러 추론 경로의 "최종 답"을 추출해 다수결하므로, 수학처럼 정답이 딱 떨어지는 과제에서만 쓸 수 있었습니다. USC는 여러 후보 응답을 하나의 프롬프트에 넣고 **LLM 자신에게 "가장 일관된 응답"을 고르게** 하여, 요약·코드·자유형 QA 같은 열린 생성 과제로 Self-Consistency를 확장합니다.

### 핵심 기여
- 답 추출 규칙 없이 LLM이 후보 간 합의를 판단하는 **범용 집계(aggregation) 방식** 제안
- 수학(GSM8K, MATH)에서는 기존 Self-Consistency와 동등한 성능, 코드 생성(BIRD-SQL, ARCADE)에서는 실행 기반 투표와 비슷한 성능을 **코드 실행 없이** 달성
- 장문 요약(GovReport, SummScreen)·TruthfulQA 등 기존 SC가 불가능했던 과제에서 greedy 디코딩 대비 일관된 향상

### 이 논문이 중요한 이유
Self-Consistency 계열에서 "투표 대상이 정형 답이어야 한다"는 가장 큰 제약을 풀었습니다. 또한 "LLM-as-a-Judge로 후보를 선택한다"는 발상은 이후 Best-of-N 선택, 생성형 검증기(generative verifier), 테스트타임 스케일링 연구의 기반 패턴이 됩니다.

### 사전 지식
- Self-Consistency (Wang et al., 2022)의 샘플링 + 다수결 원리
- 온도(temperature) 샘플링과 디코딩 전략
- LLM-as-a-Judge 평가 방식의 장단점 (위치 편향 등)

### 관련 논문
- [Self-Consistency Improves Chain of Thought Reasoning (Wang et al., 2022)](https://arxiv.org/abs/2203.11171)
- [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena (Zheng et al., 2023)](https://arxiv.org/abs/2306.05685)
- [Atomic Self-Consistency for Better Long Form Generations (Thirukovalluru et al., 2024)](https://arxiv.org/abs/2405.13131)

### 실무 적용
- 요약·이메일 초안·SQL 생성처럼 정답이 하나가 아닌 기능에서 N개 샘플 → USC 선택 단계를 추가해 품질 편차를 줄일 수 있습니다.
- 후보 수가 많아지면 컨텍스트 길이·위치 편향 문제가 생기므로, 실무에선 N=5~8 정도와 후보 순서 셔플을 함께 쓰는 것이 안전합니다.
- 평가 파이프라인에서 "사람이 고른 베스트 vs USC가 고른 베스트" 일치율을 측정하면 자동 선택기의 신뢰도를 데이터로 검증할 수 있습니다.

---

## Paper 3 (Recent): Mutual Reasoning Makes Smaller LLMs Stronger Problem-Solvers (rStar)
- **Authors:** Zhenting Qi, Mingyuan Ma, Jiahang Xu, Li Lyna Zhang, Fan Yang, Mao Yang
- **Year:** 2024 (ICLR 2025)
- **arXiv:** https://arxiv.org/abs/2408.06195
- **PDF:** [./rstar-mutual-reasoning-qi-2024.pdf](./rstar-mutual-reasoning-qi-2024.pdf)
- **Code:** https://github.com/zhentingqi/rStar
- **Citation Count:** ~150+

### 요약
rStar는 파인튜닝이나 더 큰 교사 모델 없이 소형 언어모델(SLM)의 추론력을 끌어올리는 **self-play 상호 추론(mutual reasoning)** 기법입니다. 하나의 SLM이 인간과 유사한 5가지 추론 액션으로 MCTS 트리 탐색을 하며 추론 경로를 만들고, 비슷한 크기의 다른 SLM이 판별자(discriminator)로 각 경로를 검증해 서로 "동의"하는 경로를 최종 답으로 채택합니다.

### 핵심 기여
- MCTS에 "한 단계 생각 / 남은 단계 한 번에 풀기 / 하위 질문 생성·답변 / 하위 질문 재답변 / 질문 재구성" 등 **풍부한 인간형 추론 액션 공간** 도입
- 생성자-판별자 SLM 간 **상호 일관성(mutual consistency)** 으로 보상 모델 없이 경로 선택
- GSM8K 정확도: LLaMA2-7B 12.51% → 63.91%, Mistral-7B 36.46% → 81.88%, LLaMA3-8B-Instruct 74.53% → 91.13%

### 이 논문이 중요한 이유
Tree of Thoughts(탐색) + Self-Consistency(합의) + 검증(CoVe류)을 하나의 추론 시점(test-time) 알고리즘으로 결합한 사례입니다. "모델을 키우지 않고 추론 시간 연산으로 성능을 산다"는 2024~2025년 테스트타임 스케일링 흐름을 소형 모델 관점에서 보여주며, 후속작 rStar-Math로 이어집니다.

### 사전 지식
- Tree of Thoughts, Self-Consistency, Least-to-Most / Self-Ask 등 분해형 프롬프팅
- Monte Carlo Tree Search (UCT, rollout, backpropagation)의 기본 개념
- 테스트타임 컴퓨트 스케일링과 Best-of-N 개념

### 관련 논문
- [rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking (Guan et al., 2025)](https://arxiv.org/abs/2501.04519)
- [Tree of Thoughts (Yao et al., 2023)](https://arxiv.org/abs/2305.10601)
- [Scaling LLM Test-Time Compute Optimally (Snell et al., 2024)](https://arxiv.org/abs/2408.03314)

### 실무 적용
- 온디바이스·사내 구축형처럼 대형 모델을 쓰기 어려운 환경에서, 7~8B급 모델 2개를 생성자/검증자로 묶어 정확도를 끌어올리는 설계 참고가 됩니다.
- 다만 MCTS 롤아웃으로 호출 수가 수십 배 늘어나므로, 지연시간이 허용되는 배치형 작업(리포트 생성, 데이터 라벨링, 오프라인 평가)에 적합합니다.
- "상호 일관성" 아이디어는 서로 다른 벤더 모델 2개로 교차 검증하는 프로덕션 가드레일로도 응용할 수 있습니다.

---

## 추천 읽기 순서
1. **Universal Self-Consistency** — 이미 아는 Self-Consistency를 자유형 생성으로 확장하는 가장 가벼운 출발점
2. **Chain-of-Verification** — "여러 번 풀기" 대신 "스스로 검증하기"라는 다른 축을 이해
3. **rStar** — 탐색(ToT) + 합의(SC) + 검증(CoVe)이 하나의 알고리즘으로 합쳐지는 모습을 확인

## 핵심 테이크어웨이
- 고급 프롬프팅은 크게 **탐색(ToT/MCTS)**, **합의(Self-Consistency/USC)**, **검증(CoVe/판별자)** 세 축으로 정리할 수 있습니다.
- 정답이 정형이 아니어도 LLM 자체를 선택기/검증기로 쓰면 앙상블 효과를 얻을 수 있습니다 (USC).
- 검증은 초안과 **분리된 컨텍스트**에서 해야 효과적입니다 — 같은 컨텍스트에서의 자기 검증은 오류를 반복하기 쉽습니다 (CoVe).
- 소형 모델도 추론 시점 연산(탐색+상호 검증)을 늘리면 큰 폭의 성능 향상이 가능하지만, 비용·지연시간 트레이드오프를 반드시 설계해야 합니다 (rStar).

## 다음 토픽과의 연결
다음 토픽 **Automatic Prompt Optimization**에서는 이런 추론 프롬프트를 사람이 손으로 설계하는 대신, APE·DSPy·TextGrad처럼 **프롬프트 자체를 자동으로 탐색·최적화**하는 방법을 다룹니다. 오늘 본 "후보 생성 → 평가 → 선택" 루프가 프롬프트 공간에 그대로 적용된다는 점에 주목하면 연결이 자연스럽습니다.
