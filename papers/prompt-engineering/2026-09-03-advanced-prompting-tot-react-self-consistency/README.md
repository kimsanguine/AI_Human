# Daily AI Paper Recommendations

> **Date:** 2026-09-03
> **Module:** Module 7: Prompt Engineering
> **Topic:** Advanced Prompting — ToT, ReAct, Self-Consistency

---

## Paper 1 (Classic): Let's Verify Step by Step
- **Authors:** Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, Karl Cobbe
- **Year:** 2023
- **arXiv:** https://arxiv.org/abs/2305.20050
- **PDF:** [./lets-verify-step-by-step-lightman-2023.pdf](./lets-verify-step-by-step-lightman-2023.pdf)
- **Citation Count:** ~2,500+ (매우 높음)

### 요약
OpenAI 연구팀이 "최종 답만 채점하는 방식(outcome supervision)"과 "추론의 각 단계를 채점하는 방식(process supervision)"을 정면으로 비교한 논문입니다. MATH 데이터셋에서 프로세스 보상 모델(PRM)이 결과 보상 모델(ORM)을 크게 앞질렀고, 대표 테스트 서브셋 문제의 78%를 풀어냈습니다. 80만 개의 단계별 사람 피드백 라벨로 구성된 PRM800K 데이터셋도 함께 공개했습니다.

### 핵심 기여
- 프로세스 감독이 결과 감독보다 수학 추론에서 일관되게 우수함을 대규모로 실증
- 단계별 라벨 80만 개 데이터셋(PRM800K) 공개 — 이후 추론 모델 연구의 공용 인프라가 됨
- 액티브 러닝을 결합하면 프로세스 감독의 데이터 효율이 약 2.6배 개선됨을 보임
- 정렬(alignment) 관점에서 프로세스 감독이 "사람이 승인한 추론 과정"을 직접 보상하므로 더 안전하다는 논거 제시

### 이 논문이 중요한 이유
ToT, MCTS, Best-of-N 같은 탐색 기반 추론은 모두 "여러 후보 중 무엇이 좋은가"를 판단할 채점자(verifier)를 필요로 합니다. 이 논문은 그 채점자를 어떻게 만들어야 하는가에 대한 사실상의 표준 답을 제시했고, 이후 o1·DeepSeek-R1 계열 추론 모델과 test-time scaling 연구의 토대가 되었습니다. AI 엔지니어 입장에서는 "생성 품질을 올리는 가장 확실한 방법은 더 좋은 프롬프트가 아니라 더 좋은 검증기"라는 교훈을 얻을 수 있습니다.

### 사전 지식
- Chain-of-Thought 프롬프팅의 기본 개념
- 보상 모델(reward model)과 RLHF의 기초, Best-of-N 샘플링
- MATH / GSM8K 벤치마크의 성격
- generator–verifier 구조(생성기와 검증기의 분리)

### 관련 논문
- [Training Verifiers to Solve Math Word Problems (Cobbe et al., 2021)](https://arxiv.org/abs/2110.14168)
- [Self-Consistency Improves Chain of Thought Reasoning (Wang et al., 2022)](https://arxiv.org/abs/2203.11171)
- [Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning (Setlur et al., 2024)](https://arxiv.org/abs/2410.08146)

### 실무 적용
LLM 제품에서 정확도가 중요한 구간(계산, 정책 판단, 코드 생성)에는 생성 모델 하나만 두지 말고 단계별 검증기를 붙이는 구조가 효과적입니다. 실무에서는 (1) 여러 후보를 샘플링하고 (2) PRM 또는 LLM-as-judge로 단계별 점수를 매겨 (3) 최고 점수 경로만 사용자에게 노출하는 파이프라인으로 구현합니다. 비용이 문제라면 검증기를 작은 모델로 두고 생성기만 큰 모델을 쓰는 비대칭 구성이 가성비가 좋습니다.

---

## Paper 2 (Classic): Language Agent Tree Search Unifies Reasoning, Acting, and Planning in Language Models
- **Authors:** Andy Zhou, Kai Yan, Michal Shlapentokh-Rothman, Haohan Wang, Yu-Xiong Wang
- **Year:** 2023 (ICML 2024)
- **arXiv:** https://arxiv.org/abs/2310.04406
- **PDF:** [./language-agent-tree-search-zhou-2023.pdf](./language-agent-tree-search-zhou-2023.pdf)
- **Citation Count:** ~700+ (높음)

### 요약
LATS는 Tree of Thoughts의 "탐색", ReAct의 "행동", Reflexion의 "자기반성"을 하나의 프레임워크로 통합한 연구입니다. Monte Carlo Tree Search(MCTS)를 LLM 에이전트의 의사결정 루프에 넣고, LLM 자신을 가치 함수(value function)와 반성 생성기로 활용합니다. 프로그래밍, 대화형 QA, 웹 내비게이션, 수학 등 여러 도메인에서 기존 단일 경로 방식보다 일관되게 나은 성능을 보였습니다.

### 핵심 기여
- 추론(reasoning)·행동(acting)·계획(planning)을 통합한 최초의 범용 LLM 에이전트 프레임워크 제시
- MCTS의 선택–확장–평가–시뮬레이션–역전파 사이클을 LLM 프롬프팅만으로 구현(추가 학습 불필요)
- 환경으로부터의 외부 피드백을 탐색 신호로 편입해, 내부 자기평가만 쓰는 ToT의 한계를 보완
- 실패 경로에 대한 자기반성을 메모리에 축적해 다음 탐색에 재사용하는 구조

### 이 논문이 중요한 이유
ToT와 ReAct를 각각 배운 뒤 자연스럽게 떠오르는 질문이 "탐색과 도구 사용을 어떻게 합치는가"입니다. LATS는 그 질문에 대한 정석적인 답이며, 오늘날 LangGraph·OpenAI Agents SDK 등에서 쓰이는 "계획 → 실행 → 평가 → 재시도" 에이전트 루프의 이론적 원형입니다. Agentic AI 제품을 설계한다면 반드시 읽어야 할 논문입니다.

### 사전 지식
- ReAct의 Thought–Action–Observation 루프
- Tree of Thoughts의 상태·평가·탐색 구조
- Reflexion의 언어적 자기반성 메커니즘
- MCTS의 UCT(Upper Confidence bounds applied to Trees) 기본 개념

### 관련 논문
- [ReAct: Synergizing Reasoning and Acting in Language Models (Yao et al., 2022)](https://arxiv.org/abs/2210.03629)
- [Tree of Thoughts: Deliberate Problem Solving with LLMs (Yao et al., 2023)](https://arxiv.org/abs/2305.10601)
- [Reflexion: Language Agents with Verbal Reinforcement Learning (Shinn et al., 2023)](https://arxiv.org/abs/2303.11366)
- [Reasoning with Language Model is Planning with World Model / RAP (Hao et al., 2023)](https://arxiv.org/abs/2305.14992)

### 실무 적용
코딩 에이전트, 웹 자동화, 복잡한 워크플로 에이전트처럼 "한 번에 맞히기 어렵지만 시도 결과를 검증할 수 있는" 작업에 잘 맞습니다. 다만 LATS는 LLM 호출 수가 폭증하므로 실서비스에서는 탐색 폭(k)과 깊이를 작게 잡고, 테스트 실행·API 응답 같은 저렴한 외부 신호를 가치 함수 대용으로 쓰는 것이 현실적입니다. 프리미엄 티어에서만 깊은 탐색을 허용하는 식의 티어링 설계도 자주 쓰입니다.

---

## Paper 3 (Recent): Chain-in-Tree: Back to Sequential Reasoning in LLM Tree Search
- **Authors:** Xinzhe Li
- **Year:** 2025 (updated 2026)
- **arXiv:** https://arxiv.org/abs/2509.25835
- **PDF:** [./chain-in-tree-li-2025.pdf](./chain-in-tree-li-2025.pdf)
- **Citation Count:** 신규 논문 (인용 축적 중)

### 요약
트리 탐색 기반 추론(LITS)은 성능은 좋지만 단순 프롬프팅 대비 10~20배 느리다는 치명적 약점이 있습니다. Chain-in-Tree(CiT)는 "모든 단계에서 분기하지 말고, 분기가 정말 필요한 지점에서만 분기하라"는 아이디어를 플러그인 형태로 구현합니다. 분기 필요성(Branching Necessity)을 직접 프롬프팅(BN-DP)과 자기일관성(BN-SC)으로 평가하며, ToT·ReST-MCTS·RAP에 붙였을 때 GSM8K와 Math500에서 토큰·호출·실행시간을 75~85% 줄이면서 정확도 손실은 거의 없었습니다.

### 핵심 기여
- "언제 분기할 것인가"를 트리 탐색의 독립된 단계로 정식화하고 경량 평가법 2종(BN-DP, BN-SC) 제안
- BN-DP가 기존 대비 정책 모델 호출 수를 절대 늘리지 않음을 이론적으로 증명
- 2개 벤치마크 × 2개 백본(Qwen3-32B, LLaMA3-8B) × 3개 프레임워크, 총 14개 설정에서 대규모 검증
- ToT-BS·RAP·ReST-MCTS를 동일 인터페이스로 재구현한 통합 오픈소스 코드베이스 공개

### 이 논문이 중요한 이유
2023~2024년이 "추론 성능을 어떻게 올릴까"의 시대였다면, 2025~2026년은 "그 성능을 얼마에 살 수 있는가"의 시대입니다. CiT는 오늘 함께 읽는 ToT·ReAct·RAP·Self-Consistency를 모두 대상으로 삼아 비용 축을 최적화하기 때문에, 고전 논문들을 실제 제품에 옮길 때 반드시 마주치는 문제를 정확히 다룹니다. 또한 BN-SC가 14개 설정 중 1~4개에서 불안정했다는 솔직한 보고는, 자기일관성 기반 판단이 만능이 아님을 보여주는 실무적으로 값진 데이터입니다.

### 사전 지식
- Tree of Thoughts, RAP, MCTS 기반 추론의 기본 구조
- Self-Consistency 샘플링과 다수결 투표
- 정책(policy) / 보상 모델(reward model) / 전이 모델(transition model)로 LLM 역할을 분리하는 관점
- test-time scaling(추론 시점 연산 확장)의 개념과 비용 구조

### 관련 논문
- [Scaling LLM Test-Time Compute Optimally (Snell et al., 2024)](https://arxiv.org/abs/2408.03314)
- [Wider or Deeper? Adaptive Branching Tree Search (Inoue et al., 2025)](https://arxiv.org/abs/2503.04412)
- [A Survey on LLM Test-Time Compute via Search (Li, 2025)](https://arxiv.org/abs/2501.10069)

### 실무 적용
에이전트에 트리 탐색을 넣었더니 응답이 30초씩 걸리고 토큰 비용이 폭증하는 상황에 바로 쓸 수 있는 처방입니다. 구현은 어렵지 않습니다 — 각 추론 단계 직전에 "이 단계는 자명한가, 아니면 여러 경로를 봐야 하는가"를 작은 모델에게 한 번 묻고, 자명하면 그대로 이어 붙이고 애매할 때만 k개로 확장하면 됩니다. 비용의 75~85%를 줄인다는 것은 곧 같은 예산으로 탐색 깊이를 3~4배 늘릴 수 있다는 뜻이기도 합니다.

---

## 추천 읽기 순서

1. **Language Agent Tree Search (LATS)** — 먼저 읽으세요. 어제까지 배운 ToT·ReAct·Reflexion이 하나로 합쳐지는 그림을 보면 오늘의 나머지 두 논문이 왜 필요한지가 자연스럽게 드러납니다.
2. **Let's Verify Step by Step** — LATS의 가치 함수 자리에 무엇을 넣어야 하는가에 대한 답입니다. 탐색은 결국 채점 품질만큼만 좋아집니다.
3. **Chain-in-Tree** — 앞의 두 편을 실제 서비스에 올릴 때 부딪히는 비용 문제와 그 해법입니다.

즉 **탐색 구조 → 평가 신호 → 비용 최적화** 순서로, 하나의 시스템을 세 각도에서 보는 흐름입니다.

## 핵심 테이크어웨이

- 고급 프롬프팅의 본질은 문장 다듬기가 아니라 **탐색(search) + 평가(evaluation) + 예산 배분(budget)** 의 시스템 설계입니다.
- 탐색의 상한은 검증기가 결정합니다. 채점 신호가 나쁘면 아무리 넓게 탐색해도 잘못된 경로를 고르게 됩니다(Paper 1).
- 외부 환경 피드백은 LLM의 자기평가보다 훨씬 신뢰할 수 있는 보상 신호입니다. 실행 결과·테스트 통과 여부를 쓸 수 있다면 반드시 쓰세요(Paper 2).
- 모든 단계에서 분기할 필요는 없습니다. 추론 단계의 **75~85%는 자명하며**, 여기서 분기를 생략해도 정확도는 거의 유지됩니다(Paper 3).
- 자기일관성(self-consistency)은 강력하지만 만능이 아닙니다. 극단적으로 긴 추론 단계가 섞이면 불안정해질 수 있습니다.
- PM 관점: 추론 품질은 이제 모델 선택이 아니라 **추론 시점 예산(test-time budget)을 어떻게 배분하느냐**의 제품 결정 문제입니다.

## 다음 토픽과의 연결

내일은 **Automatic Prompt Optimization(자동 프롬프트 최적화)** 로 넘어갑니다. 오늘 배운 내용이 "사람이 설계한 탐색 구조 안에서 LLM이 더 잘 추론하게 만드는 법"이었다면, 내일은 그 프롬프트와 구조 자체를 **사람이 아니라 알고리즘이 찾게 하는 법**을 다룹니다. 오늘의 검증기(Paper 1)는 내일 APE·DSPy·TextGrad에서 프롬프트 후보를 채점하는 목적 함수로 그대로 재등장하고, 오늘의 분기 예산 문제(Paper 3)는 내일 "몇 개의 프롬프트 후보를 평가할 것인가"라는 탐색 예산 문제로 형태를 바꿔 다시 나타납니다.
