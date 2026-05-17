# Daily AI Paper Recommendations

> **Date:** 2026-05-18
> **Module:** Module 7 — Prompt Engineering
> **Topic:** Advanced Prompting — ToT / ReAct / Self-Consistency

---

## Paper 1 (Classic): Self-Consistency Improves Chain of Thought Reasoning in Language Models
- **Authors:** Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, Denny Zhou
- **Year:** 2022 (ICLR 2023)
- **arXiv:** https://arxiv.org/abs/2203.11171
- **PDF:** [./self-consistency-wang-2022.pdf](./self-consistency-wang-2022.pdf)
- **Citation Count:** 4,500+ (Google Scholar)

### 요약
복잡한 추론 문제는 정답에 도달하는 경로가 여러 갈래일 수 있다는 직관에서 출발한 디코딩 전략 논문이다. 기존 Chain-of-Thought(CoT)의 그리디 디코딩을 버리고, 다양한 추론 경로를 샘플링한 뒤 답변을 다수결(marginalize)로 결정하는 **Self-Consistency**를 제안한다. 추가 학습이나 별도 verifier 없이 GSM8K +17.9%, SVAMP +11.0% 등 산술/상식 추론 벤치마크에서 큰 폭의 성능 향상을 보였다.

### 핵심 기여
- 다양한 추론 경로(reasoning paths)를 샘플링하고, 답변을 다수결로 선택하는 단순하지만 강력한 디코딩 전략 제안
- 추가 학습, 보조 모델, fine-tuning 없이 LLM 위에 그대로 얹을 수 있는 **inference-time scaling**의 초기 정석
- 모델 크기와 무관하게 효과가 일관되며, 자유형식 텍스트 답변에도 적용 가능함을 실험적으로 입증

### 이 논문이 중요한 이유
지금 우리가 당연하게 쓰는 "여러 번 생성해서 다수결로 고르기"라는 패턴의 원형이다. AI 엔지니어 입장에서는 "추론 능력은 학습이 아니라 디코딩 단계에서도 끌어올릴 수 있다"는 인식의 전환점을 만든 논문이다. 2024–2025년 OpenAI o1, DeepSeek-R1로 대표되는 **test-time compute scaling** 흐름의 출발점이며, 비용 대비 정확도를 다이얼처럼 조절할 수 있다는 실용적 함의가 크다.

### 사전 지식
- Chain-of-Thought (Wei et al., 2022) 프롬프팅의 기본 동작 방식
- Greedy decoding vs. temperature sampling 차이
- Few-shot in-context learning의 기본 구조
- 산술/상식 추론 벤치마크(GSM8K, SVAMP, CommonsenseQA)에 대한 감각

### 관련 논문
- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models (Wei et al., 2022)](https://arxiv.org/abs/2201.11903)
- [Tree of Thoughts: Deliberate Problem Solving with LLMs (Yao et al., 2023)](https://arxiv.org/abs/2305.10601)
- [Self-Refine: Iterative Refinement with Self-Feedback (Madaan et al., 2023)](https://arxiv.org/abs/2303.17651)

### 실무 적용
- 수학/코드/JSON 생성처럼 정답이 검증 가능한(verifiable) 태스크에서, `n=5~40` 샘플 다수결만으로 정확도 큰 폭 향상
- Agentic AI 워크플로우에서 "툴 호출 시그니처가 일치하는 답이 다수인가"를 합의 기준으로 활용 (예: SQL 쿼리, 함수 인자)
- A/B 테스트에서 latency-비용-정확도 trade-off를 다이얼링하는 표준 레버. p95 응답시간을 깨지 않는 선에서 샘플 수를 늘려 품질 SLA를 확보하는 용도로 즉시 적용 가능
- RAG 평가/검수에서 동일 질의를 여러 번 돌려 답이 안정적인지 보는 "consistency score" 디자인의 이론적 근거

---

## Paper 2 (Classic): Reflexion: Language Agents with Verbal Reinforcement Learning
- **Authors:** Noah Shinn, Federico Cassano, Edward Berman, Ashwin Gopinath, Karthik Narasimhan, Shunyu Yao
- **Year:** 2023 (NeurIPS 2023)
- **arXiv:** https://arxiv.org/abs/2303.11366
- **PDF:** [./reflexion-shinn-2023.pdf](./reflexion-shinn-2023.pdf)
- **Citation Count:** 2,000+ (Google Scholar)

### 요약
가중치 업데이트 없이 **언어적 피드백(verbal feedback)** 만으로 LLM 에이전트를 강화학습시키는 프레임워크다. 에이전트는 환경에서 받은 보상/실패 신호를 **자연어 reflection**으로 변환해 에피소드 메모리에 저장하고, 다음 시도에서 컨텍스트로 활용한다. HumanEval(코드), AlfWorld(의사결정), HotpotQA(질문응답) 등 다양한 태스크에서 ReAct 대비 큰 폭의 성능 향상을 보였다.

### 핵심 기여
- "Actor – Evaluator – Self-Reflection"의 3-모듈 구조로, LLM이 자신의 실패를 텍스트로 진단하고 다음 trial에 반영하는 루프 정식화
- Weight update 없이 **메모리만으로** 학습 효과를 모사하는, RLHF 대비 극도로 가벼운 self-improvement 패러다임 제시
- 스칼라/이진/자유형식 등 어떤 종류의 피드백이든 받아 처리할 수 있는 **modality-agnostic** 설계

### 이 논문이 중요한 이유
오늘날 거의 모든 Agentic AI 제품(Devin, Cursor Composer, Claude Code, OpenAI Codex 등)의 "실패 → 자가 반성 → 재시도" 루프의 직계 조상이다. AI 엔지니어 입장에서 "추론 품질을 높이려면 더 큰 모델"이라는 한계를 넘어, **에이전트 루프 설계**가 곧 제품 성능이라는 관점을 확립했다. CPO 관점에서도 "verifier 또는 critic 노드를 어디에 둘 것인가"라는 워크플로우 설계의 기준이 된다.

### 사전 지식
- ReAct (Yao et al., 2022) 의 Thought–Action–Observation 루프
- LangGraph / OpenAI Agents SDK 류의 에이전트 그래프 개념
- 단기/장기/에피소드 메모리(short-term, episodic, long-term)의 구분
- Code execution sandbox나 unit test 기반 verifier의 동작

### 관련 논문
- [ReAct: Synergizing Reasoning and Acting in Language Models (Yao et al., 2022)](https://arxiv.org/abs/2210.03629)
- [Self-Refine: Iterative Refinement with Self-Feedback (Madaan et al., 2023)](https://arxiv.org/abs/2303.17651)
- [Voyager: An Open-Ended Embodied Agent with LLMs (Wang et al., 2023)](https://arxiv.org/abs/2305.16291)

### 실무 적용
- 코드 생성 에이전트의 **실패 → unit test 결과를 reflection으로 변환 → 재생성** 루프가 가장 직접적 적용
- 고객 지원 봇이 "답변 → CSAT/escalation 신호를 자연어로 회고 → 다음 응답에 반영"하는 self-improving 운영
- RAG 시스템에서 답변이 근거(citation)와 어긋났을 때, 검색-생성 루프를 reflection으로 재기동하는 Corrective RAG의 기반
- Agentic AI 제품에서 비용 통제 관점에서 중요: **fine-tuning 없이** 모델 변경에 무관한 self-improvement 레이어를 둘 수 있어 모델 교체 비용이 낮음

---

## Paper 3 (Recent): DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning
- **Authors:** DeepSeek-AI (Daya Guo, Dejian Yang, Haowei Zhang, et al.)
- **Year:** 2025 (arXiv Jan 2025, Nature 2025)
- **arXiv:** https://arxiv.org/abs/2501.12948
- **PDF:** [./deepseek-r1-deepseek-ai-2025.pdf](./deepseek-r1-deepseek-ai-2025.pdf)
- **Citation Count:** 3,000+ (2025년 상반기 기준, 가장 빠르게 인용 누적된 LLM 논문 중 하나)

### 요약
사람이 라벨링한 reasoning trajectory 없이도, **순수 강화학습(GRPO, rule-based reward)** 만으로 LLM이 self-reflection·verification·long CoT 같은 고급 추론 패턴을 스스로 학습할 수 있음을 보인 논문이다. DeepSeek-R1-Zero(SFT 없이 RL만)와 DeepSeek-R1(cold-start SFT + 다단계 RL) 두 모델을 공개했으며, AIME 2024 pass@1을 15.6% → 71.0%(다수결 86.7%)로 끌어올려 OpenAI-o1 수준에 도달했다.

### 핵심 기여
- **RL-only로 reasoning이 emergent**하다는 실증 — SFT 없이도 long chain-of-thought, self-correction, "aha moment"가 자발적으로 발현
- **GRPO(Group Relative Policy Optimization)** — value model을 제거하고 group-relative baseline을 쓰는 가벼운 PPO 변형. 학습 효율과 안정성을 동시에 확보
- **Open-weight reasoning model 시대의 개막** — Qwen/Llama 기반 distilled 모델(1.5B~70B)을 함께 공개해 reasoning 능력의 민주화 촉진

### 이 논문이 중요한 이유
2024년의 OpenAI o1이 "test-time compute scaling"이 가능함을 보였다면, DeepSeek-R1은 그 레시피를 **재현 가능한 형태로 공개**한 논문이다. 프롬프트 엔지니어링의 관점에서 본 시사점이 크다: **모델 자체가 self-consistency, reflection, ToT 같은 패턴을 내재화**하면, 우리가 외부에서 짜던 복잡한 프롬프트 체인이 더 이상 필요 없거나 단순화된다. Agentic AI 제품 리딩 입장에서는 "어디까지 모델에게 맡기고, 어디부터 오케스트레이션할 것인가"의 경계선을 다시 그려야 함을 의미한다.

### 사전 지식
- PPO / RLHF의 기본 구조와 reward model의 역할
- Chain-of-Thought, Self-Consistency, ToT의 차이
- Pass@k, AIME, MATH, Codeforces 등 reasoning 벤치마크
- Knowledge distillation의 기본 개념(student–teacher)

### 관련 논문
- [Self-Consistency Improves Chain of Thought Reasoning (Wang et al., 2022)](https://arxiv.org/abs/2203.11171)
- [Direct Preference Optimization (Rafailov et al., 2023)](https://arxiv.org/abs/2305.18290)
- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning (Shao et al., 2024)](https://arxiv.org/abs/2402.03300)
- [OpenAI o1 System Card (OpenAI, 2024)](https://cdn.openai.com/o1-system-card.pdf)

### 실무 적용
- **모델 선택 전략 재편**: 추론 집약 워크로드(코드, 수학, 분석)에서는 R1-distill 7B/32B로도 o1-preview 수준 품질이 나오므로, GPT-4 클래스 모델 대비 추론 비용 5~10배 절감 가능
- **프롬프트 단순화**: ToT/Self-Consistency를 외부에서 구현하던 코드를 걷어내고, system prompt에서 "think step by step"만으로 충분한 케이스가 증가
- **사내 fine-tuning 파이프라인**: GRPO + rule-based reward(unit test pass, JSON schema 일치, 정규식 매칭)는 RLHF 대비 라벨링 비용이 거의 0에 가까워, AI 네이티브 스타트업의 자체 reasoning 모델 학습 진입장벽을 크게 낮춤
- **Latency–품질 트레이드오프**: long CoT는 응답이 느려지므로, 제품 UX에서 "thinking" 토큰을 스트리밍으로 보여주는(예: ChatGPT, Claude의 thinking 모드) UI 패턴이 필수

---

## 추천 읽기 순서

1. **Self-Consistency (Wang, 2022)** — "여러 번 뽑아서 다수결"이라는 가장 단순한 inference-time scaling의 출발점. 30분이면 핵심 직관 파악 가능.
2. **Reflexion (Shinn, 2023)** — Self-Consistency가 "병렬 다양화"라면, Reflexion은 "직렬 자기교정". 둘을 비교하며 읽으면 추론 향상의 두 축이 명확해진다.
3. **DeepSeek-R1 (DeepSeek-AI, 2025)** — 위 두 패턴이 "외부 프롬프트 트릭"에서 "모델 내재화"로 어떻게 옮겨갔는지 추적. GRPO 파트는 부록까지 정독 권장.

## 핵심 테이크어웨이

- **추론 품질은 학습 시점뿐 아니라 디코딩/에이전트 루프에서도 만든다.** Self-Consistency는 디코딩에서, Reflexion은 에이전트 루프에서, DeepSeek-R1은 학습에서 같은 목표를 푼다.
- **"외부에서 짜던 패턴이 모델로 흡수된다"는 추세를 직시해야 한다.** 2026년 시점의 프롬프트 엔지니어링은 더 이상 복잡한 사슬을 짜는 일이 아니라, **어떤 추론을 모델에 맡기고 어떤 검증을 외부에 둘지를 결정하는 시스템 설계**다.
- **Verifiable reward가 있는 곳에서 가장 큰 가치가 나온다.** 코드, 수학, JSON, SQL처럼 정답 판정이 자동화 가능한 도메인에서 Self-Consistency/Reflexion/RL-driven reasoning이 모두 가장 잘 동작한다. 제품 백로그를 짤 때 우선순위 판단 기준으로 쓸 수 있다.
- **Cost dial이 늘었다.** 같은 모델에서 샘플 수, reflection 횟수, thinking budget 토큰 수가 모두 품질-비용 다이얼이 된다. 그로스 마케팅 관점에서 "free tier → paid tier" 전환 시 차별화 포인트로 설계 가능.

## 다음 토픽과의 연결

- **Module 7 Day 20 — Automatic Prompt Optimization**: 오늘 본 3편이 "사람이 짜는 패턴"이라면, 다음 주제는 그 패턴을 **자동으로 발견·최적화**하는 DSPy, TextGrad, APE 계열로 이어진다. Self-Consistency / Reflexion의 외부 패턴이 모델 안으로 들어가는 흐름과, 프롬프트 설계가 자동화되는 흐름이 만나는 지점이 다음 강의의 주제다.
- **Module 8 — AI Agents and Tool Use**: Reflexion의 self-reflection 루프는 차후 강의에서 다룰 Tool-Use, Planning, Multi-Agent Debate의 직접적 전신이다.
- **Module 9 — Advanced RAG (Self-RAG, CRAG)**: "검색 → 생성 → 검증 → 재검색"의 corrective 루프 설계는 Reflexion의 verbal reinforcement 아이디어를 RAG 도메인에 적용한 것이다.
