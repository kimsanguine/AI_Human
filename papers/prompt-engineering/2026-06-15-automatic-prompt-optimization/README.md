# Daily AI Paper Recommendations

> **Date:** 2026-06-15
> **Module:** Module 7: Prompt Engineering
> **Topic:** Automatic Prompt Optimization

---

## Paper 1 (Classic): Large Language Models as Optimizers (OPRO)
- **Authors:** Chengrun Yang, Xuezhi Wang, Yifeng Lu, Hanxiao Liu, Quoc V. Le, Denny Zhou, Xinyun Chen
- **Year:** 2023
- **arXiv:** [https://arxiv.org/abs/2309.03409](https://arxiv.org/abs/2309.03409)
- **PDF:** [./opro-llms-as-optimizers-yang-2023.pdf](./opro-llms-as-optimizers-yang-2023.pdf)
- **Citation Count:** approx. 1,400+

### 요약
OPRO(Optimization by PROmpting)는 LLM 자체를 옵티마이저로 사용해 최적화 문제를 자연어로 기술하고 푸는 방법이다. 매 스텝마다 LLM은 "지금까지 생성된 해(solution)와 그 점수"가 담긴 프롬프트를 보고 더 나은 해를 새로 생성하며, 이를 평가해 다시 프롬프트에 누적하는 반복 과정을 거친다. 프롬프트 최적화에 적용하면 사람이 직접 만든 프롬프트보다 GSM8K에서 최대 8%, Big-Bench Hard에서 최대 50%까지 정확도를 높인다.

### 핵심 기여
- 그래디언트 없이 자연어 설명만으로 최적화를 수행하는 OPRO 프레임워크를 제안 (선형회귀·외판원 문제 같은 수학 최적화도 시연)
- "이전 해 + 점수"를 정렬해 보여주는 메타 프롬프트(meta-prompt) 설계로 LLM이 개선 방향을 학습하도록 유도
- "Take a deep breath and work on this problem step by step" 같은 비직관적이지만 강력한 프롬프트를 자동 발견, 프롬프트 최적화에서 인간 수준을 능가함을 입증

### 이 논문이 중요한 이유
프롬프트 엔지니어링을 사람의 직관과 수작업에서 "탐색·평가·반복"이라는 자동 최적화 루프로 전환한 대표 논문이다. AI 엔지니어가 LLM을 단순 생성기가 아니라 블랙박스 옵티마이저로 활용하는 사고방식을 익히는 데 핵심이며, 이후 거의 모든 자동 프롬프트 최적화 연구가 OPRO를 비교 기준으로 삼는다.

### 사전 지식
- 그래디언트 기반 최적화와 그래디언트-프리(블랙박스) 최적화의 차이
- In-context learning과 few-shot 프롬프트의 기본 동작
- GSM8K, Big-Bench Hard 등 reasoning 벤치마크의 개념

### 관련 논문
- [Large Language Models Are Human-Level Prompt Engineers (Zhou et al., 2022)](https://arxiv.org/abs/2211.01910)
- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models (Wei et al., 2022)](https://arxiv.org/abs/2201.11903)

### 실무 적용
프로덕션에서 시스템 프롬프트·few-shot 예시를 사람이 A/B로 수작업 튜닝하는 대신, 평가셋과 점수 함수만 정의해두면 OPRO 루프로 프롬프트를 자동 개선할 수 있다. 에이전트의 도구 사용 지시문, RAG 질의 재작성 프롬프트, 분류 태스크 instruction 자동 탐색 등에 직접 적용된다.

---

## Paper 2 (Classic): RLPrompt — Optimizing Discrete Text Prompts with Reinforcement Learning
- **Authors:** Mingkai Deng, Jianyu Wang, Cheng-Ping Hsieh, Yihan Wang, Han Guo, Tianmin Shu, Meng Song, Eric P. Xing, Zhiting Hu
- **Year:** 2022
- **arXiv:** [https://arxiv.org/abs/2205.12548](https://arxiv.org/abs/2205.12548)
- **PDF:** [./rlprompt-deng-2022.pdf](./rlprompt-deng-2022.pdf)
- **Citation Count:** approx. 700+

### 요약
RLPrompt는 강화학습(RL)으로 이산(discrete) 텍스트 프롬프트를 자동 최적화하는 방법이다. 작은 정책 네트워크가 프롬프트 토큰을 생성하고, 다운스트림 태스크 성능을 보상으로 받아 학습한다. 보상 안정화(reward stabilization) 기법으로 학습 효율을 크게 높였고, BERT 같은 마스킹 모델과 GPT 같은 좌→우 모델 모두에, 분류·생성 태스크에 두루 적용된다.

### 핵심 기여
- 연속 임베딩이 아닌 "사람이 읽을 수 있는 토큰 단위"의 이산 프롬프트를 RL로 직접 최적화하는 프레임워크 제안
- 희소·불안정한 보상을 정규화하는 reward stabilization으로 학습 안정성과 효율을 동시에 개선
- 최적 프롬프트가 종종 비문법적 "gibberish"임에도 서로 다른 LM 간 전이(transfer)되어 성능을 유지함을 발견 — LM 프롬프트가 인간 언어 패턴을 따르지 않을 수 있음을 시사

### 이 논문이 중요한 이유
OPRO가 LLM에게 프롬프트를 "글로 다시 쓰게" 한다면, RLPrompt는 보상 신호로 토큰을 직접 학습시키는 정반대 축의 접근이다. 두 방법을 함께 보면 자동 프롬프트 최적화의 설계 공간(LLM 기반 탐색 vs. 그래디언트/RL 기반 최적화)을 입체적으로 이해할 수 있다. 또한 "프롬프트가 꼭 자연어일 필요는 없다"는 통찰은 프롬프트의 본질을 다시 생각하게 한다.

### 사전 지식
- 강화학습 기본(정책, 보상, policy gradient)
- soft prompt / discrete prompt의 차이
- 마스킹 LM(BERT)과 autoregressive LM(GPT)의 동작 차이

### 관련 논문
- [AutoPrompt: Eliciting Knowledge from Language Models with Automatically Generated Prompts (Shin et al., 2020)](https://arxiv.org/abs/2010.15980)
- [The Power of Scale for Parameter-Efficient Prompt Tuning (Lester et al., 2021)](https://arxiv.org/abs/2104.08691)

### 실무 적용
라벨이 적은 few-shot 분류나 스타일 전이 태스크에서, 파인튜닝 없이 RL로 태스크 특화 프롬프트를 학습해 비용을 절감할 수 있다. 학습된 프롬프트가 모델 간 전이된다는 점은, 작은 모델에서 탐색한 프롬프트를 큰 모델에 재사용하는 비용 효율 전략으로 활용 가능하다.

---

## Paper 3 (Recent): Optimizing Instructions and Demonstrations for Multi-Stage Language Model Programs (MIPRO)
- **Authors:** Krista Opsahl-Ong, Michael J. Ryan, Josh Purtell, David Broman, Christopher Potts, Matei Zaharia, Omar Khattab
- **Year:** 2024
- **arXiv:** [https://arxiv.org/abs/2406.11695](https://arxiv.org/abs/2406.11695)
- **PDF:** [./mipro-optimizing-instructions-demonstrations-opsahl-ong-2024.pdf](./mipro-optimizing-instructions-demonstrations-opsahl-ong-2024.pdf)
- **Citation Count:** approx. 200+

### 요약
MIPRO(Multi-prompt Instruction PRoposal Optimizer)는 여러 LLM 호출이 연결된 다단계 "언어모델 프로그램(LM program)"에서, 모듈별 라벨이나 그래디언트 없이 각 모듈의 instruction과 few-shot 데모를 함께 최적화하는 알고리즘이다. 문제를 (1) 효과적인 instruction 제안, (2) 데모 선택으로 분해하고, 베이지안 surrogate 모델과 미니배치 평가로 탐색을 효율화한다. 오픈소스 Llama-3-8B에서 7개 다단계 프로그램 중 5개에서 베이스라인 대비 최대 13% 정확도 향상을 달성했고, 결과는 DSPy에 공개되었다.

### 핵심 기여
- 단일 프롬프트가 아닌 "다단계 LM 파이프라인 전체"의 프롬프트를 최적화하는 문제를 정식화하고, instruction과 데모를 분리 최적화하는 전략 제시
- 프로그램·데이터를 인지하는(program- and data-aware) instruction 제안 기법과, 모듈 간 credit assignment를 다루는 메타 최적화 절차 도입
- 확률적 미니배치 평가로 목적함수의 surrogate 모델을 학습해 탐색 비용을 절감, DSPy에 옵티마이저와 벤치마크를 오픈소스로 제공

### 이 논문이 중요한 이유
실무의 LLM 시스템은 단일 프롬프트가 아니라 RAG·툴 호출·다단계 추론이 엮인 파이프라인이다. MIPRO는 OPRO류의 단일 프롬프트 최적화를 "컴파운드 AI 시스템" 단위로 확장한 대표 연구로, 프롬프트 최적화를 프로그래밍·컴파일 관점에서 보는 DSPy 생태계의 핵심 옵티마이저다. AI 엔지니어가 프롬프트를 수작업이 아닌 "컴파일 대상"으로 다루게 해준다.

### 사전 지식
- DSPy의 모듈/시그니처/컴파일 개념
- few-shot 데모 선택과 instruction tuning의 차이
- 베이지안 최적화·surrogate 모델의 기본 아이디어

### 관련 논문
- [DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines (Khattab et al., 2023)](https://arxiv.org/abs/2310.03714)
- [Large Language Models as Optimizers (Yang et al., 2023)](https://arxiv.org/abs/2309.03409)

### 실무 적용
RAG 파이프라인이나 멀티스텝 에이전트에서 각 단계의 프롬프트를 손으로 튜닝하던 작업을, 평가셋만 준비하면 `dspy.MIPROv2`로 자동 컴파일할 수 있다. 모델 교체(예: GPT→오픈소스 Llama) 시 프롬프트를 새 모델에 맞춰 재최적화하는 마이그레이션 비용을 크게 줄여준다.

---

## 추천 읽기 순서
1. **OPRO (2023)** — "LLM을 옵티마이저로 본다"는 핵심 직관을 먼저 잡는다.
2. **RLPrompt (2022)** — 같은 목표를 RL/이산 최적화라는 다른 축으로 접근, 설계 공간을 넓힌다.
3. **MIPRO (2024)** — 단일 프롬프트에서 다단계 파이프라인 최적화로 확장되는 흐름을 본다.

## 핵심 테이크어웨이
- 자동 프롬프트 최적화의 본질은 **"탐색(propose) → 평가(score) → 반복(refine)"** 루프이며, 탐색 엔진을 LLM(OPRO/MIPRO)으로 두느냐 RL 정책(RLPrompt)으로 두느냐가 핵심 설계 선택이다.
- 최적 프롬프트는 인간 직관과 다를 수 있다 (OPRO의 "deep breath", RLPrompt의 gibberish 전이) — 사람이 만든 프롬프트가 최적이라는 가정을 버려야 한다.
- 실무 LLM 시스템은 단일 프롬프트가 아니라 다단계 파이프라인이므로, MIPRO처럼 **시스템 전체를 컴파일 대상으로** 보는 관점이 프로덕션에 직접적이다.

## 다음 토픽과의 연결
다음 모듈(Module 8: LangChain & LLM Orchestration)에서 다루는 다단계 에이전트·툴 사용 파이프라인은, 바로 MIPRO가 최적화 대상으로 삼는 "LM program"이다. 오늘 배운 자동 프롬프트 최적화는 오케스트레이션된 시스템의 각 노드 품질을 끌어올리는 도구로 이어진다.
