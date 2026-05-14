# Daily AI Paper Recommendations

> **Date:** 2026-05-14
> **Module:** Module 6 - LLM for Natural Language Generation
> **Topic:** Instruction Tuning and RLHF

---

## Paper 1 (Classic): Deep Reinforcement Learning from Human Preferences
- **Authors:** Paul Christiano, Jan Leike, Tom B. Brown, Miljan Martic, Shane Legg, Dario Amodei
- **Year:** 2017
- **arXiv:** https://arxiv.org/abs/1706.03741
- **PDF:** [./deep-rl-from-human-preferences-christiano-2017.pdf](./deep-rl-from-human-preferences-christiano-2017.pdf)
- **Citation Count:** ~4,500+

### 요약
명시적 보상 함수가 정의되기 어려운 복잡한 RL 문제에서, 인간이 두 trajectory segment 중 어느 쪽이 더 나은지 선택한 비교 데이터만으로 보상 모델을 학습하고, 그 보상 모델을 이용해 정책을 최적화하는 방법을 제안한다. Atari 게임과 시뮬레이션 로봇 제어에서 전체 상호작용의 1% 미만 데이터에만 인간 피드백을 받고도 학습 가능함을 보였다.

### 핵심 기여
- 인간의 pairwise preference 데이터로부터 latent 보상 함수를 학습하는 framework 정립
- 보상 모델 학습과 정책 학습을 비동기적으로(asynchronously) 수행하여 인간 피드백 비용을 크게 절감
- 명시적 보상 정의가 어려운 backflip, 자동차 운전 등 복잡한 novel behavior를 1시간 분량의 인간 피드백만으로 학습 가능함을 실증

### 이 논문이 중요한 이유
오늘날 ChatGPT, Claude, Gemini 등 모든 주요 LLM의 alignment 파이프라인의 출발점이 되는 논문이다. "보상 함수를 직접 설계하지 않고, 인간 선호로부터 학습한다"는 RLHF의 핵심 아이디어가 여기서 정립되었다. InstructGPT와 ChatGPT의 RLHF 단계는 본 논문의 framework를 LLM 도메인에 그대로 적용한 것이며, 이후 DPO, RLAIF, Constitutional AI 등 모든 alignment 연구의 출발점이다. AI 엔지니어가 alignment 파이프라인을 이해하기 위해 반드시 읽어야 하는 origin paper다.

### 사전 지식
- 강화학습 기초 (정책, 가치함수, PPO/TRPO)
- Bradley-Terry 모델과 pairwise comparison 학습
- 딥러닝 기초 (지도학습 손실, 신경망 학습)
- 시뮬레이션 환경 (MuJoCo, Atari) 개념

### 관련 논문
- [Fine-Tuning Language Models from Human Preferences (Ziegler et al., 2019)](https://arxiv.org/abs/1909.08593)
- [Learning to summarize from human feedback (Stiennon et al., 2020)](https://arxiv.org/abs/2009.01325)
- [Training language models to follow instructions / InstructGPT (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155)

### 실무 적용
사내 챗봇이나 코드 어시스턴트를 만들 때, "정답"이 명확하지 않은 응답 품질(친절함, 안전성, 길이 적절성)에 대해 사용자 클릭/좋아요/A-B 비교 데이터를 모아 reward model을 학습하고, 이를 기반으로 SFT 모델을 PPO/DPO로 추가 튜닝하는 표준 파이프라인의 이론적 토대가 된다. 또한 게임 NPC 행동, 로봇 동작 등 task-specific 보상 설계가 어려운 도메인에서도 human-in-the-loop 학습으로 활용 가능하다.

---

## Paper 2 (Classic): Constitutional AI: Harmlessness from AI Feedback
- **Authors:** Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, et al. (Anthropic)
- **Year:** 2022
- **arXiv:** https://arxiv.org/abs/2212.08073
- **PDF:** [./constitutional-ai-bai-2022.pdf](./constitutional-ai-bai-2022.pdf)
- **Citation Count:** ~2,500+

### 요약
인간이 모든 harmful 응답을 직접 라벨링하지 않고, 짧은 원칙(constitution) 목록만 제공해도 AI가 스스로 자기 응답을 비평·수정하며 harmless하게 학습할 수 있음을 보인 논문이다. SL-CAI(Self-Critique)와 RL-CAI(RLAIF) 두 단계로 구성되어, RLHF의 인간 라벨을 AI 라벨로 대체하면서도 helpfulness를 유지하는 방법을 제시했다.

### 핵심 기여
- **Constitution 기반 self-critique:** 모델이 자신의 응답을 constitution 원칙에 비추어 비평하고 수정한 데이터로 SFT 학습 (SL-CAI 단계)
- **RLAIF (RL from AI Feedback):** 두 응답 중 어느 쪽이 더 harmless한지 AI가 판단한 preference 데이터로 reward model을 학습하고 PPO 적용
- **Helpfulness-Harmlessness trade-off 개선:** 기존 RLHF는 harmlessness가 높아질수록 evasive(회피적)해지는 문제가 있었으나, CAI는 거부 대신 "이유 있는 설명" 방식으로 응답하도록 학습되어 helpful 유지
- **Chain-of-Thought 추론:** harmlessness 판단에 CoT를 결합하여 투명성과 성능 동시 향상

### 이 논문이 중요한 이유
Claude 모델의 핵심 alignment 기법이며, RLHF의 가장 큰 비용 병목인 "인간 라벨러 의존성"을 AI 자체로 대체할 수 있음을 입증한 패러다임 전환적 연구다. RLAIF는 이후 self-rewarding LM, AI-generated preference 등 수많은 후속 연구의 시초가 되었고, 산업계에서 alignment를 확장 가능한 비용 구조로 만드는 핵심 방법론이다. 또한 "원칙 기반 alignment"라는 새 패러다임을 제시하여, 단순 reward maximization 너머의 가치 정렬 논의를 본격화했다.

### 사전 지식
- RLHF 기본 파이프라인 (SFT → Reward Model → PPO)
- InstructGPT (Ouyang et al., 2022)의 흐름
- Chain-of-Thought prompting 개념
- Helpful & Harmless assistant 평가 framework

### 관련 논문
- [Training a Helpful and Harmless Assistant with RLHF (Bai et al., 2022)](https://arxiv.org/abs/2204.05862)
- [RLAIF vs. RLHF: Scaling Reinforcement Learning from Human Feedback with AI Feedback (Lee et al., 2023)](https://arxiv.org/abs/2309.00267)
- [Self-Rewarding Language Models (Yuan et al., 2024)](https://arxiv.org/abs/2401.10020)

### 실무 적용
B2B SaaS에서 도메인별 안전 정책(의료 정보 disclaimer, 금융 규제 준수, 아동 보호 등)을 constitution으로 정의하면, 인간 안전팀의 라벨링 부담 없이 모델을 정책에 맞게 정렬할 수 있다. Anthropic, OpenAI, Google 등 주요 LLM 제공자가 채택한 방법으로, 자체 모델을 fine-tune하는 경우 RLAIF 파이프라인을 구축하여 cost-effective하게 안전성을 강화할 수 있다. Agentic AI 제품에서 tool use 안전성 확보에도 유용하다.

---

## Paper 3 (Recent): SimPO: Simple Preference Optimization with a Reference-Free Reward
- **Authors:** Yu Meng, Mengzhou Xia, Danqi Chen (Princeton NLP)
- **Year:** 2024 (NeurIPS 2024)
- **arXiv:** https://arxiv.org/abs/2405.14734
- **PDF:** [./simpo-meng-2024.pdf](./simpo-meng-2024.pdf)
- **Citation Count:** ~600+

### 요약
DPO를 단순화하면서도 성능을 끌어올린 alignment 알고리즘. 시퀀스의 평균 log-probability를 implicit reward로 사용하고 reference model을 완전히 제거하여, 연산·메모리 비용을 줄였다. 또한 Bradley-Terry 목적함수에 target reward margin을 도입해 winning/losing 응답 간 마진을 명시적으로 키운다. Gemma-2-9B-it-SimPO는 AlpacaEval 2에서 72.4% LC win rate를 달성했다.

### 핵심 기여
- **Reference-free 보상:** DPO의 reference model 의존성을 제거하여 학습 시 메모리 ~50% 절감, 학습 속도 향상
- **Length-normalized reward:** 평균 log-prob을 보상으로 사용해 length bias를 자연스럽게 완화 (DPO의 만성적 문제 해결)
- **Target margin γ:** Bradley-Terry loss에 마진 항을 추가하여 win/lose 응답 간 분리를 명시적으로 유도
- **벤치마크 SOTA:** AlpacaEval 2에서 DPO 대비 최대 +6.4, Arena-Hard에서 최대 +7.5 포인트 향상

### 이 논문이 중요한 이유
DPO 이후 alignment 알고리즘은 단순성·효율성·성능 세 축에서 빠르게 진화 중인데, SimPO는 그 흐름의 대표 사례다. Reference model 제거는 (1) 학습 인프라 비용 감소, (2) 메모리 절약으로 더 큰 모델 fine-tune 가능, (3) 구현 단순화 등 산업적 이점이 크다. 또한 length bias 문제를 손실 함수 설계로 해결한 점은 RLHF 평가 메트릭(특히 LC win rate)의 의미를 다시 생각하게 만든다. 2024년 이후 ORPO, KTO, IPO와 함께 "post-DPO" 알고리즘 군의 표준 비교 대상이 되었다.

### 사전 지식
- DPO (Direct Preference Optimization) 수식과 reference model의 역할
- Bradley-Terry preference model
- AlpacaEval 2, Arena-Hard, MT-Bench 평가 방법
- Length bias 문제와 length-controlled win rate

### 관련 논문
- [Direct Preference Optimization / DPO (Rafailov et al., 2023)](https://arxiv.org/abs/2305.18290)
- [KTO: Model Alignment as Prospect Theoretic Optimization (Ethayarajh et al., 2024)](https://arxiv.org/abs/2402.01306)
- [ORPO: Monolithic Preference Optimization without Reference Model (Hong et al., 2024)](https://arxiv.org/abs/2403.07691)

### 실무 적용
사내 모델 alignment 비용을 줄이고 싶을 때 가장 먼저 시도해볼 알고리즘이다. Reference model 없이 학습 가능하므로 단일 GPU 노드에서도 더 큰 base model에 적용할 수 있고, length bias 완화로 사용자 평가(특히 짧고 정확한 응답 선호)에 더 자연스럽게 부합한다. Agentic AI에서 tool-call 응답이나 RAG 답변 품질을 사람 피드백으로 정렬할 때, DPO/SimPO를 빠르게 A/B 비교하여 자사 도메인에 맞는 알고리즘을 선택하는 것이 권장된다. HuggingFace TRL 라이브러리에 구현되어 있어 즉시 사용 가능하다.

---

## 추천 읽기 순서
1. **Christiano et al. (2017)** — RLHF의 origin paper. Preference learning의 수학적 framework와 직관을 먼저 잡는다.
2. **Bai et al. (2022) Constitutional AI** — 인간 라벨을 AI 라벨로 대체하는 패러다임 전환을 이해. RLHF가 scale-up되는 방향을 본다.
3. **Meng et al. (2024) SimPO** — DPO 계열의 최신 효율화 흐름. 실무에서 즉시 활용 가능한 알고리즘 디테일까지 학습.

이 순서는 "이론적 기원 → 패러다임 확장 → 실무 효율화"의 자연스러운 흐름을 따른다.

## 핵심 테이크어웨이
- **Preference learning은 명시적 보상 함수보다 강력하다.** 인간이 선호를 비교하는 것이 보상을 직접 설계하는 것보다 훨씬 쉽고 확장 가능하다.
- **RLHF의 비용 병목은 인간 라벨링이다.** Constitutional AI/RLAIF는 이 병목을 AI로 대체하며, 원칙 기반 alignment라는 새 패러다임을 열었다.
- **DPO 이후 alignment는 단순성·효율성 경쟁 중이다.** SimPO처럼 reference model을 제거하고 length bias를 완화하는 등 손실 함수 설계가 빠르게 진화하고 있다.
- **Alignment는 더 이상 RL만의 영역이 아니다.** Offline preference optimization(DPO, SimPO, KTO 등)이 산업 표준으로 자리잡고 있다.
- **Trade-off는 여전히 존재한다.** Helpfulness vs harmlessness, in-distribution 성능 vs OOD generalization, reward hacking 방지 등 미해결 문제가 많다.

## 다음 토픽과의 연결
다음 주제는 **"LLM Evaluation and Benchmarks"**이다. 오늘 학습한 alignment 알고리즘들(InstructGPT, CAI, DPO, SimPO 등)의 성능을 어떻게 측정할 것인가가 자연스러운 다음 질문이다. AlpacaEval 2, Arena-Hard, MMLU-Pro, LiveBench 같은 현대적 벤치마크는 모두 RLHF/DPO로 정렬된 모델의 강점과 약점을 드러내기 위해 설계된 것이며, length bias, contamination, judge model 신뢰성 등의 문제는 SimPO의 length-normalized reward와 직접 연결된다. 또한 reward model 자체의 평가(RewardBench)도 RLHF 파이프라인 품질을 좌우하는 핵심 주제다.
