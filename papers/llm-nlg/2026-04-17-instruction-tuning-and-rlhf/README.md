# Daily AI Paper Recommendations

> **Date:** 2026-04-17
> **Module:** LLM for Natural Language Generation
> **Topic:** Instruction Tuning and RLHF

---

## Paper 1 (Classic): Training language models to follow instructions with human feedback (InstructGPT)
- **Authors:** Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, Ryan Lowe
- **Year:** 2022
- **arXiv:** https://arxiv.org/abs/2203.02155
- **PDF:** [./instructgpt-ouyang-2022.pdf](./instructgpt-ouyang-2022.pdf)
- **Citation Count:** ~7,000+

### 요약
InstructGPT는 인간 피드백을 활용한 강화학습(RLHF)을 통해 언어 모델이 사용자의 의도에 맞는 응답을 생성하도록 훈련하는 방법을 제시한 논문이다. GPT-3를 기반으로 지도학습(SFT)과 보상 모델 훈련, 그리고 PPO(Proximal Policy Optimization)를 결합하여 1.3B 파라미터의 InstructGPT가 175B GPT-3보다 인간 평가에서 더 선호되는 결과를 달성했다.

### 핵심 기여
- RLHF 파이프라인의 3단계 프로세스(SFT → 보상 모델 → PPO)를 체계적으로 정립하여 이후 ChatGPT, Claude 등 모든 instruction-following 모델의 표준 훈련 방법론이 됨
- 모델 크기를 키우는 것만으로는 사용자 의도 정렬이 안 된다는 "alignment tax" 개념을 실증적으로 증명
- 인간 레이블러를 활용한 데이터 수집 및 평가 파이프라인의 실용적 설계 방법을 제시

### 이 논문이 중요한 이유
InstructGPT는 현대 LLM 정렬(alignment)의 출발점이다. ChatGPT의 직접적인 전신이며, RLHF라는 개념을 산업 수준에서 처음으로 성공적으로 적용한 사례다. AI 엔지니어라면 RLHF의 전체 파이프라인을 이해하는 것이 필수이며, 이 논문은 그 핵심 레퍼런스다. 보상 모델 설계, 인간 평가 프로토콜, 정렬과 성능 사이의 트레이드오프 등 실무적 통찰이 풍부하다.

### 사전 지식
- GPT 계열 언어 모델의 기본 아키텍처 (Transformer decoder)
- 강화학습의 기초 개념 (정책, 보상, PPO 알고리즘)
- 지도학습 기반 미세조정(SFT)의 원리

### 관련 논문
- [Learning to summarize from human feedback (Stiennon et al., 2020)](https://arxiv.org/abs/2009.01325)
- [Deep reinforcement learning from human preferences (Christiano et al., 2017)](https://arxiv.org/abs/1706.03741)
- [Constitutional AI: Harmlessness from AI Feedback (Bai et al., 2022)](https://arxiv.org/abs/2212.08073)

### 실무 적용
ChatGPT, Claude, Gemini 등 모든 상용 LLM의 핵심 훈련 파이프라인이 이 논문에서 출발했다. 실무에서는 RLHF를 직접 구현하기보다 이 프레임워크를 이해하고, 커스텀 보상 모델 설계나 인간 피드백 수집 프로토콜을 설계할 때 참고한다. 특히 B2B SaaS에서 도메인 특화 모델을 정렬할 때 InstructGPT의 3단계 접근법이 기본 설계 패턴으로 활용된다.

---

## Paper 2 (Classic): Direct Preference Optimization: Your Language Model is Secretly a Reward Model (DPO)
- **Authors:** Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, Chelsea Finn
- **Year:** 2023
- **arXiv:** https://arxiv.org/abs/2305.18290
- **PDF:** [./dpo-rafailov-2023.pdf](./dpo-rafailov-2023.pdf)
- **Citation Count:** ~4,000+

### 요약
DPO는 RLHF의 복잡한 강화학습 단계를 제거하고, 인간 선호도 데이터로부터 직접 정책을 최적화하는 간단한 분류 손실 함수를 제안한 논문이다. 보상 모델을 명시적으로 학습하지 않고도 동일한 최적 정책에 도달할 수 있음을 수학적으로 증명하여, LLM 정렬을 획기적으로 단순화했다.

### 핵심 기여
- RLHF의 보상 모델 + PPO라는 2단계 과정을 단일 분류 손실 함수로 대체하여 훈련 안정성과 효율성을 크게 개선
- Bradley-Terry 모델 기반의 선호도 학습이 암묵적으로 보상 함수를 내포하고 있음을 이론적으로 증명
- PPO 대비 하이퍼파라미터 튜닝이 거의 불필요하며, 구현이 매우 간단함을 실증

### 이 논문이 중요한 이유
DPO는 RLHF 이후 LLM 정렬 분야에서 가장 영향력 있는 논문 중 하나다. 기존 RLHF가 보상 모델 학습, PPO 최적화, KL 발산 제약 등 복잡한 파이프라인을 요구했다면, DPO는 이를 한 줄의 손실 함수로 압축했다. Llama 2, Zephyr 등 주요 오픈소스 모델들이 DPO를 채택했으며, 이후 KTO, IPO, ORPO 등 수많은 후속 연구의 기반이 되었다.

### 사전 지식
- InstructGPT 논문에서 제시한 RLHF 파이프라인 이해
- Bradley-Terry 선호도 모델의 기본 개념
- KL 발산(Kullback-Leibler divergence)과 정책 최적화의 관계

### 관련 논문
- [Training language models to follow instructions with human feedback / InstructGPT (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155)
- [A General Theoretical Paradigm to Understand Learning from Human Feedback / KTO (Ethayarajh et al., 2024)](https://arxiv.org/abs/2310.12036)
- [ORPO: Monolithic Preference Optimization without Reference Model (Hong et al., 2024)](https://arxiv.org/abs/2403.07691)

### 실무 적용
DPO는 현재 오픈소스 LLM 정렬의 사실상 표준이다. Hugging Face TRL 라이브러리에서 DPOTrainer를 통해 몇 줄의 코드로 구현 가능하며, 커스텀 도메인 모델 정렬 시 PPO 대비 훨씬 적은 GPU 자원으로 유사한 성능을 달성할 수 있다. AI SaaS 제품에서 사용자 피드백 기반으로 모델을 지속적으로 개선할 때, DPO는 가장 실용적인 선택지다.

---

## Paper 3 (Recent): DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (GRPO)
- **Authors:** Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y.K. Li, Y. Wu, Daya Guo
- **Year:** 2024
- **arXiv:** https://arxiv.org/abs/2402.03300
- **PDF:** [./deepseekmath-grpo-shao-2024.pdf](./deepseekmath-grpo-shao-2024.pdf)
- **Citation Count:** ~500+

### 요약
DeepSeekMath는 수학적 추론 능력을 극대화하기 위해 GRPO(Group Relative Policy Optimization)라는 새로운 강화학습 알고리즘을 제안한 논문이다. GRPO는 PPO에서 필수적이었던 크리틱(critic) 모델을 제거하고, 그룹 내 상대적 보상을 기반으로 정책을 최적화하여 메모리 사용량을 대폭 줄이면서도 우수한 성능을 달성했다.

### 핵심 기여
- PPO의 크리틱 모델을 제거하고 그룹 내 상대 점수로 베이스라인을 추정하는 GRPO 알고리즘 제안, 훈련 자원을 대폭 절감
- DeepSeekMath 7B 모델이 GSM8K 88.2%, MATH 51.7%를 달성하며 오픈소스 수학 추론 모델의 새로운 기준 수립
- 영어 instruction tuning 데이터만으로도 수학 추론 능력을 크게 향상시킬 수 있음을 입증

### 이 논문이 중요한 이유
GRPO는 2025년 LLM 업계를 뒤흔든 DeepSeek-R1의 핵심 훈련 알고리즘이다. DeepSeek-R1이 GPT-4 수준의 추론 능력을 보여주면서 GRPO는 오픈소스 커뮤니티에서 LLM 후훈련(post-training)의 표준 방법론으로 급부상했다. PPO, DPO에 이어 제3의 정렬 패러다임으로서, 특히 추론 능력 강화에 최적화된 접근법을 제시한다.

### 사전 지식
- PPO(Proximal Policy Optimization) 알고리즘의 기본 구조
- DPO와 RLHF의 차이점 이해
- 언어 모델의 수학적 추론 평가 벤치마크 (GSM8K, MATH)

### 관련 논문
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning (DeepSeek-AI, 2025)](https://arxiv.org/abs/2501.12948)
- [Direct Preference Optimization / DPO (Rafailov et al., 2023)](https://arxiv.org/abs/2305.18290)
- [Proximal Policy Optimization Algorithms / PPO (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)

### 실무 적용
GRPO는 Hugging Face TRL의 GRPOTrainer로 구현 가능하며, 특히 추론 집약적 태스크(코드 생성, 수학, 논리적 분석)에서 모델 성능을 개선할 때 효과적이다. DPO가 선호도 데이터에 의존하는 반면, GRPO는 정답 여부만으로도 훈련이 가능하여 라벨링 비용을 줄일 수 있다. Agentic AI 제품에서 에이전트의 추론 정확도를 높이는 후훈련 기법으로 주목받고 있다.

---

## 추천 읽기 순서

1. **InstructGPT (2022)** → RLHF의 전체 파이프라인을 이해하는 것이 첫 번째다. SFT → 보상 모델 → PPO라는 3단계 프로세스가 왜 필요한지, 어떤 문제를 해결하는지를 먼저 파악해야 한다.
2. **DPO (2023)** → InstructGPT의 복잡한 RLHF 파이프라인을 어떻게 단순화했는지 이해한다. 수학적 유도 과정을 따라가면 "보상 모델이 왜 불필요한지"가 명확해진다.
3. **DeepSeekMath/GRPO (2024)** → PPO와 DPO의 장단점을 이해한 상태에서, GRPO가 제시하는 제3의 접근법을 살펴본다. 특히 추론 태스크에서의 강점과 DeepSeek-R1으로의 확장을 주목한다.

## 핵심 테이크어웨이

LLM 정렬(alignment)은 3년간 급격한 진화를 겪었다. InstructGPT가 정립한 RLHF 파이프라인(2022)은 효과적이었으나 복잡하고 불안정했다. DPO(2023)는 이를 단일 손실 함수로 압축하며 민주화했고, GRPO(2024)는 크리틱 모델 없이 그룹 상대 평가로 효율성을 극대화했다. 이 세 논문은 "복잡한 파이프라인 → 단순한 손실 함수 → 효율적인 그룹 최적화"라는 정렬 기술의 진화 방향을 보여준다. 실무적으로는 태스크 특성에 따라 DPO(선호도 데이터 활용)와 GRPO(정답 기반 추론 강화)를 선택적으로 적용하는 것이 현재 최선의 전략이다.

## 다음 토픽과의 연결

다음 토픽인 **LLM Evaluation and Benchmarks**에서는 오늘 학습한 정렬 기법들의 효과를 어떻게 측정하고 비교하는지를 다룬다. MMLU, HumanEval 등의 벤치마크가 InstructGPT, DPO, GRPO로 훈련된 모델들의 성능을 어떻게 평가하는지, 그리고 벤치마크 자체의 한계와 최신 평가 프레임워크(Arena, LiveBench)는 무엇인지를 탐구한다.
