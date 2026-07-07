# Daily AI Paper Recommendations

> **Date:** 2026-07-07
> **Module:** Module 6 - LLM for Natural Language Generation
> **Topic:** Instruction Tuning and RLHF

---

## Paper 1 (Classic): Finetuned Language Models Are Zero-Shot Learners (FLAN)
- **Authors:** Jason Wei, Maarten Bosma, Vincent Y. Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M. Dai, Quoc V. Le
- **Year:** 2021
- **arXiv:** https://arxiv.org/abs/2109.01652
- **PDF:** [./flan-finetuned-lm-zero-shot-wei-2021.pdf](./flan-finetuned-lm-zero-shot-wei-2021.pdf)
- **Citation Count:** approximately 4,500+

### 요약
137B 규모의 사전학습 언어모델을 60여 개의 NLP 태스크에 대해 "자연어 지시문(instruction)" 형태로 파인튜닝하면, 학습에 사용하지 않은 새로운 태스크에 대한 제로샷 성능이 크게 향상된다는 것을 보인 논문이다. 이 방식으로 만든 모델을 FLAN이라 부르며, 25개 평가 태스크 중 20개에서 제로샷 175B GPT-3를 앞섰다. "Instruction Tuning"이라는 개념을 정립한 대표 논문이다.

### 핵심 기여
- 여러 태스크를 자연어 지시문 템플릿으로 표현해 파인튜닝하는 "instruction tuning" 패러다임을 제안
- 지시문 튜닝된 FLAN이 제로샷/퓨샷 GPT-3를 다수 태스크에서 능가함을 실험적으로 입증
- 어블레이션을 통해 태스크 수와 모델 규모가 instruction tuning 성공의 핵심 요인임을 규명

### 이 논문이 중요한 이유
오늘날 ChatGPT, Claude 등 모든 지시-따르기(instruction-following) 모델의 기초가 되는 "지시문 기반 파인튜닝" 개념을 확립했다. RLHF 이전 단계인 SFT(Supervised Fine-Tuning)의 이론적/실증적 토대를 제공하므로, AI 엔지니어가 정렬(alignment) 파이프라인을 이해하려면 반드시 읽어야 한다.

### 사전 지식
- 사전학습(pre-training)과 파인튜닝(fine-tuning)의 차이
- 제로샷/퓨샷 러닝과 in-context learning의 개념
- Transformer 기반 대규모 언어모델의 기본 구조

### 관련 논문
- [Scaling Instruction-Finetuned Language Models (Chung et al., 2022)](https://arxiv.org/abs/2210.11416)
- [Multitask Prompted Training Enables Zero-Shot Task Generalization / T0 (Sanh et al., 2021)](https://arxiv.org/abs/2110.08207)

### 실무 적용
사내 도메인 데이터를 지시문-응답 쌍으로 구성해 SFT를 수행하면, 별도의 프롬프트 엔지니어링 없이도 원하는 태스크를 잘 수행하는 특화 모델을 만들 수 있다. 오픈소스 LLM(Llama, Qwen 등) 커스터마이징의 첫 단계가 바로 이 instruction tuning이다.

---

## Paper 2 (Classic): Proximal Policy Optimization Algorithms (PPO)
- **Authors:** John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov
- **Year:** 2017
- **arXiv:** https://arxiv.org/abs/1707.06347
- **PDF:** [./ppo-schulman-2017.pdf](./ppo-schulman-2017.pdf)
- **Citation Count:** approximately 25,000+

### 요약
정책 경사(policy gradient) 기반 강화학습에서, 환경과 상호작용해 데이터를 수집하는 단계와 "surrogate" 목적함수를 확률적 경사 상승으로 최적화하는 단계를 번갈아 수행하는 새로운 방법군을 제안한다. TRPO의 장점(안정적 학습)을 유지하면서도 구현이 훨씬 단순하고 샘플 효율이 좋아, RLHF의 표준 최적화 알고리즘으로 자리잡았다.

### 핵심 기여
- 클리핑된 surrogate 목적함수(clipped objective)로 정책 업데이트 폭을 제한해 안정적 학습을 달성
- 단일 샘플당 한 번만 업데이트하던 기존 방식과 달리, 미니배치 다중 에폭 업데이트를 가능하게 함
- 로보틱스, 아타리 등 다양한 벤치마크에서 단순함과 성능의 우수한 균형을 실증

### 이 논문이 중요한 이유
InstructGPT/ChatGPT의 RLHF 파이프라인에서 실제 정책 최적화에 사용되는 알고리즘이 바로 PPO다. 보상 모델로부터 얻은 신호로 LLM 정책을 업데이트하는 핵심 엔진이므로, RLHF의 내부 동작을 이해하려면 PPO의 원리를 반드시 알아야 한다. DPO, GRPO 등 최신 정렬 기법도 모두 PPO와의 비교를 통해 설명된다.

### 사전 지식
- 강화학습 기초: 정책(policy), 가치함수(value function), 어드밴티지(advantage)
- 정책 경사 정리와 TRPO의 신뢰 영역(trust region) 개념
- 온폴리시(on-policy) vs 오프폴리시(off-policy) 학습의 차이

### 관련 논문
- [Trust Region Policy Optimization / TRPO (Schulman et al., 2015)](https://arxiv.org/abs/1502.05477)
- [Training language models to follow instructions with human feedback / InstructGPT (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155)

### 실무 적용
RLHF로 LLM을 정렬할 때 TRL, OpenRLHF 같은 라이브러리가 내부적으로 PPO를 구현한다. 보상 모델 설계, KL 페널티 튜닝, 클리핑 계수 조정 등 실무 RLHF 학습의 하이퍼파라미터는 모두 이 논문의 개념에서 출발한다.

---

## Paper 3 (Recent): ORPO: Monolithic Preference Optimization without Reference Model
- **Authors:** Jiwoo Hong, Noah Lee, James Thorne
- **Year:** 2024
- **arXiv:** https://arxiv.org/abs/2403.07691
- **PDF:** [./orpo-hong-2024.pdf](./orpo-hong-2024.pdf)
- **Citation Count:** approximately 500+

### 요약
ORPO는 별도의 참조 모델(reference model)과 별도의 선호도 정렬 단계 없이, SFT 과정 자체에 선호도 최적화를 통합한 단일(monolithic) 알고리즘이다. 오즈비(odds ratio)를 이용해 선호되지 않는 생성 스타일에 작은 페널티를 부여하는 것만으로 선호도 정렬이 가능함을 보였다. UltraFeedback만으로 파인튜닝한 Phi-2, Llama-2, Mistral이 Llama-2 Chat, Zephyr 등 SOTA 모델을 능가했다.

### 핵심 기여
- 참조 모델 없이 SFT 단계에 선호도 정렬을 통합한 monolithic odds-ratio 목적함수 제안
- 별도의 정렬 단계를 제거해 학습 파이프라인과 메모리 비용을 크게 단순화
- 125M~7B 규모에 걸쳐 오즈비가 선호/비선호 스타일 대비에 적합함을 이론·실험적으로 입증

### 이 논문이 중요한 이유
DPO도 참조 모델을 필요로 하는데, ORPO는 이마저 제거해 정렬 파이프라인을 SFT 한 단계로 압축했다. RLHF → DPO → ORPO로 이어지는 정렬 기법 간소화 흐름의 최신 지점을 보여주며, 제한된 자원에서 정렬 모델을 만드는 실무자에게 매우 실용적인 선택지를 제공한다.

### 사전 지식
- SFT와 RLHF/DPO의 차이 및 각각의 학습 단계 구조
- DPO의 참조 모델 기반 선호도 손실(preference loss)
- 오즈비(odds ratio)와 로그 확률(log-likelihood)의 개념

### 관련 논문
- [Direct Preference Optimization / DPO (Rafailov et al., 2023)](https://arxiv.org/abs/2305.18290)
- [SimPO: Simple Preference Optimization with a Reference-Free Reward (Meng et al., 2024)](https://arxiv.org/abs/2405.14734)

### 실무 적용
GPU 메모리가 제한된 환경에서 참조 모델을 별도로 올리지 않고도 정렬 모델을 학습할 수 있어, 소규모 팀이나 온프레미스 파인튜닝에 적합하다. Hugging Face TRL의 `ORPOTrainer`로 바로 적용 가능하며, SFT와 정렬을 한 번에 수행해 실험 반복 속도를 높인다.

---

## 추천 읽기 순서
1. **FLAN (2021)** — 먼저 instruction tuning(SFT)의 개념을 이해한다. 정렬 파이프라인의 출발점이다.
2. **PPO (2017)** — RLHF의 핵심 최적화 엔진을 학습한다. 강화학습 기반 정렬의 내부 동작을 파악한다.
3. **ORPO (2024)** — RLHF/DPO의 복잡성을 어떻게 단순화하는지 최신 흐름으로 마무리한다.

## 핵심 테이크어웨이
- LLM 정렬은 **SFT(instruction tuning) → 선호도 정렬(RLHF/DPO/ORPO)** 의 단계로 발전해왔다.
- **PPO**는 보상 모델 신호로 정책을 업데이트하는 RLHF의 표준 엔진이지만, 참조 모델·보상 모델·다단계 학습으로 무겁다.
- **DPO → ORPO** 로 이어지며 정렬 파이프라인은 점점 단순해지고, 참조 모델과 별도 정렬 단계를 제거하는 방향으로 진화하고 있다.
- 실무에서는 태스크·자원·데이터 형태에 따라 SFT-only, RLHF, DPO, ORPO 중 적절한 방법을 선택해야 한다.

## 다음 토픽과의 연결
다음 토픽인 **LLM Evaluation and Benchmarks**는 이렇게 정렬된 모델이 실제로 얼마나 잘 지시를 따르고 안전한지를 어떻게 정량 평가하는지를 다룬다. 정렬 기법의 효과는 결국 평가 벤치마크(MMLU, Arena-Hard 등)로 검증되므로, 오늘의 정렬 파이프라인 이해가 다음 평가 주제의 전제가 된다.
