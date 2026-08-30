# Daily AI Paper Recommendations

> **Date:** 2026-06-10
> **Module:** Module 6: LLM for Natural Language Generation
> **Topic:** Instruction Tuning and RLHF

---

## Paper 1 (Classic): Fine-Tuning Language Models from Human Preferences
- **Authors:** Daniel M. Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B. Brown, Alec Radford, Dario Amodei, Paul Christiano, Geoffrey Irving
- **Year:** 2019
- **arXiv:** https://arxiv.org/abs/1909.08593
- **PDF:** [./finetuning-lm-from-human-preferences-ziegler-2019.pdf](./finetuning-lm-from-human-preferences-ziegler-2019.pdf)
- **Citation Count:** approx. 1,500+

### 요약
사람의 선호(human preference)로부터 보상 모델(reward model)을 학습하고, 이를 강화학습 보상 신호로 사용해 언어 모델을 미세조정하는 방법을 처음으로 대규모 언어 모델에 적용한 논문이다. 텍스트 스타일 이어쓰기(긍정적 감성, 묘사적 표현)와 요약(TL;DR, CNN/Daily Mail) 등 네 가지 자연어 과제에서, 명시적 정답 라벨 대신 "둘 중 어느 출력이 더 나은가"라는 사람의 비교 판단만으로 모델을 정렬할 수 있음을 보였다. 오늘날 RLHF 파이프라인의 직접적인 원형(prototype)에 해당한다.

### 핵심 기여
- 사람의 쌍대 비교(pairwise comparison) 데이터로 보상 모델을 학습하고, 이를 RL(PPO) 보상으로 사용하는 RLHF 학습 루프를 LM에 정립
- 스타일 이어쓰기 과제에서는 5,000건의 비교만으로도 양질의 정렬이 가능함을 실증 (데이터 효율성)
- 보상 모델 과최적화(reward hacking) 문제와 KL 페널티의 필요성을 초기에 식별

### 이 논문이 중요한 이유
InstructGPT, ChatGPT, Claude 등 현대 정렬 LLM의 학습 방식인 RLHF의 출발점이다. "보상 모델 + 정책 모델 + KL 제약"이라는 구조가 여기서 정립되었기 때문에, AI 엔지니어가 정렬 파이프라인의 설계 의도와 한계를 이해하려면 반드시 읽어야 하는 1차 문헌이다.

### 사전 지식
- 강화학습 기초, 특히 정책 경사(policy gradient)와 PPO
- 언어 모델 사전학습(GPT-2 수준)과 미세조정의 차이
- KL 발산(KL divergence)과 정규화 개념

### 관련 논문
- [Learning to summarize from human feedback (Stiennon et al., 2020)](https://arxiv.org/abs/2009.01325)
- [Training language models to follow instructions with human feedback / InstructGPT (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155)
- [Proximal Policy Optimization Algorithms (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)

### 실무 적용
챗봇/어시스턴트의 톤·안전성·유용성을 라벨 정답이 아닌 사람의 상대 선호로 조정할 때 사용된다. 실무에서는 보상 모델 학습 → PPO 미세조정 → KL 제약으로 reward hacking 방지라는 흐름으로, 고객 응대 봇이나 콘텐츠 생성 제품의 출력 품질 정렬에 그대로 적용된다.

---

## Paper 2 (Classic): Learning to summarize from human feedback
- **Authors:** Nisan Stiennon, Long Ouyang, Jeff Wu, Daniel M. Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, Paul Christiano
- **Year:** 2020
- **arXiv:** https://arxiv.org/abs/2009.01325
- **PDF:** [./learning-to-summarize-from-human-feedback-stiennon-2020.pdf](./learning-to-summarize-from-human-feedback-stiennon-2020.pdf)
- **Citation Count:** approx. 2,500+

### 요약
요약(summarization)이라는 구체적 과제에서 RLHF를 본격적으로 확장해, 사람 선호로 학습한 보상 모델을 PPO로 최적화하면 ROUGE 같은 자동 지표를 직접 최적화하거나 지도학습만 하는 것보다 사람이 더 선호하는 요약을 생성함을 보였다. 모델이 사람 레퍼런스 요약보다도 선호되는 결과를 얻으며, RLHF가 단순 모방을 넘어 인간 효용에 맞춘 최적화로 이어질 수 있음을 입증했다.

### 핵심 기여
- RLHF를 실제 생성 품질 과제(요약)에 적용해 사람 선호 기준에서 지도학습·자동지표 최적화를 능가함을 실증
- 보상 모델 규모와 데이터를 늘릴수록 정렬 성능이 향상되는 스케일 경향 관찰
- 자동 지표(ROUGE)와 사람 선호 간 괴리를 정량적으로 드러내 평가 방법론에 시사점 제공

### 이 논문이 중요한 이유
InstructGPT 직전의 핵심 디딤돌로, RLHF가 "특정 작업에서 사람보다 선호되는 출력"을 만들 수 있음을 처음으로 설득력 있게 보였다. 정렬이 왜 SFT만으로 충분하지 않은지, 보상 모델 기반 최적화가 어떤 이점을 주는지를 이해하는 데 필수적이다.

### 사전 지식
- Paper 1(Ziegler 2019)의 RLHF 루프
- 요약 과제와 ROUGE 등 자동 평가 지표의 한계
- 보상 모델 과최적화와 KL 제약의 역할

### 관련 논문
- [Fine-Tuning Language Models from Human Preferences (Ziegler et al., 2019)](https://arxiv.org/abs/1909.08593)
- [Training language models to follow instructions with human feedback (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155)
- [Direct Preference Optimization (Rafailov et al., 2023)](https://arxiv.org/abs/2305.18290)

### 실무 적용
문서 요약, 회의록 요약, RAG 답변 생성 등에서 "정답 요약 모방"이 아니라 "사람이 실제로 더 유용하다고 느끼는 출력"을 목표로 모델을 조정할 때의 표준 접근이다. 자동 지표만 보고 최적화하면 실제 사용자 만족과 어긋날 수 있다는 교훈은 제품 평가 지표 설계에도 직접 적용된다.

---

## Paper 3 (Recent): KTO: Model Alignment as Prospect Theoretic Optimization
- **Authors:** Kawin Ethayarajh, Winnie Xu, Niklas Muennighoff, Dan Jurafsky, Douwe Kiela
- **Year:** 2024
- **arXiv:** https://arxiv.org/abs/2402.01306
- **PDF:** [./kto-ethayarajh-2024.pdf](./kto-ethayarajh-2024.pdf)
- **Citation Count:** approx. 600+

### 요약
행동경제학의 전망 이론(prospect theory)에 착안해, 쌍대 선호 데이터 없이 출력의 "바람직함/바람직하지 않음"이라는 이진 신호만으로 LLM을 정렬하는 KTO(Kahneman-Tversky Optimization)를 제안한다. DPO 등 선호쌍 기반 방법이 요구하는 비싼 (chosen, rejected) 쌍 데이터 대신, 단순한 좋음/나쁨 라벨로도 1B~30B 규모에서 동등하거나 더 나은 정렬 성능을 달성한다.

### 핵심 기여
- 선호쌍이 아닌 이진 신호(desirable/undesirable)만으로 정렬 가능한 손실 함수 제시 — 데이터 수집 비용 대폭 절감
- 인간 효용 함수를 전망 이론(손실 회피, 비대칭 가치 함수)으로 모델링해 정렬을 일반화
- DPO 대비 클래스 불균형·노이즈에 강건하며, 광범위한 모델 규모에서 성능 검증

### 이 논문이 중요한 이유
실무에서 (chosen, rejected) 선호쌍 구축은 가장 큰 병목이다. KTO는 제품 로그에서 자연스럽게 얻는 좋아요/싫어요·전환/이탈 같은 이진 피드백을 바로 정렬에 활용할 수 있게 해, 데이터 수집 관점에서 정렬을 크게 실용화한다. 2024년 정렬 기법 흐름(DPO 이후의 단순화·효율화)을 대표한다.

### 사전 지식
- DPO(Direct Preference Optimization)와 선호쌍 기반 정렬의 기본 구조
- RLHF 보상 모델과 KL 제약 개념 (Paper 1, 2)
- 전망 이론의 기본 직관(손실 회피, 준거점)

### 관련 논문
- [Direct Preference Optimization (Rafailov et al., 2023)](https://arxiv.org/abs/2305.18290)
- [SimPO: Simple Preference Optimization with a Reference-Free Reward (Meng et al., 2024)](https://arxiv.org/abs/2405.14734)
- [ORPO: Monolithic Preference Optimization without Reference Model (Hong et al., 2024)](https://arxiv.org/abs/2403.07691)

### 실무 적용
SaaS 제품에서 사용자 좋아요/싫어요, 답변 채택/재생성, 클릭/이탈 같은 이진 피드백을 모아 바로 모델 정렬 데이터로 전환할 수 있다. 별도의 선호쌍 라벨링 파이프라인 없이 그로스 지표·제품 로그를 정렬 신호로 재활용하므로, 데이터 드리븐 정렬 루프를 저비용으로 구축하려는 팀에 특히 적합하다.

---

## 추천 읽기 순서
1. **Fine-Tuning Language Models from Human Preferences (2019)** — RLHF의 기본 루프(보상 모델 + PPO + KL)를 먼저 이해
2. **Learning to summarize from human feedback (2020)** — RLHF가 실제 생성 과제에서 사람보다 선호되는 출력을 만드는 과정을 확인
3. **KTO (2024)** — 선호쌍의 한계를 넘어 이진 신호로 정렬을 단순화하는 최신 흐름으로 마무리

## 핵심 테이크어웨이
- 정렬의 본질은 "정답 모방(SFT)"이 아니라 "사람 효용에 맞춘 최적화"이며, 그 출발점이 사람 선호 기반 보상 학습이다.
- 자동 지표(ROUGE 등)만 최적화하면 실제 사용자 만족과 어긋날 수 있다 — 평가 지표 설계가 정렬만큼 중요하다.
- 2019→2020→2024로 이어지는 흐름은 "RL 루프 정립 → 실과제 검증 → 데이터·계산 비용의 단순화"라는 방향성을 보여준다. KTO처럼 이진 신호로 정렬하는 방식은 제품 로그를 정렬 자산으로 바꾼다.

## 다음 토픽과의 연결
다음 토픽인 **LLM Evaluation and Benchmarks**로 자연스럽게 이어진다. 오늘 본 논문들이 공통적으로 강조한 "자동 지표와 사람 선호의 괴리"는 곧 LLM을 어떻게 평가할 것인가의 문제로 직결되며, MMLU·Arena 류 벤치마크와 사람 선호 평가의 설계 논의로 연결된다.
