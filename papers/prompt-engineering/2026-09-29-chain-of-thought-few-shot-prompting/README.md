# Daily AI Paper Recommendations

> **Date:** 2026-09-29
> **Module:** Module 7: Prompt Engineering
> **Topic:** Chain-of-Thought and Few-Shot Prompting

---

## Paper 1 (Classic): Show Your Work: Scratchpads for Intermediate Computation with Language Models
- **Authors:** Maxwell Nye, Anders Johan Andreassen, Guy Gur-Ari, Henryk Michalewski, Jacob Austin, David Bieber, David Dohan, Aitor Lewkowycz, Maarten Bosma, David Luan, Charles Sutton, Augustus Odena
- **Year:** 2021
- **arXiv:** https://arxiv.org/abs/2112.00114
- **PDF:** [./show-your-work-scratchpads-nye-2021.pdf](./show-your-work-scratchpads-nye-2021.pdf)
- **Citation Count:** ~900+

### 요약
Transformer 언어 모델은 한 번에 답을 내야 하는 다단계 계산(긴 덧셈, 다항식 계산, 파이썬 프로그램 실행 결과 예측)에 약합니다. 이 논문은 모델이 최종 답 전에 중간 계산 과정을 "스크래치패드(scratchpad)"에 먼저 쓰도록 학습·프롬프트하면 이런 과제의 정확도가 크게 오른다는 것을 보였습니다. Chain-of-Thought(Wei et al., 2022)보다 몇 달 앞서 "중간 단계를 토큰으로 쓰게 하라"는 아이디어를 실험으로 증명한 선행 연구입니다.

### 핵심 기여
- 최종 답만 예측하는 대신 중간 계산 상태를 텍스트로 출력하게 하는 **스크래치패드 형식** 제안
- 긴 정수 덧셈에서 학습 때보다 더 긴 자릿수로 **길이 일반화(out-of-distribution)** 성능이 좋아짐을 확인
- 프로그램 실행 추적(line-by-line trace)을 출력하게 해 **코드 실행 결과 예측** 정확도를 크게 향상 — 파인튜닝과 few-shot 모두에서 효과 확인

### 이 논문이 중요한 이유
"모델에게 생각할 공간(토큰)을 주면 더 어려운 계산을 할 수 있다"는 원리의 출발점입니다. 오늘날 o1·o3, Claude의 extended thinking, DeepSeek-R1 같은 추론 모델의 "생각 토큰"은 결국 스크래치패드의 확장판입니다. CoT를 "프롬프트 요령"이 아니라 **계산량(test-time compute)을 늘리는 방법**으로 이해하게 해 주는 논문입니다.

### 사전 지식
- Transformer 디코더의 자기회귀(autoregressive) 생성 방식
- 파인튜닝과 few-shot 프롬프팅의 차이
- 한 번의 forward pass에서 할 수 있는 계산량에 한계가 있다는 직관

### 관련 논문
- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models (Wei et al., 2022)](https://arxiv.org/abs/2201.11903)
- [Program Synthesis with Large Language Models (Austin et al., 2021)](https://arxiv.org/abs/2108.07732)
- [Let's Verify Step by Step (Lightman et al., 2023)](https://arxiv.org/abs/2305.20050)

### 실무 적용
- 에이전트가 도구 호출 전에 계획/중간 상태를 구조화된 필드(`reasoning`, `plan`)로 먼저 쓰게 하면 오류가 줄어듭니다.
- 계산·데이터 변환 작업은 "과정을 단계별로 적고 마지막 줄에 답만" 형식으로 출력을 설계하면 파싱과 디버깅이 쉬워집니다.
- 추론 모델의 thinking budget을 조절하는 것 자체가 스크래치패드 길이를 조절하는 설계 결정입니다.

---

## Paper 2 (Classic): An Explanation of In-context Learning as Implicit Bayesian Inference
- **Authors:** Sang Michael Xie, Aditi Raghunathan, Percy Liang, Tengyu Ma
- **Year:** 2021 (ICLR 2022)
- **arXiv:** https://arxiv.org/abs/2111.02080
- **PDF:** [./icl-implicit-bayesian-inference-xie-2021.pdf](./icl-implicit-bayesian-inference-xie-2021.pdf)
- **Citation Count:** ~900+

### 요약
GPT-3는 가중치를 바꾸지 않고 프롬프트 속 예시 몇 개만 보고 새 과제를 수행합니다(in-context learning). 이 논문은 그 이유를 "사전학습 데이터가 여러 숨은 개념(latent concept)으로 생성된 문서들의 혼합이라면, 모델은 다음 토큰 예측을 잘하기 위해 문맥에서 어떤 개념인지 **암묵적으로 베이지안 추론**하는 법을 배운다"는 이론으로 설명합니다. 합성 데이터셋 GINC로 이를 실험적으로 보여줍니다.

### 핵심 기여
- In-context learning을 **잠재 개념에 대한 암묵적 베이지안 추론**으로 형식화한 이론 프레임워크
- 예시와 사전학습 분포 사이에 차이(분포 불일치)가 있어도 예시 수가 늘면 ICL이 성공하는 조건을 증명
- 합성 데이터 GINC 실험으로 예시 순서 민감도, 모델 크기 효과, 제로샷이 few-shot보다 나은 경우 등 실제 현상 재현

### 이 논문이 중요한 이유
Few-shot 프롬프팅을 "마법"이 아니라 **"모델에게 어떤 과제인지 알려주는 신호"**로 이해하게 해 줍니다. 예시가 정답 라벨보다 형식·분포·도메인을 알려주는 역할이 크다는 이후 연구(Min et al., 2022)와 연결되며, 예시를 어떻게 고를지 판단하는 기준을 제공합니다.

### 사전 지식
- 베이즈 정리와 사후 확률(posterior)의 기본 개념
- 은닉 마르코프 모델(HMM)의 개념 수준 이해
- GPT-3의 few-shot 프롬프팅 방식

### 관련 논문
- [Language Models are Few-Shot Learners (Brown et al., 2020)](https://arxiv.org/abs/2005.14165)
- [Rethinking the Role of Demonstrations: What Makes In-Context Learning Work? (Min et al., 2022)](https://arxiv.org/abs/2202.12837)
- [Transformers learn in-context by gradient descent (von Oswald et al., 2022)](https://arxiv.org/abs/2212.07677)

### 실무 적용
- Few-shot 예시는 "정답 모음"이 아니라 **과제 정의서**로 설계하세요. 실제 입력과 같은 도메인·형식·길이를 가진 예시가 효과적입니다.
- 모델이 과제를 이미 잘 알아듣는다면(개념 추론이 쉬운 경우) 예시를 줄이고 명확한 지시로 대체해 토큰 비용을 줄일 수 있습니다.
- 예시 순서에 결과가 흔들린다면, 모델이 개념을 확신하지 못한다는 신호로 보고 예시를 더 대표성 있게 바꾸는 것이 좋습니다.

---

## Paper 3 (Recent): Many-Shot In-Context Learning
- **Authors:** Rishabh Agarwal, Avi Singh, Lei M. Zhang, Bernd Bohnet, Luis Rosias, Stephanie Chan, Biao Zhang, Ankesh Anand, Zaheer Abbas, Azade Nova, John D. Co-Reyes, Eric Chu, Feryal Behbahani, Aleksandra Faust, Hugo Larochelle
- **Year:** 2024 (NeurIPS 2024 Spotlight)
- **arXiv:** https://arxiv.org/abs/2404.11018
- **PDF:** [./many-shot-in-context-learning-agarwal-2024.pdf](./many-shot-in-context-learning-agarwal-2024.pdf)
- **Citation Count:** ~300+

### 요약
컨텍스트 창이 수십만~백만 토큰으로 커지면서, 예시를 몇 개가 아니라 수백~수천 개 넣는 "many-shot" 프롬프팅이 가능해졌습니다. 이 논문은 Gemini 1.5 Pro로 번역, 요약, 수학 추론, 계획 등 다양한 과제에서 few-shot → many-shot으로 갈 때 성능이 크게 오른다는 것을 보였습니다. 또한 사람이 쓴 예시가 부족할 때를 위해 **Reinforced ICL**(모델이 생성하고 정답 검증된 CoT를 예시로 사용)과 **Unsupervised ICL**(질문만 넣기)을 제안했습니다.

### 핵심 기여
- 많은 과제에서 예시 수를 늘릴수록 성능이 계속 오르며, 일부 과제는 **파인튜닝에 근접하거나 넘어섬**을 확인
- 모델 생성 CoT 풀이로 예시를 대체하는 **Reinforced ICL**, 질문만 주는 **Unsupervised ICL** 제안 — 복잡한 추론 과제에서 효과적
- Many-shot ICL이 사전학습 편향을 **뒤집을 수 있고**, 고차원 수치 함수 학습도 가능함을 보임. 또한 다음 토큰 NLL(perplexity)이 다운스트림 성능의 좋은 지표가 아닐 수 있음을 지적

### 이 논문이 중요한 이유
"파인튜닝할까, 프롬프트에 예시를 많이 넣을까?"라는 실무 결정에 데이터를 제공합니다. 긴 컨텍스트 + 프롬프트 캐싱 시대에는 many-shot이 **파인튜닝 없이 빠르게 전문화하는 수단**이 됩니다. 또한 Reinforced ICL은 오늘날 추론 모델 학습(STaR, rejection sampling)과 같은 아이디어를 프롬프트 수준에서 구현한 것입니다.

### 사전 지식
- Few-shot 프롬프팅과 Chain-of-Thought
- 긴 컨텍스트 모델(long-context LLM)과 토큰 비용 구조
- STaR(Self-Taught Reasoner) 같은 자기 생성 데이터 학습 개념(선택)

### 관련 논문
- [STaR: Bootstrapping Reasoning With Reasoning (Zelikman et al., 2022)](https://arxiv.org/abs/2203.14465)
- [In-Context Learning with Long-Context Models: An In-Depth Exploration (Bertsch et al., 2024)](https://arxiv.org/abs/2405.00200)
- [Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context (Gemini Team, 2024)](https://arxiv.org/abs/2403.05530)

### 실무 적용
- 분류·추출·스타일 변환 같은 과제는 파인튜닝 전에 **수백 개 예시를 넣은 many-shot 프롬프트 + 프롬프트 캐싱**으로 먼저 검증하세요. 반복 호출 비용이 크게 줄어듭니다.
- 라벨된 CoT 데이터가 없다면, 모델에게 풀이를 생성시키고 정답이 맞은 것만 걸러 예시로 쓰는 Reinforced ICL을 적용할 수 있습니다.
- 예시 수에 따른 성능 곡선을 먼저 그려 보고, 성능이 포화되는 지점에서 예시 수를 정하는 것이 비용 대비 효과가 좋습니다.

---

## 추천 읽기 순서
1. **Show Your Work (Scratchpads)** — "중간 단계를 쓰게 하면 왜 좋아지는가"를 가장 직관적인 실험으로 먼저 이해합니다.
2. **ICL as Implicit Bayesian Inference** — few-shot 예시가 모델 안에서 어떤 역할을 하는지 이론적 관점을 얻습니다.
3. **Many-Shot In-Context Learning** — 앞의 두 아이디어(CoT 풀이 + 예시 기반 학습)가 긴 컨텍스트 시대에 어떻게 확장되는지 확인합니다.

## 핵심 테이크어웨이
- **생각할 토큰 = 계산량**: CoT와 스크래치패드는 모델에게 추가 계산 공간을 주는 방법이며, 오늘날 추론 모델의 기반입니다.
- **예시는 과제를 알려주는 신호**: few-shot 예시는 정답을 가르치기보다 "어떤 과제·형식인지"를 모델이 추론하게 돕습니다.
- **예시의 규모도 설계 변수**: 긴 컨텍스트에서는 예시 수를 수백 개로 늘리는 것이 파인튜닝의 현실적 대안이 됩니다.
- **모델이 만든 풀이도 예시가 된다**: 검증된 자기 생성 CoT(Reinforced ICL)는 데이터 부족 문제를 해결하는 실용적 방법입니다.

## 다음 토픽과의 연결
다음 토픽은 **Advanced Prompting — Tree of Thoughts, ReAct, Self-Consistency**입니다. 오늘 배운 "한 줄로 이어지는 중간 추론"을 넘어, 여러 추론 경로를 탐색·투표하거나(ToT, Self-Consistency) 추론 중간에 외부 도구를 호출하는(ReAct) 방식으로 확장합니다. 스크래치패드가 "생각 공간"이었다면, 다음 토픽은 그 공간을 **탐색하고 행동과 연결하는 구조**를 다룹니다.
