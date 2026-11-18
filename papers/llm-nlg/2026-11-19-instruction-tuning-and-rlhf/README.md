# Daily AI Paper Recommendations

> **Date:** 2026-11-19
> **Module:** Module 6: LLM for Natural Language Generation
> **Topic:** Instruction Tuning and RLHF

> 이번 사이클에서는 이전에 다룬 InstructGPT, DPO, FLAN, Flan-T5, T0, Self-Instruct, LIMA, Constitutional AI, HH-RLHF, PPO, PRM, GRPO/R1/DAPO 등과 겹치지 않도록 **"감독 신호를 어디서 얻는가 — 사람이 만든 태스크 → AI가 만든 피드백 → 검증 가능한 보상"** 이라는 흐름으로 논문을 골랐습니다.

---

## Paper 1 (Classic): Super-NaturalInstructions: Generalization via Declarative Instructions on 1600+ NLP Tasks
- **Authors:** Yizhong Wang, Swaroop Mishra, Pegah Alipoormolabashi, Yeganeh Kordi, Amirreza Mirzaei, Anjana Arunkumar, Arjun Ashok, Arut Selvan Dhanasekaran, Atharva Naik, David Stap, Eshaan Pathak, Giannis Karamanolakis, Haizhi Gary Lai, Ishan Purohit, Ishani Mondal, Jacob Anderson, Kirby Kuznia, Krima Doshi, Maitreya Patel, Kuntal Kumar Pal, Mehrad Moradshahi, Mihir Parmar, Mirali Purohit, Neeraj Varshney, Phani Rohitha Kaza, Pulkit Verma, Ravsehaj Singh Puri, Rushang Karia, Shailaja Keyur Sampat, Savan Doshi, Siddhartha Mishra, Sujan Reddy, Sumanta Patro, Tanay Dixit, Xudong Shen, Chitta Baral, Yejin Choi, Noah A. Smith, Hannaneh Hajishirzi, Daniel Khashabi
- **Year:** 2022 (EMNLP 2022)
- **arXiv:** https://arxiv.org/abs/2204.07705
- **PDF:** [./super-naturalinstructions-wang-2022.pdf](./super-naturalinstructions-wang-2022.pdf)
- **Citation Count:** ~1,000+

### 요약
76개 유형, 1,616개 NLP 태스크를 "자연어 지시문(정의 + 긍정/부정 예시)" 형식으로 통일한 대규모 벤치마크이자 학습 데이터셋입니다. 이 데이터로 T5를 학습한 Tk-Instruct는 **한 번도 본 적 없는 태스크**에서도 지시문만 보고 수행하는 능력이 크게 향상되어, 훨씬 큰 InstructGPT(당시)보다도 일부 지표에서 앞섰습니다. 태스크 수·예시 수·모델 크기가 일반화에 미치는 영향도 체계적으로 분석했습니다.

### 핵심 기여
- 1,600+ 태스크, 55개 언어를 포괄하는 지시문 기반 데이터셋 공개 — 이후 Flan Collection, Tulu 등 지시 튜닝 데이터 믹스의 핵심 구성요소
- "보지 않은 태스크(unseen task)" 일반화를 엄격하게 평가하는 프로토콜 정립
- 태스크 **다양성(개수)** 이 태스크당 데이터 양보다 일반화에 더 중요하다는 스케일링 분석
- 지시문 구성요소(정의 vs. 예시 vs. 설명) 중 무엇이 효과적인지에 대한 ablation

### 이 논문이 중요한 이유
"지시 튜닝 = 데이터를 다양한 태스크 형식으로 모아 학습시키는 것"이라는 개념을 데이터 엔지니어링 관점에서 가장 구체적으로 보여줍니다. 자체 모델을 SFT할 때 "데이터 양보다 태스크 다양성"이라는 원칙은 지금도 그대로 유효합니다.

### 사전 지식
- T5 같은 Encoder-Decoder 모델과 text-to-text 프레이밍
- Zero-shot / Few-shot 일반화 개념
- ROUGE-L 등 생성 평가 지표의 기본

### 관련 논문
- [Cross-Task Generalization via Natural Language Crowdsourcing Instructions (Mishra et al., 2021)](https://arxiv.org/abs/2104.08773)
- [Finetuned Language Models Are Zero-Shot Learners / FLAN (Wei et al., 2021)](https://arxiv.org/abs/2109.01652)
- [The Flan Collection: Designing Data and Methods for Effective Instruction Tuning (Longpre et al., 2023)](https://arxiv.org/abs/2301.13688)

### 실무 적용
- 사내 SFT 데이터셋 설계 시 "업무 유형별 태스크 카탈로그"를 먼저 만들고, 유형마다 지시문 템플릿(정의 + 좋은/나쁜 예시)을 정의
- 에이전트 제품의 시스템 프롬프트를 "정의 + 긍정 예시 + 부정 예시" 구조로 작성하는 근거
- 모델 평가 시 학습에 쓰지 않은 태스크 유형을 hold-out 해서 일반화를 측정

---

## Paper 2 (Classic): RLAIF vs. RLHF: Scaling Reinforcement Learning from Human Feedback with AI Feedback
- **Authors:** Harrison Lee, Samrat Phatale, Hassan Mansoor, Thomas Mesnard, Johan Ferret, Kellie Lu, Colton Bishop, Ethan Hall, Victor Carbune, Abhinav Rastogi, Sushant Prakash
- **Year:** 2023 (ICML 2024)
- **arXiv:** https://arxiv.org/abs/2309.00267
- **PDF:** [./rlaif-lee-2023.pdf](./rlaif-lee-2023.pdf)
- **Citation Count:** ~500+

### 요약
사람 대신 **기성 LLM이 선호 라벨을 매기는 RLAIF** 가 RLHF와 비교해 얼마나 좋은지를 요약·유용한 대화·무해한 대화 태스크에서 정면 비교한 논문입니다. 사람 평가자 기준으로 RLAIF는 RLHF와 비슷한 수준의 성능을 보였고, 라벨러 LLM이 정책 모델과 같은 크기여도 개선이 일어났습니다. 보상 모델을 아예 건너뛰고 LLM 점수를 바로 보상으로 쓰는 **direct-RLAIF(d-RLAIF)** 도 제안했습니다.

### 핵심 기여
- RLAIF ≈ RLHF 임을 사람 평가로 체계적으로 입증 — "사람 라벨 병목"을 풀 수 있는 근거 제시
- 라벨링 프롬프트 기법(상세 서문, CoT, 위치 편향 보정) 중 무엇이 AI 라벨 품질을 올리는지 분석
- d-RLAIF: 보상 모델 학습 없이 LLM 평가 점수를 RL 보상으로 직접 사용해 보상 모델 노후화(staleness) 문제 완화
- 자기 자신과 같은 크기의 라벨러로도 개선 가능함을 보여 "자기 개선" 가능성 제시

### 이 논문이 중요한 이유
오늘날 대부분의 포스트트레이닝 파이프라인(합성 선호 데이터, LLM-as-a-Judge)은 "AI 피드백이 사람 피드백을 대체할 수 있는가"라는 질문 위에 서 있습니다. 이 논문은 그 질문에 대한 가장 직접적인 실험적 답이며, AI 라벨러를 설계할 때 어떤 프롬프트 기법이 실제로 효과가 있는지 알려줍니다.

### 사전 지식
- RLHF 파이프라인(SFT → 보상 모델 → PPO)
- Bradley-Terry 기반 쌍대 선호 모델
- Constitutional AI의 AI 피드백 개념

### 관련 논문
- [Constitutional AI: Harmlessness from AI Feedback (Bai et al., 2022)](https://arxiv.org/abs/2212.08073)
- [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena (Zheng et al., 2023)](https://arxiv.org/abs/2306.05685)
- [UltraFeedback: Boosting Language Models with Scaled AI Feedback (Cui et al., 2023)](https://arxiv.org/abs/2310.01377)

### 실무 적용
- 고객 로그에 대해 LLM 라벨러로 선호 쌍을 대량 생성 → DPO/RM 학습 데이터로 활용해 라벨링 비용 절감
- LLM-as-a-Judge 평가 시 위치 편향 보정(A/B 순서 바꿔 두 번 평가)과 CoT 기반 판단을 기본으로 적용
- 루브릭 기반 점수를 RL 보상으로 바로 쓰는 d-RLAIF 방식을 에이전트 품질 개선 루프에 적용

---

## Paper 3 (Recent): Understanding R1-Zero-Like Training: A Critical Perspective (Dr. GRPO)
- **Authors:** Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, Min Lin
- **Year:** 2025
- **arXiv:** https://arxiv.org/abs/2503.20783
- **PDF:** [./understanding-r1-zero-dr-grpo-liu-2025.pdf](./understanding-r1-zero-dr-grpo-liu-2025.pdf)
- **Code:** https://github.com/sail-sg/understand-r1-zero
- **Citation Count:** ~300+

### 요약
DeepSeek-R1-Zero식 "베이스 모델에 바로 RL" 학습을 **베이스 모델**과 **RL 알고리즘** 두 측면에서 비판적으로 재검토한 논문입니다. 베이스 모델(DeepSeek-V3-Base, Qwen2.5)이 RL 이전부터 이미 "Aha moment"와 강한 추론 능력을 보인다는 점을 밝혔고, GRPO의 길이·표준편차 정규화 항이 **틀린 응답일수록 길이를 늘리는 편향**을 만든다는 것을 찾아냈습니다. 이 항을 제거한 **Dr. GRPO** 와 미니멀 레시피로 7B 베이스 모델에서 AIME 2024 43.3%를 달성했다고 보고합니다(저자 주장).

### 핵심 기여
- "RL로 추론 능력이 새로 생긴다"는 통념에 대한 반례: 사전학습 단계에서 이미 자기 성찰 행동이 존재
- GRPO의 최적화 편향(응답 길이 정규화, 질문 난이도별 std 정규화) 분석 및 수정판 Dr. GRPO 제안
- "응답 길이 증가 = 추론 향상"이라는 해석이 편향의 부산물일 수 있음을 경고 → 토큰 효율 개선
- 프롬프트 템플릿과 베이스 모델 선택이 RL 결과에 미치는 영향 정리

### 이 논문이 중요한 이유
GRPO는 2025년 이후 오픈소스 추론 모델 학습의 사실상 표준이 되었지만, 그 동작을 맹목적으로 따라 하면 길어지기만 하고 똑똑해지지 않는 모델을 얻을 수 있습니다. 이 논문은 **알고리즘의 작은 정규화 항이 제품 지표(지연·비용)에 직결된다**는 것을 보여주어, 포스트트레이닝을 직접 다루는 AI 엔지니어에게 필수 점검 리스트를 제공합니다.

### 사전 지식
- PPO의 advantage, clipping 개념
- GRPO(그룹 상대 advantage) — DeepSeekMath 논문
- 검증 가능한 보상 기반 RL(RLVR) 개념과 DeepSeek-R1-Zero

### 관련 논문
- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning / GRPO (Shao et al., 2024)](https://arxiv.org/abs/2402.03300)
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL (DeepSeek-AI, 2025)](https://arxiv.org/abs/2501.12948)
- [DAPO: An Open-Source LLM Reinforcement Learning System at Scale (Yu et al., 2025)](https://arxiv.org/abs/2503.14476)

### 실무 적용
- 사내 도메인(코드·SQL·수학) 추론 모델을 RL로 튜닝할 때 Dr. GRPO 방식으로 정규화 항을 바꿔 불필요한 장문 출력 억제 → 추론 비용·지연 감소
- RL 실험 대시보드에서 "정답/오답별 응답 길이"를 분리 모니터링해 편향 조기 탐지
- RL 전에 베이스 모델의 잠재 능력(템플릿 유무, pass@k)을 먼저 측정해 "RL이 정말 필요한가"를 판단

---

## 추천 읽기 순서
1. **Super-NaturalInstructions** — 지시 튜닝 데이터가 무엇이고 왜 "다양성"이 핵심인지 이해
2. **RLAIF vs. RLHF** — 선호 신호를 사람 대신 AI로 얻는 방법과 한계
3. **Dr. GRPO** — 사람·AI 피드백 없이 검증 가능한 보상으로 RL할 때 생기는 알고리즘 함정

## 핵심 테이크어웨이
- **Q. 지시 튜닝 데이터는 많을수록 좋은가?** → 양보다 태스크 다양성이 일반화를 좌우한다.
- **Q. 사람 라벨이 꼭 필요한가?** → 잘 설계된 AI 라벨러(CoT, 위치 편향 보정)는 RLHF에 근접한 성능을 낸다. 단, 라벨러의 편향이 그대로 전이된다는 점은 별도 검증이 필요하다.
- **Q. 응답이 길어지면 추론이 좋아진 것인가?** → 아닐 수 있다. 목적 함수의 정규화 편향일 수 있으니 정답/오답별 길이를 분리해 봐야 한다.
- 감독 신호의 진화: **사람이 만든 태스크 → AI 피드백 → 검증 가능한 보상**. 어떤 신호를 쓰느냐가 비용·품질·편향의 트레이드오프를 결정한다.

## 다음 토픽과의 연결
다음 토픽은 **LLM Evaluation and Benchmarks** 입니다. 오늘 본 RLAIF의 LLM 라벨러와 Dr. GRPO의 "길이 편향"은 결국 **"무엇으로, 어떻게 측정하는가"** 의 문제로 이어집니다. 평가 지표가 잘못되면 정렬·RL 모두 잘못된 방향으로 최적화된다는 점을 염두에 두고 벤치마크 설계를 살펴보세요.
