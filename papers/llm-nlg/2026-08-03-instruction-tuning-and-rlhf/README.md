# Daily AI Paper Recommendations

> **Date:** 2026-08-03
> **Module:** Module 6: LLM for Natural Language Generation
> **Topic:** Instruction Tuning and RLHF

---

## Paper 1 (Classic): Self-Instruct: Aligning Language Models with Self-Generated Instructions
- **Authors:** Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, Hannaneh Hajishirzi
- **Year:** 2022
- **arXiv:** [https://arxiv.org/abs/2212.10560](https://arxiv.org/abs/2212.10560)
- **PDF:** [./self-instruct-wang-2022.pdf](./self-instruct-wang-2022.pdf)
- **Citation Count:** approx. 3,500+

### 요약
인스트럭션 튜닝은 사람이 직접 작성한 지시-응답 데이터에 크게 의존하는데, 이 데이터는 양·다양성·창의성 측면에서 한계가 있습니다. Self-Instruct는 언어 모델이 스스로 지시문과 입력/출력 예시를 생성하고, 저품질·중복 샘플을 필터링한 뒤 그 데이터로 다시 자기 자신을 파인튜닝하는 부트스트래핑 프레임워크입니다. 175개의 시드 태스크만으로 5만 개 이상의 인스트럭션 데이터를 만들어냈습니다.

### 핵심 기여
- 사람이 라벨링한 데이터 없이도 언어 모델의 자체 생성만으로 대규모 인스트럭션 데이터를 구축하는 방법 제시
- 지시문 생성 → 태스크 분류 → 인스턴스 생성 → 필터링으로 이어지는 반자동 파이프라인 정립
- GPT-3에 적용해 InstructGPT_001에 근접한 성능을 저비용으로 달성함을 실증

### 이 논문이 중요한 이유
합성 데이터(synthetic data)로 정렬(alignment)을 수행하는 현대적 흐름의 출발점입니다. Alpaca, WizardLM 등 오픈소스 인스트럭션 튜닝 모델이 모두 이 아이디어에 기반하며, AI 엔지니어가 데이터 라벨링 비용 없이 모델을 정렬하는 실전 전략을 이해하는 데 필수입니다.

### 사전 지식
- 인스트럭션 튜닝(instruction tuning)과 지도학습 파인튜닝(SFT)의 개념
- GPT-3 계열 few-shot 프롬프팅과 in-context learning
- 데이터 필터링(ROUGE 유사도 기반 중복 제거) 기초

### 관련 논문
- [Finetuned Language Models Are Zero-Shot Learners / FLAN (Wei et al., 2021)](https://arxiv.org/abs/2109.01652)
- [Training language models to follow instructions with human feedback / InstructGPT (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155)

### 실무 적용
소량의 시드 예시만으로 도메인 특화 인스트럭션 데이터셋을 자동 생성해 사내 SFT 데이터 파이프라인을 구축할 수 있습니다. 챗봇·에이전트의 콜드스타트 문제를 완화하고, 사람 라벨링 예산을 검증(QA) 단계에 집중시킬 수 있습니다.

---

## Paper 2 (Classic): Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback
- **Authors:** Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, et al. (Anthropic)
- **Year:** 2022
- **arXiv:** [https://arxiv.org/abs/2204.05862](https://arxiv.org/abs/2204.05862)
- **PDF:** [./helpful-harmless-assistant-rlhf-bai-2022.pdf](./helpful-harmless-assistant-rlhf-bai-2022.pdf)
- **Citation Count:** approx. 3,000+

### 요약
사람 선호 피드백으로 언어 모델을 "도움이 되면서(helpful)" 동시에 "해롭지 않게(harmless)" 정렬하는 RLHF 파이프라인을 대규모로 정립한 Anthropic의 대표 논문입니다. 선호 데이터 수집 → 보상 모델 학습 → PPO 강화학습의 전체 워크플로우와, 데이터를 계속 수집하며 모델을 갱신하는 온라인 반복 학습(iterated online RLHF)을 상세히 다룹니다.

### 핵심 기여
- helpfulness와 harmlessness라는 두 목표 사이의 긴장과 트레이드오프를 정량적으로 분석
- RLHF가 거의 모든 NLP 평가에서 "정렬 세금(alignment tax)" 없이 성능을 개선하거나 유지함을 실증
- 보상 모델의 스케일링 법칙과 온라인 반복 학습의 효과를 체계적으로 보고

### 이 논문이 중요한 이유
InstructGPT와 함께 상용 LLM 어시스턴트(Claude, ChatGPT 등)의 정렬 방법론적 토대를 제공합니다. 보상 모델 설계, 선호 데이터 수집 방식, 안전성 평가까지 RLHF 실무의 전 과정을 담고 있어 정렬 엔지니어의 레퍼런스로 자주 인용됩니다.

### 사전 지식
- 강화학습 기초와 PPO(Proximal Policy Optimization)
- 보상 모델(reward model)과 선호 학습(preference learning)의 개념
- KL 페널티를 통한 정책 정규화

### 관련 논문
- [Proximal Policy Optimization Algorithms (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)
- [Constitutional AI: Harmlessness from AI Feedback (Bai et al., 2022)](https://arxiv.org/abs/2212.08073)

### 실무 적용
사내 어시스턴트를 배포할 때 helpfulness와 safety를 동시에 최적화하는 보상 설계·평가 프로토콜의 청사진으로 활용할 수 있습니다. 온라인 반복 RLHF 개념은 실사용자 피드백을 지속적으로 반영하는 그로스 루프(feedback loop) 설계에도 직접 응용됩니다.

---

## Paper 3 (Recent): Self-Rewarding Language Models
- **Authors:** Weizhe Yuan, Richard Yuanzhe Pang, Kyunghyun Cho, Xian Li, Sainbayar Sukhbaatar, Jing Xu, Jason Weston (Meta AI / NYU)
- **Year:** 2024
- **arXiv:** [https://arxiv.org/abs/2401.10020](https://arxiv.org/abs/2401.10020)
- **PDF:** [./self-rewarding-language-models-yuan-2024.pdf](./self-rewarding-language-models-yuan-2024.pdf)
- **Citation Count:** approx. 700+

### 요약
사람이나 별도의 고정된 보상 모델 대신, 학습 중인 언어 모델 자신이 LLM-as-a-Judge 프롬프팅으로 스스로에게 보상을 매기는 방식을 제안합니다. 모델은 자기 응답을 생성하고 스스로 평가해 선호 쌍을 만든 뒤 Iterative DPO로 자신을 갱신하며, 이 과정을 반복할수록 지시 수행 능력과 "심판 능력" 자체가 함께 향상됩니다.

### 핵심 기여
- 정책 모델과 보상 모델을 하나로 통합해 사람 피드백 병목을 제거하는 self-rewarding 학습 루프 제안
- 반복(iteration)마다 지시 수행 능력뿐 아니라 자기 평가(보상 부여) 능력도 함께 개선됨을 실증
- Llama 2 70B에 3회 반복 적용해 AlpacaEval에서 GPT-4, Claude 2 등을 능가하는 성능 달성

### 이 논문이 중요한 이유
"사람 성능 상한(human ceiling)"에 갇히지 않는 초인적(superhuman) 정렬을 향한 대표적 방향을 제시합니다. 최근의 self-improvement, iterative preference optimization 연구 흐름의 핵심 레퍼런스로, 데이터 확장이 어려운 환경에서 정렬을 확장하는 전략을 이해하는 데 중요합니다.

### 사전 지식
- DPO(Direct Preference Optimization)와 Iterative DPO
- LLM-as-a-Judge 평가 방식과 그 편향 문제
- SFT 시드 데이터와 선호 데이터의 차이

### 관련 논문
- [Direct Preference Optimization / DPO (Rafailov et al., 2023)](https://arxiv.org/abs/2305.18290)
- [Self-Instruct: Aligning Language Models with Self-Generated Instructions (Wang et al., 2022)](https://arxiv.org/abs/2212.10560)

### 실무 적용
사람 선호 데이터 수집이 비싸거나 느린 상황에서, 모델 자체 평가로 선호 데이터를 부트스트랩해 정렬 비용을 낮출 수 있습니다. 다만 self-judge 편향으로 인한 리워드 해킹·품질 붕괴 위험이 있어, 반복 횟수 제한과 외부 벤치마크 검증을 병행하는 운영 전략이 필요합니다.

---

## 추천 읽기 순서
1. **Self-Instruct (Wang, 2022)** — 합성 데이터로 지시 수행 능력을 부여하는 SFT 단계의 출발점
2. **Training a Helpful and Harmless Assistant with RLHF (Bai, 2022)** — SFT 이후 사람 선호로 정렬하는 RLHF 표준 파이프라인
3. **Self-Rewarding Language Models (Yuan, 2024)** — 사람 없이 모델 스스로 보상을 만드는 최신 self-improvement 방향

## 핵심 테이크어웨이
- 정렬은 (1) 인스트럭션 데이터 확보 → (2) 사람 선호 기반 보상 학습 → (3) 자기 개선 루프로 진화해 왔습니다.
- 데이터 병목이 핵심 제약이며, Self-Instruct는 SFT 데이터를, Self-Rewarding은 선호 데이터를 각각 합성으로 돌파합니다.
- RLHF는 helpfulness와 harmlessness의 트레이드오프를 다루는 다목적 최적화 문제이며, 보상 설계가 성패를 좌우합니다.
- 자동화(합성 데이터·self-judge)는 비용을 낮추지만 리워드 해킹·편향 위험을 동반하므로 외부 검증이 필수입니다.

## 다음 토픽과의 연결
다음 토픽인 **LLM Evaluation and Benchmarks**로 자연스럽게 이어집니다. 정렬된 모델이 실제로 얼마나 helpful·harmless한지, self-reward가 실제 성능 향상인지 리워드 해킹인지를 판별하려면 신뢰할 수 있는 평가 벤치마크와 방법론이 반드시 필요하기 때문입니다.
