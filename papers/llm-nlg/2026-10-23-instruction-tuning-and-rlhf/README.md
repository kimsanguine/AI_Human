# Daily AI Paper Recommendations

> **Date:** 2026-10-23
> **Module:** Module 6: LLM for Natural Language Generation
> **Topic:** Instruction Tuning and RLHF

> 이번 사이클에서는 이전에 다룬 InstructGPT, DPO, Constitutional AI, HH-RLHF, FLAN, LIMA 등과 겹치지 않도록 "정렬(Alignment)을 무엇으로 측정하고 어떻게 보상하는가"라는 관점에서 새로운 논문을 골랐습니다.

---

## Paper 1 (Classic): A General Language Assistant as a Laboratory for Alignment
- **Authors:** Amanda Askell, Yuntao Bai, Anna Chen, Dawn Drain, Deep Ganguli, Tom Henighan, Andy Jones, Nicholas Joseph, Ben Mann, Nova DasSarma, Nelson Elhage, Zac Hatfield-Dodds, Danny Hernandez, Jackson Kernion, Kamal Ndousse, Catherine Olsson, Dario Amodei, Tom Brown, Jack Clark, Sam McCandlish, Chris Olah, Jared Kaplan
- **Year:** 2021
- **arXiv:** https://arxiv.org/abs/2112.00861
- **PDF:** [./general-language-assistant-askell-2021.pdf](./general-language-assistant-askell-2021.pdf)
- **Citation Count:** ~1,000+

### 요약
"도움이 되고(Helpful), 정직하며(Honest), 해롭지 않은(Harmless)" 어시스턴트, 즉 HHH라는 정렬 목표를 처음으로 명확히 정의한 논문입니다. 단순 프롬프팅만으로도 큰 모델은 정렬 성향이 좋아진다는 점을 보이고, 모방 학습·이진 판별·랭킹 선호 모델링을 비교해 **랭킹 기반 선호 모델(Preference Model)이 가장 잘 스케일한다**는 것을 확인했습니다. 또한 선호 모델 사전학습(PMP)으로 샘플 효율을 높이는 방법을 제안했습니다.

### 핵심 기여
- HHH(Helpful·Honest·Harmless) 프레임워크로 정렬 목표를 정의 — 이후 Anthropic HH-RLHF, Constitutional AI의 출발점
- 모델 크기가 커질수록 "정렬 세금(alignment tax)"이 작아지거나 오히려 이득이 됨을 실험으로 제시
- 랭킹 선호 모델링 > 이진 판별 > 모방 학습 순으로 스케일링이 좋다는 비교 분석
- Preference Model Pre-training(PMP): 공개 데이터(Stack Exchange, Reddit 등)로 선호 모델을 미리 학습해 파인튜닝 데이터 효율 향상

### 이 논문이 중요한 이유
RLHF의 "R(보상)"이 무엇을 목표로 해야 하는지를 정의한 논문입니다. AI 엔지니어가 보상 모델이나 평가 기준(rubric)을 설계할 때, "무엇이 좋은 응답인가"를 Helpful/Honest/Harmless로 분해하는 사고방식은 지금도 그대로 쓰입니다.

### 사전 지식
- GPT 계열 디코더 언어 모델과 스케일링 법칙(Kaplan et al., 2020)
- 쌍대 비교(pairwise comparison)와 Bradley-Terry 모델 개념
- 퓨샷 프롬프팅의 기본

### 관련 논문
- [Training a Helpful and Harmless Assistant with RLHF (Bai et al., 2022)](https://arxiv.org/abs/2204.05862)
- [Constitutional AI: Harmlessness from AI Feedback (Bai et al., 2022)](https://arxiv.org/abs/2212.08073)
- [Deep Reinforcement Learning from Human Preferences (Christiano et al., 2017)](https://arxiv.org/abs/1706.03741)

### 실무 적용
- 고객용 챗봇/에이전트의 **평가 루브릭**을 HHH 축으로 나누어 LLM-as-a-Judge 프롬프트 설계
- 시스템 프롬프트(HHH 프롬프트)만으로 베이스라인 정렬을 확보한 뒤, 부족한 부분만 파인튜닝하는 단계적 전략
- 사내 리뷰/피드백 로그로 선호 모델을 사전학습해 라벨링 비용 절감

---

## Paper 2 (Classic): Let's Verify Step by Step
- **Authors:** Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, Karl Cobbe
- **Year:** 2023
- **arXiv:** https://arxiv.org/abs/2305.20050
- **PDF:** [./lets-verify-step-by-step-lightman-2023.pdf](./lets-verify-step-by-step-lightman-2023.pdf)
- **Citation Count:** ~1,500+

### 요약
최종 답만 보고 보상을 주는 **결과 보상 모델(ORM)** 과 풀이의 각 단계마다 정답 여부를 판단하는 **과정 보상 모델(PRM)** 을 비교한 논문입니다. MATH 데이터셋에서 PRM으로 best-of-N 샘플을 고르면 78%를 풀어 ORM을 크게 앞섰고, 단계별 인간 피드백 80만 개를 담은 PRM800K 데이터셋을 공개했습니다.

### 핵심 기여
- 과정 감독(process supervision)이 결과 감독(outcome supervision)보다 신뢰성 높은 보상 신호임을 대규모로 입증
- 능동 학습(active learning)으로 "그럴듯하지만 틀린" 풀이를 우선 라벨링해 데이터 효율 약 2.6배 향상
- PRM800K(단계별 라벨 80만 개) 공개
- 과정 감독은 "올바른 추론 과정 자체"를 보상하므로 정렬 관점에서도 이점(negative alignment tax)이 있음을 주장

### 이 논문이 중요한 이유
o1, DeepSeek-R1 이후 추론 모델 시대에 "보상을 어디에 줄 것인가"는 핵심 설계 결정입니다. PRM/ORM 구분, verifier 기반 best-of-N, 테스트 타임 컴퓨트 스케일링 같은 개념을 이해하려면 반드시 읽어야 하는 논문입니다.

### 사전 지식
- RLHF의 보상 모델 개념(Paper 1, InstructGPT)
- Chain-of-Thought 프롬프팅
- Best-of-N 샘플링, rejection sampling

### 관련 논문
- [Training Verifiers to Solve Math Word Problems (Cobbe et al., 2021)](https://arxiv.org/abs/2110.14168)
- [Solving math word problems with process- and outcome-based feedback (Uesato et al., 2022)](https://arxiv.org/abs/2211.14275)
- [Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations (Wang et al., 2023)](https://arxiv.org/abs/2312.08935)

### 실무 적용
- 에이전트 워크플로우에서 최종 결과만이 아니라 **각 스텝(툴 호출, 중간 판단)을 검증하는 verifier** 설계
- 코드/수학/데이터 분석 기능에서 여러 후보를 생성하고 PRM 또는 LLM judge로 고르는 best-of-N 파이프라인
- 평가 데이터셋 구축 시 "틀린데 설득력 있는" 케이스를 우선 수집하는 능동 학습 전략

---

## Paper 3 (Recent): DAPO: An Open-Source LLM Reinforcement Learning System at Scale
- **Authors:** Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, et al. (ByteDance Seed, Tsinghua AIR, HKU)
- **Year:** 2025
- **arXiv:** https://arxiv.org/abs/2503.14476
- **PDF:** [./dapo-yu-2025.pdf](./dapo-yu-2025.pdf)
- **Citation Count:** ~700+ (NeurIPS 2025)

### 요약
DeepSeek-R1, o1처럼 강화학습으로 추론 능력을 끌어올리는 방법은 핵심 디테일이 공개되지 않아 재현이 어려웠습니다. DAPO(Decoupled Clip and Dynamic sAmpling Policy Optimization)는 GRPO를 대규모로 돌릴 때 생기는 엔트로피 붕괴·보상 노이즈·학습 불안정 문제를 4가지 기법으로 해결해, Qwen2.5-32B 베이스로 AIME 2024 50점을 달성했고 코드(verl 기반)와 데이터셋을 모두 공개했습니다.

### 핵심 기여
- **Clip-Higher:** 상한 클리핑 범위를 분리(ε_high 확대)해 탐색을 유지하고 엔트로피 붕괴 방지
- **Dynamic Sampling:** 정답률 0% 또는 100%인 프롬프트(그래디언트 0)를 걸러 학습 효율과 안정성 향상
- **Token-level Policy Gradient Loss:** 긴 CoT 응답에서도 토큰 단위로 손실을 균형 있게 반영
- **Overlong Reward Shaping:** 최대 길이를 넘긴 응답에 대한 보상 노이즈를 필터링/소프트 패널티로 완화
- 학습 코드·데이터·레시피 전체 오픈소스 공개

### 이 논문이 중요한 이유
2024~2025년 post-training의 중심은 "선호 학습(DPO 계열)"에서 "검증 가능한 보상 기반 RL(RLVR, GRPO 계열)"로 이동했습니다. DAPO는 이 흐름에서 **실제로 대규모 RL을 안정적으로 돌리기 위한 엔지니어링 디테일**을 가장 구체적으로 공개한 논문이라, 추론 모델을 직접 튜닝하려는 엔지니어에게 사실상의 레시피 역할을 합니다.

### 사전 지식
- PPO의 클리핑 목적함수(Schulman et al., 2017)
- GRPO(DeepSeekMath, 2024)와 그룹 기반 advantage 추정
- 검증 가능한 보상(정답 비교, 단위 테스트 등)의 개념

### 관련 논문
- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning (Shao et al., 2024)](https://arxiv.org/abs/2402.03300)
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL (DeepSeek-AI, 2025)](https://arxiv.org/abs/2501.12948)
- [Understanding R1-Zero-Like Training: A Critical Perspective / Dr. GRPO (Liu et al., 2025)](https://arxiv.org/abs/2503.20783)

### 실무 적용
- 사내 도메인(SQL 생성, 코드, 수식 계산 등)처럼 **정답 검증이 가능한 태스크**에 GRPO/DAPO로 소형 모델 RL 파인튜닝
- 학습 중 엔트로피·응답 길이·정답률 0/100% 비율을 모니터링 지표로 두는 RL 운영 대시보드 설계
- verl, TRL 등 오픈소스 프레임워크에서 DAPO 옵션을 켜고 GRPO 대비 A/B 실험

---

## 추천 읽기 순서
1. **A General Language Assistant as a Laboratory for Alignment** — "무엇을 정렬 목표로 삼는가(HHH)"와 선호 모델의 기본을 먼저 잡습니다.
2. **Let's Verify Step by Step** — 보상을 "결과"가 아닌 "과정"에 주는 아이디어로, 보상 설계의 해상도를 높입니다.
3. **DAPO** — 검증 가능한 보상으로 대규모 RL을 실제로 안정적으로 돌리는 최신 레시피를 확인합니다.

## 핵심 테이크어웨이
- **Q. RLHF에서 가장 어려운 것은 무엇인가?** → 최적화 알고리즘보다 "무엇을 보상할지" 정의하는 것. HHH(2021) → PRM(2023) → 검증 가능한 보상(2025)으로 보상 정의가 점점 구체화되었습니다.
- **Q. 결과만 맞으면 충분한가?** → 아닙니다. 과정 감독은 더 신뢰할 수 있는 신호를 주고, 에이전트 시스템의 스텝별 검증에도 그대로 적용됩니다.
- **Q. 최신 post-training의 실무 포인트는?** → 알고리즘 이름(GRPO, DAPO)보다 클리핑, 샘플 필터링, 길이 처리 같은 **안정화 디테일**이 성능을 좌우합니다.

## 다음 토픽과의 연결
다음 토픽은 **LLM Evaluation and Benchmarks**입니다. 오늘 다룬 보상 모델·PRM·검증기는 결국 "모델을 어떻게 평가하는가"의 문제와 맞닿아 있습니다. HHH 루브릭은 LLM-as-a-Judge 평가로, PRM과 verifier는 AIME·MATH 같은 추론 벤치마크 평가 방식으로 이어지므로, 평가 설계가 곧 보상 설계라는 관점으로 다음 논문들을 읽어보세요.
