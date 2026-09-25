# Daily AI Paper Recommendations

> **Date:** 2026-09-26
> **Module:** Module 6: LLM for Natural Language Generation
> **Topic:** Instruction Tuning and RLHF

> 이번 사이클에서는 이전에 다룬 InstructGPT, DPO, FLAN, Constitutional AI, LIMA 등과 겹치지 않도록 **"멀티태스크 프롬프트 학습(T0)"**, **"보상 모델 과최적화(Reward Hacking)"**, 그리고 **"순수 RL로 추론 능력을 끌어낸 DeepSeek-R1"** 을 골랐습니다. 질문 하나로 묶으면: *"모델에게 '지시를 따르는 법'과 '잘 생각하는 법'을 가르칠 때, 우리는 무엇을 보상하고 있으며 그 보상은 언제 배신하는가?"*

---

## Paper 1 (Classic): Multitask Prompted Training Enables Zero-Shot Task Generalization
- **Authors:** Victor Sanh, Albert Webson, Colin Raffel, Stephen H. Bach, Lintang Sutawika, et al. (BigScience)
- **Year:** 2021 (ICLR 2022)
- **arXiv:** https://arxiv.org/abs/2110.08207
- **PDF:** [./multitask-prompted-training-t0-sanh-2021.pdf](./multitask-prompted-training-t0-sanh-2021.pdf)
- **Citation Count:** ~1,800+

### 요약
수많은 NLP 데이터셋을 사람이 작성한 다양한 자연어 프롬프트 템플릿(PromptSource)으로 변환하고, 이 멀티태스크 혼합 데이터로 T5 기반 인코더-디코더 모델(T0, 11B)을 파인튜닝했습니다. 학습에 쓰지 않은 태스크에서도 제로샷 성능이 크게 올라, 최대 16배 큰 GPT-3보다 앞서는 경우가 많았습니다.

### 핵심 기여
- **PromptSource**: 170여 개 데이터셋에 대해 2,000개 이상의 프롬프트를 크라우드소싱으로 만든 오픈 도구·자산
- **태스크 단위 hold-out 평가**: "학습에서 본 적 없는 태스크 유형"으로 일반화를 측정하는 엄격한 평가 설계
- **프롬프트 다양성 효과 검증**: 데이터셋당 프롬프트 수가 많을수록 성능 중앙값이 오르고 프롬프트 표현에 대한 민감도(분산)가 줄어듦을 보임

### 이 논문이 중요한 이유
FLAN과 거의 동시에 나온 "Instruction Tuning" 쌍둥이 논문입니다. "모델 크기를 키우는 대신 **지시문 형태의 다양한 데이터**로 가르치면 작은 모델도 일반화된다"는 명제를 오픈 커뮤니티(BigScience)가 재현 가능하게 증명했다는 점에서, 이후 Alpaca·Dolly·Tulu 같은 오픈 SFT 데이터 운동의 출발점이 됩니다.

### 사전 지식
- T5(Text-to-Text Transfer Transformer)와 인코더-디코더 구조
- 제로샷/퓨샷 프롬프팅(GPT-3) 개념
- 멀티태스크 학습의 기본 아이디어

### 관련 논문
- [Finetuned Language Models Are Zero-Shot Learners / FLAN (Wei et al., 2021)](https://arxiv.org/abs/2109.01652)
- [Super-NaturalInstructions (Wang et al., 2022)](https://arxiv.org/abs/2204.07705)
- [Crosslingual Generalization through Multitask Finetuning / BLOOMZ, mT0 (Muennighoff et al., 2022)](https://arxiv.org/abs/2211.01786)

### 실무 적용
사내 SFT 데이터셋을 만들 때 "같은 태스크를 여러 표현의 지시문으로 쓰기"가 곧 강건성입니다. AI 더빙/아바타 제품처럼 사용자가 자연어로 다양하게 요청하는 서비스라면, 하나의 기능(예: "톤을 밝게", "좀 더 활기차게")에 대해 복수의 프롬프트 변형을 학습·평가셋에 넣어 프롬프트 민감도를 낮추는 것이 실전 팁입니다.

---

## Paper 2 (Classic): Scaling Laws for Reward Model Overoptimization
- **Authors:** Leo Gao, John Schulman, Jacob Hilton (OpenAI)
- **Year:** 2022 (ICML 2023)
- **arXiv:** https://arxiv.org/abs/2210.10760
- **PDF:** [./reward-model-overoptimization-gao-2022.pdf](./reward-model-overoptimization-gao-2022.pdf)
- **Citation Count:** ~700+

### 요약
RLHF는 사람 선호를 흉내 낸 "대리(proxy) 보상 모델"을 최적화하는데, 이를 너무 세게 최적화하면 진짜 목표(골드 보상)는 오히려 떨어지는 Goodhart 현상이 생깁니다. 이 논문은 합성 "골드 보상 모델"을 세워 이 과최적화를 정량 측정하고, Best-of-n과 RL 각각에서 골드 점수가 KL 거리에 따라 어떤 함수 형태로 변하는지 스케일링 법칙으로 정리했습니다.

### 핵심 기여
- **합성 골드 RM 실험 설계**: 비싼 사람 라벨 없이 과최적화를 반복 측정할 수 있는 실험 프레임워크
- **함수 형태 발견**: 골드 점수는 Best-of-n에서는 √KL에 대한 2차 형태, RL에서는 로그 형태를 따르며 정점 이후 하락
- **스케일 효과**: 보상 모델 파라미터 수·학습 데이터가 커질수록 과최적화가 늦게·약하게 나타남을 계수로 제시

### 이 논문이 중요한 이유
"보상 점수가 오르는데 실제 품질은 떨어진다"는 Reward Hacking을 감(感)이 아닌 **곡선과 계수**로 보여준 논문입니다. KL 페널티, 조기 종료, RM 앙상블 같은 실무 기법이 왜 필요한지의 이론적 근거가 되며, DPO 계열의 과최적화 연구나 LLM-as-a-Judge의 위험성 논의로도 직결됩니다.

### 사전 지식
- RLHF 파이프라인(SFT → 보상 모델 → PPO)
- KL divergence와 KL 페널티의 역할
- Best-of-n(리젝션) 샘플링
- Goodhart의 법칙: "측정치가 목표가 되면 좋은 측정치가 아니게 된다"

### 관련 논문
- [Learning to summarize from human feedback (Stiennon et al., 2020)](https://arxiv.org/abs/2009.01325)
- [Reward Model Ensembles Help Mitigate Overoptimization (Coste et al., 2023)](https://arxiv.org/abs/2310.02743)
- [Scaling Laws for Reward Model Overoptimization in Direct Alignment Algorithms (Rafailov et al., 2024)](https://arxiv.org/abs/2406.02900)

### 실무 적용
PM 관점에서 가장 실용적인 교훈은 **"자동 평가 지표만 보고 모델을 고르지 말 것"** 입니다. LLM Judge 점수·자체 품질 스코어로 프롬프트나 모델을 반복 튜닝하면 곧 그 지표의 약점을 파고드는 출력(장황함, 특정 키워드 남발)이 나옵니다. 운영 시에는 (1) 튜닝용 지표와 홀드아웃 검증 지표를 분리하고, (2) 주기적으로 사람 평가를 섞으며, (3) 기준 모델 대비 출력 분포가 얼마나 벗어났는지(KL·길이·스타일 드리프트)를 모니터링하는 것이 좋습니다.

---

## Paper 3 (Recent): DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning
- **Authors:** DeepSeek-AI (Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, et al.)
- **Year:** 2025 (Nature, 2025)
- **arXiv:** https://arxiv.org/abs/2501.12948
- **PDF:** [./deepseek-r1-guo-2025.pdf](./deepseek-r1-guo-2025.pdf)
- **Citation Count:** ~5,000+

### 요약
DeepSeek-R1-Zero는 SFT 없이 베이스 모델에 GRPO 기반 대규모 강화학습만 적용해, 긴 사고 사슬·자기 검증·"아하 모먼트" 같은 추론 행동이 자발적으로 나타남을 보였습니다. 가독성 저하·언어 혼용 문제를 해결하기 위해 DeepSeek-R1은 소량의 cold-start 데이터 → 추론 RL → 리젝션 샘플링 SFT → 전 영역 RL의 다단계 파이프라인을 쓰고, 그 결과를 작은 모델로 증류(distillation)해 공개했습니다.

### 핵심 기여
- **"RL만으로도 추론이 창발한다"**: 사람이 쓴 CoT 시연 없이 정답 검증 가능한 보상(rule-based reward)만으로 추론 능력을 끌어올림
- **규칙 기반 보상 설계**: 정확도 보상 + 형식 보상만 사용하고, 신경망 보상 모델은 Reward Hacking 위험 때문에 추론 단계에서 의도적으로 배제
- **추론 능력 증류**: R1의 출력으로 Qwen/Llama 기반 1.5B~70B 모델을 SFT해 작은 모델에서도 강한 추론 성능 달성

### 이 논문이 중요한 이유
InstructGPT 이후 "RLHF = 사람 선호 맞추기"였던 공식을 **"RLVR(검증 가능한 보상 기반 RL) = 생각하는 능력 기르기"** 로 확장한 전환점입니다. 흥미롭게도 Paper 2(과최적화)의 교훈을 직접 반영해 학습형 보상 모델 대신 규칙 기반 보상을 택했다는 점에서 오늘의 세 논문이 하나의 흐름으로 이어집니다. 2025년 이후의 추론 모델·에이전트 학습 논의는 대부분 이 논문을 출발점으로 삼습니다.

### 사전 지식
- PPO와 GRPO(그룹 상대 이점 추정, 가치 모델 없음)의 차이
- Chain-of-Thought 프롬프팅
- 지식 증류(Knowledge Distillation)와 리젝션 샘플링
- MoE 구조(DeepSeek-V3 베이스) 기본 개념

### 관련 논문
- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning / GRPO (Shao et al., 2024)](https://arxiv.org/abs/2402.03300)
- [Kimi k1.5: Scaling Reinforcement Learning with LLMs (Kimi Team, 2025)](https://arxiv.org/abs/2501.12599)
- [DAPO: An Open-Source LLM Reinforcement Learning System at Scale (Yu et al., 2025)](https://arxiv.org/abs/2503.14476)

### 실무 적용
에이전트 제품에서 "정답을 자동으로 검증할 수 있는 태스크"(코드 실행 통과, SQL 결과 일치, 구조화 출력 스키마 검증, 툴 호출 성공 여부)를 찾으면 사람 라벨 없이도 RL/리젝션 샘플링으로 모델을 개선할 수 있습니다. 또한 대형 추론 모델의 출력을 작은 모델로 증류하는 방식은 **추론 비용·지연시간이 핵심 KPI인 B2B SaaS**에서 현실적인 선택지입니다. 단, 긴 사고 사슬은 토큰 비용과 응답 지연을 키우므로 UX 관점에서 "생각 과정 노출 여부"와 "사고 예산(thinking budget)"을 제품 설계에 함께 고려해야 합니다.

---

## 추천 읽기 순서
1. **T0 (Sanh et al., 2021)** — "지시문으로 가르치면 일반화된다"는 Instruction Tuning의 기본 명제부터 확인
2. **Reward Model Overoptimization (Gao et al., 2022)** — RLHF의 핵심 약점인 보상 해킹을 정량적으로 이해
3. **DeepSeek-R1 (2025)** — 앞의 두 교훈(데이터 다양성·보상 신뢰성)이 최신 추론 모델 학습에서 어떻게 결합되는지 확인

## 핵심 테이크어웨이
- **Q. 모델에게 무엇을 가르치는가?** → T0: 태스크 자체보다 "지시를 따르는 형식"과 표현의 다양성이 일반화를 만든다.
- **Q. 보상은 언제 배신하는가?** → Gao: 대리 보상을 세게 밀수록 진짜 품질은 정점 이후 하락한다. 보상 모델을 키우고, KL로 묶고, 지표를 분리하라.
- **Q. 그렇다면 믿을 수 있는 보상은?** → DeepSeek-R1: 검증 가능한 규칙 기반 보상은 해킹에 강하고, 이것만으로도 추론이 창발한다.
- **PM 관점 가설:** "우리 제품에서 자동 검증 가능한 성공 신호는 무엇인가?"를 먼저 정의하는 팀이 후처리 학습(post-training)과 평가 자동화에서 우위를 가진다.

## 다음 토픽과의 연결
다음 토픽은 **LLM Evaluation and Benchmarks** 입니다. 오늘 Gao 논문이 보여준 "지표를 최적화하면 지표가 망가진다"는 문제는 벤치마크 오염·포화 문제와 같은 뿌리를 가집니다. MMLU·HumanEval 같은 벤치마크가 왜 계속 새 버전(MMLU-Pro, LiveBench 등)으로 교체되는지, 그리고 DeepSeek-R1이 사용한 "검증 가능한 정답"이 평가 설계에서는 어떤 의미인지 연결해서 살펴보세요.
