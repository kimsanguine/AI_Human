# Daily AI Paper Recommendations

> **Date:** 2026-07-08
> **Module:** Module 6: LLM for Natural Language Generation
> **Topic:** LLM Evaluation and Benchmarks

---

## Paper 1 (Classic): HellaSwag: Can a Machine Really Finish Your Sentence?
- **Authors:** Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, Yejin Choi
- **Year:** 2019
- **arXiv:** https://arxiv.org/abs/1905.07830
- **PDF:** [./hellaswag-zellers-2019.pdf](./hellaswag-zellers-2019.pdf)
- **Citation Count:** ~4,500+

### 요약
HellaSwag는 "이 문장 다음에 무엇이 올까?"를 맞히는 상식 추론(commonsense inference) 벤치마크다. 사람에게는 95% 이상 자명하지만 당시 최고 성능 모델은 48% 미만을 기록할 만큼 어려웠다. 핵심은 Adversarial Filtering(AF)이라는 데이터 수집 기법으로, 판별기가 반복적으로 "기계는 속지만 사람은 알아채는" 오답을 선별해 데이터를 어렵게 만든다.

### 핵심 기여
- Adversarial Filtering(AF)로 "Goldilocks 존"(사람에겐 우스꽝스럽지만 모델은 자주 틀리는 난이도)을 겨냥한 데이터셋 구축
- 예제의 길이와 복잡도를 스케일업하면서 모델이 상식 추론에서 여전히 취약함을 실증
- 사람-모델 성능 격차(>95% vs <48%)를 정량화하여 상식 추론 평가의 새로운 표준 제시

### 이 논문이 중요한 이유
HellaSwag는 GPT-3 이후 거의 모든 LLM 기술 리포트의 표준 평가 항목으로 자리 잡았다. AI 엔지니어가 모델의 "상식" 능력을 비교할 때 가장 먼저 보는 지표 중 하나이며, adversarial 데이터 생성이라는 아이디어는 이후 오염되지 않은(contamination-free) 벤치마크 설계 철학의 뿌리가 되었다.

### 사전 지식
- 다지선다(multiple-choice) 평가 방식과 accuracy 지표
- 언어모델의 perplexity/likelihood 기반 선택지 스코어링
- 데이터 편향(annotation artifacts)과 adversarial 데이터셋의 개념

### 관련 논문
- [SWAG: A Large-Scale Adversarial Dataset for Grounded Commonsense Inference (Zellers et al., 2018)](https://arxiv.org/abs/1808.05326)
- [WinoGrande: An Adversarial Winograd Schema Challenge at Scale (Sakaguchi et al., 2019)](https://arxiv.org/abs/1907.10641)

### 실무 적용
사내 LLM을 파인튜닝하거나 오픈소스 모델을 선정할 때 HellaSwag 점수는 "기본 상식이 무너지지 않았는지"를 확인하는 회귀 테스트로 쓰인다. lm-evaluation-harness 같은 평가 프레임워크에 기본 포함되어 있어 릴리스 게이트 지표로 바로 활용 가능하다.

---

## Paper 2 (Classic): Training Verifiers to Solve Math Word Problems (GSM8K)
- **Authors:** Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, John Schulman
- **Year:** 2021
- **arXiv:** https://arxiv.org/abs/2110.14168
- **PDF:** [./gsm8k-cobbe-2021.pdf](./gsm8k-cobbe-2021.pdf)
- **Citation Count:** ~4,000+

### 요약
이 논문은 초등학교 수준의 수학 문장제 8.5K 문항으로 구성된 GSM8K 데이터셋을 공개하고, 모델이 만든 여러 풀이를 채점하는 "verifier"를 학습시키는 방법을 제안한다. 테스트 시 여러 후보 풀이를 생성한 뒤 verifier가 가장 높게 평가한 풀이를 고르는데, 이 방식이 단순 파인튜닝보다 데이터 증가에 따라 훨씬 잘 스케일함을 보였다.

### 핵심 기여
- 다단계 산술 추론을 요구하는 고품질 벤치마크 GSM8K 공개(현재 LLM 수학 추론의 사실상 표준)
- 생성 후 검증(generate-then-verify) 패러다임 제시 — 후속 self-consistency, best-of-n, RL 보상모델의 원류
- verifier 기반 검증이 finetuning 대비 데이터 스케일링 효율이 높다는 실증적 근거 제공

### 이 논문이 중요한 이유
GSM8K는 Chain-of-Thought, self-consistency, 프로세스 보상 모델(PRM) 등 현대 추론 연구가 성능을 겨루는 공통 무대가 되었다. AI 엔지니어에게는 "추론형 모델을 평가·개선할 때 반드시 이해해야 하는" 기준점이며, verifier 개념은 오늘날 RLHF·RLVR 보상 설계의 기초 사고다.

### 사전 지식
- Chain-of-Thought(단계적 추론) 프롬프팅
- 언어모델 샘플링(temperature, best-of-n)과 다수결(self-consistency)
- 보상모델/검증기(reward model, verifier)의 기본 개념

### 관련 논문
- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models (Wei et al., 2022)](https://arxiv.org/abs/2201.11903)
- [Let's Verify Step by Step (Lightman et al., 2023)](https://arxiv.org/abs/2305.20050)

### 실무 적용
수학·코딩·에이전트 등 정답 검증이 가능한 태스크에서 best-of-n 샘플링 + verifier 조합은 지금도 실서비스 정확도를 끌어올리는 표준 레시피다. GSM8K 점수는 사내 추론 모델의 성능 추적과 데이터 오염 점검의 기본 지표로 활용된다.

---

## Paper 3 (Recent): LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code
- **Authors:** Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, Ion Stoica
- **Year:** 2024
- **arXiv:** https://arxiv.org/abs/2403.07974
- **PDF:** [./livecodebench-jain-2024.pdf](./livecodebench-jain-2024.pdf)
- **Citation Count:** ~600+

### 요약
LiveCodeBench는 LeetCode·AtCoder·CodeForces 대회에서 새 문제를 시간에 따라 지속적으로 수집해 데이터 오염(contamination)을 원천 차단하는 코드 평가 벤치마크다. 코드 생성뿐 아니라 self-repair, 코드 실행, 테스트 출력 예측 등 폭넓은 능력을 측정하며, 문제마다 공개일을 기록해 모델의 학습 컷오프 이후 문제로만 평가하는 "generalization" 측정이 가능하다.

### 핵심 기여
- 문제 공개일 기반 시간 분할로 오염 없는 평가를 실현한 "라이브" 벤치마크 설계
- 코드 생성 외 self-repair/실행/테스트 예측 등 다면적(holistic) 코드 역량 평가
- 18개 base + 34개 instruction-tuned 모델 비교로 HumanEval 과적합 등 기존 벤치마크의 한계를 드러냄

### 이 논문이 중요한 이유
HumanEval 같은 정적 벤치마크가 학습 데이터에 유출되어 점수가 부풀려지는 문제를 정면으로 다룬다. AI 엔지니어에게 "벤치마크 점수를 어떻게 신뢰할 것인가"라는 2024–2025년 핵심 화두(오염, 오버피팅)를 실무적으로 해결하는 방법론을 제시한다.

### 사전 지식
- 코드 생성 평가 지표 pass@k와 HumanEval/MBPP
- 데이터 오염(train-test leakage)과 벤치마크 포화(saturation) 문제
- 실행 기반 평가(테스트 케이스 통과 여부) 파이프라인

### 관련 논문
- [Evaluating Large Language Models Trained on Code (HumanEval / Codex) (Chen et al., 2021)](https://arxiv.org/abs/2107.03374)
- [LiveBench: A Challenging, Contamination-Limited LLM Benchmark (White et al., 2024)](https://arxiv.org/abs/2406.19314)

### 실무 적용
코딩 에이전트나 코드 어시스턴트를 평가할 때 정적 벤치마크만 믿으면 실제 성능을 과대평가하기 쉽다. LiveCodeBench처럼 "모델 컷오프 이후 문제"로 평가하는 롤링 방식은 사내 코드 모델의 실사용 성능을 정직하게 추적하는 실무 표준이 되고 있다.

---

## 추천 읽기 순서
1. **HellaSwag (2019)** — 상식 추론 평가와 adversarial 데이터셋 개념으로 "벤치마크가 왜 어려워야 하는가"를 이해
2. **GSM8K (2021)** — 추론 평가 + generate-then-verify 패러다임으로 현대 추론 연구의 출발점 학습
3. **LiveCodeBench (2024)** — 오염·포화 문제를 해결하는 최신 평가 철학으로 마무리

## 핵심 테이크어웨이
- 좋은 벤치마크는 "사람은 쉽지만 모델은 어렵게" 만드는 것이 핵심이며, adversarial 필터링이 그 대표적 방법이다.
- 정답 검증이 가능한 태스크에서는 생성 후 검증(verifier) 패러다임이 성능·데이터 효율 모두에서 우위를 가진다.
- 정적 벤치마크는 시간이 지나면 오염·포화되므로, 시간 분할·롤링 수집 같은 contamination-free 설계가 2024–2025 평가의 중심이다.

## 다음 토픽과의 연결
다음 토픽(Efficient LLM: Quantization and Distillation)에서는 모델을 경량화하면서 성능 저하를 어떻게 측정·방어할지가 관건이다. 오늘 배운 평가 벤치마크들은 압축·증류된 모델의 품질 회귀를 정량적으로 검증하는 도구로 그대로 이어진다.
