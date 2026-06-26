# Daily AI Paper Recommendations

> **Date:** 2026-06-27
> **Module:** Module 3 — Machine Learning and Deep Learning
> **Topic:** Optimization and Regularization

---

## Paper 1 (Classic): ADADELTA: An Adaptive Learning Rate Method
- **Authors:** Matthew D. Zeiler
- **Year:** 2012
- **arXiv:** https://arxiv.org/abs/1212.5701
- **PDF:** [./adadelta-zeiler-2012.pdf](./adadelta-zeiler-2012.pdf)
- **Citation Count:** ~15,000+

### 요약
ADADELTA는 학습률(learning rate)을 사람이 직접 튜닝하지 않고 차원별로 자동 적응시키는 1차(first-order) 최적화 기법이다. AdaGrad가 학습이 진행될수록 누적 제곱 그래디언트 때문에 학습률이 0으로 사라지는 문제를 해결하기 위해, 그래디언트 제곱의 "고정 윈도우" 이동 평균과 파라미터 업데이트 제곱의 이동 평균을 함께 사용한다. 그 결과 전역 학습률 하이퍼파라미터 없이도 안정적으로 동작한다.

### 핵심 기여
- AdaGrad의 학습률 단조 감소(monotonic decay) 문제를 지수 이동 평균(EMA) 윈도우로 해결
- 업데이트 단위(units)를 맞추기 위해 분자에 파라미터 변화량의 RMS를 도입 — 별도의 전역 학습률이 필요 없음
- 노이즈가 큰 그래디언트, 다양한 모델 구조·데이터 형태에 강건하며 SGD 대비 추가 계산 비용이 거의 없음

### 이 논문이 중요한 이유
Adam/RMSProp으로 이어지는 적응형 옵티마이저 계보의 핵심 디딤돌이다. "왜 그래디언트 제곱의 이동 평균이 필요한가", "왜 학습률이 사라지면 안 되는가"라는 직관은 오늘날 거의 모든 딥러닝 옵티마이저의 설계 원리이므로, AI 엔지니어가 옵티마이저를 이해하는 출발점으로 필독이다.

### 사전 지식
경사하강법(SGD), 학습률 개념, AdaGrad의 누적 제곱 그래디언트 방식, 지수 이동 평균(EMA), RMS(제곱평균제곱근)의 의미.

### 관련 논문
- [Adaptive Subgradient Methods for Online Learning and Stochastic Optimization (AdaGrad) (Duchi et al., 2011)](https://jmlr.org/papers/v12/duchi11a.html)
- [Adam: A Method for Stochastic Optimization (Kingma & Ba, 2014)](https://arxiv.org/abs/1412.6980)

### 실무 적용
하이퍼파라미터 튜닝 예산이 적은 초기 프로토타이핑, 분산·온라인 학습 환경에서 학습률 스케줄을 직접 설계하기 어려울 때 합리적인 기본값으로 활용된다. PyTorch `torch.optim.Adadelta` 등으로 바로 사용 가능하며, 적응형 옵티마이저의 동작을 디버깅·비교할 때 기준선(baseline)으로 자주 쓰인다.

---

## Paper 2 (Classic): Averaging Weights Leads to Wider Optima and Better Generalization (SWA)
- **Authors:** Pavel Izmailov, Dmitrii Podoprikhin, Timur Garipov, Dmitry P. Vetrov, Andrew Gordon Wilson
- **Year:** 2018
- **arXiv:** https://arxiv.org/abs/1803.05407
- **PDF:** [./swa-averaging-weights-izmailov-2018.pdf](./swa-averaging-weights-izmailov-2018.pdf)
- **Citation Count:** ~3,000+

### 요약
SWA(Stochastic Weight Averaging)는 SGD 학습 궤적 위의 여러 지점에서 얻은 가중치를 단순 평균하는 것만으로 일반화 성능을 끌어올린다. 순환(cyclical) 또는 일정한 학습률로 학습을 이어가며 주기적으로 가중치를 모아 평균하면, 더 평평한(flat) 최솟값으로 수렴해 테스트 정확도가 향상된다. 구현이 매우 간단하고 추가 계산 비용이 거의 없다.

### 핵심 기여
- 단일 모델의 학습 궤적에서 가중치를 평균하면 더 넓고 평평한 최적점에 도달함을 실험·기하학적으로 보임
- Fast Geometric Ensembling을 단일 모델로 근사 — 앙상블 비용 없이 앙상블 효과의 일부를 획득
- CIFAR-10/100, ImageNet의 ResNet·PyramidNet·DenseNet·Shake-Shake 등에서 일관된 일반화 향상

### 이 논문이 중요한 이유
"평평한 최솟값이 더 잘 일반화된다"는 손실 지형(loss landscape) 관점을 실용적인 정규화 기법으로 바꾼 대표 사례다. 가중치 평균이라는 단순한 아이디어가 강력한 정규화 효과를 낸다는 점에서, 최적화와 정규화가 분리되지 않음을 보여준다. EMA·모델 평균은 오늘날 대규모 모델 학습의 표준 기법이라 AI 엔지니어 필독이다.

### 사전 지식
SGD와 학습률 스케줄(순환 학습률), 손실 지형/평평한 최솟값 개념, 앙상블과 일반화 갭, 배치 정규화 통계 재계산의 필요성.

### 관련 논문
- [Sharpness-Aware Minimization for Efficiently Improving Generalization (Foret et al., 2020)](https://arxiv.org/abs/2010.01412)
- [SGDR: Stochastic Gradient Descent with Warm Restarts (Loshchilov & Hutter, 2016)](https://arxiv.org/abs/1608.03983)

### 실무 적용
거의 모든 비전·NLP 모델 학습에서 막판 성능을 손쉽게 끌어올리는 후처리로 쓰인다. PyTorch `torch.optim.swa_utils`로 기본 제공되며, 확산 모델·LLM 학습의 가중치 EMA, 모델 머징(model merging/soup)의 이론적 토대로도 이어진다.

---

## Paper 3 (Recent): The AdEMAMix Optimizer: Better, Faster, Older
- **Authors:** Matteo Pagliardini, Pierre Ablin, David Grangier
- **Year:** 2024 (ICLR 2025 accepted)
- **arXiv:** https://arxiv.org/abs/2409.03137
- **PDF:** [./ademamix-optimizer-pagliardini-2024.pdf](./ademamix-optimizer-pagliardini-2024.pdf)
- **Citation Count:** ~80+ (빠르게 증가 중)

### 요약
AdEMAMix는 Adam의 단일 그래디언트 EMA가 오래된 그래디언트의 기여를 너무 빨리 지워버린다는 문제의식에서 출발한다. 빠르게 반응하는 EMA와 매우 느리게 감쇠하는 EMA, 두 개의 모멘텀을 혼합해 "최근 그래디언트"와 "수만 스텝 전 그래디언트"를 동시에 활용한다. 실험적으로 그래디언트가 수만 스텝 동안 유효함을 보이며, 13억 파라미터 모델에서 101B 토큰으로 학습한 AdEMAMix가 197B 토큰으로 학습한 AdamW와 비슷한 성능을 낸다.

### 핵심 기여
- Adam 모멘텀을 두 개의 EMA 혼합으로 확장 — 최근/과거 그래디언트를 모두 반영
- 과거 그래디언트가 예상보다 훨씬 오래(수만 스텝) 유효함을 실증
- 언어 모델·이미지 분류에서 동일 성능 대비 토큰/스텝 효율을 크게 개선하고 모델 망각(forgetting)을 늦춤

### 이 논문이 중요한 이유
LLM 사전학습 비용이 사실상 옵티마이저 효율에 좌우되는 시대에, "모멘텀을 어떻게 설계하면 같은 데이터로 더 많이 배우는가"라는 질문에 구체적 답을 제시한다. Adam 이후 정체되어 있던 옵티마이저 설계에 다시 활력을 준 2024-2025 대표 연구로, schedule-free·Muon·ADOPT 등과 함께 차세대 옵티마이저 흐름을 이해하는 데 필수다.

### 사전 지식
Adam/AdamW의 1차·2차 모멘텀(EMA) 구조, 모멘텀 계수 β의 의미, LLM 사전학습의 토큰 효율·스케일링, 학습률 스케줄과 워밍업.

### 관련 논문
- [Decoupled Weight Decay Regularization (AdamW) (Loshchilov & Hutter, 2017)](https://arxiv.org/abs/1711.05101)
- [The Road Less Scheduled (Schedule-Free) (Defazio et al., 2024)](https://arxiv.org/abs/2405.15682)

### 실무 적용
대규모 LLM·비전 모델 사전학습에서 동일 compute로 더 높은 성능을 얻거나, 목표 성능까지의 학습 비용을 줄이는 데 직접 적용된다. 다만 β를 매우 크게(예: 0.9999) 쓰는 만큼 워밍업 스케줄과 안정성 튜닝이 필요하며, 장기 학습일수록 이점이 커지므로 사전학습·지속학습(continual training) 시나리오에 특히 유효하다.

---

## 추천 읽기 순서
1. **ADADELTA (2012)** — 적응형 학습률의 직관을 먼저 잡는다. "학습률을 자동으로 정한다"는 아이디어의 원형.
2. **AdEMAMix (2024)** — 같은 적응형 옵티마이저 계보에서 "모멘텀을 어떻게 더 잘 쓸 것인가"로 한 단계 나아간다.
3. **SWA (2018)** — 옵티마이저 자체가 아니라 "가중치를 평균해 일반화를 높인다"는 정규화 관점으로 시야를 넓힌다.

## 핵심 테이크어웨이
- 최적화는 **학습률 적응(ADADELTA)**, **모멘텀 설계(AdEMAMix)**, **해의 평탄성/평균화(SWA)** 라는 세 축으로 이해할 수 있다.
- 적응형 옵티마이저의 핵심은 "그래디언트 정보를 시간축으로 어떻게 누적·감쇠시킬 것인가"이며, 윈도우 EMA(ADADELTA) → 이중 EMA(AdEMAMix)로 정교해졌다.
- 좋은 일반화는 옵티마이저 선택만이 아니라 **수렴한 해의 기하학적 성질(평평함)** 에서 나온다 — SWA는 이를 거의 공짜로 얻는 실용적 방법이다.

## 다음 토픽과의 연결
다음 토픽인 **Transfer Learning and Foundation Models**에서는 이렇게 안정적으로 학습된 가중치를 "어떻게 재사용하고 전이하는가"로 초점이 옮겨간다. 오늘 다룬 옵티마이저·정규화 기법은 대규모 파운데이션 모델을 처음부터 학습하거나(pre-training) 파인튜닝할 때 학습 안정성과 일반화를 좌우하는 토대가 된다.
